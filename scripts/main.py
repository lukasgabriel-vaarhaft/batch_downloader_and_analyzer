#!/usr/bin/env python3
import argparse
import base64
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import logging
import re

try:
  import google.generativeai as genai
except Exception as e:  # pragma: no cover
  genai = None  # type: ignore

try:
  import hdbscan  # type: ignore
except Exception:
  hdbscan = None  # type: ignore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}


@dataclass
class Config:
  gemini_api_key: str
  caption_model: str
  embedding_model: str
  images_dir: str
  misclassified_dir: Optional[str]
  outputs_dir: str
  image_max_size: int
  batch_size_captions: int
  batch_size_embeddings: int
  per_request_sleep_seconds: float
  max_retries: int
  retry_backoff_seconds: float
  clustering_mode: str
  kmeans_k: Any
  min_cluster_size: int
  min_samples: Any
  tfidf_max_features: int
  tfidf_top_keywords: int
  thumbnail_size: int
  report_examples_per_cluster: int


def load_config(path: str) -> Config:
  with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  data = cfg.get("data", {})
  processing = cfg.get("processing", {})
  clustering = cfg.get("clustering", {})
  tfidf = cfg.get("tfidf", {})
  report = cfg.get("report", {})
  return Config(
    gemini_api_key=cfg.get("gemini_api_key", ""),
    caption_model=cfg.get("caption_model", "gemini-1.5-flash-8b"),
    embedding_model=cfg.get("embedding_model", "text-embedding-004"),
    images_dir=data.get("images_dir", "data/input/baseline"),
    misclassified_dir=data.get("misclassified_dir"),
    outputs_dir=data.get("outputs_dir", "data/outputs"),
    image_max_size=int(processing.get("image_max_size", 768)),
    batch_size_captions=int(processing.get("batch_size_captions", 8)),
    batch_size_embeddings=int(processing.get("batch_size_embeddings", 64)),
    per_request_sleep_seconds=float(processing.get("per_request_sleep_seconds", 0.0)),
    max_retries=int(processing.get("max_retries", 3)),
    retry_backoff_seconds=float(processing.get("retry_backoff_seconds", 2.0)),
    clustering_mode=str(clustering.get("mode", "auto")),
    kmeans_k=clustering.get("kmeans_k", "auto"),
    min_cluster_size=int(clustering.get("min_cluster_size", 15)),
    min_samples=clustering.get("min_samples", "auto"),
    tfidf_max_features=int(tfidf.get("max_features", 3000)),
    tfidf_top_keywords=int(tfidf.get("n_top_keywords", 10)),
    thumbnail_size=int(report.get("thumbnail_size", 256)),
    report_examples_per_cluster=int(report.get("examples_per_cluster", 12)),
  )


def ensure_dirs(outputs_dir: str) -> Dict[str, str]:
  thumbnails_dir = os.path.join(outputs_dir, "thumbnails")
  os.makedirs(outputs_dir, exist_ok=True)
  os.makedirs(thumbnails_dir, exist_ok=True)
  return {"thumbnails": thumbnails_dir}


def setup_logging(outputs_dir: str) -> None:
  os.makedirs(outputs_dir, exist_ok=True)
  log_path = os.path.join(outputs_dir, "run.log")
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
      logging.FileHandler(log_path, mode="a", encoding="utf-8"),
      logging.StreamHandler(sys.stdout),
    ],
  )


def _render_progress(current: int, total: int) -> None:
  try:
    width = 30
    ratio = 0 if total <= 0 else min(max(current / float(total), 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({int(ratio*100)}%)")
    sys.stdout.flush()
  except Exception:
    pass


def _progress_newline() -> None:
  try:
    sys.stdout.write("\n")
    sys.stdout.flush()
  except Exception:
    pass


def list_images(images_dir: str) -> List[str]:
  paths: List[str] = []
  for root, _dirs, files in os.walk(images_dir):
    for name in files:
      ext = os.path.splitext(name)[1].lower()
      if ext in IMAGE_EXTS:
        paths.append(os.path.join(root, name))
  paths.sort()
  return paths


def list_tagged_images(primary_dir: str, secondary_dir: Optional[str]) -> Tuple[List[str], List[str]]:
  base = list_images(primary_dir)
  if secondary_dir and os.path.isdir(secondary_dir):
    mis = list_images(secondary_dir)
  else:
    mis = []
  return base, mis


def load_existing_captions(outputs_dir: str) -> pd.DataFrame:
  path = os.path.join(outputs_dir, "captions.parquet")
  if os.path.exists(path):
    try:
      return pd.read_parquet(path)
    except Exception:
      pass
  return pd.DataFrame(columns=["image_path", "caption_text", "caption_json"])  # type: ignore


def pil_resize_max(image: Image.Image, max_size: int) -> Image.Image:
  w, h = image.size
  scale = min(max_size / max(w, h), 1.0)
  if scale < 1.0:
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)
  return image


def image_to_data_part(image_path: str, max_size: int) -> Dict[str, Any]:
  with Image.open(image_path) as im:
    im = im.convert("RGB")
    im = pil_resize_max(im, max_size)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
  return {"mime_type": "image/jpeg", "data": data}


def get_caption_prompt() -> str:
  return (
    "You are a meticulous vision analyst. Describe the image as STRICT JSON only, "
    "no prose, no code fences. Use this schema: "
    "{\n"
    "  \"summary\": string,\n"
    "  \"objects\": [ { \"name\": string, \"attributes\": [string] } ],\n"
    "  \"scene\": { \"indoor_outdoor\": string, \"environment\": [string] },\n"
    "  \"style\": { \"genre\": [string], \"lighting\": [string], \"colors\": [string] },\n"
    "  \"camera\": { \"shot\": string, \"fov\": string, \"lens\": string },\n"
    "  \"text\": [string],\n"
    "  \"quality\": { \"aesthetic\": string, \"issues\": [string] }\n"
    "}\n"
    "Rules: arrays <= 8 items, concise tokens, use 'unknown' when unsure."
  )


def parse_json_strict(s: str) -> Dict[str, Any]:
  s = s.strip()
  # If response includes extra prose, try to extract the first JSON object
  first = s.find("{")
  last = s.rfind("}")
  if first != -1 and last != -1 and last > first:
    s = s[first : last + 1]
  return json.loads(s)


def caption_one(
  model: Any,
  image_path: str,
  cfg: Config,
) -> Tuple[str, Dict[str, Any]]:
  prompt = get_caption_prompt()
  data_part = image_to_data_part(image_path, cfg.image_max_size)
  for attempt in range(cfg.max_retries):
    try:
      resp = model.generate_content([prompt, data_part], request_options={"timeout": 60})
      text = resp.text if hasattr(resp, "text") else str(resp)
      parsed = parse_json_strict(text)
      summary = parsed.get("summary")
      if not isinstance(summary, str) or not summary:
        # Fallback: make a compact text from fields
        summary = _json_to_compact_text(parsed)
      return summary, parsed
    except Exception as e:  # pragma: no cover
      logging.exception("Caption error for %s (attempt %d): %s", os.path.basename(image_path), attempt + 1, e)
      if attempt + 1 >= cfg.max_retries:
        raise
      # Backoff with server hint if present
      delay = cfg.retry_backoff_seconds * (2 ** attempt)
      try:
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", str(e))
        if m:
          delay = max(delay, float(m.group(1)))
      except Exception:
        pass
      time.sleep(delay)
  # Unreachable
  return "", {}


def _json_to_compact_text(d: Dict[str, Any]) -> str:
  try:
    parts: List[str] = []
    if "objects" in d and isinstance(d["objects"], list):
      obj_tokens = []
      for o in d["objects"]:
        if not isinstance(o, dict):
          continue
        name = o.get("name")
        attrs = o.get("attributes", [])
        if isinstance(attrs, list):
          attrs = ",".join([str(a) for a in attrs[:5]])
        obj_tokens.append(f"{name}:{attrs}")
      if obj_tokens:
        parts.append("objs=" + ";".join(obj_tokens))
    scene = d.get("scene", {})
    if isinstance(scene, dict):
      env = scene.get("environment", [])
      if isinstance(env, list) and env:
        parts.append("env=" + ",".join(env[:5]))
    style = d.get("style", {})
    if isinstance(style, dict):
      genres = style.get("genre", [])
      light = style.get("lighting", [])
      colors = style.get("colors", [])
      for label, arr in [("genre", genres), ("light", light), ("colors", colors)]:
        if isinstance(arr, list) and arr:
          parts.append(f"{label}=" + ",".join(arr[:5]))
    txt = d.get("text", [])
    if isinstance(txt, list) and txt:
      parts.append("text=" + ";".join(txt[:5]))
    return " | ".join(parts)[:1000]
  except Exception:
    return json.dumps(d)[:1000]


def do_captioning(images: List[str], cfg: Config, outputs_dir: str) -> pd.DataFrame:
  if genai is None:
    raise RuntimeError("google-generativeai not installed; check requirements.txt")
  genai.configure(api_key=cfg.gemini_api_key)
  try:
    model = genai.GenerativeModel(
      cfg.caption_model,
      generation_config={"response_mime_type": "application/json"},
    )
  except Exception:
    model = genai.GenerativeModel(cfg.caption_model)

  logging.info("Starting captioning for %d images", len(images))
  existing = load_existing_captions(outputs_dir)
  have: set = set(existing["image_path"].tolist()) if len(existing) else set()

  rows: List[Dict[str, Any]] = []
  todo_paths = [p for p in images if p not in have]
  if not todo_paths:
    return existing

  for idx, img_path in enumerate(todo_paths, 1):
    logging.info("Captioning %d/%d: %s", idx, len(todo_paths), os.path.basename(img_path))
    _render_progress(idx - 1, len(todo_paths))
    try:
      summary, parsed = caption_one(model, img_path, cfg)
      row = {
        "image_path": img_path,
        "caption_text": summary,
        "caption_json": json.dumps(parsed, ensure_ascii=False),
      }
      rows.append(row)
      logging.info("Captioned %d/%d", idx, len(todo_paths))
      _render_progress(idx, len(todo_paths))
    except Exception as e:  # pragma: no cover
      row = {
        "image_path": img_path,
        "caption_text": "",
        "caption_json": json.dumps({"error": str(e)}),
      }
      rows.append(row)
    if cfg.per_request_sleep_seconds > 0:
      time.sleep(cfg.per_request_sleep_seconds)

  df_new = pd.DataFrame(rows)
  df_all = pd.concat([existing, df_new], ignore_index=True)
  df_all.to_parquet(os.path.join(outputs_dir, "captions.parquet"), index=False)
  logging.info("Captioning done. Total rows: %d", len(df_all))
  _progress_newline()
  return df_all


def embed_texts(texts: List[str], cfg: Config) -> np.ndarray:
  if genai is None:
    raise RuntimeError("google-generativeai not installed; check requirements.txt")
  genai.configure(api_key=cfg.gemini_api_key)

  logging.info("Embedding %d captions", len(texts))
  vecs: List[List[float]] = []
  for i, t in enumerate(texts, 1):
    for attempt in range(cfg.max_retries):
      try:
        resp = genai.embed_content(model=cfg.embedding_model, content=t, request_options={"timeout": 30})
        emb = resp["embedding"] if isinstance(resp, dict) else resp.embedding
        vecs.append(emb)
        break
      except Exception as e:  # pragma: no cover
        logging.exception("Embedding error (i=%d, attempt %d): %s", i, attempt + 1, e)
        if attempt + 1 >= cfg.max_retries:
          raise
        delay = cfg.retry_backoff_seconds * (2 ** attempt)
        try:
          m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)\s*\}", str(e))
          if m:
            delay = max(delay, float(m.group(1)))
        except Exception:
          pass
        time.sleep(delay)
    if i % 200 == 0:
      logging.info("Embedded %d/%d", i, len(texts))
    if cfg.per_request_sleep_seconds > 0 and (i % cfg.batch_size_embeddings == 0):
      time.sleep(cfg.per_request_sleep_seconds)
  arr = np.array(vecs, dtype=np.float32)
  logging.info("Embeddings done. Shape: %s", arr.shape)
  return arr


def cluster_embeddings(emb: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
  n = emb.shape[0]
  meta: Dict[str, Any] = {}

  mode = cfg.clustering_mode.lower()
  if mode == "auto":
    use_hdb = hdbscan is not None
  elif mode == "hdbscan":
    use_hdb = hdbscan is not None
  else:
    use_hdb = False

  if use_hdb:
    min_samples = None if cfg.min_samples == "auto" else int(cfg.min_samples)
    min_samples_val = min_samples if min_samples is not None else max(5, int(math.sqrt(n)))
    clusterer = hdbscan.HDBSCAN(
      min_cluster_size=max(5, cfg.min_cluster_size),
      min_samples=min_samples_val,
      metric="euclidean",
      core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(emb)
    meta = {
      "method": "hdbscan",
      "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
      "noise": int(np.sum(labels == -1)),
    }
    logging.info("HDBSCAN clusters: %d, noise: %d", meta["n_clusters"], meta["noise"])
    return labels, meta

  # Fallback: MiniBatchKMeans
  if cfg.kmeans_k == "auto":
    k = max(2, min(1000, int(round(math.sqrt(n)))))
  else:
    k = int(cfg.kmeans_k)
  km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
  labels = km.fit_predict(emb)
  meta = {"method": "kmeans", "n_clusters": int(k)}
  logging.info("KMeans clusters: %d", int(k))
  return labels, meta


def compute_keywords_per_cluster(texts: List[str], labels: np.ndarray, cfg: Config) -> pd.DataFrame:
  df = pd.DataFrame({"text": texts, "cluster": labels})
  # Drop noise cluster -1 for keyword computation
  df_valid = df[df["cluster"] != -1].copy()
  if df_valid.empty:
    return pd.DataFrame(columns=["cluster", "keyword", "score"])  # type: ignore

  vectorizer = TfidfVectorizer(max_features=cfg.tfidf_max_features, stop_words="english")
  X = vectorizer.fit_transform(df_valid["text"].tolist())
  vocab = np.array(vectorizer.get_feature_names_out())

  rows: List[Dict[str, Any]] = []
  for clust_id, g in df_valid.groupby("cluster"):
    idxs = g.index.to_list()
    if not idxs:
      continue
    sub = X[df_valid.index.get_indexer(idxs)]
    scores = np.asarray(sub.mean(axis=0)).ravel()
    top_idx = np.argsort(scores)[::-1][: cfg.tfidf_top_keywords]
    for k_idx in top_idx:
      rows.append({
        "cluster": int(clust_id),
        "keyword": str(vocab[k_idx]),
        "score": float(scores[k_idx]),
      })
  kw_df = pd.DataFrame(rows)
  return kw_df


def _safe_prop(count: int, total: int, alpha: float = 0.5, k: int = 1) -> float:
  # Laplace smoothing
  return (count + alpha) / (total + alpha * k)


def compute_shift_metrics(
  labels: np.ndarray,
  datasets: List[str],
  top_keywords_df: pd.DataFrame,
  texts: List[str],
  tfidf_max_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Cluster shift
  df = pd.DataFrame({"cluster": labels, "dataset": datasets})
  df = df[df["cluster"] != -1]
  counts = df.groupby(["dataset", "cluster"]).size().reset_index(name="count")
  totals = df.groupby("dataset").size().to_dict()
  clusters_list = sorted(df["cluster"].unique().tolist())
  k = max(1, len(clusters_list))
  rows = []
  for c in clusters_list:
    nb = int(counts.query("dataset=='baseline' and cluster==@c")["count"].sum())
    nm = int(counts.query("dataset=='misclassified' and cluster==@c")["count"].sum())
    pb = _safe_prop(nb, totals.get("baseline", 0), k=k)
    pm = _safe_prop(nm, totals.get("misclassified", 0), k=k)
    lift = (pm / max(pb, 1e-9))
    log2_lift = float(np.log2(max(lift, 1e-9)))
    # counts-based log2 lift with Laplace smoothing (+1) to avoid division by zero
    log2_lift_counts = float(np.log2((nm + 1.0) / (nb + 1.0)))
    rows.append({"cluster": int(c), "count_base": nb, "count_mis": nm,
                 "pct_base": pb, "pct_mis": pm,
                 "log2_lift": log2_lift,
                 "log2_lift_counts": log2_lift_counts})
  clusters_shift = pd.DataFrame(rows).sort_values("log2_lift", ascending=False)

  # Keyword shift
  # Build simple counts per dataset using CountVectorizer
  vectorizer = CountVectorizer(max_features=tfidf_max_features, stop_words="english")
  X = vectorizer.fit_transform(texts)
  vocab = np.array(vectorizer.get_feature_names_out())
  ds = np.array(datasets)
  mask_b = ds == "baseline"
  mask_m = ds == "misclassified"
  cb = np.asarray(X[mask_b].sum(axis=0)).ravel().astype(int)
  cm = np.asarray(X[mask_m].sum(axis=0)).ravel().astype(int)
  Nb = int(cb.sum())
  Nm = int(cm.sum())
  kw_rows = []
  for i, w in enumerate(vocab):
    pb = _safe_prop(int(cb[i]), Nb, k=len(vocab))
    pm = _safe_prop(int(cm[i]), Nm, k=len(vocab))
    lift = (pm / max(pb, 1e-12))
    log2_lift = float(np.log2(max(lift, 1e-12)))
    log2_lift_counts = float(np.log2((int(cm[i]) + 1.0) / (int(cb[i]) + 1.0)))
    kw_rows.append({"keyword": w, "count_base": int(cb[i]), "count_mis": int(cm[i]),
                    "pct_base": pb, "pct_mis": pm,
                    "log2_lift": log2_lift,
                    "log2_lift_counts": log2_lift_counts})
  keywords_shift = pd.DataFrame(kw_rows).sort_values("log2_lift", ascending=False)
  return clusters_shift, keywords_shift


def compute_global_keyword_counts(texts: List[str], max_features: int, top_n: int = 20) -> List[Tuple[str, int]]:
  vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
  X = vectorizer.fit_transform(texts)
  vocab = np.array(vectorizer.get_feature_names_out())
  counts = np.asarray(X.sum(axis=0)).ravel().astype(int)
  order = np.argsort(counts)[::-1][:top_n]
  result: List[Tuple[str, int]] = [(str(vocab[i]), int(counts[i])) for i in order]
  logging.info("Top keywords: %s", result[:5])
  return result


def compute_underrepresented_keywords(texts: List[str], max_features: int, top_n: int = 25) -> List[Tuple[str, int]]:
  # Deprecated/unused per user request; keep to avoid import churn
  return []


def slugify(value: str) -> str:
  s = value.lower().strip()
  s = re.sub(r"[^a-z0-9\-\s]", "", s)
  s = re.sub(r"\s+", "-", s)
  s = re.sub(r"-+", "-", s)
  return s or "kw"


def compute_keyword_to_image_paths(
  texts: List[str],
  image_paths: List[str],
  keywords: List[str],
  max_features: int,
) -> Dict[str, List[str]]:
  # Pair texts with their corresponding image paths and drop empty/whitespace-only entries
  valid_pairs = [
    (p, t)
    for p, t in zip(image_paths, texts)
    if isinstance(t, str) and t.strip()
  ]
  if not valid_pairs:
    return {}

  filtered_paths, nonempty_texts = zip(*valid_pairs)
  nonempty_texts = list(nonempty_texts)
  filtered_paths = list(filtered_paths)

  vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
  try:
    X = vectorizer.fit_transform(nonempty_texts)
  except ValueError as e:
    # Handle cases where texts reduce to an empty vocabulary (e.g., all stop words)
    if "empty vocabulary" in str(e).lower():
      return {}
    raise

  vocab = {w: i for i, w in enumerate(vectorizer.get_feature_names_out())}
  mapping: Dict[str, List[str]] = {}
  for w in keywords:
    idx = vocab.get(w)
    if idx is None:
      continue
    col = X[:, idx]
    rows = col.nonzero()[0].tolist()
    mapping[w] = [filtered_paths[r] for r in rows]
  return mapping


def save_thumbnails(image_paths: List[str], labels: np.ndarray, cfg: Config, thumbs_dir: str) -> None:
  for img_path in image_paths:
    try:
      with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((cfg.thumbnail_size, cfg.thumbnail_size), Image.LANCZOS)
        base = os.path.basename(img_path)
        name, _ = os.path.splitext(base)
        out_name = f"{name}.jpg"
        out_path = os.path.join(thumbs_dir, out_name)
        im.save(out_path, format="JPEG", quality=85)
    except Exception:
      continue


def build_html_report(
  images: List[str],
  labels: np.ndarray,
  kw_df: pd.DataFrame,
  datasets: List[str],
  global_counts: List[Tuple[str, int]],
  rare_counts: List[Tuple[str, int]],
  all_texts: List[str],
  clusters_shift: Optional[pd.DataFrame],
  keywords_shift: Optional[pd.DataFrame],
  cfg: Config,
  outputs_dir: str,
) -> None:
  # Select example images per cluster (misclassified only view)
  data = pd.DataFrame({
    "image_path": images,
    "cluster": labels,
    "dataset": datasets,
    "text": all_texts,
  })
  mis_df = data[(data["dataset"] == "misclassified") & (data["cluster"] != -1)].copy()
  ex_per = cfg.report_examples_per_cluster
  # Determine cluster order by count (descending), excluding noise -1
  counts_df = (
    mis_df
    .groupby("cluster")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
  )
  ordered_clusters: List[int] = counts_df["cluster"].astype(int).tolist()

  cluster_cards: List[Dict[str, Any]] = []
  # Prepare per-cluster member lists (for detail pages)
  cluster_to_all_paths: Dict[int, List[str]] = {}
  for c in ordered_clusters:
    members = mis_df[mis_df["cluster"] == c]
    ex_paths = members["image_path"].head(ex_per).tolist()
    all_paths = members["image_path"].tolist()
    kw = kw_df[kw_df["cluster"] == c].sort_values("score", ascending=False)
    keywords = kw["keyword"].head(12).tolist()
    thumbs = [os.path.join("thumbnails", os.path.basename(p).rsplit(".", 1)[0] + ".jpg") for p in ex_paths]
    cluster_cards.append({
      "cluster": c,
      "count": int(len(members)),
      "keywords": ", ".join(keywords),
      "thumbs": thumbs,
    })
    cluster_to_all_paths[int(c)] = all_paths

  # Build similarities bar chart from global keyword counts (misclassified only)
  max_count = max([c for _w, c in global_counts], default=1)
  bars_html = ""
  sim_links: List[Tuple[str, int, str]] = []
  for word, count in global_counts:
    pct = int(round((count / max_count) * 100))
    link = f"similarities/{slugify(word)}.html"
    sim_links.append((word, count, link))
    bars_html += (
      f'<div class="bar">'
      f'<div class="bar-label"><a href="{link}" style="color:inherit;text-decoration:none;">{word}</a></div>'
      f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;"></div></div>'
      f'<div class="bar-count">{count}</div>'
      f'</div>'
    )

  # Underrepresented section removed per user request
  rare_html = ""

  # Bubble scatter removed per user request
  pca_json = "[]"

  # Surging sections (tables)
  surging_clusters_html = ""
  if clusters_shift is not None and not clusters_shift.empty:
    top_clusters = clusters_shift.copy()
    top_clusters = top_clusters[top_clusters["log2_lift"] > 0]
    top_clusters = top_clusters.sort_values("log2_lift", ascending=False).head(12)
    rows_html = []
    for _i, r in top_clusters.iterrows():
      c = int(r["cluster"]) if not pd.isna(r["cluster"]) else -1
      link = f"clusters/cluster_{c}.html"
      hi = " highlight" if float(r["log2_lift"]) >= 1.0 else ""
      # Top keywords for this cluster (linked to keyword pages)
      kw_for_c = kw_df[kw_df["cluster"] == c].sort_values("score", ascending=False)
      kw_links = ", ".join([
        f"<a href=\"similarities/{slugify(str(w))}.html\" style=\"color:inherit;text-decoration:none;\">{str(w)}</a>"
        for w in kw_for_c["keyword"].head(6).tolist()
      ])
      rows_html.append(
        (
          f"<tr>"
          f"<td data-label=\"Cluster\"><a href=\"{link}\" style=\"color:inherit;text-decoration:none;\">{c}</a></td>"
          f"<td data-label=\"Keywords\" class=\"kw\">{kw_links}</td>"
          f"<td data-label=\"Mis\">{int(r['count_mis'])}</td>"
          f"<td data-label=\"Base\">{int(r['count_base'])}</td>"
          f"<td data-label=\"p_mis\">{float(r['pct_mis']):.3f}</td>"
          f"<td data-label=\"p_base\">{float(r['pct_base']):.3f}</td>"
          f"<td data-label=\"log2_lift\" class=\"num{hi}\">{float(r['log2_lift']):.2f}</td>"
          f"</tr>"
        )
      )
    surging_clusters_html = (
      "<h2>Surging clusters (misclassified vs baseline)</h2>"
      + "<table class=\"tbl\">"
      + "<caption>Clusters with highest normalized lift</caption>"
      + "<thead><tr><th>Cluster</th><th>Keywords</th><th>Mis</th><th>Base</th><th>p_mis</th><th>p_base</th><th>log2_lift</th></tr></thead>"
      + "<tbody>" + "".join(rows_html) + "</tbody></table>"
    )

  surging_keywords_html = ""
  if keywords_shift is not None and not keywords_shift.empty:
    top_kw = keywords_shift.copy()
    top_kw = top_kw[top_kw["log2_lift"] > 0]
    top_kw = top_kw.sort_values("log2_lift", ascending=False).head(25)
    rows_html = []
    for _i, r in top_kw.iterrows():
      w = str(r["keyword"]) if not pd.isna(r["keyword"]) else ""
      link = f"similarities/{slugify(w)}.html"
      hi = " highlight" if float(r["log2_lift"]) >= 1.0 else ""
      rows_html.append(
        (
          f"<tr>"
          f"<td data-label=\"Keyword\"><a href=\"{link}\" style=\"color:inherit;text-decoration:none;\">{w}</a></td>"
          f"<td data-label=\"Mis\">{int(r['count_mis'])}</td>"
          f"<td data-label=\"Base\">{int(r['count_base'])}</td>"
          f"<td data-label=\"p_mis\">{float(r['pct_mis']):.3f}</td>"
          f"<td data-label=\"p_base\">{float(r['pct_base']):.3f}</td>"
          f"<td data-label=\"log2_lift\" class=\"num{hi}\">{float(r['log2_lift']):.2f}</td>"
          f"</tr>"
        )
      )
    surging_keywords_html = (
      "<h2>Surging keywords (misclassified vs baseline)</h2>"
      + "<table class=\"tbl\">"
      + "<caption>Keywords with highest normalized lift</caption>"
      + "<thead><tr><th>Keyword</th><th>Mis</th><th>Base</th><th>p_mis</th><th>p_base</th><th>log2_lift</th></tr></thead>"
      + "<tbody>" + "".join(rows_html) + "</tbody></table>"
    )

  html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Image Clusters Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
.thumbs {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
.thumb img {{ width: 100%; height: auto; border-radius: 4px; object-fit: cover; }}
.header {{ display:flex; justify-content: space-between; align-items: baseline; margin-bottom:8px; }}
.bars {{ max-width: 960px; margin-bottom: 24px; }}
.bar {{ display: grid; grid-template-columns: 200px 1fr 80px; align-items: center; gap: 10px; margin: 6px 0; }}
.bar-label {{ font-size: 14px; color: #333; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar-track {{ background: #f0f0f0; border-radius: 4px; height: 16px; position: relative; }}
.bar-fill {{ background: #4a8ef0; height: 100%; border-radius: 4px; }}
.bar-count {{ font-size: 12px; color: #666; text-align: right; }}
.tbl {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
.tbl th, .tbl td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
.tbl th {{ background-color: #f7f7f7; }}
.tbl tr:nth-child(even) {{ background-color: #fafafa; }}
.tbl td.num.highlight {{ background-color: #fffae0; font-weight: 600; }}
@media screen and (max-width: 720px) {{
  .tbl, .tbl thead, .tbl tbody, .tbl th, .tbl td, .tbl tr {{ display:block; }}
  .tbl thead {{ display:none; }}
  .tbl tr {{ margin-bottom: 10px; }}
  .tbl td {{ text-align: right; padding-left: 50%; position: relative; }}
  .tbl td::before {{ content: attr(data-label); position: absolute; left: 12px; text-align: left; font-weight: 600; }}
}}
</style>
</head>
<body>
<h1>Misclassified — Similarities</h1>
<div class=\"bars\">{bars_html}</div>

{surging_clusters_html}
{surging_keywords_html}

<!-- Underrepresented section removed per user request -->

<h2>Clusters</h2>
<div style=\"margin: 8px 0 16px; font-size:14px; color:#333;\">Ranked by number of images</div>
<div class=\"grid\">
"""
  # Ensure clusters subdirectory exists
  clusters_dir = os.path.join(outputs_dir, "clusters")
  os.makedirs(clusters_dir, exist_ok=True)

  for card in cluster_cards:
    html += (
      f'<div class="card">'
      f'<div class="header"><h3><a href="clusters/cluster_{card["cluster"]}.html" style="color:inherit;text-decoration:none;">'
      f'Cluster {card["cluster"]}</a></h3>'
      f'<span>{card["count"]} images</span></div>'
      f'<div><strong>Keywords:</strong> {card["keywords"]}</div>'
    )
    # No thumbnails on main page per user request
    html += "</div>"
  html += "</div></body></html>"

  out_path = os.path.join(outputs_dir, "report.html")
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
  logging.info("Report written: %s", out_path)

  # Write per-cluster detail pages with all thumbnails
  for c, paths in cluster_to_all_paths.items():
    thumbs = [os.path.join("..", "thumbnails", os.path.basename(p).rsplit(".", 1)[0] + ".jpg") for p in paths]
    # keywords for this cluster (top 12) for header context
    kw = kw_df[kw_df["cluster"] == c].sort_values("score", ascending=False)
    keyline = ", ".join(kw["keyword"].head(12).tolist())
    page = (
      "<!DOCTYPE html><html><head><meta charset=\"utf-8\" />"
      "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
      f"<title>Cluster {c} - Images</title>"
      "<style>body{font-family:Arial,sans-serif;margin:24px;}"
      ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;}"
      ".thumb img{width:100%;height:auto;border-radius:6px;object-fit:cover;border:1px solid #ddd;}"
      "a.link{color:#2962ff;text-decoration:none;} a.link:hover{text-decoration:underline;}"
      "</style></head><body>"
      f"<h1>Cluster {c} (misclassified) - {len(thumbs)} images</h1>"
      f"<div style=\"margin:6px 0 16px;color:#333;\"><strong>Keywords:</strong> {keyline}</div>"
      "<div style=\"margin-bottom:12px;\"><a class=\"link\" href=\"../report.html\">← Back to report</a></div>"
      "<div class=\"grid\">"
    )
    for t in thumbs:
      page += f"<div class=\"thumb\"><img src=\"{t}\" loading=\"lazy\"></div>"
    page += "</div></body></html>"

    page_path = os.path.join(clusters_dir, f"cluster_{c}.html")
    with open(page_path, "w", encoding="utf-8") as f:
      f.write(page)
    logging.info("Cluster page written: %s", page_path)

  # Write per-keyword similarity pages (misclassified subset only)
  sim_dir = os.path.join(outputs_dir, "similarities")
  os.makedirs(sim_dir, exist_ok=True)
  # Build mapping from top keywords to images containing them
  top_words = [w for (w, _c) in global_counts]
  mis_paths_for_kw = mis_df["image_path"].tolist()
  mis_texts_for_kw = mis_df["text"].astype(str).tolist()
  kw_to_paths = compute_keyword_to_image_paths(
    texts=mis_texts_for_kw,
    image_paths=mis_paths_for_kw,
    keywords=top_words,
    max_features=cfg.tfidf_max_features,
  )
  for w, paths in kw_to_paths.items():
    thumbs = [os.path.join("..", "thumbnails", os.path.basename(p).rsplit(".", 1)[0] + ".jpg") for p in paths]
    page = (
      "<!DOCTYPE html><html><head><meta charset=\"utf-8\" />"
      "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
      f"<title>Similarity — {w}</title>"
      "<style>body{font-family:Arial,sans-serif;margin:24px;}"
      ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;}"
      ".thumb img{width:100%;height:auto;border-radius:6px;object-fit:cover;border:1px solid #ddd;}"
      "a.link{color:#2962ff;text-decoration:none;} a.link:hover{text-decoration:underline;}"
      "</style></head><body>"
      f"<h1>Similarity: {w} — {len(thumbs)} images</h1>"
      "<div style=\"margin-bottom:12px;\"><a class=\"link\" href=\"../report.html\">← Back to report</a></div>"
      "<div class=\"grid\">"
    )
    for t in thumbs:
      page += f"<div class=\"thumb\"><img src=\"{t}\" loading=\"lazy\"></div>"
    page += "</div></body></html>"
    page_path = os.path.join(sim_dir, f"{slugify(w)}.html")
    with open(page_path, "w", encoding="utf-8") as f:
      f.write(page)
    logging.info("Similarity page written: %s", page_path)


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(description="Local image captioning & clustering pipeline")
  parser.add_argument("--config", required=True, help="Path to config.yaml")
  args = parser.parse_args(argv)

  cfg = load_config(args.config)
  if not cfg.gemini_api_key:
    print("ERROR: gemini_api_key missing in config.yaml", file=sys.stderr)
    return 2

  dirs = ensure_dirs(cfg.outputs_dir)
  setup_logging(cfg.outputs_dir)
  logging.info("Pipeline started")
  # Ensure API key is configured for all subsequent SDK calls
  try:
    os.environ["GOOGLE_API_KEY"] = cfg.gemini_api_key
  except Exception:
    pass
  try:
    genai.configure(api_key=cfg.gemini_api_key)
  except Exception as e:
    logging.exception("Failed to configure Gemini SDK: %s", e)
    return 3
  # Quick API health check
  try:
    _ = genai.embed_content(model=cfg.embedding_model, content="hello", request_options={"timeout": 10})
    logging.info("Gemini API health check OK")
  except Exception as e:
    logging.exception("Gemini API health check failed: %s", e)
    return 3
  base_images, mis_images = list_tagged_images(cfg.images_dir, cfg.misclassified_dir)
  images = base_images + mis_images
  if not images:
    logging.warning("No images found in %s", cfg.images_dir)
    return 0
  # Optional limiter for free-tier runs
  try:
    max_images_env = os.getenv("MAX_IMAGES")
    if max_images_env:
      k = int(max_images_env)
      if k > 0 and len(images) > k:
        logging.info("Limiting run to first %d images due to MAX_IMAGES", k)
        images = images[:k]
  except Exception:
    pass

  # 1) Captioning with resume
  captions_df = do_captioning(images, cfg, cfg.outputs_dir)
  captions_df = captions_df.sort_values("image_path").reset_index(drop=True)
  # Keep only rows for images in this run to avoid length mismatches with embeddings/labels
  images_set = set(images)
  captions_df = captions_df[captions_df["image_path"].isin(images_set)].reset_index(drop=True)
  # Sync images order with captions_df
  images = captions_df["image_path"].tolist()

  # Tag dataset
  def which_dataset(p: str) -> str:
    if p in mis_images:
      return "misclassified"
    return "baseline"
  captions_df["dataset"] = captions_df["image_path"].apply(which_dataset)

  # 2) Embeddings for all captions (empty captions become empty strings)
  texts = [t if isinstance(t, str) else "" for t in captions_df["caption_text"].tolist()]
  emb = embed_texts(texts, cfg)
  np.save(os.path.join(cfg.outputs_dir, "text_embeddings.npy"), emb)

  # 3) Clustering
  labels, meta = cluster_embeddings(emb, cfg)
  if len(labels) != len(captions_df):
    raise RuntimeError(f"Label count {len(labels)} does not match captions {len(captions_df)}")
  clusters_df = pd.DataFrame({"image_path": captions_df["image_path"], "cluster": labels})
  clusters_df.to_csv(os.path.join(cfg.outputs_dir, "clusters.csv"), index=False)

  # 4) Keywords + cluster summary table (ranked by number of images)
  kw_df = compute_keywords_per_cluster(texts, labels, cfg)
  label_stats = (
    clusters_df.groupby("cluster").size().reset_index(name="count").sort_values("count", ascending=False)
  )
  kw_stats = (
    kw_df.groupby(["cluster"]).head(10).groupby("cluster")["keyword"].apply(lambda s: ", ".join(s)).reset_index()
  )
  cluster_labels = label_stats.merge(kw_stats, on="cluster", how="left")
  cluster_labels.rename(columns={"keyword": "keywords"}, inplace=True)
  cluster_labels.to_csv(os.path.join(cfg.outputs_dir, "cluster_labels.csv"), index=False)

  # 4b) If misclassified present, compute shift metrics
  if captions_df["dataset"].nunique() > 1:
    clusters_shift, keywords_shift = compute_shift_metrics(
      labels=labels,
      datasets=captions_df["dataset"].tolist(),
      top_keywords_df=kw_df,
      texts=texts,
      tfidf_max_features=cfg.tfidf_max_features,
    )
    clusters_shift.to_csv(os.path.join(cfg.outputs_dir, "clusters_shift.csv"), index=False)
    keywords_shift.to_csv(os.path.join(cfg.outputs_dir, "keywords_shift.csv"), index=False)

  # 5) Thumbnails + HTML report
  save_thumbnails(images, labels, cfg, dirs["thumbnails"])
  # Global keyword counts for misclassified only view in report
  mis_mask = [d == "misclassified" for d in captions_df["dataset"].tolist()]
  mis_texts = [t for t, m in zip(texts, mis_mask) if m]
  global_counts = compute_global_keyword_counts(mis_texts if mis_texts else texts, cfg.tfidf_max_features, top_n=25)
  rare_counts = compute_underrepresented_keywords(texts, cfg.tfidf_max_features, top_n=25)
  # Save rare keywords CSV
  rare_csv = os.path.join(cfg.outputs_dir, "underrepresented_keywords.csv")
  pd.DataFrame(rare_counts, columns=["keyword", "count"]).to_csv(rare_csv, index=False)
  clusters_shift_df = None
  keywords_shift_df = None
  if os.path.exists(os.path.join(cfg.outputs_dir, "clusters_shift.csv")):
    try:
      clusters_shift_df = pd.read_csv(os.path.join(cfg.outputs_dir, "clusters_shift.csv"))
    except Exception:
      clusters_shift_df = None
  if os.path.exists(os.path.join(cfg.outputs_dir, "keywords_shift.csv")):
    try:
      keywords_shift_df = pd.read_csv(os.path.join(cfg.outputs_dir, "keywords_shift.csv"))
    except Exception:
      keywords_shift_df = None
  build_html_report(
    images=images,
    labels=labels,
    kw_df=kw_df,
    datasets=captions_df["dataset"].tolist(),
    global_counts=global_counts,
    rare_counts=rare_counts,
    all_texts=texts,
    clusters_shift=clusters_shift_df,
    keywords_shift=keywords_shift_df,
    cfg=cfg,
    outputs_dir=cfg.outputs_dir,
  )

  # 6) Save captions parquet again to ensure persistence
  captions_df.to_parquet(os.path.join(cfg.outputs_dir, "captions.parquet"), index=False)

  logging.info("Done. %s", json.dumps({"clustering": meta}))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
