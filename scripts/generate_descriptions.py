#!/usr/bin/env python3
import argparse
import io
import json
import os
import sys
import time
import re
import csv
from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from PIL import Image
from tqdm import tqdm

try:
  import google.generativeai as genai
except ImportError:
  genai = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

@dataclass
class Config:
  gemini_api_key: str
  caption_model: str
  image_max_size: int
  max_retries: int
  retry_backoff_seconds: float
  per_request_sleep_seconds: float

def load_config(path: str) -> Config:
  with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  processing = cfg.get("processing", {})
  return Config(
    gemini_api_key=cfg.get("gemini_api_key", ""),
    caption_model=cfg.get("caption_model", "gemini-1.5-flash-8b"),
    image_max_size=int(processing.get("image_max_size", 768)),
    max_retries=int(processing.get("max_retries", 3)),
    retry_backoff_seconds=float(processing.get("retry_backoff_seconds", 2.0)),
    per_request_sleep_seconds=float(processing.get("per_request_sleep_seconds", 0.0)),
  )

def list_images(images_dir: str) -> List[str]:
  paths: List[str] = []
  for root, _dirs, files in os.walk(images_dir):
    for name in files:
      ext = os.path.splitext(name)[1].lower()
      if ext in IMAGE_EXTS:
        paths.append(os.path.join(root, name))
  paths.sort()
  return paths

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
  first = s.find("{")
  last = s.rfind("}")
  if first != -1 and last != -1 and last > first:
    s = s[first : last + 1]
  return json.loads(s)

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
    return " | ".join(parts)[:1000]
  except Exception:
    return json.dumps(d)[:1000]

def caption_one(
  model: Any,
  image_path: str,
  cfg: Config,
) -> str:
  prompt = get_caption_prompt()
  data_part = image_to_data_part(image_path, cfg.image_max_size)
  for attempt in range(cfg.max_retries):
    try:
      resp = model.generate_content([prompt, data_part], request_options={"timeout": 60})
      text = resp.text if hasattr(resp, "text") else str(resp)
      parsed = parse_json_strict(text)
      summary = parsed.get("summary")
      if not isinstance(summary, str) or not summary:
        summary = _json_to_compact_text(parsed)
      return summary
    except Exception as e:
      logging.exception("Caption error for %s (attempt %d): %s", os.path.basename(image_path), attempt + 1, e)
      if attempt + 1 >= cfg.max_retries:
        return f"ERROR: {e}"
      delay = cfg.retry_backoff_seconds * (2 ** attempt)
      time.sleep(delay)
  return "ERROR: Max retries exceeded"

def caption_one_wrapper(args):
    """Helper function to unpack arguments for caption_one."""
    model, image_path, cfg = args
    return os.path.basename(image_path), caption_one(model, image_path, cfg)

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

def main():
    parser = argparse.ArgumentParser(description="Generate image descriptions using Gemini.")
    parser.add_argument("--dataset", choices=["real", "gen"], required=True,
                        help="Dataset to process (real or gen)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers.")
    args = parser.parse_args()
    
    # Configure paths based on dataset
    if args.dataset == "real":
        images_dir = "REAL"
        output_csv = "data/real_descriptions.csv"
    else:
        images_dir = "GEN"
        output_csv = "data/gen_descriptions.csv"

    setup_logging()

    if not os.path.exists(args.config):
        logging.error(f"Config file not found at: {args.config}")
        return 1
    
    cfg = load_config(args.config)

    if not cfg.gemini_api_key:
        logging.error("gemini_api_key missing in config.yaml")
        return 1
    
    if genai is None:
        logging.error("google-generativeai not installed; check requirements.txt")
        return 1

    genai.configure(api_key=cfg.gemini_api_key)
    try:
        model = genai.GenerativeModel(
            cfg.caption_model,
            generation_config={"response_mime_type": "application/json"},
        )
    except Exception:
        model = genai.GenerativeModel(cfg.caption_model)

    image_paths = list_images(images_dir)
    if not image_paths:
        logging.warning(f"No images found in {images_dir}")
        return 0

    processed_images = set()
    if os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    for row in reader:
                        if row:
                            processed_images.add(row[0])
            logging.info(f"Found {len(processed_images)} already processed images in {output_csv}")
        except (IOError, csv.Error) as e:
            logging.error(f"Could not read existing CSV file: {e}")


    images_to_process = [p for p in image_paths if os.path.basename(p) not in processed_images]
    
    if not images_to_process:
        logging.info("All images have already been processed.")
        return 0

    logging.info(f"Resuming with {len(images_to_process)} images to process using {args.num_workers} workers.")

    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        # Write header only if the file is new/empty or just contains a header
        file_is_empty = os.path.getsize(output_csv) == 0
        if not processed_images and file_is_empty:
            f.write('image_name,description\n')
            f.flush()

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            tasks = [(model, path, cfg) for path in images_to_process]
            future_to_path = {executor.submit(caption_one_wrapper, task): task[1] for task in tasks}

            for future in tqdm(as_completed(future_to_path), total=len(images_to_process), desc="Generating descriptions"):
                try:
                    image_name, description = future.result()
                    
                    s = io.StringIO()
                    csv.writer(s, quoting=csv.QUOTE_ALL).writerow([description])
                    quoted_description = s.getvalue().strip()
                    
                    f.write(f'{image_name},{quoted_description}\n')
                    f.flush()
                except Exception as exc:
                    image_path = future_to_path[future]
                    logging.error(f'{os.path.basename(image_path)} generated an exception: {exc}')

    logging.info(f"Descriptions saved to {output_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
