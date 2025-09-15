#!/usr/bin/env python3
import argparse
import os
import sys
import time
import csv
import logging
import multiprocessing
from typing import List, Tuple

import yaml
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from generate_descriptions import Config, load_config

# --- Worker Functions for Multiprocessing ---
worker_model = None
worker_cfg = None
worker_categories = None

def init_worker_categorize(cfg_arg: Config, categories_arg: List[str]):
    """Initializer for multiprocessing pool to set up the model and categories."""
    global worker_model, worker_cfg, worker_categories
    worker_cfg = cfg_arg
    worker_categories = categories_arg
    
    genai.configure(api_key=worker_cfg.gemini_api_key)
    try:
        worker_model = genai.GenerativeModel(
            worker_cfg.caption_model,
            generation_config={"response_mime_type": "application/json"},
        )
    except Exception:
        worker_model = genai.GenerativeModel(worker_cfg.caption_model)

def get_category_prompt(description: str, categories: List[str]) -> str:
    """Creates the prompt for the Gemini model."""
    category_list = ", ".join(f'"{cat}"' for cat in categories)
    return (
        "You are a precise data categorization engine. "
        "Analyze the following image description and determine which of the given categories it belongs to. "
        "The description is: '{}'\n\n"
        "The available categories are: [{}].\n\n"
        "Rules:\n"
        "1. Respond with JSON only, like this: {{\"categories\": [\"Category1\", \"Category2\"]}}\n"
        "2. Assign one or at most two relevant categories.\n"
        "3. DO NOT assign the same category twice.\n"
        "4. If NO category is a good fit, respond with: {{\"categories\": [\"unknown\"]}}\n"
        "5. Only use categories from the provided list."
    ).format(description, category_list)

def categorize_description(model: any, description: str, cfg: Config, categories: List[str]) -> str:
    """Uses the Gemini model to categorize a single description."""
    if not description or description.startswith("ERROR:"):
        return "unknown"
        
    prompt = get_category_prompt(description, categories)
    for attempt in range(cfg.max_retries):
        try:
            resp = model.generate_content([prompt], request_options={"timeout": 60})
            
            # Simple parsing for the JSON response
            text_resp = resp.text.strip()
            start = text_resp.find('[')
            end = text_resp.rfind(']')
            if start != -1 and end != -1:
                cat_list_str = text_resp[start+1:end]
                # Remove quotes and split
                assigned_categories = [cat.strip().replace('"', '') for cat in cat_list_str.split(',')]
                # Filter out empty strings, convert to lowercase, remove duplicates, then sort
                unique_categories = sorted(list(set(cat.lower() for cat in assigned_categories if cat)))
                
                if not unique_categories:
                    return "unknown"
                
                return ", ".join(unique_categories)

            return "unknown"
            
        except Exception as e:
            logging.exception("Categorization error (attempt %d): %s", attempt + 1, e)
            if attempt + 1 >= cfg.max_retries:
                return "error"
            delay = cfg.retry_backoff_seconds * (2 ** attempt)
            time.sleep(delay)
    return "error"

def categorize_row_worker(row: List[str]) -> Tuple[str, str, str]:
    """Wrapper function for worker processes to categorize a CSV row."""
    image_name, description = row
    if worker_model is None or worker_cfg is None or worker_categories is None:
        categories = "ERROR: Worker not initialized"
    else:
        categories = categorize_description(worker_model, description, worker_cfg, worker_categories)
    return image_name, description, categories

# --- End Worker Functions ---

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def main():
    parser = argparse.ArgumentParser(description="Categorize image descriptions.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input-csv", default="data/real_descriptions.csv", help="Input CSV with descriptions")
    parser.add_argument("--categories-csv", default="data/categorys/unique_categories.csv", help="CSV with unique categories")
    parser.add_argument("--output-csv", default="reports/real_descriptions_categorized.csv", help="Path to final output CSV file")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    setup_logging()

    # --- Load Config and Categories ---
    if not os.path.exists(args.config):
        logging.error(f"Config file not found at: {args.config}")
        return 1
    cfg = load_config(args.config)

    if not os.path.exists(args.categories_csv):
        logging.error(f"Categories file not found at: {args.categories_csv}")
        return 1
    with open(args.categories_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        unique_categories = [row[0] for row in reader if row]

    if not unique_categories:
        logging.error("No categories found in the categories file.")
        return 1
    logging.info(f"Loaded {len(unique_categories)} categories.")

    # --- Load and filter data to process ---
    if not os.path.exists(args.input_csv):
        logging.error(f"Input description file not found at: {args.input_csv}")
        return 1
        
    rows_to_process = []
    processed_images = set()

    if os.path.exists(args.output_csv):
        with open(args.output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader) # skip header
                for row in reader:
                    if row:
                        processed_images.add(row[0])
            except StopIteration:
                pass # File is empty
    
    with open(args.input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
            for row in reader:
                if row and row[0] not in processed_images:
                    rows_to_process.append(row)
        except StopIteration:
            pass # Input file is empty

    if not rows_to_process:
        logging.info("No new descriptions to categorize. All done.")
        return 0
    logging.info(f"Found {len(rows_to_process)} new descriptions to categorize.")

    # --- Process Data ---
    init_args = (cfg, unique_categories)
    with multiprocessing.Pool(processes=args.num_workers, initializer=init_worker_categorize, initargs=init_args) as pool:
        
        file_exists = os.path.exists(args.output_csv)
        
        with open(args.output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or f.tell() == 0:
                writer.writerow(['image_name', 'description', 'categories'])
            
            total = len(rows_to_process)
            for i, result in enumerate(pool.imap_unordered(categorize_row_worker, rows_to_process), 1):
                writer.writerow(result)
                f.flush()
                logging.info(f"Categorized {i}/{total}: {result[0]}")

    logging.info(f"Categorization complete. Output saved to {args.output_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
