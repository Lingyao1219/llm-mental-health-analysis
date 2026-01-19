"""
Direct API processing with real-time JSONL saving and parallel workers.

This script:
1. Reads input JSONL file from data/ folder
2. Processes texts in parallel using multiple workers
3. Saves results in real-time to JSONL (one text = one line)
4. Shows progress bar
"""

import json
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI

from . import config as cfg
from . import prompt

# Configure logging - disable HTTP request logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)
# Disable HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def process_single_row(row_data):
    """Process a single row (thread-safe)."""
    idx, row = row_data
    text = row.get(cfg.TEXT_COLUMN, '')

    # Validate input
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return {
            "index": idx,
            "date": str(row.get('Date', '')),
            "url": row.get('Url', ''),
            "full_text": text,
            "impacts": [],
            "error": "Invalid or empty text"
        }

    # Each worker creates its own client
    try:
        client = OpenAI(api_key=cfg.API_KEY)

        response = client.chat.completions.create(
            model=cfg.MODEL_NAME,
            temperature=cfg.TEMPERATURE,
            messages=[
                {"role": "system", "content": prompt.SYSTEM_PROMPT},
                {"role": "user", "content": prompt.TASK_PROMPT.format(text=text)}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)
        impacts = parsed.get("impacts", [])

        return {
            "index": idx,
            "date": str(row.get('Date', '')),
            "url": row.get('Url', ''),
            "full_text": text,
            "impacts": impacts,
            "error": None
        }

    except Exception as e:
        return {
            "index": idx,
            "date": str(row.get('Date', '')),
            "url": row.get('Url', ''),
            "full_text": text,
            "impacts": [],
            "error": f"{type(e).__name__}: {str(e)[:100]}"
        }


def main():
    logging.info("=" * 80)
    logging.info("Direct API Processing with Real-time JSONL Saving")
    logging.info("=" * 80)

    # Find input files
    data_folder = Path("data")
    input_files = list(data_folder.glob("*.jsonl"))

    if not input_files:
        logging.error(f"No JSONL files found in {data_folder}")
        return

    # Select input file
    if len(input_files) == 1:
        input_file = input_files[0]
        logging.info(f"Found input file: {input_file}")
    else:
        logging.info("Multiple input files found:")
        for idx, file in enumerate(input_files, 1):
            logging.info(f"  {idx}. {file.name}")
        choice = int(input(f"\nSelect a file (1-{len(input_files)}): "))
        input_file = input_files[choice - 1]

    # Setup output
    base_name = input_file.stem
    output_folder = Path(cfg.OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{base_name}_results.jsonl"

    logging.info(f"Input: {input_file}")
    logging.info(f"Output: {output_file}")

    # Load data
    df = pd.read_json(input_file, lines=True)
    logging.info(f"Loaded {len(df)} records")
    logging.info(f"Using {cfg.MAX_WORKERS} parallel workers")

    # Process with parallel workers
    success_count = 0
    error_count = 0
    write_lock = threading.Lock()

    with open(output_file, 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=cfg.MAX_WORKERS) as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_row, (idx, row)): idx
                      for idx, row in df.iterrows()}

            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(df), desc="Processing"):
                result = future.result()

                # Count successes/errors
                if result["error"]:
                    error_count += 1
                else:
                    success_count += 1

                # Thread-safe writing
                with write_lock:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

    logging.info("=" * 80)
    logging.info(f"Processing complete!")
    logging.info(f"  Output: {output_file}")
    logging.info(f"  Total: {len(df)}")
    logging.info(f"  Success: {success_count}")
    logging.info(f"  Errors: {error_count}")
    logging.info("=" * 80)
    logging.info(f"Run 'python parse_result.py' to parse results into CSV")


if __name__ == "__main__":
    main()
