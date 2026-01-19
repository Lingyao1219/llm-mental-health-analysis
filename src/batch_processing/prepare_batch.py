"""
Prepare data for OpenAI Batch API by:
1. Deduplicating unique texts to save budget
2. Formatting into Batch API JSONL format
"""

import pandas as pd
import json
import logging
import hashlib
from pathlib import Path

import config as cfg
import prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")


def hash_text(text: str) -> str:
    """Generate a hash for text to identify duplicates."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_and_deduplicate(input_file: str):
    """Load data from JSONL file and deduplicate by unique text content."""
    logging.info(f"Loading data from: {input_file}")

    # Load JSONL file
    df = pd.read_json(input_file, lines=True)
    logging.info(f"Loaded {len(df)} total records")

    # Add hash column for deduplication using Full Text column
    df['text_hash'] = df[cfg.TEXT_COLUMN].apply(hash_text)

    # Keep track of original indices for each unique text
    unique_texts = {}
    for idx, row in df.iterrows():
        text_hash = row['text_hash']
        if text_hash not in unique_texts:
            unique_texts[text_hash] = {
                'text': row[cfg.TEXT_COLUMN],
                'original_indices': [idx],
                'original_records': [row.to_dict()]
            }
        else:
            unique_texts[text_hash]['original_indices'].append(idx)
            unique_texts[text_hash]['original_records'].append(row.to_dict())

    logging.info(f"Found {len(unique_texts)} unique texts (saved {len(df) - len(unique_texts)} duplicate API calls)")

    return unique_texts, df


def create_batch_request(custom_id: str, text: str) -> dict:
    """Create a single batch API request in the required format."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": cfg.MODEL_NAME,
            "temperature": cfg.TEMPERATURE,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt.SYSTEM_PROMPT},
                {"role": "user", "content": prompt.TASK_PROMPT.format(text=text)}
            ]
        }
    }


def prepare_batch_file(unique_texts: dict, input_filename: str):
    """Prepare the batch input file for OpenAI Batch API."""
    logging.info("Preparing batch API input file...")

    batch_requests = []

    for text_hash, data in unique_texts.items():
        text = data['text']

        # Validate text
        if not isinstance(text, str) or len(text.strip()) < 10:
            logging.warning(f"Skipping invalid text with hash {text_hash}")
            continue

        # Create batch request with hash as custom_id
        request = create_batch_request(text_hash, text)
        batch_requests.append(request)

    # Create batch folder if it doesn't exist
    batch_folder = Path(cfg.BATCH_FOLDER)
    batch_folder.mkdir(exist_ok=True)

    # Create filename based on input file: sample.jsonl -> sample_batch_input.jsonl
    base_name = Path(input_filename).stem  # Gets 'sample' from 'sample.jsonl'
    output_filename = f"{base_name}_batch_input.jsonl"
    output_path = batch_folder / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    logging.info(f"✅ Created batch input file: {output_path}")
    logging.info(f"   Total unique requests: {len(batch_requests)}")
    logging.info(f"   Estimated cost savings: ~50% vs standard API")

    return output_path


def save_mapping(unique_texts: dict, input_filename: str):
    """Save the mapping from text_hash to original indices."""
    mapping = {}
    for text_hash, data in unique_texts.items():
        mapping[text_hash] = {
            'original_indices': data['original_indices'],
            'num_duplicates': len(data['original_indices'])
        }

    # Save to batch folder with input filename: sample_text_hash_mapping.json
    batch_folder = Path(cfg.BATCH_FOLDER)
    batch_folder.mkdir(exist_ok=True)

    base_name = Path(input_filename).stem
    mapping_filename = f"{base_name}_text_hash_mapping.json"
    mapping_path = batch_folder / mapping_filename

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

    logging.info(f"✅ Saved hash mapping to: {mapping_path}")
    return mapping_path


def main():
    """Main function to prepare batch API input."""
    import sys

    logging.info("=" * 80)
    logging.info("Preparing Data for OpenAI Batch API")
    logging.info("=" * 80)

    # Find all JSONL files in data folder
    data_folder = Path('data')
    jsonl_files = sorted(data_folder.glob('*.jsonl'))

    if not jsonl_files:
        logging.error(f"No JSONL files found in {data_folder}/")
        logging.error("Please add JSONL files to the data/ folder first.")
        sys.exit(1)

    # Let user pick a file
    logging.info("\nAvailable JSONL files:")
    for i, file in enumerate(jsonl_files, 1):
        logging.info(f"  {i}. {file.name}")

    if len(jsonl_files) == 1:
        choice = 1
        logging.info(f"\nAutomatically selected: {jsonl_files[0].name}")
    else:
        try:
            choice = int(input(f"\nSelect a file (1-{len(jsonl_files)}): "))
            if choice < 1 or choice > len(jsonl_files):
                logging.error(f"Invalid choice. Please select between 1 and {len(jsonl_files)}")
                sys.exit(1)
        except ValueError:
            logging.error("Invalid input. Please enter a number.")
            sys.exit(1)

    selected_file = jsonl_files[choice - 1]
    logging.info(f"Selected: {selected_file}")

    # Load and deduplicate
    unique_texts, original_df = load_and_deduplicate(str(selected_file))

    # Use the input filename as base name
    input_filename = selected_file.stem

    # Prepare batch file
    batch_file = prepare_batch_file(unique_texts, input_filename)

    # Save mapping for later reconstruction
    mapping_file = save_mapping(unique_texts, input_filename)

    logging.info("\n" + "=" * 80)
    logging.info("Preparation Complete!")
    logging.info(f"Batch files location: {cfg.BATCH_FOLDER}")
    logging.info(f"Next steps:")
    logging.info(f"1. Run: python submit_batch.py")
    logging.info(f"2. Wait for batch completion (check status)")
    logging.info(f"3. Run: python process_results.py")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
