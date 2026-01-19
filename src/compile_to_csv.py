"""
Compile batch API results into a single CSV file.

This script:
1. Reads the batch output JSONL file
2. Extracts the impacts from each response
3. Merges with original data (Date, Full Text, Url)
4. Saves to a CSV file with all fields
"""

import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import config as cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def load_batch_output(output_file: str) -> list:
    """Load batch API output from JSONL file."""
    results = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_text_hash_mapping(mapping_file: str) -> dict:
    """Load the text hash mapping."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_batch_results(batch_results: list, text_hash_mapping: dict, original_df: pd.DataFrame) -> list:
    """Parse batch results and extract impacts with original data."""
    all_rows = []

    for result in batch_results:
        custom_id = result.get('custom_id')

        # Get original data from mapping
        if custom_id not in text_hash_mapping:
            logging.warning(f"Custom ID {custom_id} not found in mapping")
            continue

        original_data = text_hash_mapping[custom_id]
        original_indices = original_data.get('original_indices', [])

        if not original_indices:
            logging.warning(f"No original indices for custom ID {custom_id}")
            continue

        # Get the first original record for metadata
        first_idx = original_indices[0]
        first_record = original_df.iloc[first_idx]
        date = first_record.get('Date')
        full_text = first_record.get(cfg.TEXT_COLUMN, '')
        url = first_record.get('Url', '')

        # Extract impacts from response
        response = result.get('response', {})
        body = response.get('body', {})
        choices = body.get('choices', [])

        if not choices:
            logging.warning(f"No choices in response for custom ID {custom_id}")
            continue

        message = choices[0].get('message', {})
        content = message.get('content', '')

        try:
            # Parse the JSON response
            parsed_content = json.loads(content)
            impacts = parsed_content.get('impacts', [])

            if not impacts:
                # No LLM-mental health relationship found
                all_rows.append({
                    'Date': date,
                    'Full Text': full_text,
                    'Url': url,
                    'llm_product': None,
                    'llm_impact': None,
                    'mental_health_condition': None,
                    'user_perspective_category': None,
                    'supporting_quote': None,
                    'user_value_expressed': None
                })
            else:
                # Add a row for each impact
                for impact in impacts:
                    all_rows.append({
                        'Date': date,
                        'Full Text': full_text,
                        'Url': url,
                        'llm_product': impact.get('llm_product'),
                        'llm_impact': impact.get('llm_impact'),
                        'mental_health_condition': impact.get('mental_health_condition'),
                        'user_perspective_category': impact.get('user_perspective_category'),
                        'supporting_quote': impact.get('supporting_quote'),
                        'user_value_expressed': impact.get('user_value_expressed')
                    })

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON for custom ID {custom_id}: {e}")
            logging.error(f"Content: {content[:200]}...")
            continue

    return all_rows


def save_to_csv(rows: list, base_name: str) -> str:
    """Save parsed results to CSV file."""
    # Create output folder if it doesn't exist
    output_folder = Path(cfg.OUTPUT_FOLDER) / base_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_folder / f"compiled_{timestamp}.csv"

    # Convert to DataFrame and save
    df = pd.DataFrame(rows)

    # Convert Date from timestamp to readable format
    if 'Date' in df.columns and df['Date'].notna().any():
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', errors='coerce')

    df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    return str(csv_file)


def main():
    logging.info("="*80)
    logging.info("Compiling Batch Results to CSV")
    logging.info("="*80)

    # Find the batch output file
    batch_folder = Path(cfg.BATCH_FOLDER)

    # List available batch output files
    output_files = list(batch_folder.glob("*_batch_output.jsonl"))

    if not output_files:
        logging.error(f"No batch output files found in {batch_folder}")
        return

    if len(output_files) == 1:
        output_file = output_files[0]
        logging.info(f"Found batch output file: {output_file}")
    else:
        logging.info("Multiple batch output files found:")
        for idx, file in enumerate(output_files, 1):
            logging.info(f"  {idx}. {file.name}")

        choice = int(input(f"\nSelect a file (1-{len(output_files)}): "))
        output_file = output_files[choice - 1]

    # Derive base name from output file
    base_name = output_file.stem.replace("_batch_output", "")

    # Find corresponding files
    mapping_file = batch_folder / f"{base_name}_text_hash_mapping.json"
    original_data_file = Path("data") / f"{base_name}.jsonl"

    if not mapping_file.exists():
        logging.error(f"Mapping file not found: {mapping_file}")
        return

    if not original_data_file.exists():
        logging.error(f"Original data file not found: {original_data_file}")
        return

    logging.info(f"Using mapping file: {mapping_file}")
    logging.info(f"Using original data file: {original_data_file}")

    # Load data
    logging.info("Loading original data...")
    original_df = pd.read_json(original_data_file, lines=True)
    logging.info(f"Loaded {len(original_df)} original records")

    logging.info("Loading batch results...")
    batch_results = load_batch_output(str(output_file))
    logging.info(f"Loaded {len(batch_results)} batch results")

    logging.info("Loading text hash mapping...")
    text_hash_mapping = load_text_hash_mapping(str(mapping_file))
    logging.info(f"Loaded mapping for {len(text_hash_mapping)} unique texts")

    # Parse results
    logging.info("Parsing batch results...")
    rows = parse_batch_results(batch_results, text_hash_mapping, original_df)
    logging.info(f"Extracted {len(rows)} total rows")

    # Save to CSV
    logging.info("Saving to CSV...")
    csv_file = save_to_csv(rows, base_name)

    logging.info("="*80)
    logging.info(f"✅ CSV file created: {csv_file}")
    logging.info(f"   Total rows: {len(rows)}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
