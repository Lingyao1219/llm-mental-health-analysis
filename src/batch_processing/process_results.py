"""
Process batch API results:
1. Load batch output and original data
2. Map results back to all original records (including duplicates)
3. Save in batches to separate files in OUTPUT_FOLDER
"""

import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")


def load_batch_results(base_name: str):
    """Load the batch output file."""
    batch_folder = Path(cfg.BATCH_FOLDER)
    result_path = batch_folder / f"{base_name}_batch_output.jsonl"

    if not result_path.exists():
        logging.error(f"Batch output file not found: {result_path}")
        logging.error("Make sure the batch job is completed and results are downloaded.")
        return None, None

    logging.info(f"Loading batch results from: {result_path}")

    results = {}
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']  # This is our text_hash
            response = result['response']

            if response['status_code'] == 200:
                # Extract the assistant's message content
                content = response['body']['choices'][0]['message']['content']
                try:
                    extracted_data = json.loads(content)
                    results[custom_id] = {
                        "data": extracted_data.get('impacts', []),
                        "error_message": None
                    }
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse response for {custom_id}: {e}")
                    results[custom_id] = {
                        "data": [],
                        "error_message": f"JSON parse error: {str(e)}"
                    }
            else:
                # API error
                error = response['body'].get('error', {})
                results[custom_id] = {
                    "data": [],
                    "error_message": f"API error: {error.get('message', 'Unknown error')}"
                }

    logging.info(f"✅ Loaded {len(results)} batch results")
    return results, result_path


def load_hash_mapping(base_name: str):
    """Load the text hash to original indices mapping."""
    batch_folder = Path(cfg.BATCH_FOLDER)
    mapping_path = batch_folder / f"{base_name}_text_hash_mapping.json"

    if not mapping_path.exists():
        logging.error(f"Mapping file not found: {mapping_path}")
        return None

    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    logging.info(f"✅ Loaded hash mapping for {len(mapping)} unique texts")
    return mapping


def load_original_data(base_name: str):
    """Load the original dataset from JSONL file."""
    # Find the JSONL file matching the base_name
    data_folder = Path('data')
    input_file = data_folder / f"{base_name}.jsonl"

    if not input_file.exists():
        logging.error(f"Original JSONL file not found: {input_file}")
        raise FileNotFoundError(f"JSONL file not found: {input_file}")

    logging.info(f"Loading original data from: {input_file}")
    df = pd.read_json(input_file, lines=True)
    logging.info(f"✅ Loaded {len(df)} original records")
    return df


def reconstruct_full_results(batch_results: dict, hash_mapping: dict, original_df: pd.DataFrame):
    """Map batch results back to all original records."""
    import hashlib

    logging.info("Reconstructing full results with duplicates...")

    # Create hash for each row
    original_df['text_hash'] = original_df[cfg.TEXT_COLUMN].apply(
        lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()
    )

    # Add results to each row
    results_list = []
    processed_hashes = set()

    for idx, row in original_df.iterrows():
        text_hash = row['text_hash']

        # Get the extraction result for this hash
        if text_hash in batch_results:
            extraction = batch_results[text_hash]
        else:
            # Hash not found in results (maybe failed validation)
            extraction = {
                "data": [],
                "error_message": "Text not processed (validation failed or missing)"
            }

        # Track which unique texts we've processed
        if text_hash not in processed_hashes:
            processed_hashes.add(text_hash)

        # Create output record
        output_record = {
            **row.to_dict(),
            f"{cfg.MODEL_NAME}_results": extraction["data"],
            f"{cfg.MODEL_NAME}_impact_count": len(extraction["data"]),
            f"{cfg.MODEL_NAME}_error": extraction["error_message"]
        }

        # Remove the text_hash column from output
        output_record.pop('text_hash', None)

        results_list.append(output_record)

    logging.info(f"✅ Reconstructed {len(results_list)} total records")
    logging.info(f"   Unique texts processed: {len(processed_hashes)}")

    return results_list


def save_in_batches(results: list, base_name: str):
    """Save results in batches to separate files in a subfolder."""
    # Create subfolder based on input filename (without .jsonl extension)
    output_folder = Path(cfg.OUTPUT_FOLDER) / base_name
    output_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving results to folder: {output_folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_num = 0
    total_saved = 0

    for i in range(0, len(results), cfg.BATCH_SIZE):
        batch = results[i:i + cfg.BATCH_SIZE]
        batch_num += 1

        # Create batch filename
        output_file = output_folder / f"batch_{batch_num:03d}_{timestamp}.{cfg.OUTPUT_FORMAT}"

        # Save batch
        if cfg.OUTPUT_FORMAT == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in batch:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

        elif cfg.OUTPUT_FORMAT == 'csv':
            # Flatten for CSV
            flat_batch = []
            for rec in batch:
                flat_rec = {k: v for k, v in rec.items() if not k.startswith(cfg.MODEL_NAME)}
                flat_rec[f"{cfg.MODEL_NAME}_error"] = rec.get(f"{cfg.MODEL_NAME}_error")
                flat_rec[f"{cfg.MODEL_NAME}_impact_count"] = rec.get(f"{cfg.MODEL_NAME}_impact_count")
                flat_rec[f"{cfg.MODEL_NAME}_results"] = json.dumps(rec.get(f"{cfg.MODEL_NAME}_results"))
                flat_batch.append(flat_rec)

            df_batch = pd.DataFrame(flat_batch)
            df_batch.to_csv(output_file, index=False)

        elif cfg.OUTPUT_FORMAT == 'pkl':
            df_batch = pd.DataFrame(batch)
            df_batch.to_pickle(output_file)

        total_saved += len(batch)
        logging.info(f"✅ Saved batch {batch_num}: {output_file} ({len(batch)} records)")

    logging.info(f"\n✅ All results saved! Total records: {total_saved}")
    logging.info(f"   Total batch files: {batch_num}")
    logging.info(f"   Output folder: {output_folder}")


def main():
    """Main function to process batch results."""
    logging.info("=" * 80)
    logging.info("Processing Batch API Results")
    logging.info("=" * 80)

    # Find the most recent batch output file
    batch_folder = Path(cfg.BATCH_FOLDER)
    batch_output_files = list(batch_folder.glob("*_batch_output.jsonl"))

    if not batch_output_files:
        logging.error(f"No batch output files found in {batch_folder}")
        logging.error("Make sure the batch job is completed and results are downloaded.")
        return

    # Use the most recent one
    output_file = sorted(batch_output_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    base_name = output_file.stem.replace("_batch_output", "")

    logging.info(f"Processing results for: {base_name}")

    # Load batch results
    batch_results, result_path = load_batch_results(base_name)
    if batch_results is None:
        return

    # Load mapping
    hash_mapping = load_hash_mapping(base_name)
    if hash_mapping is None:
        return

    # Load original data
    original_df = load_original_data(base_name)

    # Reconstruct full results
    full_results = reconstruct_full_results(batch_results, hash_mapping, original_df)

    # Save in batches to subfolder named after input file
    save_in_batches(full_results, base_name)

    logging.info("\n" + "=" * 80)
    logging.info("Processing Complete!")
    logging.info(f"Base name: {base_name}")
    logging.info(f"Results saved to: {Path(cfg.OUTPUT_FOLDER) / base_name}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
