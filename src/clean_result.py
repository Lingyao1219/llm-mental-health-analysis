"""
Clean the mapped JSONL file by removing invalid rows.

This script:
1. Removes rows with non-output llm_impact values (must be: positive, negative, or neutral)
2. Removes rows with null mapped_llm_product
3. Cleans "Others(...)" wrapper from mapped_mental_terms, keeping only content
4. Cleans "Others(...)" in mapped_primary_mental to "Other"
5. Saves the cleaned JSONL file
"""

import json
import argparse
import logging
import re
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)


def clean_mental_terms(val):
    """Remove 'Others(...)' wrapper, keep only content inside parentheses."""
    if not val:
        return val
    val_str = str(val).strip()
    # Match "Others (...)" or "Other (...)"
    match = re.search(r'others?\s*\((.*?)\)', val_str, re.IGNORECASE)
    if match:
        # Return only the content inside ()
        return match.group(1).strip()
    return val_str


def clean_primary_mental(val):
    """Convert 'Others(...)' to 'Other'."""
    if not val:
        return val
    val_str = str(val).strip()
    # Match "Others (...)" - replace with "Other"
    if re.search(r'others\s*\(.*?\)', val_str, re.IGNORECASE):
        return "Other"
    return val_str


def clean_impact(impact):
    """Clean a single impact dictionary."""
    # 1. Check llm_impact is valid
    valid_impacts = ['positive', 'negative', 'neutral']
    if impact.get('llm_impact') not in valid_impacts:
        return None

    # 2. Check mapped_llm_product is not null
    if not impact.get('mapped_llm_product'):
        return None

    # 3. Clean mapped_mental_terms
    if 'mapped_mental_terms' in impact:
        impact['mapped_mental_terms'] = clean_mental_terms(impact['mapped_mental_terms'])

    # 4. Clean mapped_primary_mental
    if 'mapped_primary_mental' in impact:
        impact['mapped_primary_mental'] = clean_primary_mental(impact['mapped_primary_mental'])

    return impact


def clean_mapped_jsonl(input_file, output_file=None):
    """Clean the mapped JSONL file by removing invalid rows and cleaning Others()."""

    # Count total lines
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    logging.info(f"Loading: {input_file}")
    logging.info(f"Total lines: {total_lines}")

    initial_impacts = 0
    removed_impact = 0
    removed_llm = 0
    final_impacts = 0

    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_cleaned.jsonl"

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, total=total_lines, desc="Processing", unit="line"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                initial_impacts += 1

                # Check if this is a valid impact record
                valid_impacts = ['positive', 'negative', 'neutral']
                if data.get('llm_impact') not in valid_impacts:
                    removed_impact += 1
                    continue

                if not data.get('mapped_llm_product'):
                    removed_llm += 1
                    continue

                # Clean mapped_mental_terms
                if 'mapped_mental_terms' in data:
                    data['mapped_mental_terms'] = clean_mental_terms(data['mapped_mental_terms'])

                # Clean mapped_primary_mental
                if 'mapped_primary_mental' in data:
                    data['mapped_primary_mental'] = clean_primary_mental(data['mapped_primary_mental'])

                # Write to output
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                final_impacts += 1

            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line: {e}")
                continue

    total_removed = initial_impacts - final_impacts

    logging.info("=" * 80)
    logging.info(f"Cleaning complete:")
    logging.info(f"  Initial impacts: {initial_impacts}")
    logging.info(f"  Removed (invalid llm_impact): {removed_impact}")
    logging.info(f"  Removed (null mapped_llm_product): {removed_llm}")
    logging.info(f"  Final impacts: {final_impacts}")
    if initial_impacts > 0:
        logging.info(f"  Total removed: {total_removed} ({total_removed/initial_impacts*100:.1f}%)")
    else:
        logging.info(f"  Total removed: {total_removed}")
    logging.info("=" * 80)
    logging.info(f"Cleaned data saved to: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Clean mapped JSONL by removing invalid rows'
    )
    parser.add_argument('-i', '--input', required=True,
                       help='Input JSONL file (mapped results)')
    parser.add_argument('-o', '--output',
                       help='Output JSONL file (default: input_cleaned.jsonl)')

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("Cleaning Mapped JSONL File")
    logging.info("=" * 80)

    clean_mapped_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
