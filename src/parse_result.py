"""
Parse JSONL results into CSV format.

This script:
1. Reads the JSONL results file
2. Expands impacts into separate rows
3. Saves to CSV with all columns
"""

import json
import pandas as pd
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)


def parse_results(input_file: str, output_file: str = None):
    """Parse JSONL results into CSV."""
    input_path = Path(input_file)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_file}")
        return

    # Default output file
    if output_file is None:
        output_file = input_path.with_suffix('.csv')

    logging.info(f"Parsing: {input_file}")

    # Read JSONL
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    logging.info(f"Loaded {len(rows)} records")

    # Expand impacts into separate rows
    expanded_rows = []
    for row in rows:
        impacts = row.get('impacts', [])

        if not impacts or len(impacts) == 0:
            # No impacts - still include the row with null values
            expanded_rows.append({
                'index': row.get('index'),
                'date': row.get('date'),
                'url': row.get('url'),
                'full_text': row.get('full_text'),
                'llm_product': None,
                'llm_impact': None,
                'mental_health_condition': None,
                'user_perspective_category': None,
                'supporting_quote': None,
                'user_value_expressed': None,
                'mapped_llm_product': None,
                'mapped_primary_mental': None,
                'mapped_mental_terms': None,
                'mapped_user_perspective_category': None,
                'error': row.get('error')
            })
        else:
            # One row per impact
            for impact in impacts:
                expanded_rows.append({
                    'index': row.get('index'),
                    'date': row.get('date'),
                    'url': row.get('url'),
                    'full_text': row.get('full_text'),
                    'llm_product': impact.get('llm_product'),
                    'llm_impact': impact.get('llm_impact'),
                    'mental_health_condition': impact.get('mental_health_condition'),
                    'user_perspective_category': impact.get('user_perspective_category'),
                    'supporting_quote': impact.get('supporting_quote'),
                    'user_value_expressed': impact.get('user_value_expressed'),
                    'mapped_llm_product': impact.get('mapped_llm_product'),
                    'mapped_primary_mental': impact.get('mapped_primary_mental'),
                    'mapped_mental_terms': impact.get('mapped_mental_terms'),
                    'mapped_user_perspective_category': impact.get('mapped_user_perspective_category'),
                    'error': row.get('error')
                })

    # Create DataFrame, sort by index, and save
    df = pd.DataFrame(expanded_rows)
    df = df.sort_values('index').reset_index(drop=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    logging.info(f"Saved {len(expanded_rows)} rows to: {output_file}")

    # Summary stats
    impact_count = df[df['llm_product'].notna()].shape[0]
    no_impact_count = df[df['llm_product'].isna()].shape[0]
    error_count = df[df['error'].notna()].shape[0]

    logging.info(f"Summary:")
    logging.info(f"  Total rows: {len(expanded_rows)}")
    logging.info(f"  With impacts: {impact_count}")
    logging.info(f"  No impacts: {no_impact_count}")
    logging.info(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description='Parse JSONL results into CSV')
    parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
    parser.add_argument('-o', '--output', help='Output CSV file (default: same name as input)')

    args = parser.parse_args()
    parse_results(args.input, args.output)


if __name__ == "__main__":
    main()
