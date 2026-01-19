"""
Apply mapping dictionaries to CSV file - ROBUST VERSION

This script properly handles:
1. Other(...) content extraction and mapping
2. Category: term splitting
3. Duplicate category handling (e.g., "General: General: loneliness")
4. Clean primary categories (no wrappers)
5. Proper CSV quoting to prevent misalignment
6. NaN handling
"""

import pandas as pd
import json
import argparse
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)


def load_mappings(mappings_dir):
    """Load all mapping files from the mappings directory."""
    mappings = {}
    mappings_dir = Path(mappings_dir)

    mapping_files = list(mappings_dir.glob("*_mapping.json"))

    if not mapping_files:
        logging.error(f"No mapping files found in {mappings_dir}")
        return mappings

    for mapping_file in mapping_files:
        field_name = mapping_file.stem.replace('_mapping', '')

        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings[field_name] = json.load(f)

        logging.info(f"Loaded {field_name}: {len(mappings[field_name])} mappings")

    return mappings


def extract_other_content(value):
    """
    Extract content from 'Other (...)' or 'other (...)' format.

    Returns: (content, has_wrapper)
    """
    if pd.isna(value) or not value:
        return None, False

    value_str = str(value).strip()
    match = re.search(r'other\s*\((.*?)\)', value_str, re.IGNORECASE)

    if match:
        content = match.group(1).strip()
        return content, True

    return value_str, False


def map_llm_product(value, mapping_dict):
    """Map LLM product value."""
    if pd.isna(value):
        return None

    value_str = str(value).strip()
    return mapping_dict.get(value_str, value_str)


def map_mental_health(value, mapping_dict):
    """
    Map mental health condition value.

    Returns: (primary, term)
    """
    if pd.isna(value):
        return None, None

    value_str = str(value).strip()

    # Extract content from Other(...) if present
    content, has_wrapper = extract_other_content(value_str)

    # Try to find mapping
    mapped = None

    # Priority 1: If has wrapper, try mapping the content inside
    if has_wrapper and content:
        mapped = mapping_dict.get(content)

    # Priority 2: Try mapping the full value
    if mapped is None:
        mapped = mapping_dict.get(value_str)

    # If no mapping found, use original
    if mapped is None:
        # Return original as both primary and term
        if has_wrapper:
            return value_str, value_str
        return value_str, value_str

    # Process mapped value
    mapped = str(mapped).strip()

    # Check if mapped has "Category: term" format
    if ':' in mapped:
        parts = mapped.split(':', 1)
        primary = parts[0].strip()
        term = parts[1].strip() if len(parts) > 1 else ''

        # Handle duplicate like "General: General: loneliness"
        if term.startswith(primary + ':'):
            term = term[len(primary)+1:].strip()

        # PRIMARY IS ALWAYS CLEAN - NO WRAPPER
        # Add wrapper to term only if:
        # 1. Original had Other(...) wrapper
        # 2. Primary is NOT "Other" (because "Other" category doesn't need wrapper)
        if has_wrapper and primary.lower() != 'other':
            term = f"other({term})"

        return primary, term
    else:
        # No colon - use mapped as both
        # PRIMARY IS ALWAYS CLEAN - NO WRAPPER
        if has_wrapper:
            return mapped, f"other({mapped})"
        return mapped, mapped


def map_user_perspective(value, mapping_dict):
    """Map user perspective category value."""
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    # Extract content from other(...) if present
    content, has_wrapper = extract_other_content(value_str)

    # Try to find mapping
    mapped = None

    # Priority 1: If has wrapper, try mapping the content inside
    if has_wrapper and content:
        mapped = mapping_dict.get(content)

    # Priority 2: Try mapping the full value
    if mapped is None:
        mapped = mapping_dict.get(value_str)

    # If no mapping found, use original
    if mapped is None:
        return value_str

    # Add wrapper if original had it
    if has_wrapper:
        return f"other({mapped})"

    return mapped


def apply_all_mappings(df, mappings):
    """Apply all mappings to the dataframe."""
    df_result = df.copy()

    # Map LLM product
    if 'llm_product' in mappings and 'llm_product' in df_result.columns:
        logging.info("Mapping llm_product...")
        df_result['mapped_llm_product'] = df_result['llm_product'].apply(
            lambda x: map_llm_product(x, mappings['llm_product'])
        )
        before = df_result['llm_product'].nunique()
        after = df_result['mapped_llm_product'].nunique()
        logging.info(f"  {before} unique -> {after} unique")

    # Map mental health condition
    if 'mental_health_condition' in mappings and 'mental_health_condition' in df_result.columns:
        logging.info("Mapping mental_health_condition...")

        results = df_result['mental_health_condition'].apply(
            lambda x: map_mental_health(x, mappings['mental_health_condition'])
        )

        df_result['mapped_primary_mental'] = results.apply(lambda x: x[0])
        df_result['mapped_mental_terms'] = results.apply(lambda x: x[1])

        before = df_result['mental_health_condition'].nunique()
        primary_count = df_result['mapped_primary_mental'].nunique()
        term_count = df_result['mapped_mental_terms'].nunique()
        logging.info(f"  {before} unique -> {primary_count} primary, {term_count} terms")

    # Map user perspective category
    if 'user_perspective_category' in mappings and 'user_perspective_category' in df_result.columns:
        logging.info("Mapping user_perspective_category...")
        df_result['mapped_user_perspective_category'] = df_result['user_perspective_category'].apply(
            lambda x: map_user_perspective(x, mappings['user_perspective_category'])
        )
        before = df_result['user_perspective_category'].nunique()
        after = df_result['mapped_user_perspective_category'].nunique()
        logging.info(f"  {before} unique -> {after} unique")

    return df_result


def main():
    parser = argparse.ArgumentParser(description='Apply mappings to CSV file (v2 - robust)')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-m', '--mappings-dir', default='mappings', help='Mappings directory')
    parser.add_argument('-o', '--output', help='Output CSV file')

    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_mapped.csv"

    logging.info("=" * 80)
    logging.info("Applying Mappings (Robust Version)")
    logging.info("=" * 80)

    # Load data
    logging.info(f"Loading: {args.input}")
    df = pd.read_csv(args.input, encoding='utf-8-sig')
    logging.info(f"Loaded {len(df):,} rows")

    # Load mappings
    mappings = load_mappings(args.mappings_dir)
    if not mappings:
        logging.error("No mappings loaded. Exiting.")
        return

    # Apply mappings
    logging.info("")
    df_mapped = apply_all_mappings(df, mappings)

    # Save outputs
    logging.info("")
    output_path = Path(args.output)

    # Save CSV with proper settings
    logging.info(f"Saving CSV: {args.output}")
    df_mapped.to_csv(
        args.output,
        index=False,
        encoding='utf-8-sig',
        quotechar='"',
        doublequote=True,
        lineterminator='\n'
    )

    # Save JSONL
    jsonl_path = output_path.parent / f"{output_path.stem}.jsonl"
    logging.info(f"Saving JSONL: {jsonl_path}")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, row in df_mapped.iterrows():
            # Convert row to dict, handle NaN properly
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val
            f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')

    logging.info("=" * 80)
    logging.info("COMPLETE!")
    logging.info("=" * 80)

    # Summary
    logging.info("\nMapped columns summary:")
    if 'mapped_llm_product' in df_mapped.columns:
        logging.info(f"  mapped_llm_product: {df_mapped['mapped_llm_product'].nunique()} unique")
    if 'mapped_primary_mental' in df_mapped.columns:
        logging.info(f"  mapped_primary_mental: {df_mapped['mapped_primary_mental'].nunique()} unique")
    if 'mapped_mental_terms' in df_mapped.columns:
        logging.info(f"  mapped_mental_terms: {df_mapped['mapped_mental_terms'].nunique()} unique")
    if 'mapped_user_perspective_category' in df_mapped.columns:
        logging.info(f"  mapped_user_perspective_category: {df_mapped['mapped_user_perspective_category'].nunique()} unique")


if __name__ == "__main__":
    main()
