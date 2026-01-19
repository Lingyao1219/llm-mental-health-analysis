"""
Generate mapping dictionaries using GPT to intelligently group similar terms.

This script:
1. Reads unique values or frequency JSON file
2. Uses GPT to map each term to a canonical term
3. Builds a dictionary where similar terms map to the same canonical term
4. Saves the mapping dictionary
"""

import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

import config as cfg

# Configure logging - disable HTTP request logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)
# Disable HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def get_field_specific_prompt(field_name: str, term: str, canonical_terms: list) -> str:
    """Generate field-specific prompts for different fields."""

    base_task = f"""Current canonical terms: {json.dumps(canonical_terms)}

New term to process: "{term}"

Task: Determine if this new term is semantically similar or synonymous to any existing canonical term.

Rules:
1. If the term is similar/synonymous to an existing canonical term, return that canonical term
2. If the term is distinct and should be its own canonical term, return the new term itself
"""

    if field_name == "mental_health_condition":
        return f"""You are helping to standardize mental health condition terminology for research.

{base_task}

3. Consider these mental health condition categories and their variations. You MUST classify each term into one of these major groups:
   - **General:** ONLY when the user does not specify ANY specific mental health condition (e.g., "mental health", "mental illness")
   - **Schizophrenia spectrum disorders:** schizophrenia, paranoid schizophrenia, schizo
   - **Depressive disorders:** depression, major depressive disorder (MDD), dysthymia, postpartum depression, suicidal ideation, suicidality, suicide
   - **Idiopathic developmental intellectual disability:** slow learner, special education, developmental delay, intellectual disability, developmental disability
   - **Bipolar disorders:** bipolar disorder, manic depression, cyclothymia
   - **Anxiety disorders:** anxiety, panic disorder, social phobia, OCD, obsessive-compulsive disorder, PTSD, post-traumatic stress disorder, GAD
   - **Eating disorders:** eating disorder, anorexia, bulimia, ARFID
   - **Autism spectrum disorders:** autism spectrum disorder (ASD), Asperger's syndrome, autistic
   - **Attention-deficit/hyperactivity disorder:** ADHD, ADD, attention deficit, attention-deficit hyperactivity disorder
   - **Conduct disorders:** conduct disorder, anger issues, aggressive behavior, ODD
   - **Other:** any specific mental health condition not listed above

4. Return the canonical term as a dictionary with the major category as key and specific term as value.
   Examples:
   - If term is "depression" or "MDD" → {{"canonical_term": {{"Depressive disorders": "depression"}}}}
   - If term is "ADHD" or "ADD" → {{"canonical_term": {{"Attention-deficit/hyperactivity disorder": "ADHD"}}}}
   - If term is "suicidal" or "suicidality" → {{"canonical_term": {{"Depressive disorders": "suicidal ideation"}}}}
   - If term is "anxiety" or "GAD" → {{"canonical_term": {{"Anxiety disorders": "anxiety"}}}}

Return ONLY a JSON object in this format:
{{"canonical_term": {{"Major Category Name": "specific term"}}}}"""

    elif field_name == "llm_product":
        return f"""You are helping to standardize LLM product names for research.

{base_task}

3. You MUST categorize each LLM product into one of these groups:
   - **GPT**: ChatGPT, GPT-5.1, GPT-5, GPT-4, GPT-4o, GPT-3.5, GPT, OpenAI, o1, o3, o1-preview, o1-mini, any GPT variant
   - **Claude**: Claude, Claude AI, Anthropic Claude, Claude 3, Claude 2, any Claude variant
   - **Gemini**: Gemini, Google Gemini, Bard, any Gemini variant
   - **Qwen**: Qwen, QwQ, any Qwen variant
   - **Grok**: Grok, xAI Grok, any Grok variant
   - **Llama**: Llama, Meta Llama, Llama 2, Llama 3, any Llama variant
   - **DeepSeek**: DeepSeek, DeepSeek-R1, any DeepSeek variant
   - **Others**: Any LLM not in the above categories, OR if the text mentions multiple different LLMs

4. Return the canonical term as a dictionary with the product family as key and empty string as value.
   Examples:
   - "ChatGPT" or "GPT-4" or "o1" → {{"canonical_term": {{"GPT": ""}}}}
   - "Claude 3" or "Anthropic" → {{"canonical_term": {{"Claude": ""}}}}
   - "Gemini" or "Bard" → {{"canonical_term": {{"Gemini": ""}}}}
   - "Llama 3" → {{"canonical_term": {{"Llama": ""}}}}
   - "ChatGPT and Claude" or "multiple LLMs" → {{"canonical_term": {{"Others": ""}}}}
   - "Perplexity" or "Cohere" → {{"canonical_term": {{"Others": ""}}}}

Return ONLY a JSON object in this format:
{{"canonical_term": {{"Product Family": ""}}}}"""

    elif field_name == "user_perspective_category":
        return f"""You are helping to standardize user perspective categories for research.

{base_task}
    
3. Consider variations like:
   - emotional_support vs emotional support → use "emotional support" (no underscore)
   - over-reliance vs over reliance vs overreliance → use "over-reliance" (hyphenated)
   - misinformation vs mis-information → use "misinformation" (no hyphen)
   - privacy_concern vs privacy concern → use "privacy concern" (no underscore)

4. IMPORTANT Rules:
   - Always use spaces instead of underscores in the canonical term
   - Use concise, clear terminology (keep the canonical term to 5 words or fewer)

   Examples of simplification:
   - "criticism of LLM replacing traditional GP roles in mental health treatment" → "LLM replacing therapists"
   - "announcement of mental health related update and guardrails" → "mental health guardrails"
   - "ethical communication balancing truth and harm prevention" → "ethical communication"
   - "concern about negative labeling by LLMs" → "labeling concern"
   - "limitations_of_AI_in_therapy" → "therapy limitations"

Return ONLY a JSON object with the canonical term (no underscores, max 5 words):
{{"canonical_term": "the canonical term here"}}"""

    else:
        # Default prompt for other fields
        return f"""You are helping to standardize terminology for mental health research.

{base_task}

Return ONLY a JSON object with the canonical term:
{{"canonical_term": "the canonical term here"}}"""


def map_term_to_canonical(term: str, canonical_dict: dict, client: OpenAI, field_name: str) -> str:
    """
    Use GPT to determine if a term should map to an existing canonical term
    or become a new canonical term.

    Args:
        term: The term to map
        canonical_dict: Dictionary of canonical_term -> [list of variations]
        client: OpenAI client
        field_name: Name of the field being mapped

    Returns:
        The canonical term (either existing or the term itself as new canonical)
    """
    # Get list of existing canonical terms
    canonical_terms = list(canonical_dict.keys())

    if not canonical_terms:
        # First term - it becomes canonical
        # For user_perspective_category, convert underscores to spaces
        if field_name == "user_perspective_category":
            return term.replace("_", " ")
        return term

    # Get field-specific prompt
    prompt = get_field_specific_prompt(field_name, term, canonical_terms)

    try:
        response = client.chat.completions.create(
            model=cfg.MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a terminology standardization expert for mental health research."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        canonical = result.get("canonical_term", term)

        # For mental health conditions and llm_product, canonical is a dict with category as key and term as value
        if field_name in ["mental_health_condition", "llm_product"] and isinstance(canonical, dict):
            # Expected format: {"Depressive disorders": "depression"} or {"GPT": ""}
            # Convert to string format: "Depressive disorders: depression" or just "GPT"
            if canonical:
                category = list(canonical.keys())[0]
                term_value = canonical[category]

                # For llm_product with empty value, use only category name
                if field_name == "llm_product" and not term_value:
                    canonical_str = category
                else:
                    canonical_str = f"{category}: {term_value}"

                # Check if this canonical exists or if it's new
                if canonical_str not in canonical_terms and term not in canonical_terms:
                    # New canonical term
                    return canonical_str
                elif canonical_str in canonical_terms:
                    return canonical_str
                else:
                    # Find matching canonical term
                    for existing in canonical_terms:
                        if field_name == "llm_product":
                            # For llm_product, just match the category name
                            if existing == canonical_str or existing.startswith(f"{category}:"):
                                return canonical_str
                        else:
                            if existing.endswith(f": {term_value}") or existing == canonical_str:
                                return existing
                    return canonical_str
            else:
                # Empty dict, use term as canonical
                return term

        # For other fields, canonical is a simple string
        # For user_perspective_category, ensure no underscores and max 5 words
        if field_name == "user_perspective_category":
            canonical = canonical.replace("_", " ")

            # Check if longer than 5 words
            word_count = len(canonical.split())
            if word_count > 5:
                # Ask LLM to shorten it
                shorten_prompt = f"""The term "{canonical}" is too long ({word_count} words).

Please provide a concise version that is 5 words or fewer while preserving the core meaning.

Return ONLY a JSON object:
{{"shortened_term": "the shortened term here (max 5 words)"}}"""

                try:
                    shorten_response = client.chat.completions.create(
                        model=cfg.MODEL_NAME,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You are a terminology standardization expert."},
                            {"role": "user", "content": shorten_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )
                    shorten_result = json.loads(shorten_response.choices[0].message.content)
                    shortened = shorten_result.get("shortened_term", canonical)
                    canonical = shortened.replace("_", " ")
                except Exception:
                    pass

        # Verify the canonical term is either existing or the new term
        if canonical not in canonical_terms and canonical != term:
            # GPT returned something unexpected, default to new term
            return term

        return canonical

    except Exception:
        return term


def build_gpt_mapping(terms: list, client: OpenAI, mapping_file: Path, field_name: str) -> dict:
    """
    Build mapping dictionary using GPT to group similar terms.
    Saves in real-time after each term is processed.

    Args:
        terms: List of terms to process
        client: OpenAI client
        mapping_file: Path to save the mapping file
        field_name: Name of the field being mapped

    Returns:
        Dictionary mapping each term to its canonical term
    """
    # Track canonical terms and their variations
    canonical_dict = {}  # canonical_term -> [list of variations]
    term_to_canonical = {}  # term -> canonical_term

    # Load existing mapping if it exists (resume from where it stopped)
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                term_to_canonical = json.load(f)

            # Rebuild canonical_dict from existing mappings
            for term, canonical in term_to_canonical.items():
                if canonical not in canonical_dict:
                    canonical_dict[canonical] = []
                if term != canonical:
                    canonical_dict[canonical].append(term)

            already_processed = len(term_to_canonical)
            logging.info(f"Resuming from existing mapping file: {already_processed} terms already processed")
        except Exception as e:
            logging.warning(f"Could not load existing mapping file: {e}. Starting fresh.")

    logging.info("Building mapping using GPT (saving in real-time)...")

    for term in tqdm(terms, desc="Processing terms", unit="term"):
        # Skip if already processed
        if term in term_to_canonical:
            continue

        # Get canonical term for this term
        canonical = map_term_to_canonical(term, canonical_dict, client, field_name)

        # For user_perspective_category, ensure no underscores in canonical
        if field_name == "user_perspective_category":
            canonical = canonical.replace("_", " ")

        # Update mappings
        term_to_canonical[term] = canonical

        # Add to canonical dictionary
        if canonical not in canonical_dict:
            canonical_dict[canonical] = []

        # If term is different from canonical, add as variation
        if term != canonical:
            canonical_dict[canonical].append(term)

        # Save to file in real-time
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(term_to_canonical, f, indent=2, ensure_ascii=False)

    # Show summary
    logging.info(f"\n" + "=" * 80)
    logging.info(f"Mapping summary:")
    logging.info(f"  Total unique terms: {len(terms)}")
    logging.info(f"  Canonical terms: {len(canonical_dict)}")

    # Show groups with variations
    has_variations = False
    for canonical, variations in canonical_dict.items():
        if variations:
            has_variations = True
            break

    if has_variations:
        logging.info(f"\nGroups with variations:")
        for canonical, variations in canonical_dict.items():
            if variations:
                logging.info(f"  '{canonical}' <- {variations}")
    else:
        logging.info("\nNo variations found - all terms are unique")

    return term_to_canonical


def main():
    parser = argparse.ArgumentParser(description='Generate mapping dictionaries using GPT')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file (unique_values or frequency)')
    parser.add_argument('-o', '--output-dir', default='mappings', help='Output directory for mapping file')

    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("Generating Field Mapping using GPT")
    logging.info("=" * 80)

    # Setup directories
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check input file exists
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        return

    # Extract field name from filename
    field_name = input_file.stem.replace('_unique_values', '').replace('_frequency', '')

    logging.info(f"Processing field: {field_name}")
    logging.info(f"Input file: {input_file}")

    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if it's a list (unique_values) or dict (frequency)
    if isinstance(data, list):
        terms = data
        logging.info(f"Loaded {len(terms)} unique values\n")
    elif isinstance(data, dict):
        terms = list(data.keys())
        logging.info(f"Loaded {len(terms)} unique terms from frequency dict\n")
    else:
        logging.error(f"Invalid JSON format. Expected list or dict.")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=cfg.API_KEY)

    # Prepare output file path
    mapping_file = output_dir / f"{field_name}_mapping.json"

    # Build mapping with GPT (saves in real-time)
    build_gpt_mapping(terms, client, mapping_file, field_name)

    logging.info("\n" + "=" * 80)
    logging.info(f"Mapping saved to: {mapping_file}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
