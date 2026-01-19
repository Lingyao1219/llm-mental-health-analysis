"""
Helper utility functions for data processing.
"""

import pandas as pd
import json
import os
import logging
from typing import Optional

import config as cfg


def validate_input_text(text: str) -> Optional[str]:
    """Validates the input text against a series of basic checks.

    Returns:
        None if valid, error message string if invalid
    """
    if not isinstance(text, str) or not text.strip():
        return "Input validation failed: Empty or non-string input"
    if len(text.strip()) < 10:
        return "Content too short: Text contains fewer than 10 characters"
    if "i am a bot" in text.lower() or "this action was performed automatically" in text.lower():
        return "Content type: Automated message or moderation notice"
    return None


def json_converter(obj):
    """Custom converter to handle special data types like pandas Timestamps."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_jsonl(file_path: str):
    """Load JSONL file and return as list of dictionaries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise


def save_jsonl(data: list, file_path: str):
    """Save list of dictionaries to JSONL file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False, default=json_converter) + '\n')
    except Exception as e:
        logging.error(f"Error saving to {file_path}: {e}")
        raise
