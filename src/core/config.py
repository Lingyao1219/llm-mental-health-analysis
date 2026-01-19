import os

# --- API and Model Configuration ------------------------------------------------
API_KEY = os.getenv('OPENAI_API_KEY', '')  # Set via environment variable
MODEL_NAME = 'gpt-4.1-mini'
TEMPERATURE = 0.0

# --- Input/Output Configuration -------------------------------------------------
TEXT_COLUMN = 'Full Text'  # Column name in JSONL files containing the text to analyze
OUTPUT_FOLDER = "result/"  # Folder to save results

# --- Processing Configuration ---------------------------------------------------
MAX_WORKERS = 4  # Number of parallel workers