# LLM-Mental Health Bidirectional Impact Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** This work has been accepted to the **2026 ACM Web Conference (WWW '26)**.

## Overview

This repository contains the code and analysis pipeline for studying the **bidirectional relationship** between Large Language Models (LLMs) and mental health. We analyze social media data to understand:

1. **Positive Impact**: How people use LLMs (ChatGPT, Claude, Gemini, etc.) for mental health support, therapy, and coping
2. **Negative Impact**: How LLMs might contribute to mental health challenges (anxiety, addiction, misinformation, etc.)

The research framework to implement the study is presented below. 
<img width="1256" height="624" alt="framework" src="https://github.com/user-attachments/assets/b5055d3d-6e77-4141-b213-1cefda14f3b8" />



## Repository Structure

```
llm-mental/
├── src/                          # Source code
│   ├── core/                     # Core processing modules
│   │   ├── config.py            # Configuration settings
│   │   ├── prompt.py            # LLM prompts for extraction
│   │   ├── utils.py             # Utility functions
│   │   └── main.py              # Main processing script
│   ├── batch_processing/         # OpenAI Batch API scripts
│   │   ├── prepare_batch.py     # Prepare batch requests
│   │   ├── submit_batch.py      # Submit to OpenAI Batch API
│   │   └── process_results.py   # Process batch results
│   ├── analysis/                 # Analysis scripts
│   │   ├── analyze_timeseries.py     # Time series analysis
│   │   ├── analyze_correlation.py    # Correlation analysis
│   │   ├── analyze_impact.py         # Impact analysis
│   │   ├── analyze_combined.py       # Combined analysis
│   │   ├── analyze_fingerprint.py    # Fingerprint visualization
│   │   └── analyze_value.py          # Value analysis
│   ├── parse_result.py          # Parse JSONL to CSV
│   ├── clean_result.py          # Clean and validate results
│   ├── apply_mappings.py        # Apply category mappings
│   ├── generate_mappings.py     # Generate category mappings
│   └── compile_to_csv.py        # Compile final CSV
├── data/                         # Input data (not included in repo)
├── result/                       # Processing results (not included)
├── mappings/                     # Category mapping files
├── batch_files/                  # Batch API files
├── figures/
│   └── paper_figures/           # Figures used in the paper
├── .env.example                  # Example environment variables
├── .gitignore                    # Git ignore file
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

- **Multi-threaded Processing**: Parallel API calls for efficient processing
- **Robust Extraction**: JSON-based structured extraction with validation
- **Comprehensive Analysis**: Time series, correlation, and impact analysis
- **Value-Sensitive Design**: Identifies core human values in user perspectives
- **Batch Processing**: Support for OpenAI Batch API for cost-effective processing

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-mental.git
cd llm-mental
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Usage

### 1. Data Processing

Place JSONL data files in the `data/` folder. Each line should be a JSON object with at least a `Full Text` field. 

Run the main processing script:
```bash
python -m src.core.main
```

This will:
- Read JSONL files from `data/`
- Process texts using GPT-4o-mini
- Extract structured information about LLM-mental health relationships
- Save results to `result/` folder

### 2. Parse Results

Convert JSONL results to CSV:
```bash
python -m src.parse_result result/your_results.jsonl
```

### 3. Clean and Process

Clean the results:
```bash
python -m src.clean_result
```

Apply category mappings:
```bash
python -m src.apply_mappings
```

### 4. Analysis

Run various analyses:

```bash
# Time series analysis
python -m src.analysis.analyze_timeseries

# Correlation analysis
python -m src.analysis.analyze_correlation

# Impact analysis
python -m src.analysis.analyze_impact

# Combined analysis
python -m src.analysis.analyze_combined

# Fingerprint visualization
python -m src.analysis.analyze_fingerprint
```

### Batch Processing (Optional)

For large datasets, use OpenAI's Batch API for cost savings:

```bash
# 1. Prepare batch file
python -m src.batch_processing.prepare_batch

# 2. Submit to OpenAI Batch API
python -m src.batch_processing.submit_batch

# 3. Process results (after batch completes)
python -m src.batch_processing.process_results
```

## Extracted Information

For each social media post, the system extracts:

| Field | Description |
|-------|-------------|
| `supporting_quote` | Direct quote from the post supporting the extraction |
| `llm_product` | Specific LLM mentioned (ChatGPT, Claude, Gemini, etc.) |
| `llm_impact` | Impact type: `positive`, `negative`, `neutral`, `not_applicable` |
| `mental_health_condition` | Specific condition mentioned (anxiety, depression, ADHD, etc.) |
| `user_perspective_category` | User's perspective on LLM use (emotional_support, addiction, etc.) |
| `user_value_expressed` | Core human value implied (human welfare, autonomy, privacy, etc.) |

## Extraction Criteria

The system only extracts information when there is a **direct relationship** between LLM use and mental health:

✅ **Include:**
- Causal: "Using ChatGPT for therapy helped my anxiety"
- Instrumental: "As someone with ADHD, I use ChatGPT to organize thoughts"
- Experiential: "My depression makes me rely on Claude more than friends"
- Evaluative: "Claude's mental health screening is too sensitive"

❌ **Exclude:**
- No LLM mention
- LLM and mental health mentioned separately without connection
- Mental health terms used as jokes/insults
- Mental health terms describing AI behavior (not human)

## Configuration

Edit [src/core/config.py](src/core/config.py) to customize:

```python
MODEL_NAME = 'gpt-4.1-mini'      # OpenAI model to use
TEMPERATURE = 0.0               # Temperature for deterministic outputs
TEXT_COLUMN = 'Full Text'      # Column name in input data
OUTPUT_FOLDER = "result/"       # Output directory
MAX_WORKERS = 4                 # Number of parallel workers
```

## Value-Sensitive Design Framework

We identify 12 core human values based on Value-Sensitive Design principles:

1. Human welfare
2. Autonomy
3. Privacy
4. Informed consent
5. Trust
6. Accountability
7. Fairness
8. Intellectual property
9. Ownership
10. Identity
11. Calmness
12. Sustainability

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{li2025llm,
  title={LLM Use for Mental Health: Crowdsourcing Users' Sentiment-based Perspectives and Values from Social Discussions},
  author={Li, Lingyao and Huang, Xiaoshan and Ma, Renkai and Zhang, Ben Zefeng and Wu, Haolun and Yang, Fan and Chen, Chen},
  journal={arXiv preprint arXiv:2512.07797},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [lingyaol@usf.edu, xiaoshan.huang@mail.mcgill.ca, renkai.ma@uc.edu].

---

**Disclaimer**: This research result is for academic purposes only. It does not imply any medical advice of using LLM for mental health conditions. If you or someone you know is experiencing mental health issues, please seek help from qualified mental health professionals.
