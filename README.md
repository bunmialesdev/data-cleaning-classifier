# Data Cleaning & Classifier

A Python tool for preprocessing and classifying text datasets for machine learning. It cleans messy text data (removes duplicates, handles missing values, normalizes text) and classifies it into categories using a keyword-based system.

## Features

- **Data Cleaning**: Remove duplicates, handle missing values, normalize text, remove extra whitespace
- **Text Processing**: Tokenization and stopword removal
- **Classification**: Keyword-based classification with customizable rules
- **Logging**: Built-in logging for transparency
- **Flexible Config**: Define your own classification rules via JSON

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --input sample_data.csv --output output/
```

### With Custom Classification Rules

```bash
python main.py --input sample_data.csv --output output/ --config config.json
```

### All Options

```bash
python main.py --input <file> --output <dir> [--config <json>] [--text-column <name>] [--missing-strategy drop|fill]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Path to input CSV file | (required) |
| `--output, -o` | Output directory | (required) |
| `--config, -c` | JSON file with custom classification rules | Built-in rules |
| `--text-column, -t` | Name of the text column in CSV | `text` |
| `--missing-strategy` | How to handle missing values: `drop` or `fill` | `drop` |

## Example

### Input (`sample_data.csv`)

| id | text |
|----|------|
| 1 | This product is absolutely GREAT and I love it! |
| 2 | Terrible experience. The worst service I have ever had. |
| 3 | The quality is okay and seems average for the price. |
| 5 |    Bad    quality   and    very   disappointing  overall. |
| 13 | *(empty)* |

### Output (`output/classified_data.csv`)

| id | text | label |
|----|------|-------|
| 1 | this product is absolutely great and i love it | positive |
| 2 | terrible experience the worst service i have ever had | negative |
| 3 | the quality is okay and seems average for the price | neutral |
| 5 | bad quality and very disappointing overall | negative |

*(Row 13 was dropped due to missing text)*

### Terminal Summary

```
==================================================
  SUMMARY
==================================================
  Total rows processed: 20
  Class distribution:
      negative:     6  (30.0%)
      neutral:      5  (25.0%)
      positive:     7  (35.0%)
      unknown:      2  (10.0%)
==================================================
```

## Custom Classification Rules

Create a JSON file to define your own categories:

```json
{
    "tech": ["software", "hardware", "code", "programming", "computer"],
    "finance": ["stock", "market", "investment", "banking", "profit"],
    "health": ["doctor", "medicine", "exercise", "nutrition", "wellness"]
}
```

Then run:

```bash
python main.py --input data.csv --output output/ --config my_rules.json
```

## Project Structure

```
data-cleaning-classifier/
├── main.py              # CLI entry point
├── cleaner.py           # Data cleaning functions
├── classifier.py        # Keyword-based classification
├── utils.py             # Shared utilities
├── config.json          # Default classification rules
├── sample_data.csv      # Example dataset
├── output/              # Output directory
├── README.md            # This file
└── requirements.txt     # Dependencies
```

## WIP
Support for LLM model through API Key, such as Claude/OpenAI/Ollama
