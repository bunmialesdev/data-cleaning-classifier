"""
utils.py - Shared utility functions for data-cleaning-classifier.

Provides file I/O, tokenization, stopword removal, and logging setup.
"""

import os
import re
import csv
import json
import logging
import pandas as pd


# Common English stopwords
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
}


def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("data-cleaning-classifier")


def load_csv(filepath):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{filepath}': {e}")

    if df.empty:
        raise ValueError(f"CSV file '{filepath}' is empty.")

    return df


def save_csv(df, filepath):
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Data to save.
        filepath (str): Destination file path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)


def tokenize(text):
    """
    Tokenize text into a list of words using regex.

    Args:
        text (str): Input text.

    Returns:
        list[str]: List of word tokens.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def remove_stopwords(tokens):
    """
    Remove common English stopwords from a list of tokens.

    Args:
        tokens (list[str]): List of word tokens.

    Returns:
        list[str]: Filtered tokens without stopwords.
    """
    return [token for token in tokens if token.lower() not in STOPWORDS]


def load_json_config(filepath):
    """
    Load a JSON configuration file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file '{filepath}': {e}")


def print_summary(total_rows, class_distribution):
    """
    Print summary statistics to the terminal.

    Args:
        total_rows (int): Total number of rows processed.
        class_distribution (dict): Mapping of label -> count.
    """
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  Total rows processed: {total_rows}")
    print(f"  Class distribution:")
    for label, count in sorted(class_distribution.items()):
        pct = (count / total_rows * 100) if total_rows > 0 else 0
        print(f"    {label:>12s}: {count:>5d}  ({pct:.1f}%)")
    print("=" * 50 + "\n")
