"""
cleaner.py - Data cleaning functions for text datasets.

Provides functions to remove duplicates, handle missing values,
normalize text, and run a full cleaning pipeline.
"""

import re
import logging
import pandas as pd

logger = logging.getLogger("data-cleaning-classifier")


def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Data with duplicates removed.
    """
    original_count = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = original_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate row(s).")
    return df


def handle_missing(df, text_column, strategy="drop"):
    """
    Handle missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input data.
        text_column (str): Name of the text column to check.
        strategy (str): 'drop' to remove rows with missing text,
                        'fill' to replace with empty string.

    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    missing_count = df[text_column].isna().sum()

    if missing_count == 0:
        logger.info("No missing values found in text column.")
        return df

    if strategy == "drop":
        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        logger.info(f"Dropped {missing_count} row(s) with missing text.")
    elif strategy == "fill":
        df[text_column] = df[text_column].fillna("")
        logger.info(f"Filled {missing_count} missing text value(s) with empty string.")
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'drop' or 'fill'.")

    return df


def normalize_text(text):
    """
    Normalize a text string: lowercase, remove punctuation.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (keep letters, numbers, spaces)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_extra_whitespace(text):
    """
    Remove extra whitespace from text: strip and collapse multiple spaces.

    Args:
        text (str): Input text.

    Returns:
        str: Text with normalized whitespace.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_pipeline(df, text_column="text", missing_strategy="drop"):
    """
    Run the full data cleaning pipeline on a DataFrame.

    Steps:
        1. Remove duplicates
        2. Handle missing values
        3. Normalize text (lowercase, remove punctuation)
        4. Remove extra whitespace

    Args:
        df (pd.DataFrame): Input data.
        text_column (str): Name of the text column.
        missing_strategy (str): Strategy for handling missing values.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    logger.info("Starting cleaning pipeline...")

    # Step 1: Remove duplicates
    df = remove_duplicates(df)

    # Step 2: Handle missing values
    df = handle_missing(df, text_column, strategy=missing_strategy)

    # Step 3 & 4: Normalize text and remove extra whitespace
    df[text_column] = df[text_column].apply(normalize_text)
    df[text_column] = df[text_column].apply(remove_extra_whitespace)

    # Remove any rows that became empty after cleaning
    empty_count = (df[text_column] == "").sum()
    if empty_count > 0:
        df = df[df[text_column] != ""].reset_index(drop=True)
        logger.info(f"Removed {empty_count} row(s) that were empty after cleaning.")

    logger.info(f"Cleaning complete. {len(df)} rows remaining.")
    return df
