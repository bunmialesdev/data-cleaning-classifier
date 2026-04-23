"""
classifier.py - Keyword-based text classification.

Classifies text into categories based on keyword matching.
Supports loading rules from a dictionary or JSON config file.
"""

import logging
from utils import tokenize, remove_stopwords

logger = logging.getLogger("data-cleaning-classifier")

# Default classification rules
DEFAULT_RULES = {
    "positive": ["good", "great", "excellent", "love", "happy", "wonderful",
                 "amazing", "fantastic", "awesome", "best", "brilliant",
                 "beautiful", "perfect", "enjoy", "pleased", "superb"],
    "negative": ["bad", "terrible", "awful", "hate", "sad", "horrible",
                 "worst", "poor", "ugly", "disgusting", "annoying",
                 "disappointing", "dreadful", "miserable", "angry", "fail"],
    "neutral":  ["okay", "fine", "average", "normal", "moderate", "standard",
                 "acceptable", "adequate", "fair", "reasonable", "ordinary"],
}


def load_rules(config=None):
    """
    Load classification rules from a config dictionary.

    Args:
        config (dict, optional): Dictionary mapping category names to keyword lists.
                                 If None, uses DEFAULT_RULES.

    Returns:
        dict: Classification rules {category: [keywords]}.
    """
    if config is None:
        logger.info("Using default classification rules.")
        return DEFAULT_RULES

    # Validate the config structure
    if not isinstance(config, dict):
        raise ValueError("Classification rules must be a dictionary.")

    for category, keywords in config.items():
        if not isinstance(keywords, list):
            raise ValueError(f"Keywords for category '{category}' must be a list.")

    logger.info(f"Loaded custom rules with {len(config)} categories.")
    return config


def classify_text(text, rules):
    """
    Classify a single text string based on keyword matching.

    The text is tokenized and stopwords are removed. Each token is checked
    against the keyword lists. The category with the highest match count wins.
    If no keywords match, the label 'unknown' is assigned.

    Args:
        text (str): Input text (should be pre-cleaned/normalized).
        rules (dict): Classification rules {category: [keywords]}.

    Returns:
        str: Predicted category label.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)

    scores = {}
    for category, keywords in rules.items():
        keyword_set = set(kw.lower() for kw in keywords)
        score = sum(1 for token in tokens if token in keyword_set)
        scores[category] = score

    # Find the category with the highest score
    max_score = max(scores.values()) if scores else 0

    if max_score == 0:
        return "unknown"

    # Return the first category with the max score
    for category, score in scores.items():
        if score == max_score:
            return category

    return "unknown"


def classify_dataset(df, text_column, rules):
    """
    Classify all rows in a DataFrame, adding a 'label' column.

    Args:
        df (pd.DataFrame): Input data with a text column.
        text_column (str): Name of the text column.
        rules (dict): Classification rules {category: [keywords]}.

    Returns:
        pd.DataFrame: DataFrame with a new 'label' column.
    """
    logger.info("Starting classification...")
    df = df.copy()
    df["label"] = df[text_column].apply(lambda text: classify_text(text, rules))

    # Log distribution
    distribution = df["label"].value_counts().to_dict()
    logger.info(f"Classification complete. Distribution: {distribution}")

    return df
