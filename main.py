"""
main.py - CLI entry point for data-cleaning-classifier.

Usage:
    python main.py --input sample_data.csv --output output/
    python main.py --input data.csv --output results/ --config config.json
"""

import argparse
import os
import sys

from utils import setup_logging, load_csv, save_csv, load_json_config, print_summary
from cleaner import clean_pipeline
from classifier import load_rules, classify_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Cleaning & Classification Tool - "
                    "Preprocess and classify text datasets for machine learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input sample_data.csv --output output/
  python main.py --input data.csv --output results/ --config config.json
  python main.py --input data.csv --output results/ --missing-strategy fill
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to a JSON config file with custom classification rules.",
    )
    parser.add_argument(
        "--text-column", "-t",
        default="text",
        help="Name of the text column in the CSV (default: 'text').",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=["drop", "fill"],
        default="drop",
        help="Strategy for handling missing values: 'drop' or 'fill' (default: 'drop').",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 50)
    logger.info("Data Cleaning & Classification Tool")
    logger.info("=" * 50)

    # --- Load input data ---
    try:
        logger.info(f"Loading data from: {args.input}")
        df = load_csv(args.input)
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Validate text column exists
    if args.text_column not in df.columns:
        logger.error(
            f"Text column '{args.text_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        sys.exit(1)

    # --- Clean the data ---
    try:
        cleaned_df = clean_pipeline(
            df,
            text_column=args.text_column,
            missing_strategy=args.missing_strategy,
        )
    except Exception as e:
        logger.error(f"Error during cleaning: {e}")
        sys.exit(1)

    # Save cleaned data
    cleaned_path = os.path.join(args.output, "cleaned_data.csv")
    save_csv(cleaned_df, cleaned_path)
    logger.info(f"Cleaned data saved to: {cleaned_path}")

    # --- Classify the data ---
    try:
        # Load classification rules
        if args.config:
            logger.info(f"Loading custom rules from: {args.config}")
            config = load_json_config(args.config)
            rules = load_rules(config)
        else:
            rules = load_rules()

        classified_df = classify_dataset(cleaned_df, args.text_column, rules)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Save classified data
    classified_path = os.path.join(args.output, "classified_data.csv")
    save_csv(classified_df, classified_path)
    logger.info(f"Classified data saved to: {classified_path}")

    # --- Print summary ---
    class_distribution = classified_df["label"].value_counts().to_dict()
    print_summary(
        total_rows=len(classified_df),
        class_distribution=class_distribution,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
