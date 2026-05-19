#!/usr/bin/env python3

"""
Run BERTopic on a WhatsApp/group chat CSV and save topic assignments.

Default input:
    /output_data/group_chat_merged_consecutive.csv

Default output:
    /output_data/topic/topic_consecutive.csv

Expected input:
    - CSV file with a column named "message"
    - Other columns (date, time, sender, etc.) are preserved

Output:
    - Same CSV data plus a new column: "topic"

Example:
    python topic_model_bertopic.py
    python topic_model_bertopic.py --input /path/to/input.csv --output /path/to/output.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from typing import List, Tuple

import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_INPUT_FILE = "/output_data/group_chat_merged_consecutive.csv"
DEFAULT_OUTPUT_FILENAME = "/output_data/topic/topic_consecutive.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply BERTopic to a chat CSV and save topic assignments."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to input CSV file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Path to output CSV file (default: {DEFAULT_OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--message-column",
        default="message",
        help='Name of the text column containing messages (default: "message")',
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="Minimum topic size for BERTopic (default: 10)",
    )
    parser.add_argument(
        "--nr-topics",
        default=None,
        help='Number of topics after reduction, e.g. 20 or "auto" (default: None)',
    )
    parser.add_argument(
        "--language",
        default="multilingual",
        help='BERTopic language setting, e.g. "english" or "multilingual" (default: multilingual)',
    )
    return parser.parse_args()


def ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def basic_clean_text(text: str) -> str:
    """
    Light cleaning suitable for BERTopic:
    - convert to string
    - lowercase
    - remove URLs
    - normalize whitespace

    We keep most words intact because BERTopic generally benefits
    from richer text compared with aggressive preprocessing.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def validate_input_dataframe(df: pd.DataFrame, message_column: str) -> None:
    if message_column not in df.columns:
        raise ValueError(
            f'Input CSV does not contain the required column "{message_column}". '
            f"Available columns: {list(df.columns)}"
        )


def prepare_documents(
    df: pd.DataFrame,
    message_column: str,
    apply_cleaning: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare documents for BERTopic while preserving row order.

    Empty messages are still retained in the dataframe, but they will
    receive topic -1 unless BERTopic is run on valid text.
    """
    working_df = df.copy()

    if apply_cleaning:
        working_df["_topic_text"] = working_df[message_column].apply(basic_clean_text)
    else:
        working_df["_topic_text"] = working_df[message_column].fillna("").astype(str)

    working_df["_topic_text"] = working_df["_topic_text"].fillna("").astype(str)
    return working_df, working_df["_topic_text"].tolist()


def build_topic_model(
    min_topic_size: int,
    nr_topics,
    language: str,
) -> BERTopic:
    """
    Create a BERTopic model.

    We use a CountVectorizer to improve topic representations.
    """
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    model = BERTopic(
        language=language,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    return model


def assign_topics(
    df: pd.DataFrame,
    docs: List[str],
    topic_model: BERTopic,
) -> Tuple[pd.DataFrame, BERTopic]:
    """
    Fit BERTopic and assign topic ids back to the dataframe.

    Rows with empty text are assigned topic -1.
    """
    result_df = df.copy()
    result_df["topic"] = -1

    valid_mask = result_df["_topic_text"].str.strip().ne("")
    valid_docs = [doc for doc, is_valid in zip(docs, valid_mask.tolist()) if is_valid]

    if len(valid_docs) == 0:
        print("No valid non-empty messages found. Writing output with topic = -1 for all rows.")
        return result_df, topic_model

    topics, _ = topic_model.fit_transform(valid_docs)

    result_df.loc[valid_mask, "topic"] = topics
    return result_df, topic_model


def save_outputs(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    final_df = df.drop(columns=["_topic_text"], errors="ignore")
    ensure_parent_dir(output_path)
    final_df.to_csv(output_path, index=False)


def print_topic_summary(df: pd.DataFrame) -> None:
    topic_counts = df["topic"].value_counts(dropna=False).sort_index()
    print("\\nTopic counts:")
    print(topic_counts.to_string())


def main() -> int:
    args = parse_args()

    input_path = args.input
    output_path = args.output
    message_column = args.message_column
    min_topic_size = args.min_topic_size
    nr_topics = args.nr_topics
    language = args.language
    apply_cleaning = not args.no_cleaning

    if nr_topics is not None and nr_topics != "auto":
        try:
            nr_topics = int(nr_topics)
        except ValueError:
            raise ValueError('--nr-topics must be an integer, "auto", or omitted')

    print(f"Reading input CSV: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    validate_input_dataframe(df, message_column)

    print(f"Loaded {len(df)} rows.")
    print(f'Using message column: "{message_column}"')

    df_prepared, docs = prepare_documents(
        df=df,
        message_column=message_column,
        apply_cleaning=apply_cleaning,
    )

    topic_model = build_topic_model(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        language=language,
    )

    df_with_topics, _ = assign_topics(
        df=df_prepared,
        docs=docs,
        topic_model=topic_model,
    )

    save_outputs(df_with_topics, output_path)

    print(f"Saved output CSV to: {output_path}")
    print_topic_summary(df_with_topics)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
