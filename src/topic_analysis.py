from __future__ import annotations
from pathlib import Path

import argparse
import sys
import re
from typing import List, Tuple

import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_INPUT_FILE = Path("output_data/prep_logs/group_chat_merged_consecutive.csv")
DEFAULT_OUTPUT_FILE = Path("output_data/topic/topic_consecutive.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply BERTopic to a chat CSV and save topic assignments."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help=f"Path to input CSV file (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Path to output CSV file (default: {DEFAULT_OUTPUT_FILE})",
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


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def basic_clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate_input_dataframe(df: pd.DataFrame, message_column: str) -> None:
    if message_column not in df.columns:
        raise ValueError(
            f'Input CSV does not contain the required column "{message_column}". '
            f"Available columns: {list(df.columns)}"
        )


def prepare_documents(
    df: pd.DataFrame,
    message_column: str
) -> Tuple[pd.DataFrame, List[str]]:
    working_df = df.copy()
    working_df["_topic_text"] = working_df[message_column].apply(basic_clean_text)
    working_df["_topic_text"] = working_df["_topic_text"].fillna("").astype(str)
    return working_df, working_df["_topic_text"].tolist()


def build_topic_model(
    min_topic_size: int,
    nr_topics,
    language: str,
) -> BERTopic:
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


def get_topic_label_map(topic_model: BERTopic) -> dict[int, str]:
    """
    Build a mapping from topic id to a readable topic label.

    Example label:
        "pizza, dinner, tonight, order, food"
    """
    topic_info = topic_model.get_topic_info()
    topic_label_map: dict[int, str] = {}

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        if topic_id == -1:
            topic_label_map[topic_id] = "outlier"
            continue

        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            label = ", ".join(word for word, _ in topic_words[:5])
        else:
            label = f"topic_{topic_id}"

        topic_label_map[topic_id] = label

    return topic_label_map


def assign_topics(
    df: pd.DataFrame,
    docs: List[str],
    topic_model: BERTopic,
) -> Tuple[pd.DataFrame, BERTopic]:
    result_df = df.copy()
    result_df["topic"] = -1
    result_df["topic_label"] = "outlier"

    valid_mask = result_df["_topic_text"].str.strip().ne("")
    valid_docs = [doc for doc, is_valid in zip(docs, valid_mask.tolist()) if is_valid]

    if len(valid_docs) == 0:
        print("No valid non-empty messages found. Writing output with topic = -1 and topic_label = 'outlier' for all rows.")
        return result_df, topic_model

    topics, _ = topic_model.fit_transform(valid_docs)

    result_df.loc[valid_mask, "topic"] = topics

    topic_label_map = get_topic_label_map(topic_model)
    result_df["topic_label"] = result_df["topic"].map(topic_label_map).fillna("unknown")

    return result_df, topic_model


def save_outputs(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    final_df = df.drop(columns=["_topic_text"], errors="ignore")
    ensure_parent_dir(output_path)
    final_df.to_csv(output_path, index=False)


def print_topic_summary(df: pd.DataFrame) -> None:
    summary = (
        df[["topic", "topic_label"]]
        .value_counts(dropna=False)
        .reset_index(name="count")
        .sort_values(["topic"])
    )
    print("\nTopic counts:")
    print(summary.to_string(index=False))


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    message_column = args.message_column
    min_topic_size = args.min_topic_size
    nr_topics = args.nr_topics
    language = args.language

    if nr_topics is not None and nr_topics != "auto":
        try:
            nr_topics = int(nr_topics)
        except ValueError:
            raise ValueError('--nr-topics must be an integer, "auto", or omitted')

    print(f"Reading input CSV: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    validate_input_dataframe(df, message_column)

    print(f"Loaded {len(df)} rows.")
    print(f'Using message column: "{message_column}"')

    df_prepared, docs = prepare_documents(
        df=df,
        message_column=message_column
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
