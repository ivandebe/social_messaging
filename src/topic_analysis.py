from __future__ import annotations
from pathlib import Path

import argparse
import sys
import re
from typing import List, Tuple

import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_INPUT_FILE = Path("output_data/prep_logs/history_consecutive.csv")
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


def _normalize_topic_phrase(phrase: str) -> str:
    """
    Normalize a topic phrase for duplicate detection.
    Example:
        'eyes smiling_face_with_heart' -> 'eyes smiling face with heart'
    """
    phrase = phrase.lower().replace("_", " ").strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def _canonical_bow_key(phrase: str) -> tuple[str, ...]:
    """
    Build a canonical bag-of-words key so phrases with same words in different
    order collapse together.
    Example:
        'eyes smiling face with heart'
        'smiling face with heart eyes'
    both -> ('eyes', 'face', 'heart', 'smiling', 'with')
    """
    tokens = _normalize_topic_phrase(phrase).split()
    return tuple(sorted(tokens))


def _shorten_emoji_like_phrase(phrase: str) -> str:
    """
    Make emoji-style topic labels shorter and more readable.
    Example:
        'smiling_face_with_heart_eyes' -> 'smiling face heart eyes'
    """
    phrase = phrase.replace("_", " ")
    phrase = re.sub(r"\bwith\b", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase


def get_topic_label_map(topic_model: BERTopic) -> dict[int, str]:
    """
    Build a mapping from topic id to a shorter, more readable topic label.
    """
    topic_info = topic_model.get_topic_info()
    topic_label_map: dict[int, str] = {}

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        if topic_id == -1:
            topic_label_map[topic_id] = "outlier"
            continue

        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            topic_label_map[topic_id] = f"topic_{topic_id}"
            continue

        raw_terms = [word for word, _ in topic_words[:10]]

        # 1) remove exact duplicates preserving order
        dedup_terms = list(dict.fromkeys(raw_terms))

        # 2) remove near-duplicates based on same bag-of-words
        selected_terms = []
        seen_keys = set()

        for term in dedup_terms:
            bow_key = _canonical_bow_key(term)
            if bow_key not in seen_keys:
                seen_keys.add(bow_key)
                selected_terms.append(term)

        # 3) shorten labels for display
        display_terms = [_shorten_emoji_like_phrase(term) for term in selected_terms[:5]]

        label = ", ".join(display_terms) if display_terms else f"topic_{topic_id}"
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
    
    input_cols = [col for col in df.columns if col in ["message", "sender", "date", "time", "timestamp"]]
    df=df[input_cols]

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
