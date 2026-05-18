import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

DEFAULT_INPUT_FILE = "group_chat_cleaned.csv"
DEFAULT_OUTPUT_SUBFOLDER = "mental"
DEFAULT_OUTPUT_FILENAME = "mental_health_classification.csv"
DEFAULT_TEXT_COLUMN = "message"
DEFAULT_MH_SINGLE_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MH_MULTI_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"


def find_input_file(input_file: str, prep_logs_output_dir: Path) -> Path:
    input_path = Path(input_file)
    if input_path.is_absolute() and input_path.exists():
        return input_path

    if input_path.exists():
        return input_path

    fallback = prep_logs_output_dir / input_file
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not locate input file: {input_file}. Checked: {input_path} and {fallback}"
    )

def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(data: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    return output_path


def apply_text_classification(
    data: pd.DataFrame,
    text_column: str,
    model_name: str,
    prefix: str,
    batch_size: int = 16,
    device: int = -1,
) -> pd.DataFrame:
    classifier = pipeline(
        "text-classification",
        model=model_name,
        return_all_scores=True,
        truncation=True,
        device=device,
    )

    data = data.copy()
    data[text_column] = data[text_column].fillna("").astype(str)
    scores_list = []
    texts = data[text_column].tolist()
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        try:
            results = classifier(batch_texts)
        except Exception as exc:
            print(f"Error processing {prefix} batch {start}:{end}: {exc}")
            results = [{} for _ in batch_texts]

        for result in results:
            if isinstance(result, list):
                scores = {item["label"]: item["score"] for item in result}

            elif isinstance(result, dict) and "label" in result and "score" in result:
                scores = {result["label"]: result["score"]}
            else:
                scores = {}
            scores_list.append(scores)

        if (start // batch_size + 1) % 10 == 0 or end == total:
            print(f"  {prefix}: processed {end}/{total}")

    scores_df = pd.DataFrame(scores_list).fillna(0.0)
    scores_df.columns = [f"{prefix}_{col}" for col in scores_df.columns]
    if not scores_df.empty:
        label_columns = list(scores_df.columns)
        scores_df[f"{prefix}_predicted"] = (
            scores_df[label_columns].idxmax(axis=1).str.replace(f"{prefix}_", "", regex=False)
        )
    else:
        scores_df[f"{prefix}_predicted"] = ""

    return pd.concat([data.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)


def apply_mental_health_classification(
    data: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
    model_name: str = DEFAULT_MH_SINGLE_MODEL,
) -> pd.DataFrame:
    return apply_text_classification(
        data=data,
        text_column=text_column,
        model_name=model_name,
        prefix="mh_single",
        batch_size=16,
        device=0 if torch.cuda.is_available() else -1,
    )


def apply_mental_health_multi_classification(
    data: pd.DataFrame,
    text_column: str = DEFAULT_TEXT_COLUMN,
    model_name: str = DEFAULT_MH_MULTI_MODEL,
) -> pd.DataFrame:
    return apply_text_classification(
        data=data,
        text_column=text_column,
        model_name=model_name,
        prefix="mh_multi",
        batch_size=16,
        device=0 if torch.cuda.is_available() else -1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate combined mental health classification from cleaned WhatsApp chat data."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Input cleaned chat CSV file name or path.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the input CSV. Defaults to output_data/prep_logs/.",
    )
    parser.add_argument(
        "--output-subfolder",
        type=str,
        default=DEFAULT_OUTPUT_SUBFOLDER,
        help="Subfolder under output_data where results are saved.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help="Text column in the cleaned CSV file.",
    )
    parser.add_argument(
        "--mh-model",
        type=str,
        default=DEFAULT_MH_SINGLE_MODEL,
        help="Model name or path for single-label mental health classification.",
    )
    parser.add_argument(
        "--mh-multi-model",
        type=str,
        default=DEFAULT_MH_MULTI_MODEL,
        help="Model name or path for multi-label emotion classification.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir or (root / "output_data" / "prep_logs")
    output_dir = ensure_output_dir(root / "output_data" / args.output_subfolder)

    input_path = find_input_file(args.input_file, input_dir)
    print(f"Reading input from: {input_path}")

    data = pd.read_csv(input_path)
    print(f"Loaded {len(data)} rows.")

    print("Applying single-label mental health classification...")
    data = apply_mental_health_classification(
        data,
        text_column=args.text_column,
        model_name=args.mh_model,
    )
    print("Single-label mental health classification complete.")

    print("Applying multi-label emotion-based mental health classification...")
    data = apply_mental_health_multi_classification(
        data,
        text_column=args.text_column,
        model_name=args.mh_multi_model,
    )
    print("Multi-label mental health classification complete.")

    output_path = save_dataframe(data, output_dir, DEFAULT_OUTPUT_FILENAME)
    print(f"Saved combined mental health results to: {output_path}")


if __name__ == "__main__":
    main()
