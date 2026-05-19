import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_NAME = "ethandavey/mental-health-diagnosis-bert"
DEFAULT_INPUT_FILE = Path("output_data/prep_logs/group_chat_merged_consecutive.csv")
DEFAULT_OUTPUT_FILE = Path("output_data/mental/group_chat_merged_consecutive_mental_health_scores.csv")
DEFAULT_TEXT_COLUMN = "message"


def add_mental_health_scores(
    df: pd.DataFrame,
    text_col: str = DEFAULT_TEXT_COLUMN,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = None,
) -> pd.DataFrame:
    """Add mental health class probability scores to a DataFrame."""
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataframe")

    out = df.copy()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise ValueError("The model config does not expose an id2label mapping.")

    label_names = {
        int(idx): str(label).strip().lower().replace(" ", "_").replace("-", "_")
        for idx, label in id2label.items()
    }

    texts = out[text_col].fillna("").astype(str).tolist()

    if len(texts) == 0:
        for idx in sorted(label_names.keys()):
            out[f"{label_names[idx]}_score"] = pd.Series(dtype="float64")
        out["mental_health_top_label"] = pd.Series(dtype="object")
        out["mental_health_top_score"] = pd.Series(dtype="float64")
        return out

    all_probs = []
    all_top_labels = []
    all_top_scores = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu()

        top_scores, top_idx = torch.max(probs, dim=1)
        all_probs.append(probs)
        all_top_scores.extend(top_scores.tolist())
        all_top_labels.extend([label_names[int(idx)] for idx in top_idx.tolist()])

    probs_tensor = torch.cat(all_probs, dim=0)

    for idx in range(probs_tensor.shape[1]):
        label = label_names[idx]
        out[f"{label}_score"] = probs_tensor[:, idx].numpy()

    out["mental_health_top_label"] = all_top_labels
    out["mental_health_top_score"] = all_top_scores

    return out


def load_dataframe(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    return pd.read_csv(input_file)


def save_dataframe(df: pd.DataFrame, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score WhatsApp chat data for mental health labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to input CSV file. Default: {DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to save enriched CSV. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help=f"Name of the text column. Default: {DEFAULT_TEXT_COLUMN}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for model inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length for the tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: cuda, mps, or cpu. Auto-detects if not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file = args.input
    output_file = args.output

    print(f"Loading input file: {input_file}")
    df = load_dataframe(input_file)

    print(f"Scoring {len(df)} rows with model {args.model}")
    scored_df = add_mental_health_scores(
        df,
        text_col=args.text_col,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    saved_path = save_dataframe(scored_df, output_file)
    print(f"Saved enriched dataframe to: {saved_path}")
    print("Sample columns:")
    preview_cols = [
        args.text_col,
        "anxiety_score",
        "normal_score",
        "depression_score",
        "suicidal_score",
        "stress_score",
        "mental_health_top_label",
        "mental_health_top_score",
    ]
    existing_preview_cols = [col for col in preview_cols if col in scored_df.columns]
    print(scored_df[existing_preview_cols].head(5))


if __name__ == "__main__":
    main()
