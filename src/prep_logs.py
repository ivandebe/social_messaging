import argparse
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


DEFAULT_INPUT_FILE = Path(__file__).resolve().parents[1] / "input_data" / "group_chat_history.txt"
DEFAULT_OUTPUT_SUBFOLDER = "prep_logs"
DEFAULT_MAX_CHUNK_SIZE = 512
DEFAULT_TOKENIZER = "bert-base-uncased"


def parse_whatsapp_chat(file_path: Path) -> pd.DataFrame:
    """Parse a WhatsApp export file into a cleaned DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    messages = []
    current_message = None
    pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.+?): (.+)")

    for line in lines:
        line = line.strip()
        match = pattern.match(line)

        if match:
            if current_message:
                messages.append(current_message)
            date, time, sender, message = match.groups()
            current_message = {
                "date": date,
                "time": time,
                "sender": sender,
                "message": message,
            }
        elif current_message:
            current_message["message"] += "\n" + line

    if current_message:
        messages.append(current_message)

    data = pd.DataFrame(messages)
    data["message"] = (
        data["message"]
        .astype(str)
        .replace(r"<Media omitted>", "", regex=True)
        .str.strip()
    )
    data = data[data["message"].notna() & (data["message"] != "")].reset_index(drop=True)

    return data

def merge_consecutive_messages(data: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive messages from the same sender on the same day."""
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce").dt.date
    data["time"] = pd.to_datetime(data["time"], format="%H:%M", errors="coerce").dt.time

    merged_data = []
    current_message = None

    for _, row in data.iterrows():
        if (
            current_message
            and row["sender"] == current_message["sender"]
            and row["date"] == current_message["date"]
        ):
            current_message["message"] += "\n" + row["message"]
        else:
            if current_message:
                merged_data.append(current_message)
            current_message = row.to_dict()

    if current_message:
        merged_data.append(current_message)

    return pd.DataFrame(merged_data)


def chunk_messages_by_token_limit(
    data: pd.DataFrame,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> pd.DataFrame:
    """Group messages into chunks not exceeding max token size per sender per day."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce").dt.date

    chunked_data = []
    grouped = data.groupby(["sender", "date"])

    for (sender, date), group in grouped:
        messages = group.sort_values(["time"]).reset_index(drop=True)
        current_chunk = []
        current_tokens = 0

        for _, row in messages.iterrows():
            msg = str(row["message"]).strip()
            token_count = len(tokenizer.encode(msg, add_special_tokens=True))

            if token_count > max_chunk_size:
                print(f"Skipping long message from {sender} on {date} (tokens={token_count})")
                continue

            if current_tokens + token_count <= max_chunk_size:
                current_chunk.append(msg)
                current_tokens += token_count
            else:
                chunked_data.append(
                    {
                        "sender": sender,
                        "date": date,
                        "chunk": " ".join(current_chunk),
                        "messages_in_chunk": len(current_chunk),
                        "token_count": current_tokens,
                    }
                )
                current_chunk = [msg]
                current_tokens = token_count

        if current_chunk:
            chunked_data.append(
                {
                    "sender": sender,
                    "date": date,
                    "chunk": " ".join(current_chunk),
                    "messages_in_chunk": len(current_chunk),
                    "token_count": current_tokens,
                }
            )

    chunked_df = pd.DataFrame(chunked_data)
    chunked_df = chunked_df.dropna(subset=["chunk"]).reset_index(drop=True)
    return chunked_df


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(data: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    return output_path


def find_input_file(input_file: Path) -> Path:
    if input_file.exists():
        return input_file

    root = Path(__file__).resolve().parents[1]
    fallback = root / "input_data" / "group_chat_history.txt"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not locate the WhatsApp chat file. Checked: {input_file} and {fallback}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WhatsApp chat logs and save cleaned, aggregated, and chunked outputs.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the WhatsApp export text file.",
    )
    parser.add_argument(
        "--output-subfolder",
        type=str,
        default=DEFAULT_OUTPUT_SUBFOLDER,
        help="Subfolder under output_data where CSV files will be saved.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name for message chunking.",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_MAX_CHUNK_SIZE,
        help="Maximum token count per chunk.",
    )
    args = parser.parse_args()

    input_file = find_input_file(args.input_file)
    root = Path(__file__).resolve().parents[1]
    output_dir = ensure_output_dir(root / "output_data" / args.output_subfolder)

    print(f"Reading chat file from: {input_file}")
    data = parse_whatsapp_chat(input_file)
    print(f"Parsed {len(data)} messages.")

    chunked_data = chunk_messages_by_token_limit(
        data,
        tokenizer_name=args.tokenizer,
        max_chunk_size=args.max_chunk_size,
    )

    cleaned_path = save_dataframe(data, output_dir, "group_chat_cleaned_single.csv")
    merged_path = save_dataframe(merge_consecutive_messages(data), output_dir, "group_chat_merged_consecutive.csv")
    chunked_path = save_dataframe(chunked_data, output_dir, "chunked_data.csv")

    print(f"Saved cleaned messages to: {cleaned_path}")
    print(f"Saved merged messages to: {merged_path}")
    print(f"Saved chunked messages to: {chunked_path}")


if __name__ == "__main__":
    main()



