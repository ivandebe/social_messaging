import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DEFAULT_INPUT_FILE = "group_chat_merged_consecutive.csv"
DEFAULT_OUTPUT_SUBFOLDER = "sentiment"


def apply_vader_sentiment(data: pd.DataFrame, text_column: str = "message") -> pd.DataFrame:
    """Apply VADER sentiment analysis to messages."""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_score(text):
        if not isinstance(text, str) or len(str(text).strip()) == 0:
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        return analyzer.polarity_scores(text)
    
    sentiment_scores = data[text_column].apply(get_sentiment_score).apply(pd.Series)
    sentiment_scores.columns = [f"vader_{col}" for col in sentiment_scores.columns]
    
    return pd.concat([data.reset_index(drop=True), sentiment_scores.reset_index(drop=True)], axis=1)


def apply_huggingface_emotion(data: pd.DataFrame, text_column: str = "message", batch_size: int = 32) -> pd.DataFrame:
    """Apply Hugging Face emotion classification to messages using batch processing."""
    data = data.copy()
    data[text_column] = data[text_column].fillna("")
    
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        truncation=True,
        device=-1,  # Use CPU; change to 0 for GPU
    )
    
    def normalize_text(text):
        """Truncate text to a reasonable length for the model."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        # Truncate to ~500 chars to stay within token limits
        return text[:500]
    
    # Normalize all texts first
    texts = data[text_column].apply(normalize_text).tolist()
    
    # Process in batches for efficiency
    emotion_scores_list = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        batch = texts[i:batch_end]
        
        # Filter out empty strings for processing
        batch_with_idx = [(idx, text) for idx, text in enumerate(range(i, batch_end)) if texts[idx]]
        
        if not batch_with_idx:
            for _ in range(batch_end - i):
                emotion_scores_list.append({})
            continue
        
        batch_texts = [texts[idx] for idx, _ in batch_with_idx]
        
        try:
            # Classify the batch
            results = classifier(batch_texts)
            
            # Process results
            result_idx = 0
            for batch_idx in range(batch_end - i):
                original_idx = i + batch_idx
                
                if original_idx in [idx for idx, _ in batch_with_idx]:
                    if isinstance(results[result_idx], list):
                        # Format: list of dicts with 'label' and 'score'
                        scores = {item["label"]: item["score"] for item in results[result_idx]}
                    else:
                        # Single result
                        scores = {results[result_idx]["label"]: results[result_idx]["score"]}
                    emotion_scores_list.append(scores)
                    result_idx += 1
                else:
                    emotion_scores_list.append({})
        except Exception as e:
            print(f"Error processing batch {i}-{batch_end}: {e}")
            for _ in range(batch_end - i):
                emotion_scores_list.append({})
        
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Processed {min(i + batch_size, total)}/{total} messages")
    
    # Convert list of dicts to DataFrame
    emotion_df = pd.DataFrame(emotion_scores_list).fillna(0.0)
    emotion_df.columns = [f"emotion_{col}" for col in emotion_df.columns]
    
    return pd.concat([data.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)


def find_input_file(input_file: str, prep_logs_output_dir: Path) -> Path:
    """Locate the cleaned chat CSV file."""
    input_path = Path(input_file)
    
    if input_path.is_absolute() and input_path.exists():
        return input_path
    
    # Check in prep_logs output directory
    fallback = prep_logs_output_dir / input_file
    if fallback.exists():
        return fallback
    
    raise FileNotFoundError(
        f"Could not locate input file: {input_file}. "
        f"Checked: {input_path} and {fallback}"
    )


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_dataframe(data: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply VADER and Hugging Face emotion classification to cleaned chat messages."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help="Name or path to the cleaned chat CSV file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing the input CSV. Defaults to output_data/prep_logs/",
    )
    parser.add_argument(
        "--output-subfolder",
        type=str,
        default=DEFAULT_OUTPUT_SUBFOLDER,
        help="Subfolder under output_data where results will be saved.",
    )
    parser.add_argument(
        "--skip-vader",
        action="store_true",
        help="Skip VADER sentiment analysis.",
    )
    parser.add_argument(
        "--skip-huggingface",
        action="store_true",
        help="Skip Hugging Face emotion classification.",
    )
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir or (root / "output_data" / "prep_logs")
    output_dir = ensure_output_dir(root / "output_data" / args.output_subfolder)
    
    input_file = find_input_file(args.input_file, input_dir)
    
    print(f"Reading input from: {input_file}")
    data = pd.read_csv(input_file)
    print(f"Loaded {len(data)} messages.")
    
    if not args.skip_vader:
        print("Applying VADER sentiment analysis...")
        data = apply_vader_sentiment(data, text_column="message")
        print("VADER analysis complete.")
    
    if not args.skip_huggingface:
        print("Applying Hugging Face emotion classification (this may take a while)...")
        data = apply_huggingface_emotion(data, text_column="message")
        print("Hugging Face analysis complete.")
    
    output_path = save_dataframe(data, output_dir, "sentiment_analysis_results.csv")
    print(f"Saved combined results to: {output_path}")


if __name__ == "__main__":
    main()