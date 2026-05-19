import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TextClassificationPipeline, AutoModelForMaskedLM
import torch



def parse_whatsapp_chat(file_path):

    # If that doesn't work, try 'utf-16'
    # with open(file_path, 'r', encoding='utf-16') as f:
    #     print(f.readline())

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    messages = []
    message_buffer = []
    current_message = None

    # WhatsApp date-message pattern (change if your locale differs)
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.+?): (.+)')

    for line in lines:
        line = line.strip()
        match = pattern.match(line)

        if match:
            if current_message:
                messages.append(current_message)
            date, time, sender, message = match.groups()
            current_message = {
                'date': date,
                'time': time,
                'sender': sender,
                'message': message
            }
        else:
            # Continuation of previous message
            if current_message:
                current_message['message'] += '\n' + line

    if current_message:
        messages.append(current_message)

    data = pd.DataFrame(messages)

    # Remove <Media omitted>
    data['message'] = data['message'].str.replace(r'<Media omitted>', '', regex=True).str.strip()

    # Drop rows with missing messages
    data = data.dropna(subset=['message'])

    # Save to CSV for analysis
    data.to_csv('group_chat_cleaned.csv', index=False)


    ## Aggregate all data per user
    aggregated_data = data.groupby("sender")['message'].apply(lambda x: ' '.join(x.dropna())).reset_index()
    aggregated_data.columns = ['sender', 'all_messages']
    aggregated_data.to_csv('aggregated_data_by_user.csv', index=False)

    return data, aggregated_data



def chunk_messages_by_token_limit(data: pd.DataFrame, tokenizer_name: str = "bert-base-uncased", max_chunk_size: int = 512):
    """
    Groups WhatsApp messages into chunks that do not exceed a specified max token size per sender per day.
    Avoids cutting messages in the middle and keeps messages from different days separated.
    
    Parameters:
        data (pd.DataFrame): DataFrame with 'date', 'time', 'sender', and 'message' columns.
        tokenizer_name (str): Name of the tokenizer to use (e.g., 'bert-base-uncased').
        max_chunk_size (int): Maximum number of tokens allowed per chunk.

    Returns:
        pd.DataFrame: A new DataFrame with columns: ['sender', 'date', 'chunk', 'messages_in_chunk', 'token_count']
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Ensure 'date' is datetime
    data['date'] = pd.to_datetime(data['date'], dayfirst=True).dt.date

    chunked_data = []

    # Group by sender and date
    grouped = data.groupby(['sender', 'date'])

    for (sender, date), group in grouped:
        messages = group.sort_values(['time']).reset_index(drop=True)
        
        current_chunk = []
        current_tokens = 0

        for _, row in messages.iterrows():
            msg = str(row['message']).strip()
            token_count = len(tokenizer.encode(msg, add_special_tokens=True))

            if token_count > max_chunk_size:
                # Skip overly long messages
                print(f"Skipping long message from {sender} on {date} (tokens={token_count})")
                continue

            if current_tokens + token_count <= max_chunk_size:
                current_chunk.append(msg)
                current_tokens += token_count
            else:
                # Save current chunk
                chunked_data.append({
                    'sender': sender,
                    'date': date,
                    'chunk': " ".join(current_chunk),
                    'messages_in_chunk': len(current_chunk),
                    'token_count': current_tokens,
                })
                # Start new chunk
                current_chunk = [msg]
                current_tokens = token_count

        # Save the last chunk if it exists
        if current_chunk:
            chunked_data.append({
                'sender': sender,
                'date': date,
                'chunk': " ".join(current_chunk),
                'messages_in_chunk': len(current_chunk),
                'token_count': current_tokens,
            })

    # Create DataFrame once
    chunked_df = pd.DataFrame(chunked_data)

    # remove empty chunks
    chunked_df = chunked_df.dropna(subset=['chunk']) 

    # Save to file safely
    if not chunked_df.empty:
        chunked_df.to_csv("chunked_data.csv", index=False)
    else:
        print("No chunks were created. Please check your input data or chunking logic.")

    return chunked_df



def add_sentiment_analysis_by_vader(df):

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Apply sentiment analysis
    def get_sentiment_score(text):
        return analyzer.polarity_scores(text)

    # Create sentiment columns
    sentiment_scores = df['message'].apply(get_sentiment_score).apply(pd.Series)
    df = pd.concat([df, sentiment_scores], axis=1)

    # Now you have: 'neg', 'neu', 'pos', 'compound' scores

    # Group by sender and get average sentiment
    mean_sentiment_by_user = df.groupby('sender')[['neg', 'neu', 'pos', 'compound']].mean().reset_index()

    mean_sentiment_by_user.to_csv('mean_sentiment_by_user.csv')
    df.to_csv("data_sa.csv")

    return df



def hugging_face_emotion_classification(data, text_column='message'):

    #prevent NaN
    data[text_column] = data[text_column].fillna("") 

    # Load the emotion classification pipeline'joy', 'sadness', 'anger', 'fear', 'surprise', 
    classifier = pipeline("text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        # model="finiteautomata/bertweet-base-emotion-analysis",
        return_all_scores=True,
        truncation=True
        )

    
    def get_emotion_scores(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            # Use label names directly
            return {label: 0.0 for label in classifier.model.config.id2label.values()}
        
        result = classifier(text)[0]
        return {item['label']: item['score'] for item in result}



    # # this function will give you the score. If the text message is bigger than the chunk_size, it will be chuncked and score will be averaged
    # def get_emotion_scores(text, chunk_size=400):
    #     chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    #     all_scores = []

    #     for chunk in chunks:
    #         try:
    #             result = classifier(chunk)[0]
    #             scores = {item['label']: item['score'] for item in result}
    #             all_scores.append(scores)
    #         except Exception as e:
    #             print(f"Error in chunk: {e}")
    #             continue

    #     if not all_scores:
    #         return {}

    #     # Average the emotion scores over all chunks
    #     return {
    #         label: np.mean([score[label] for score in all_scores if label in score])
    #         for label in all_scores[0]
    #     }

    # Apply to your DataFrame
    emotion_scores = data[text_column].apply(get_emotion_scores)

    # Convert list of dicts to a DataFrame
    emotion_df = pd.DataFrame(emotion_scores.tolist())

    # Concatenate the emotion scores back to your original DataFrame
    data_with_emotions = pd.concat([data.reset_index(drop=True), emotion_df], axis=1)

    # Now data_with_emotions has columns like: 'love', 'neutral' etc.
    # Save it
    data_with_emotions.to_csv('data_with_emotions.csv')

    return data_with_emotions

def mental_health_classification(data, text_column = 'chunk'):

    from huggingface_hub import login
    login(token="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")


    #It needs aggregated data - not single messages

    # tokenizer = AutoTokenizer.from_pretrained("Xuhui/mental-health-bert")
    # model = AutoModelForSequenceClassification.from_pretrained("Xuhui/mental-health-bert")
    # mental_health_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Load the tokenizer and model
    model_name = "mental/mental-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Use a pipeline as a high-level helper
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0 if torch.cuda.is_available() else -1)

    # Function to get prediction for a message
    def classify_mental_health(text):
        try:
            result = pipeline(text[:512])  # Truncate to max 512 tokens - not necessary if chunked properly
            return result[0]  # Return scores for all classes
        except Exception as e:
            print(f"Error: {e} — for message: {text}")
            return None


    # Apply to your WhatsApp messages
    data['mental_scores'] = data[text_column].apply(classify_mental_health)

    # convert the list of dicts into separate columns
    def extract_scores(score_list):
        if isinstance(score_list, list):
            return {item['label']: item['score'] for item in score_list}
        else:
            return {}

    mental_scores_df = data['mental_scores'].apply(extract_scores).apply(pd.Series)
    data = pd.concat([data, mental_scores_df], axis=1)

    #save it
    data.to_csv("data_mental.csv", index=False)

    return data


def mental_health_multi_classification(data, text_column = 'chunk'):

    # from huggingface_hub import login
    # login(token="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")

    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    pipeline = pipeline("text-classification", model=model_name, return_all_scores=True)

    # tokenizer = AutoTokenizer.from_pretrained("Xuhui/mental-health-bert")
    # model = AutoModelForSequenceClassification.from_pretrained("Xuhui/mental-health-bert")
    # mental_health_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # # Load the tokenizer and model
    # model_name = "mental/mental-bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # # Use a pipeline as a high-level helper
    # pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0 if torch.cuda.is_available() else -1)

    # Function to get prediction for a message
    def classify_mental_health(text):
        try:
            result = pipeline(text[:512])  # Truncate to max 512 tokens - not necessary if chunked properly
            return result[0]  # Return scores for all classes
        except Exception as e:
            print(f"Error: {e} — for message: {text}")
            return None


    # Apply to your WhatsApp messages
    data['mental_scores'] = data[text_column].apply(classify_mental_health)

    # convert the list of dicts into separate columns
    def extract_scores(score_list):
        if isinstance(score_list, list):
            return {item['label']: item['score'] for item in score_list}
        else:
            return {}

    mental_scores_df = data['mental_scores'].apply(extract_scores).apply(pd.Series)
    data = pd.concat([data, mental_scores_df], axis=1)

    #save it
    data.to_csv("data_multi_mental.csv", index=False)

    return data




# ### DATA
# file_path = "group_chat_history.txt"
# data, _ = parse_whatsapp_chat(file_path)
# chunked_df = chunk_messages_by_token_limit(data, tokenizer_name="bert-base-uncased", max_chunk_size=512)
# print(chunked_df.head())

##### SINGLE MESSAGES
data = pd.read_csv('group_chat_cleaned.csv')
# data = add_sentiment_analysis_by_vader(data)
data = hugging_face_emotion_classification(data)

###### AGGREGATED DATA
# data = pd.read_csv('aggregated_data_by_user.csv')
# data = mental_health_classification(data)

### CHUNKED DATA
# data = pd.read_csv('chunked_data.csv')
# data = mental_health_classification(data, text_column='chunk')
# data = hugging_face_emotion_classification(data, text_column='chunk')



