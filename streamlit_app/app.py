import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import sys
from datetime import timedelta

# Add utils to path
# sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from utils.messages_dual_radial_bars import create_messages_dual_radial_bars
from utils.sentiment_heatmap import plot_sentiment_heatmap

# TODO need to check the chunk dataset and the dates format as well

@st.cache_data
def upload_history_chat() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "output_data" / "prep_logs" / "group_chat_merged_consecutive.csv"
    # csv_path = Path(__file__).parent.parent / "output_data" / "prep_logs" / "chunked_data.csv"
    try:
        chat_df = pd.read_csv(csv_path)
        sender_rename = {"IvanDB": "Ivan", "Richard McBride": "Richard"}
        if "sender" in chat_df.columns:
            chat_df["sender"] = chat_df["sender"].replace(sender_rename)
        if "chunk" in chat_df.columns:
            chat_df = chat_df.rename(columns={"chunk": "message"})
        return chat_df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def upload_sentiment_results() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "output_data" / "sentiment" / "sentiment_analysis_results_consecutive.csv"
    try:
        sentiment_df = pd.read_csv(csv_path)
        sender_rename = {"IvanDB": "Ivan", "Richard McBride": "Richard"}
        if "sender" in sentiment_df.columns:
            sentiment_df["sender"] = sentiment_df["sender"].replace(sender_rename)
        return sentiment_df
    except Exception:
        return pd.DataFrame()


def _parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


st.set_page_config(page_title="Group Chat Analysis tool", layout="wide")

# Custom color palette
CUSTOM_COLORS = ["#dd6e42", "#e8dab2", "#4f6d7a", "#c0d6df"]

def main():
    st.title("Group Chat Analysis tool")

    chat_df = upload_history_chat()
    sentiment_df = upload_sentiment_results()

    if not chat_df.empty:
        chat_df = _parse_date_column(chat_df)
    if not sentiment_df.empty:
        sentiment_df = _parse_date_column(sentiment_df)

    selected_senders = []
    selected_date_range = None

    # Sidebar
    with st.sidebar:
        images_dir = Path(__file__).parent / "images"
        logo_path = images_dir / "icon_social.png"
        if logo_path.exists():
            st.image(str(logo_path), width=300)
        else:
            st.warning("Logo image not found.")

        choice = st.radio("Select view", ("Explore chat history", "Topic Analysis", "Sentiment analysis", "Mental health analysis"))

        if chat_df.empty and sentiment_df.empty:
            st.error("Could not load chat history or sentiment results. Please check `output_data/prep_logs` and `output_data/sentiment`.")
        else:
            df_for_filters = chat_df if not chat_df.empty else sentiment_df

            if "date" in df_for_filters.columns:
                min_date = df_for_filters["date"].dropna().min()
                max_date = df_for_filters["date"].dropna().max()
                if pd.notna(min_date) and pd.notna(max_date):
                    # Convert to date objects if needed
                    min_date_obj = min_date.date() if hasattr(min_date, "date") else min_date
                    max_date_obj = max_date.date() if hasattr(max_date, "date") else max_date
                    
                    # Date range presets based on max_date (not today)
                    date_preset = st.radio(
                        "Date range preset",
                        ("Past Week", "Past Month", "Past 6 Months", "All History", "Custom")
                    )
                    
                    if date_preset == "Past Week":
                        start_date_preset = max_date_obj - timedelta(days=7)
                        selected_date_range = (start_date_preset, max_date_obj)
                    elif date_preset == "Past Month":
                        start_date_preset = max_date_obj - timedelta(days=30)
                        selected_date_range = (start_date_preset, max_date_obj)
                    elif date_preset == "Past 6 Months":
                        start_date_preset = max_date_obj - timedelta(days=180)
                        selected_date_range = (start_date_preset, max_date_obj)
                    elif date_preset == "All History":
                        selected_date_range = (min_date_obj, max_date_obj)
                    else:  # Custom
                        selected_date_range = st.date_input(
                            "Select custom date range",
                            value=(max_date_obj - timedelta(days=180), max_date_obj),
                            min_value=min_date_obj,
                            max_value=max_date_obj,
                            format="DD/MM/YYYY",
                        )
                else:
                    st.warning("The loaded data does not contain a valid date range.")
            else:
                st.warning("No `date` column found in the available data.")

            if "sender" in df_for_filters.columns:
                unique_senders = sorted(df_for_filters["sender"].dropna().astype(str).unique())
                if unique_senders:
                    selected_senders = st.multiselect("Select senders", unique_senders, default=unique_senders)
                else:
                    st.warning("No senders found in the available data.")
            else:
                st.warning("No `sender` column found in the available data.")

    # Main content - empty placeholders
    if choice == "Explore chat history":
        if chat_df.empty:
            st.error("Could not find or load the chat history DataFrame.")
        else:
            filtered_df = chat_df.copy()
            if selected_senders:
                filtered_df = filtered_df[filtered_df["sender"].isin(selected_senders)]
            if selected_date_range is not None and len(selected_date_range) == 2 and "date" in filtered_df.columns:
                start_date, end_date = selected_date_range
                filtered_df = filtered_df[
                    (filtered_df["date"] >= pd.to_datetime(start_date)) &
                    (filtered_df["date"] <= pd.to_datetime(end_date))
                ]

            st.header("Default History Chat")
            total_messages = len(filtered_df)
            sender_counts = filtered_df["sender"].value_counts() if "sender" in filtered_df.columns else pd.Series(dtype=int)
            top_sender = sender_counts.idxmax() if not sender_counts.empty else "n/a"
            active_senders = len(sender_counts)
            day_range = "n/a"
            if "date" in filtered_df.columns and total_messages > 0:
                visible_dates = filtered_df["date"].dropna()
                if not visible_dates.empty:
                    day_range = f"{visible_dates.min().date()} → {visible_dates.max().date()}"

            col1, col2, col3 = st.columns(3)
            col1.metric("Total messages", total_messages)
            col2.metric("Active senders", active_senders)
            col3.metric("Top sender", top_sender)

            if not sender_counts.empty:
                sender_df = sender_counts.rename_axis("sender").reset_index(name="count").sort_values("count", ascending=True)
                fig_bar = px.bar(
                    sender_df,
                    y="sender",
                    x="count",
                    orientation="h",
                    title="Total messages by sender",
                    labels={"count": "Number of messages", "sender": "Sender"},
                    color="sender",
                    color_discrete_sequence=CUSTOM_COLORS,
                )
                fig_bar.update_layout(showlegend=False, height=max(300, len(sender_counts) * 40))
                st.plotly_chart(fig_bar, use_container_width=True)

            if "date" in filtered_df.columns and not filtered_df["date"].dropna().empty:
                daily_counts = (
                    filtered_df.groupby([filtered_df["date"].dt.date, "sender"])
                    .size()
                    .reset_index(name="count")
                    .rename(columns={0: "date"})
                )
                daily_counts["date"] = pd.to_datetime(daily_counts["date"])
                fig_line = px.line(
                    daily_counts,
                    x="date",
                    y="count",
                    color="sender",
                    title="Messages per day by sender",
                    labels={"count": "Number of messages", "date": "Date"},
                    markers=True,
                    color_discrete_sequence=CUSTOM_COLORS,
                )
                fig_line.update_layout(hovermode="x unified", height=400)
                st.plotly_chart(fig_line, use_container_width=True)

            st.write(f"Showing {len(filtered_df)} rows")
            st.dataframe(filtered_df.head(20))

            search_query = st.text_input("Search messages", "")
            if st.button("Search"):
                if search_query.strip() == "":
                    st.warning("Enter a search query to filter messages.")
                elif "message" not in filtered_df.columns:
                    st.warning("No `message` column found in the chat history.")
                else:
                    search_df = filtered_df[
                        filtered_df["message"].astype(str).str.contains(search_query, case=False, na=False)
                    ]
                    if "date" in search_df.columns:
                        search_df = search_df.sort_values("date")
                    st.subheader("Search results")
                    st.write(f"Found {len(search_df)} matching rows")
                    st.dataframe(search_df.head(100))

            # Messages by hour chart
            st.subheader("Messages by Hour")
            if "time" in filtered_df.columns and not filtered_df["time"].dropna().empty:
                # Extract hour from time string (format: HH:MM:SS)
                filtered_df["hour"] = pd.to_datetime(filtered_df["time"], format="%H:%M:%S", errors="coerce").dt.hour
                hourly_counts = filtered_df.groupby("hour").size()
                
                # Create array of 24 hours with counts (0 if no messages in that hour)
                counts = [hourly_counts.get(hour, 0) for hour in range(24)]
                
                # Create and display the chart
                fig_radial = create_messages_dual_radial_bars(counts)
                fig_radial.update_layout(height=600)
                st.plotly_chart(fig_radial, width='stretch')
            else:
                st.warning("No time information available to create hourly message chart.")

    elif choice == "Topic Analysis":
        if chat_df.empty:
            st.error("Could not find or load the chat history DataFrame.")
        else:
            filtered_df = chat_df.copy()
            if selected_senders:
                filtered_df = filtered_df[filtered_df["sender"].isin(selected_senders)]
            if selected_date_range is not None and len(selected_date_range) == 2 and "date" in filtered_df.columns:
                start_date, end_date = selected_date_range
                filtered_df = filtered_df[
                    (filtered_df["date"] >= pd.to_datetime(start_date)) &
                    (filtered_df["date"] <= pd.to_datetime(end_date))
                ]

            st.header("Topic Analysis")

            if "message" in filtered_df.columns and not filtered_df.empty:
                all_messages = " ".join(filtered_df["message"].astype(str).dropna())
                if all_messages.strip():
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        colormap="viridis"
                    ).generate(all_messages)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation="bilinear")
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)

            topic_query = st.text_input("Topic messages", "")
            if st.button("Search topic"):
                if topic_query.strip() == "":
                    st.warning("Enter a topic query to filter messages.")
                elif "message" not in filtered_df.columns:
                    st.warning("No `message` column found in the chat history.")
                else:
                    topic_df = filtered_df[
                        filtered_df["message"].astype(str).str.contains(topic_query, case=False, na=False)
                    ]
                    if "date" in topic_df.columns:
                        topic_df = topic_df.sort_values("date")

                    st.write(f"Found {len(topic_df)} matching rows")
                    st.dataframe(topic_df.head(100))

                    if "sender" in topic_df.columns and not topic_df.empty:
                        counts = topic_df["sender"].value_counts().rename_axis("sender").reset_index(name="count")
                        counts["percentage"] = counts["count"] / counts["count"].sum() * 100
                        fig, ax = plt.subplots()
                        ax.pie(counts["count"], labels=counts["sender"], autopct="%1.1f%%")
                        ax.axis("equal")
                        st.subheader("Topic frequency by sender")
                        st.pyplot(fig)
                        st.write(counts)
    elif choice == "Sentiment analysis":
        st.header("Sentiment analysis")
        sentiment_source = st.selectbox(
            "Sentiment source",
            ["VADER", "Twitter-RoBERTa"],
            index=0,
            help="Choose which sentiment analysis source to use for both the compound trend chart and the heatmap."
        )

        source_columns = {
            "VADER": {
                "pos": "vader_pos",
                "neg": "vader_neg",
                "neu": "vader_neu",
                "compound": "vader_compound",
            },
            "Twitter-RoBERTa": {
                "pos": "twitter_roberta_positive",
                "neg": "twitter_roberta_negative",
                "neu": "twitter_roberta_neutral",
                "compound": "twitter_roberta_compound",
            },
        }
        selected_cols = source_columns[sentiment_source]

        if sentiment_source == "VADER":
            st.markdown(
                "**VADER** is a rule-based lexicon and sentiment scoring system designed for social media text. "
                "It uses a dictionary of sentiment-laden words and heuristics for punctuation, capitalization, degree modifiers, and negation. "
                "VADER returns positive, negative, neutral scores, plus a normalized `compound` score in the range [-1, 1]. "
                "A compound score near +1 means strongly positive sentiment, near -1 means strongly negative sentiment, and near 0 means neutral or mixed sentiment."
            )
        else:
            st.markdown(
                "**Twitter-RoBERTa** is a transformer-based model fine-tuned on Twitter sentiment data. "
                "It outputs probability-like scores for positive, negative, and neutral sentiment. "
                "The derived `twitter_roberta_compound` score is calculated as `positive - negative`, so positive values indicate more positive polarity and negative values indicate more negative polarity. "
                "Because RoBERTa scores are probabilities, the resulting net values can be sparse and the heatmap may appear flat if most predictions are neutral."
            )

        st.markdown(
            "A **rolling mean** smooths the compound score over a moving window of days or weeks. "
            "This makes the trend easier to read by reducing short-term spikes and noise while preserving the overall sentiment direction."
        )

        if sentiment_df.empty:
            st.error("Could not load sentiment analysis results from `output_data/sentiment/sentiment_analysis_results_consecutive.csv`.")
        else:
            filtered_sentiment = sentiment_df.copy()
            if selected_senders:
                filtered_sentiment = filtered_sentiment[filtered_sentiment["sender"].isin(selected_senders)]

            if selected_date_range is not None and len(selected_date_range) == 2 and "date" in filtered_sentiment.columns:
                start_date, end_date = selected_date_range
                filtered_sentiment = filtered_sentiment[
                    (filtered_sentiment["date"] >= pd.to_datetime(start_date)) &
                    (filtered_sentiment["date"] <= pd.to_datetime(end_date))
                ]

            missing_cols = [col for col in selected_cols.values() if col not in filtered_sentiment.columns]
            if missing_cols:
                st.warning(
                    f"The selected source '{sentiment_source}' is not available in the loaded sentiment data. Missing columns: {', '.join(missing_cols)}."
                )
            else:
                compound_col = selected_cols["compound"]
                filtered_sentiment = filtered_sentiment.dropna(subset=["date", compound_col]).sort_values("date")

                if filtered_sentiment.empty:
                    st.warning("No sentiment data available after applying the selected filters.")
                else:
                    agg_freq = st.selectbox("Aggregation frequency", ["Daily", "Weekly"], index=0)
                    smoothing_window = st.slider(
                        "Rolling average window",
                        min_value=1,
                        max_value=21,
                        value=7,
                        help="Smoothing window for the compiled mean compound sentiment time series.",
                    )

                    resample_rule = "D" if agg_freq == "Daily" else "W-MON"
                    grouped = (
                        filtered_sentiment.set_index("date")
                        .groupby("sender")[compound_col]
                        .resample(resample_rule)
                        .mean()
                        .rename("mean_compound")
                        .reset_index()
                    )

                    grouped["rolling_mean"] = grouped.groupby("sender")["mean_compound"].transform(
                        lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
                    )

                    sender_count = grouped["sender"].nunique()
                    chart_title = f"{agg_freq} mean {sentiment_source} compound sentiment (smoothed over {smoothing_window})"
                    if sender_count <= 6:
                        fig_sentiment = px.line(
                            grouped,
                            x="date",
                            y="rolling_mean",
                            color="sender",
                            title=chart_title,
                            labels={"rolling_mean": "Smoothed compound score", "date": "Date"},
                            markers=True,
                        )
                        fig_sentiment.update_layout(hovermode="x unified", height=450)
                    else:
                        fig_sentiment = px.line(
                            grouped,
                            x="date",
                            y="rolling_mean",
                            color="sender",
                            facet_col="sender",
                            facet_col_wrap=2,
                            title=chart_title,
                            labels={"rolling_mean": "Smoothed compound score", "date": "Date"},
                        )
                        fig_sentiment.update_layout(height=320 * ((sender_count + 1) // 2), showlegend=False)

                    st.plotly_chart(fig_sentiment, use_container_width=True)

                    st.subheader("Sentiment heatmap")
                    if "time" not in filtered_sentiment.columns:
                        st.warning("No `time` column available for the heatmap. The heatmap requires date+time information.")
                    else:
                        net_values = filtered_sentiment[selected_cols["pos"]].fillna(0) - filtered_sentiment[selected_cols["neg"]].fillna(0)
                        if (net_values != 0).sum() < len(net_values) * 0.05:
                            st.info(
                                "Twitter-RoBERTa net values are sparse in this dataset, so the heatmap may appear close to zero. "
                                "This is expected when most predictions are neutral or when the model only produces a few non-neutral probabilities."
                            )
                        try:
                            fig_heat, heatmap_df, df_out = plot_sentiment_heatmap(
                                filtered_sentiment,
                                date_col="date",
                                time_col="time",
                                message_col="message",
                                pos_col=selected_cols["pos"],
                                neg_col=selected_cols["neg"],
                                neu_col=selected_cols["neu"],
                                metric="net",
                                agg="mean"
                            )
                            st.plotly_chart(fig_heat, use_container_width=True)
                        except Exception as exc:
                            st.error(f"Unable to render sentiment heatmap: {exc}")

                    sentiment_summary = (
                        grouped.groupby("sender")["mean_compound"]
                        .mean()
                        .reset_index(name="avg_mean_compound")
                        .sort_values("avg_mean_compound", ascending=False)
                    )

                    st.subheader("Average compound sentiment by sender")
                    st.dataframe(sentiment_summary)

                    emotion_cols = [col for col in filtered_sentiment.columns if col.startswith("emotion_") and col!="emotion_neutral"]
                    if emotion_cols:
                        st.subheader("Emotion Analysis")
                        st.markdown(
                            "This chart shows the average emotion scores detected by the `j-hartmann/emotion-english-distilroberta-base` model. "
                            "The model predicts a probability-like score for each emotion label, and each radar chart below visualizes the mean strength of each emotion for that sender."
                        )
                        
                        # Calculate emotion scores per sender
                        senders = filtered_sentiment["sender"].dropna().unique()
                        for sender in sorted(senders):
                            sender_data = filtered_sentiment[filtered_sentiment["sender"] == sender]
                            emotion_summary = (
                                sender_data[emotion_cols]
                                .astype(float)
                                .mean(skipna=True)
                                .reset_index(name="score")
                                .rename(columns={"index": "emotion"})
                            )
                            # Sort by emotion name for consistent radar chart ordering
                            emotion_summary = emotion_summary.sort_values("emotion")
                            
                            # Calculate dynamic radialaxis max based on the highest emotion score
                            max_score = emotion_summary["score"].max()
                            # Round up to make it slightly bigger than max to ensure the border is close to max
                            radial_max = max(max_score * 1.1, 0.1)
                            
                            fig_emotion = go.Figure(
                                data=[
                                    go.Scatterpolar(
                                        r=emotion_summary["score"].fillna(0),
                                        theta=emotion_summary["emotion"].str.replace("emotion_", "", regex=False),
                                        fill="toself",
                                        name=f"Emotion scores",
                                    )
                                ]
                            )
                            fig_emotion.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, radial_max]),
                                ),
                                showlegend=False,
                                title=f"Emotion scores - {sender}",
                                height=500
                            )
                            st.plotly_chart(fig_emotion, use_container_width=True)
                    else:
                        st.warning("No emotion score columns were found in the sentiment data to build the Emotion Analysis radar chart.")
    elif choice == "Mental health analysis":
        st.header("Mental health analysis")
        st.write("Placeholder for mental health analysis tools and visualizations.")

if __name__ == "__main__":
    main()

