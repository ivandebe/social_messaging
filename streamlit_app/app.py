import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import sys
from datetime import timedelta

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


@st.cache_data
def upload_mental_health_results() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "output_data" / "mental" / "group_chat_merged_consecutive_mental_health_scores.csv"
    try:
        mental_df = pd.read_csv(csv_path)
        sender_rename = {"IvanDB": "Ivan", "Richard McBride": "Richard"}
        if "sender" in mental_df.columns:
            mental_df["sender"] = mental_df["sender"].replace(sender_rename)
        return mental_df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def upload_topic_results() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / "output_data" / "topic" / "topic_consecutive.csv"
    try:
        topic_df = pd.read_csv(csv_path)
        sender_rename = {"IvanDB": "Ivan", "Richard McBride": "Richard"}
        if "sender" in topic_df.columns:
            topic_df["sender"] = topic_df["sender"].replace(sender_rename)
        return topic_df
    except Exception:
        return pd.DataFrame()


def _aggregate_mental_health_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_date_column(df)
    if "date" not in df.columns:
        return pd.DataFrame()
    agg_columns = [col for col in ["anxiety_score", "depression_score", "suicidal_score", "stress_score", "normal_score"] if col in df.columns]
    if not agg_columns:
        return pd.DataFrame()
    if "sender" in df.columns:
        grouped = (
            df.dropna(subset=["date"])
            .groupby([df["sender"], df["date"].dt.normalize()])[agg_columns]
            .mean()
            .reset_index()
            .rename(columns={"date": "date"})
        )
    else:
        grouped = (
            df.dropna(subset=["date"])
            .groupby(df["date"].dt.normalize())[agg_columns]
            .mean()
            .reset_index()
        )
    grouped = grouped.sort_values(["sender", "date"]) if "sender" in grouped.columns else grouped.sort_values("date")
    return grouped


def _compute_baseline_threshold(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    baseline_rows = []
    for sender, sender_df in df.groupby("sender"):
        sender_df = sender_df.sort_values("date")
        row = {"sender": sender}
        for metric in metrics:
            if metric not in sender_df.columns:
                continue
            baseline = sender_df[metric].mean()
            std_dev = sender_df[metric].std(ddof=0)
            threshold = baseline + max(0.05, std_dev)
            row[f"{metric}_baseline"] = baseline
            row[f"{metric}_threshold"] = threshold
        baseline_rows.append(row)
    return pd.DataFrame(baseline_rows)


def _compute_mental_health_alerts(message_df: pd.DataFrame, baseline_df: pd.DataFrame, metrics: list[str], amber_pct: float = 0.15, red_pct: float = 0.25) -> pd.DataFrame:
    alerts = []
    for sender, sender_df in message_df.groupby("sender"):
        if sender_df.empty:
            continue
        sender_baseline = baseline_df[baseline_df["sender"] == sender]
        if sender_baseline.empty:
            continue
        total_messages = len(sender_df)
        if total_messages == 0:
            continue
        for metric in metrics:
            if metric not in sender_df.columns:
                continue
            baseline = sender_baseline[f"{metric}_baseline"].iloc[0]
            threshold = sender_baseline[f"{metric}_threshold"].iloc[0]
            count_above = (sender_df[metric] > threshold).sum()
            pct_above = count_above / total_messages
            if pct_above >= red_pct:
                status = "Red"
            elif pct_above >= amber_pct:
                status = "Amber"
            else:
                status = "Green"
            alerts.append({
                "sender": sender,
                "metric": metric,
                "baseline": baseline,
                "threshold": threshold,
                "messages_above_threshold": int(count_above),
                "message_count": int(total_messages),
                "pct_above_threshold": pct_above,
                "status": status,
            })
    return pd.DataFrame(alerts)


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
    topic_df = upload_topic_results()
    mental_df = upload_mental_health_results()

    if not chat_df.empty:
        chat_df = _parse_date_column(chat_df)
    if not sentiment_df.empty:
        sentiment_df = _parse_date_column(sentiment_df)
    if not topic_df.empty:
        topic_df = _parse_date_column(topic_df)
    if not mental_df.empty:
        mental_df = _parse_date_column(mental_df)

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
                st.plotly_chart(fig_bar, width="stretch")

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
                st.plotly_chart(fig_line, width="stretch")

            st.write(f"Showing {len(filtered_df)} rows")
            st.dataframe(filtered_df.head(20))

            st.subheader("Search messages by exact word or phrase")
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
        if topic_df.empty:
            st.error("Could not find or load the topic analysis DataFrame.")
        else:
            filtered_df = topic_df.copy()

            show_outliers = st.checkbox("Include outliers", value=False)
            if not show_outliers:
                filtered_df = filtered_df[filtered_df["topic"] != -1]

            if selected_senders:
                filtered_df = filtered_df[filtered_df["sender"].isin(selected_senders)]
            if selected_date_range is not None and len(selected_date_range) == 2 and "date" in filtered_df.columns:
                start_date, end_date = selected_date_range
                filtered_df = filtered_df[
                    (filtered_df["date"] >= pd.to_datetime(start_date)) &
                    (filtered_df["date"] <= pd.to_datetime(end_date))
                ]

            st.header("Topic Analysis")

            # WordCloud
            # if "message" in filtered_df.columns and not filtered_df.empty:
            #     all_messages = " ".join(filtered_df["message"].astype(str).dropna())
            #     if all_messages.strip():
            #         wordcloud = WordCloud(
            #             width=800,
            #             height=400,
            #             background_color="white",
            #             colormap="viridis"
            #         ).generate(all_messages)
            #         fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            #         ax_wc.imshow(wordcloud, interpolation="bilinear")
            #         ax_wc.axis("off")
            #         st.pyplot(fig_wc)

            topic_options = (filtered_df["topic_label"].value_counts().index.tolist())


            # KPIs
            total_messages = len(filtered_df)
            unique_topics = filtered_df["topic_label"].nunique()

            if total_messages > 0:
                top_topic = filtered_df["topic_label"].value_counts().idxmax()
                outlier_share = (filtered_df["topic"].eq(-1).mean()) * 100
            else:
                top_topic = "N/A"
                outlier_share = 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Messages", f"{total_messages:,}")
            c2.metric("Unique topics", unique_topics)
            c3.metric("Top topic", top_topic)
            c4.metric("Outlier %", f"{outlier_share:.1f}%")

            # ----------------------------
            # Top topics bar chart
            # ----------------------------
            st.subheader("Most discussed topics")

            topic_counts = (
                filtered_df["topic_label"]
                .value_counts()
                .reset_index()
            )
            topic_counts.columns = ["topic_label", "message_count"]

            fig_bar = px.bar(
                topic_counts.head(15),
                x="message_count",
                y="topic_label",
                orientation="h",
                title="Top 15 topics by message count",
            )
            fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_bar, width="stretch")

            # ----------------------------
            # Topic trend over time
            # ----------------------------
            st.subheader("Topic activity over time")

            trend = (
                filtered_df.groupby(["date", "topic_label"])
                .size()
                .reset_index(name="message_count")
            )

            top_trend_topics = (
                filtered_df["topic_label"]
                .value_counts()
                .head(7)
                .index
                .tolist()
            )

            trend = trend[trend["topic_label"].isin(top_trend_topics)]

            fig_line = px.line(
                trend,
                x="date",
                y="message_count",
                color="topic_label",
                markers=True,
                title="Top topic trends over time (top 7 topics by message count)",
                labels={"message_count": "Number of messages", "date": "Date"},
            )

            fig_line.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=100, b=120, l=40, r=40)
            )

            st.plotly_chart(fig_line, width="stretch")


            # ----------------------------
            # Sender-topic heatmap
            # ----------------------------
            st.subheader("Who talks about what")

            heatmap_df = (
                filtered_df.groupby(["sender", "topic_label"])
                .size()
                .reset_index(name="message_count")
            )

            top_heatmap_topics = (
                filtered_df["topic_label"]
                .value_counts()
                .head(10)
                .index
                .tolist()
            )
            heatmap_df = heatmap_df[heatmap_df["topic_label"].isin(top_heatmap_topics)]

            if not heatmap_df.empty:
                pivot_df = heatmap_df.pivot(
                    index="sender",
                    columns="topic_label",
                    values="message_count"
                ).fillna(0)

                fig_heatmap = px.imshow(
                    pivot_df,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Messages by sender and topic",
                )
                st.plotly_chart(fig_heatmap, width='stretch')

            # ----------------------------
            # Raw messages table
            # ----------------------------
            st.subheader("Messages")

            topic_for_table = st.selectbox(
                "Inspect one topic",
                options=["All"] + topic_options
            )

            table_df = filtered_df.copy()
            if topic_for_table != "All":
                table_df = table_df[table_df["topic_label"] == topic_for_table]

            st.dataframe(
                table_df[["date", "sender", "topic", "topic_label", "message"]]
                .sort_values("date", ascending=False)
                .reset_index(drop=True),
                width="stretch",
                height=500,
            )


            # st.subheader("Topic label selector")
            # selected_topics = st.sidebar.multiselect(
            #     "Topic label",
            #     options=topic_options,
            #     default=topic_options[:10] if len(topic_options) > 10 else topic_options,
            #     help="Filter messages by topic label. Showing top 10 most frequent topics by default."
            #     )

            topic_query = st.text_input("Topic messages", "")
            if st.button("Search topic"):
                if topic_query.strip() == "":
                    st.warning("Enter a topic query to filter messages.")
                elif "topic_label" not in filtered_df.columns:
                    st.warning("No `topic_label` column found in the topic analysis.")
                else:
                    topic_df = filtered_df[
                        filtered_df["topic_label"].astype(str).str.contains(topic_query, case=False, na=False)
                    ]
                    if "date" in topic_df.columns:
                        topic_df = topic_df.sort_values("date")

                    st.write(f"Found {len(topic_df)} matching rows. Only show up to 100 rows.")
                    st.dataframe(topic_df.head(100))

                    # # Pie chart and count table
                    # if "sender" in topic_df.columns and not topic_df.empty:
                    #     counts = topic_df["sender"].value_counts().rename_axis("sender").reset_index(name="count")
                    #     counts["percentage"] = counts["count"] / counts["count"].sum() * 100
                    #     fig, ax = plt.subplots()
                    #     ax.pie(counts["count"], labels=counts["sender"], autopct="%1.1f%%")
                    #     ax.axis("equal")
                    #     st.subheader("Topic frequency by sender")
                    #     st.pyplot(fig)
                    #     st.write(counts)

                    # Pie chart and count table
                    if "sender" in topic_df.columns and not topic_df.empty:
                        counts = (
                            topic_df["sender"]
                            .value_counts()
                            .rename_axis("sender")
                            .reset_index(name="count")
                        )
                        counts["percentage"] = counts["count"] / counts["count"].sum() * 100

                        st.subheader("Topic frequency by sender")

                        fig = px.pie(
                            counts,
                            names="sender",
                            values="count",
                            title="Topic frequency by sender",
                            hole=0.45,
                            color_discrete_sequence=CUSTOM_COLORS
                        )

                        fig.update_traces(
                            textinfo="label+value+percent",
                            texttemplate="%{label}<br>Count: %{value}<br>%{percent}",
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                        )
                        
                        fig.update_layout(
                            width=950,
                            height=750,
                            font=dict(size=16),
                            margin=dict(t=80, b=40, l=40, r=40)
                        )

                        st.plotly_chart(fig, width='stretch')


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

                    st.plotly_chart(fig_sentiment, width="stretch")

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
                            st.plotly_chart(fig_heat, width="stretch")
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
                            st.plotly_chart(fig_emotion, width="stretch")
                    else:
                        st.warning("No emotion score columns were found in the sentiment data to build the Emotion Analysis radar chart.")
    elif choice == "Mental health analysis":
        st.header("Mental health analysis")

        if mental_df.empty:
            st.error(
                "Could not load mental health results from `output_data/mental/group_chat_merged_consecutive_mental_health_scores.csv`. "
                "Please ensure the file exists and has the expected columns."
            )
        else:
            filtered_mental = mental_df.copy()

            if selected_senders:
                filtered_mental = filtered_mental[filtered_mental["sender"].isin(selected_senders)]

            if selected_date_range is not None and len(selected_date_range) == 2 and "date" in filtered_mental.columns:
                start_date, end_date = selected_date_range
                filtered_mental = filtered_mental[
                    (filtered_mental["date"] >= pd.to_datetime(start_date)) &
                    (filtered_mental["date"] <= pd.to_datetime(end_date))
                ]

            if filtered_mental.empty:
                st.warning("No mental health data available for the selected senders/date range.")
            else:
                daily_agg = _aggregate_mental_health_daily(filtered_mental)
                if daily_agg.empty:
                    st.warning("Unable to aggregate mental health scores by date. Check the data columns.")
                else:
                    st.subheader("Mental health score highlights by sender")
                    senders = sorted(daily_agg["sender"].dropna().unique())
                    metric_display_names = {
                        "anxiety_score": "Anxiety",
                        "depression_score": "Depression",
                        "stress_score": "Stress",
                        "normal_score": "Normal",
                    }
                    metric_order = ["anxiety_score", "depression_score", "stress_score", "normal_score"]
                    for sender in senders:
                        sender_row = daily_agg[daily_agg["sender"] == sender]
                        if sender_row.empty:
                            continue
                        latest_values = sender_row.sort_values("date").iloc[-1]
                        st.markdown(f"#### {sender}")
                        cols = st.columns(4)
                        for panel_col, metric_name in zip(cols, metric_order):
                            if metric_name not in sender_row.columns:
                                panel_col.warning(f"Missing {metric_name}")
                                continue
                            latest_value = latest_values[metric_name]
                            start_value = sender_row[metric_name].iloc[0] if len(sender_row) > 0 else 0.0
                            delta_value = latest_value - start_value
                            delta_color = "inverse" if metric_name == "stress_score" else "normal"
                            panel_col.metric(
                                metric_display_names[metric_name],
                                f"{latest_value:.3f}",
                                f"{delta_value:+.3f} since first",
                                delta_color=delta_color,
                            )

                    st.markdown(
                        "### Daily mental health summary and message-level drilldown"
                    )
                    smoothing_window = st.slider(
                        "Rolling average window (days)",
                        min_value=1,
                        max_value=30,
                        value=7,
                        help="Smooth the daily trend data with a rolling average.",
                    )

                    selected_metric = st.selectbox(
                        "Select mental health score to plot",
                        options=["anxiety_score", "depression_score", "stress_score", "normal_score"],
                        format_func=lambda x: metric_display_names.get(x, x),
                        index=0,
                    )
                    if selected_metric not in daily_agg.columns:
                        st.warning(f"Selected metric `{selected_metric}` is not available in the data.")
                    else:
                        metric_df = daily_agg.copy()
                        metric_df["metric_rolling"] = metric_df.groupby("sender")[selected_metric].transform(
                            lambda x: x.rolling(window=smoothing_window, min_periods=1).mean()
                        )
                        st.subheader(f"{metric_display_names.get(selected_metric, selected_metric)} trend by sender")
                        fig_metric = px.line(
                            metric_df,
                            x="date",
                            y="metric_rolling",
                            color="sender",
                            markers=True,
                            title=f"Rolling {metric_display_names.get(selected_metric, selected_metric)} scores by sender",
                            labels={"metric_rolling": f"Rolling {metric_display_names.get(selected_metric, selected_metric)} score", "date": "Date", "sender": "Sender"},
                        )
                        fig_metric.update_layout(hovermode="x unified", height=500)
                        st.plotly_chart(fig_metric, width="stretch")

                st.subheader("Message-level mental health predictions")
                st.markdown(
                    "Use this table to drill down into the individual message predictions while preserving the daily aggregated trend above."
                )
                alert_source_for_labels = filtered_mental.copy()
                top_label_options = [
                    label for label in sorted(filtered_mental["mental_health_top_label"].dropna().unique().tolist())
                    if label.lower() not in {"normal", "suicidal"}
                ]
                if top_label_options:
                    selected_top_label = st.selectbox("Filter by predicted mental health label", top_label_options, index=0)
                    filtered_mental = filtered_mental[filtered_mental["mental_health_top_label"] == selected_top_label]
                else:
                    selected_top_label = None

                display_columns = [
                    col for col in ["date", "time", "sender", "mental_health_top_label", "mental_health_top_score", "message"]
                    if col in filtered_mental.columns
                ]
                st.write(f"Showing {len(filtered_mental)} message rows")
                with st.expander("Show message-level data", expanded=False):
                    st.dataframe(filtered_mental[display_columns].sort_values("date", ascending=False).reset_index(drop=True))

                st.subheader("Alert Logic and Sender Baselines")
                st.markdown(
                    "These alert signals are based on all messages in the current selected date range. "
                    "Each sender's baseline is calculated from their full historical mental health profile, and the threshold is defined as baseline + one standard deviation. "
                    "A sender receives an amber signal when at least 15% of the current range messages exceed the metric threshold, and a red signal when at least 25% exceed it. "
                    "This helps surface broad, sustained elevation in anxiety, depression, or stress levels across the selected window."
                )

                alert_metrics = ["anxiety_score", "depression_score", "stress_score"]
                baseline_source = mental_df.copy()
                if selected_senders:
                    baseline_source = baseline_source[baseline_source["sender"].isin(selected_senders)]

                baseline_daily = _aggregate_mental_health_daily(baseline_source)
                baseline_threshold_df = _compute_baseline_threshold(baseline_daily, alert_metrics)

                baseline_rows = []
                display_names = {
                    "anxiety_score": "Anxiety",
                    "depression_score": "Depression",
                    "stress_score": "Stress",
                }
                for _, row in baseline_threshold_df.iterrows():
                    for metric in alert_metrics:
                        baseline_rows.append(
                            {
                                "sender": row["sender"],
                                "metric": display_names.get(metric, metric),
                                "baseline": row.get(f"{metric}_baseline", None),
                                "threshold": row.get(f"{metric}_threshold", None),
                            }
                        )
                baseline_table = pd.DataFrame(baseline_rows)
                with st.expander("Baseline and threshold values", expanded=False):
                    if not baseline_table.empty:
                        st.dataframe(baseline_table)
                    else:
                        st.warning("Baseline values could not be calculated because sender-level historical data is missing.")

                filtered_mental_for_alerts = alert_source_for_labels.copy()
                alert_df = _compute_mental_health_alerts(filtered_mental_for_alerts, baseline_threshold_df, alert_metrics)
                if alert_df.empty:
                    st.warning("No alert data could be generated for the selected senders and date range.")
                else:
                    display_order = ["sender"] + [display_names[metric] for metric in alert_metrics]
                    status_emoji = {"Green": "🟢", "Amber": "🟠", "Red": "🔴"}
                    signal_rows = []
                    for sender in sorted(alert_df["sender"].unique()):
                        row = {"sender": sender}
                        sender_df = alert_df[alert_df["sender"] == sender]
                        for metric in alert_metrics:
                            metric_name = display_names[metric]
                            status = sender_df.loc[sender_df["metric"] == metric, "status"]
                            if not status.empty:
                                row[metric_name] = status_emoji.get(status.iloc[0], "⚪")
                            else:
                                row[metric_name] = "⚪"
                        signal_rows.append(row)
                    signal_table = pd.DataFrame(signal_rows)[display_order]
                    st.write("#### Alert signal panel")
                    st.markdown(
                        "Each row shows the alert status for a sender and each metric. "
                        "A green circle means normal range, amber means elevated, and red means high confidence of sustained worsening."
                    )
                    st.table(signal_table)

if __name__ == "__main__":
    main()
