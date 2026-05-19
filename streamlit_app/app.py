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
    if not chat_df.empty:
        chat_df = _parse_date_column(chat_df)

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

        if chat_df.empty:
            st.error("Could not load `group_chat_merged_consecutive.csv`. Please check `output_data/prep_logs`.")
        else:
            if "date" in chat_df.columns:
                min_date = chat_df["date"].dropna().min()
                max_date = chat_df["date"].dropna().max()
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
                    st.warning("The loaded chat history does not contain a valid date range.")
            else:
                st.warning("No `date` column found in chat history.")

            if "sender" in chat_df.columns:
                unique_senders = sorted(chat_df["sender"].dropna().astype(str).unique())
                if unique_senders:
                    selected_senders = st.multiselect("Select senders", unique_senders, default=unique_senders)
                else:
                    st.warning("No senders found in chat history.")
            else:
                st.warning("No `sender` column found in chat history.")

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
        st.write("Placeholder for sentiment analysis charts and controls.")
    elif choice == "Mental health analysis":
        st.header("Mental health analysis")
        st.write("Placeholder for mental health analysis tools and visualizations.")

if __name__ == "__main__":
    main()

