import pandas as pd
import numpy as np
import plotly.express as px


def plot_sentiment_heatmap(
    df,
    date_col="date",
    time_col="time",
    message_col="message",
    pos_col="positive",
    neg_col="negative",
    neu_col="neutral",
    metric="net",
    agg="mean",
    combine_datetime=True,
    datetime_col=None,
    timezone=None,
    title=None
):
    """
    Create a Plotly heatmap of sentiment by weekday and hour.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with one row per message.
    date_col, time_col, message_col, pos_col, neg_col, neu_col : str
        Column names in df.
    metric : str
        Which sentiment metric to plot. Options:
        - "net"      -> positive - negative
        - "positive" -> positive score
        - "negative" -> negative score
        - "neutral"  -> neutral score
        - "dominant" -> encodes dominant class as -1 (neg), 0 (neu), 1 (pos)
    agg : str
        Aggregation within each weekday-hour cell.
        Options: "mean", "median", "count"
    combine_datetime : bool
        If True, combine date_col + time_col into a datetime.
    datetime_col : str or None
        If you already have a datetime column, pass it here.
    timezone : str or None
        Optional timezone to localize/convert datetime.
    title : str or None
        Optional custom chart title.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly heatmap figure.
    heatmap_df : pandas.DataFrame
        Pivoted dataframe used in the heatmap.
    working_df : pandas.DataFrame
        Copy of the original dataframe with derived columns added.
    """

    working_df = df.copy()

    # Build datetime column
    if datetime_col is not None:
        working_df["_dt"] = pd.to_datetime(working_df[datetime_col], errors="coerce")
    elif combine_datetime:
        working_df["_dt"] = pd.to_datetime(
            working_df[date_col].astype(str) + " " + working_df[time_col].astype(str),
            errors="coerce"
        )
    else:
        raise ValueError("Provide datetime_col or set combine_datetime=True.")

    if timezone is not None:
        if working_df["_dt"].dt.tz is None:
            working_df["_dt"] = working_df["_dt"].dt.tz_localize(timezone)
        else:
            working_df["_dt"] = working_df["_dt"].dt.tz_convert(timezone)

    working_df = working_df.dropna(subset=["_dt"]).copy()

    # Time features
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    working_df["weekday"] = pd.Categorical(
        working_df["_dt"].dt.day_name(),
        categories=weekday_order,
        ordered=True
    )
    working_df["hour"] = working_df["_dt"].dt.hour

    # Derived metric
    working_df["net_score"] = working_df[pos_col] - working_df[neg_col]
    working_df["dominant_score"] = np.select(
        [
            (working_df[pos_col] >= working_df[neg_col]) & (working_df[pos_col] >= working_df[neu_col]),
            (working_df[neg_col] > working_df[pos_col]) & (working_df[neg_col] >= working_df[neu_col]),
            (working_df[neu_col] > working_df[pos_col]) & (working_df[neu_col] > working_df[neg_col]),
        ],
        [1, -1, 0],
        default=0
    )

    metric_map = {
        "net": "net_score",
        "positive": pos_col,
        "negative": neg_col,
        "neutral": neu_col,
        "dominant": "dominant_score",
    }

    if metric not in metric_map:
        raise ValueError("metric must be one of: 'net', 'positive', 'negative', 'neutral', 'dominant'")

    value_col = metric_map[metric]

    # Aggregate
    if agg == "mean":
        grouped = (
            working_df.groupby(["weekday", "hour"], observed=False)[value_col]
            .mean()
            .reset_index()
        )
    elif agg == "median":
        grouped = (
            working_df.groupby(["weekday", "hour"], observed=False)[value_col]
            .median()
            .reset_index()
        )
    elif agg == "count":
        grouped = (
            working_df.groupby(["weekday", "hour"], observed=False)[value_col]
            .count()
            .reset_index(name=value_col)
        )
    else:
        raise ValueError("agg must be one of: 'mean', 'median', 'count'")

    # Pivot to 7 x 24 grid
    heatmap_df = (
        grouped.pivot(index="weekday", columns="hour", values=value_col)
        .reindex(index=weekday_order, columns=range(24))
    )

    # Labels and color settings
    metric_titles = {
        "net": "Average Net Sentiment (Positive - Negative)",
        "positive": "Average Positive Score",
        "negative": "Average Negative Score",
        "neutral": "Average Neutral Score",
        "dominant": "Average Dominant Sentiment Class"
    }

    if title is None:
        title = f"Sentiment Heatmap by Weekday and Hour<br><sup>{metric_titles[metric]}</sup>"

    if metric in ["net", "dominant"]:
        color_scale = "RdBu"
        zmid = 0
        range_color = None
    else:
        color_scale = "YlGnBu"
        zmid = None
        range_color = (0, 1)

    fig = px.imshow(
        heatmap_df,
        labels={"x": "Hour of day", "y": "Day of week", "color": metric_titles[metric]},
        x=[f"{h:02d}:00" for h in heatmap_df.columns],
        y=heatmap_df.index,
        color_continuous_scale=color_scale,
        zmin=range_color[0] if range_color else None,
        zmax=range_color[1] if range_color else None,
        aspect="auto",
        text_auto=".2f"
    )

    if zmid is not None:
        fig.update_coloraxes(cmid=zmid)

    fig.update_traces(
        hovertemplate=(
            "Day: %{y}<br>"
            "Hour: %{x}<br>"
            "Value: %{z:.3f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Hour of day",
        yaxis_title="Day of week",
        margin=dict(t=90, l=80, r=30, b=50),
        coloraxis_colorbar=dict(len=0.8)
    )

    return fig, heatmap_df, working_df
