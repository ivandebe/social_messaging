from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go


REQUIRED_COLUMNS = {"date", "time", "topic", "topic_label"}


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataframe is missing required columns: {sorted(missing)}"
        )


def _prepare_dataframe(
    df: pd.DataFrame,
    exclude_outlier_topic: bool = True,
) -> pd.DataFrame:
    _validate_input(df)

    data = df.copy()
    data["topic"] = pd.to_numeric(data["topic"], errors="coerce")
    data = data.dropna(subset=["topic", "topic_label", "date", "time"]).copy()
    data["topic"] = data["topic"].astype(int)

    data["topic_label"] = data["topic_label"].astype(str).str.strip()
    data = data[data["topic_label"] != ""].copy()

    if exclude_outlier_topic:
        data = data[data["topic"] != -1].copy()

    data["datetime"] = pd.to_datetime(
        data["date"].astype(str) + " " + data["time"].astype(str),
        errors="coerce",
    )
    data = data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid rows available after preprocessing.")

    return data


def _build_windows(
    df: pd.DataFrame,
    window_mode: str = "count",
    message_window_size: int = 50,
    time_window: str = "1D",
    deduplicate_topics_per_window: bool = True,
) -> list[list[int]]:
    if window_mode not in {"count", "time"}:
        raise ValueError("window_mode must be either 'count' or 'time'.")

    windows: list[list[int]] = []

    if window_mode == "count":
        if message_window_size <= 0:
            raise ValueError("message_window_size must be > 0.")

        for start in range(0, len(df), message_window_size):
            topics = df.iloc[start : start + message_window_size]["topic"].tolist()
            if deduplicate_topics_per_window:
                topics = sorted(set(topics))
            if len(topics) >= 2:
                windows.append(topics)
    else:
        grouped = df.groupby(pd.Grouper(key="datetime", freq=time_window))["topic"]
        for _, series in grouped:
            topics = series.dropna().astype(int).tolist()
            if deduplicate_topics_per_window:
                topics = sorted(set(topics))
            if len(topics) >= 2:
                windows.append(topics)

    return windows


def _compute_graph_components(
    df: pd.DataFrame,
    windows: Iterable[list[int]],
    min_edge_weight: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_counts = df["topic"].value_counts().to_dict()
    label_map = (
        df[["topic", "topic_label"]]
        .drop_duplicates(subset=["topic"])
        .set_index("topic")["topic_label"]
        .to_dict()
    )

    edge_counter: Counter[tuple[int, int]] = Counter()

    for topics in windows:
        for a, b in combinations(sorted(topics), 2):
            edge_counter[(a, b)] += 1

    edge_rows = [
        {
            "source": a,
            "target": b,
            "weight": w,
            "source_label": label_map.get(a, f"topic_{a}"),
            "target_label": label_map.get(b, f"topic_{b}"),
        }
        for (a, b), w in edge_counter.items()
        if w >= min_edge_weight
    ]
    edges_df = pd.DataFrame(edge_rows)

    connected_topics: set[int]
    if edges_df.empty:
        connected_topics = set()
    else:
        connected_topics = set(edges_df["source"]).union(set(edges_df["target"]))

    node_rows = [
        {
            "topic": topic,
            "label": label_map.get(topic, f"topic_{topic}"),
            "message_count": int(topic_counts.get(topic, 0)),
            "degree": int(
                ((edges_df["source"] == topic) | (edges_df["target"] == topic)).sum()
            )
            if not edges_df.empty
            else 0,
        }
        for topic in sorted(set(df["topic"]))
        if (not connected_topics) or (topic in connected_topics)
    ]
    nodes_df = pd.DataFrame(node_rows)

    return nodes_df, edges_df


def create_topic_cooccurrence_figure(
    df: pd.DataFrame,
    *,
    window_mode: str = "count",
    message_window_size: int = 50,
    time_window: str = "1D",
    deduplicate_topics_per_window: bool = True,
    min_edge_weight: int = 2,
    max_topics: int | None = 30,
    exclude_outlier_topic: bool = True,
    layout_seed: int = 42,
    layout_iterations: int = 100,
    title: str = "Topic co-occurrence graph",
) -> go.Figure:
    """
    Create a Plotly network graph of topic co-occurrence.

    Parameters
    ----------
    df:
        Already-filtered dataframe containing at least: date, time, topic, topic_label.
    window_mode:
        "count" for non-overlapping blocks of messages, "time" for calendar bins.
    message_window_size:
        Number of messages per block when window_mode="count".
    time_window:
        Pandas offset alias such as "1H", "1D", "7D" when window_mode="time".
    deduplicate_topics_per_window:
        If True, each topic is counted at most once per window.
    min_edge_weight:
        Minimum co-occurrence count required to draw an edge.
    max_topics:
        Keep only the top N topics by message count before building the graph.
    exclude_outlier_topic:
        If True, remove BERTopic outlier topic -1.
    layout_seed:
        Seed used by the spring layout for reproducible positioning.
    layout_iterations:
        Number of spring-layout iterations.
    title:
        Figure title.
    """
    data = _prepare_dataframe(df, exclude_outlier_topic=exclude_outlier_topic)

    if max_topics is not None and max_topics > 0:
        top_topics = data["topic"].value_counts().head(max_topics).index
        data = data[data["topic"].isin(top_topics)].copy()

    if data["topic"].nunique() < 2:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[
                dict(
                    text="Not enough distinct topics to build a co-occurrence graph.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    windows = _build_windows(
        data,
        window_mode=window_mode,
        message_window_size=message_window_size,
        time_window=time_window,
        deduplicate_topics_per_window=deduplicate_topics_per_window,
    )

    if not windows:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[
                dict(
                    text="No topic windows with at least two topics were found.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    nodes_df, edges_df = _compute_graph_components(
        data,
        windows,
        min_edge_weight=min_edge_weight,
    )

    if nodes_df.empty or edges_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[
                dict(
                    text="No edges survived the current threshold. Try a lower min_edge_weight or a larger window.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return fig

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "networkx is required for create_topic_cooccurrence_figure. Install it with `pip install networkx`."
        ) from exc

    graph = nx.Graph()

    for _, row in nodes_df.iterrows():
        graph.add_node(
            int(row["topic"]),
            label=row["label"],
            message_count=int(row["message_count"]),
            degree=int(row["degree"]),
        )

    for _, row in edges_df.iterrows():
        graph.add_edge(
            int(row["source"]),
            int(row["target"]),
            weight=int(row["weight"]),
        )

    positions = nx.spring_layout(
        graph,
        seed=layout_seed,
        weight="weight",
        iterations=layout_iterations,
    )

    edge_x = []
    edge_y = []
    edge_text_x = []
    edge_text_y = []
    edge_text = []

    for source, target, attrs in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text_x.append((x0 + x1) / 2)
        edge_text_y.append((y0 + y1) / 2)
        edge_text.append(
            f"{graph.nodes[source]['label']} ↔ {graph.nodes[target]['label']}<br>Co-occurrence count: {attrs['weight']}"
        )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_labels = []

    max_messages = max(nx.get_node_attributes(graph, "message_count").values())

    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(attrs["label"])
        node_color.append(attrs["degree"])
        size = 18 + 42 * (attrs["message_count"] / max_messages)
        node_size.append(size)
        node_text.append(
            f"Topic ID: {node}<br>Label: {attrs['label']}<br>Messages: {attrs['message_count']}<br>Connections: {attrs['degree']}"
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="rgba(120,120,120,0.45)"),
        hoverinfo="skip",
        showlegend=False,
    )

    edge_hover_trace = go.Scatter(
        x=edge_text_x,
        y=edge_text_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),
        text=edge_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="YlGnBu",
            showscale=True,
            colorbar=dict(title="Degree"),
            line=dict(width=1, color="white"),
            opacity=0.92,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace])
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=750,
    )
    return fig


def create_topic_cooccurrence_tables(
    df: pd.DataFrame,
    *,
    window_mode: str = "count",
    message_window_size: int = 50,
    time_window: str = "1D",
    deduplicate_topics_per_window: bool = True,
    min_edge_weight: int = 1,
    max_topics: int | None = 30,
    exclude_outlier_topic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return node and edge tables used to build the graph.
    Useful for debugging or showing supporting tables in Streamlit.
    """
    data = _prepare_dataframe(df, exclude_outlier_topic=exclude_outlier_topic)

    if max_topics is not None and max_topics > 0:
        top_topics = data["topic"].value_counts().head(max_topics).index
        data = data[data["topic"].isin(top_topics)].copy()

    windows = _build_windows(
        data,
        window_mode=window_mode,
        message_window_size=message_window_size,
        time_window=time_window,
        deduplicate_topics_per_window=deduplicate_topics_per_window,
    )

    return _compute_graph_components(
        data,
        windows,
        min_edge_weight=min_edge_weight,
    )
