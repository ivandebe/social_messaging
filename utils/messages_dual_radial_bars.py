import numpy as np
import plotly.graph_objects as go


def create_messages_dual_radial_bars(counts):
    """
    Create a dual radial bar chart showing messages by hour.
    
    Args:
        counts: Array-like of 24 values representing message counts per hour (0-23)
        
    Returns:
        plotly.graph_objects.Figure: The dual radial bar chart figure
    """
    counts = np.array(counts)
    hours = np.arange(24)
    max_count = counts.max()

    # Clock layout: inner ring = 12 AM-11 AM, outer ring = 12 PM-11 PM
    angles_deg = np.linspace(90, 90 - 330, 12)
    angles_rad = np.deg2rad(angles_deg)
    inner_hours = np.arange(0, 12)
    outer_hours = np.arange(12, 24)

    r_inner = 0.58
    r_outer = 1.08
    inner_bar_len = 0.10 + 0.28 * (counts[inner_hours] / max_count)
    outer_bar_len = 0.16 + 0.42 * (counts[outer_hours] / max_count)

    fig = go.Figure()

    # Decorative circle borders / guides
    for r, width, color in [
        (0.38, 1.2, 'rgba(99,110,250,0.18)'),
        (r_inner, 2.2, 'rgba(99,110,250,0.32)'),
        (0.82, 1.2, 'rgba(8,122,122,0.14)'),
        (r_outer, 2.6, 'rgba(8,122,122,0.30)'),
        (r_outer + 0.22, 1.2, 'rgba(8,122,122,0.12)')
    ]:
        theta = np.linspace(0, 2 * np.pi, 361)
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='skip',
                showlegend=False
            )
        )

    # Hour separators and labels
    for i, angle in enumerate(angles_rad):
        # Inner separators
        fig.add_trace(
            go.Scatter(
                x=[(r_inner - 0.08) * np.cos(angle), (r_inner + 0.08) * np.cos(angle)],
                y=[(r_inner - 0.08) * np.sin(angle), (r_inner + 0.08) * np.sin(angle)],
                mode='lines',
                line=dict(color='rgba(90,90,90,0.16)', width=1),
                hoverinfo='skip',
                showlegend=False
            )
        )
        # Outer separators
        fig.add_trace(
            go.Scatter(
                x=[(r_outer - 0.09) * np.cos(angle), (r_outer + 0.09) * np.cos(angle)],
                y=[(r_outer - 0.09) * np.sin(angle), (r_outer + 0.09) * np.sin(angle)],
                mode='lines',
                line=dict(color='rgba(90,90,90,0.18)', width=1),
                hoverinfo='skip',
                showlegend=False
            )
        )

        ih = inner_hours[i]
        oh = outer_hours[i]
        inner_label = '12 AM' if ih == 0 else f'{ih} AM'
        outer_label = '12 PM' if oh == 12 else f'{oh - 12} PM'

        fig.add_annotation(
            x=(r_inner + 0.16) * np.cos(angle),
            y=(r_inner + 0.16) * np.sin(angle),
            text=inner_label,
            showarrow=False,
            font=dict(size=12, color='rgba(55,55,55,0.82)')
        )
        fig.add_annotation(
            x=(r_outer + 0.19) * np.cos(angle),
            y=(r_outer + 0.19) * np.sin(angle),
            text=outer_label,
            showarrow=False,
            font=dict(size=13, color='rgba(55,55,55,0.88)')
        )

    # Inner radial bars: AM
    for i, hour in enumerate(inner_hours):
        angle = angles_rad[i]
        x0, y0 = r_inner * np.cos(angle), r_inner * np.sin(angle)
        x1, y1 = (r_inner + inner_bar_len[i]) * np.cos(angle), (r_inner + inner_bar_len[i]) * np.sin(angle)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='rgba(99,110,250,0.92)', width=10),
                hovertemplate=f'{hour:02d}:00–{hour:02d}:59<br>Messages: {counts[hour]}<extra></extra>',
                showlegend=False
            )
        )

    # Outer radial bars: PM
    for i, hour in enumerate(outer_hours):
        angle = angles_rad[i]
        x0, y0 = r_outer * np.cos(angle), r_outer * np.sin(angle)
        x1, y1 = (r_outer + outer_bar_len[i]) * np.cos(angle), (r_outer + outer_bar_len[i]) * np.sin(angle)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='rgba(8,122,122,0.96)', width=12),
                hovertemplate=f'{hour:02d}:00–{hour:02d}:59<br>Messages: {counts[hour]}<extra></extra>',
                showlegend=False
            )
        )

    # Center text
    fig.add_annotation(x=0, y=0.06, text='Messages', showarrow=False, font=dict(size=22))
    fig.add_annotation(x=0, y=-0.05, text='Inner ring: AM', showarrow=False, font=dict(size=13, color='rgba(70,70,70,0.78)'))
    fig.add_annotation(x=0, y=-0.15, text='Outer ring: PM', showarrow=False, font=dict(size=13, color='rgba(70,70,70,0.78)'))

    fig.update_layout(
        title={
            'text': 'Messages by Hour (Dual radial bars)<br><span style="font-size: 18px; font-weight: normal;">Two 12-hour circles with radial bars and guide borders</span>'
        },
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_traces(cliponaxis=False)
    return fig


if __name__ == "__main__":
    # Example usage
    counts = np.array([12, 8, 5, 3, 2, 4, 9, 18, 26, 34, 41, 38,
                       32, 29, 31, 36, 44, 52, 48, 39, 30, 24, 19, 15])
    fig = create_messages_dual_radial_bars(counts)
    fig.show()
