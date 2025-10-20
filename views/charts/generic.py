from __future__ import annotations
from typing import Dict, List, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import theme
from .helpers import _wrap_text, _hex_to_rgba

def render_bar_chart(
    df, x_col, y_col, category_col, title,
    x_order=None, colors=None, y_label=None, key=None
):
    """Generic bar chart renderer with consistent theme-based colors."""

    df = df.copy()

    # --- Determine base category order ---
    cat_order = list(df[category_col].unique())
    if pd.api.types.is_categorical_dtype(df[category_col]):
        cat_order = list(df[category_col].cat.categories)

    # --- Apply canonical order from theme if available ---
    explicit_order = None
    if colors is theme.COST_COLORS:
        explicit_order = getattr(theme, "COST_ORDER", None)
    elif colors is theme.EMISSIONS_COLORS:
        explicit_order = getattr(theme, "EMISSIONS_ORDER", None)
    elif colors is theme.LANDUSE_COLORS:
        explicit_order = getattr(theme, "LANDUSE_ORDER", None)
    elif colors is theme.FUEL_COLORS:
        explicit_order = getattr(theme, "FUEL_ORDER", None)

    if explicit_order:
        cat_order = [c for c in explicit_order if c in cat_order]

    # --- Set categorical type for stable ordering ---
    df[category_col] = pd.Categorical(df[category_col], categories=cat_order, ordered=True)
    df = df.sort_values([x_col, category_col])

    # --- Build Plotly figure with explicit color mapping ---
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=category_col,
        category_orders={category_col: cat_order},
        color_discrete_map=colors or {},
    )

    # --- Ensure correct stacking order (bottom→top = cat_order) ---
    sorted_traces = sorted(
        fig.data, key=lambda t: cat_order.index(t.name) if t.name in cat_order else 999
    )
    fig.data = tuple(reversed(sorted_traces))

    # --- Layout ---
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_label or y_col,
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title=None,
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", color="#111", size=14),
    )

    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_bar")

def render_line_chart(df, x_col, y_col, category_col, title,
                      colors=None, y_label=None, key=None):
    """Generic line chart renderer with consistent theme-based colors."""

    # Keep a copy of the data
    df = df.copy()

    # Apply category order if color map is provided
    category_orders = None
    if isinstance(colors, dict) and len(colors) > 0:
        category_orders = {category_col: list(colors.keys())}

    # Build Plotly figure with color map (this is the key fix)
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=category_col,
        color_discrete_map=colors or {},   # ✅ force use of theme colors
        category_orders=category_orders,   # ✅ ensure stable order
        title=title,
    )

    # Update layout (same defaults as before)
    fig.update_layout(
        xaxis_title="",
        yaxis_title=y_label or y_col,
        legend_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_line")

def render_grouped_bar_and_line(
    prod_df, demand_df, x_col, y_col,
    category_col, title, height=None,
    colors=None, y_label=None, key=None
):
    """Render grouped bar + overlay line with consistent theme colors."""

    prod_df = prod_df.copy()
    cat_order = list(prod_df[category_col].unique())
    if colors:
        explicit_order = [c for c in colors.keys() if c in cat_order]
        if explicit_order:
            cat_order = explicit_order

    prod_df[category_col] = pd.Categorical(prod_df[category_col], categories=cat_order, ordered=True)
    prod_df = prod_df.sort_values([x_col, category_col])

    fig = px.bar(
        prod_df,
        x=x_col,
        y=y_col,
        color=category_col,
        category_orders={category_col: cat_order},
        color_discrete_map=colors or {},
    )

    # Overlay the line chart (demand or total)
    if demand_df is not None and not demand_df.empty:
        # Default to a distinct or mapped color
        line_name = demand_df.get(category_col, pd.Series(["Line"])).iloc[0]
        line_color = None
        if isinstance(colors, dict):
            line_color = colors.get(line_name)

        fig.add_trace(
            go.Scatter(
                x=demand_df[x_col],
                y=demand_df[y_col],
                mode="lines+markers",
                name=line_name,
                line=dict(color=line_color or "#000000", width=2),
            )
        )

    fig.update_layout(
        title=title,
        height=height or getattr(theme, "CHART_HEIGHT", 500),
        xaxis_title="",
        yaxis_title=y_label or y_col,
        legend_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_grouped")

def render_sankey(
    links_df: pd.DataFrame,
    title: str,
    node_order: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    plot: bool = True,
    full_width: bool = False,
    label_wrap: int = 14,
    label_map: Optional[Dict[str, str]] = None,
    font_size: int = 15,
    node_thickness: int = 26,
    node_pad: int = 24,
    node_colors: Optional[Dict[str, str]] = None,
    link_alpha: float = 0.35,
    key=None,
):
    needed = {"source", "target", "value"}
    if not needed.issubset(links_df.columns):
        raise ValueError("Sankey links_df missing required columns")

    if node_order:
        nodes = node_order
    else:
        nodes = []
        for n in links_df["source"].tolist() + links_df["target"].tolist():
            if n not in nodes: nodes.append(n)
    idx = {n: i for i, n in enumerate(nodes)}

    raw_labels = [label_map.get(n, n) if label_map else n for n in nodes]
    wrapped = [_wrap_text(s, max_len=label_wrap) for s in raw_labels]

    default_c = "#e5e7eb"
    node_color_list = [(node_colors.get(n) if node_colors else default_c) or default_c for n in nodes]
    link_colors = [_hex_to_rgba(node_color_list[s], link_alpha) for s in links_df["source"].map(idx).tolist()]

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(label=wrapped, pad=node_pad, thickness=node_thickness,
                  line=dict(width=0.6, color="rgba(0,0,0,0.25)"), color=node_color_list),
        link=dict(
            source=links_df["source"].map(idx).tolist(),
            target=links_df["target"].map(idx).tolist(),
            value=links_df["value"].astype(float).tolist(),
            color=link_colors,
        ),
        valueformat=".3s",
        textfont=dict(size=font_size, color="#111", family="Arial, Helvetica, sans-serif"),
    )

    fig = go.Figure(data=[sankey])
    layout_kwargs = dict(
        title=title, height=height or getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Arial, Helvetica, sans-serif", color="#111"),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    if not full_width:
        layout_kwargs["width"] = width or getattr(theme, "CHART_WIDTH", 800)
    fig.update_layout(**layout_kwargs)

    if plot:
        st.plotly_chart(fig, use_container_width=full_width, key=key or f"{title}_sankey")
    return fig

def apply_theme_order(df, category_col, colors):
    """Return dataframe with stable categorical order."""
    if not isinstance(colors, dict):
        return df
    order = [c for c in colors.keys() if c in df[category_col].unique()]
    df[category_col] = pd.Categorical(df[category_col], categories=order, ordered=True)
    return df.sort_values([category_col])
