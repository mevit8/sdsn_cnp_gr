# views/charts.py
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import theme
from typing import Optional

ColorSpec = Union[Dict[str, str], List[str], None]


def _normalize(label: str) -> str:
    """Use theme.normalize if available; else a minimal normalizer."""
    if hasattr(theme, "normalize"):
        return theme.normalize(label)
    return " ".join(str(label).split())


def _build_px_colors(df, color_col: str, colors: ColorSpec):
    """
    Returns (color_discrete_map, color_discrete_sequence) for Plotly Express.
    - If colors is a dict: map known categories; fall back to DEFAULT_PALETTE if present.
    - If colors is a list: use it as a sequence.
    - If None: let Plotly defaults handle it.
    """
    color_discrete_map: Optional[Dict[str, str]] = None
    color_discrete_sequence: Optional[List[str]] = None

    if isinstance(colors, dict):
        uniques = df[color_col].dropna().astype(str).unique().tolist()
        color_discrete_map = {}
        for lbl in uniques:
            # exact key first
            if lbl in colors:
                color_discrete_map[lbl] = colors[lbl]
                continue
            # normalized key fallback (lets "oil refining" match "Oil Refining")
            norm = _normalize(lbl)
            if norm in colors:
                color_discrete_map[lbl] = colors[norm]
        color_discrete_sequence = getattr(theme, "DEFAULT_PALETTE", None)

    elif isinstance(colors, list):
        color_discrete_sequence = colors

    return color_discrete_map, color_discrete_sequence

'''
def render_bar_chart(
    df,
    x: str,
    y: str,
    color_col: str,
    title: str,
    x_order: Optional[List[str]] = None,
    colors: ColorSpec = None,          # dict[str,str] or list[str]
    barmode: str = "relative",
    height: int = 400,
    use_container_width: bool = True,
):
    """
    Generic bar chart with optional discrete color mapping and x-axis ordering.
    Matches calls in app.py, e.g.:
      render_bar_chart(melted, "YearStr", "Value", "Component", "...", [str(y) for y in years])
      render_bar_chart(..., colors=theme.FUEL_COLORS)
    """
    if color_col not in df.columns:
        raise KeyError(f"Column '{color_col}' not in DataFrame: {list(df.columns)}")

    color_discrete_map, color_discrete_sequence = _build_px_colors(df, color_col, colors)

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color_col,
        title=title,
        category_orders=({x: x_order} if x_order else None),
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=color_discrete_sequence,
        height=height,
    )
    # stacking/grouping
    fig.update_layout(barmode=barmode, legend_title_text=color_col, margin=dict(t=60, r=10, b=10, l=10))
    # force categorical axis if you pass an explicit x_order
    if x_order:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=x_order)

    st.plotly_chart(fig, use_container_width=use_container_width) 

'''
def render_bar_chart(
    df,
    x,
    y,
    color,
    title,
    x_category_order,
    colors=None,
    tick_every_years: Optional[int] = None,
    start_year: Optional[int] = None,
    plot: bool = True,
):
    """
    Renders a bar chart with optional tick_every_years to show only
    every Nth year on categorical year axes (e.g., YearStr).
    """
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        title=title,
        category_orders={x: x_category_order},
        color_discrete_map=colors if isinstance(colors, dict) else None,
        color_discrete_sequence=None if isinstance(colors, dict) else colors,
    )

    if tick_every_years:
        years_int = []
        for v in x_category_order:
            try:
                years_int.append(int(v))
            except Exception:
                pass
        if years_int:
            base = start_year if start_year is not None else min(years_int)
            base -= base % tick_every_years
            tickvals = [str(y) for y in years_int if (y - base) % tick_every_years == 0]
            if tickvals:
                fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=tickvals)

    if plot:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True)

    return fig

def render_line_chart(
    df,
    x: str,
    y: str,
    color_col: str,
    title: str,
    colors: ColorSpec = None,
    height: int = 400,
    use_container_width: bool = True,
):
    """
    Line chart with optional discrete color mapping. Signature matches app.py.
    """
    if color_col not in df.columns:
        raise KeyError(f"Column '{color_col}' not in DataFrame: {list(df.columns)}")

    color_discrete_map, color_discrete_sequence = _build_px_colors(df, color_col, colors)

    fig = px.line(
        df,
        x=x,
        y=y,
        color=color_col,
        markers=True,
        title=title,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=color_discrete_sequence,
        height=height,
    )
    fig.update_layout(legend_title_text=color_col, margin=dict(t=60, r=10, b=10, l=10))
    st.plotly_chart(fig, use_container_width=use_container_width)


def render_grouped_bar_and_line(
    prod_df,
    demand_df,
    x_col: str,
    y_col: str,
    category_col: str,
    title: str,
    colors: ColorSpec = None,
    height: int = 450,
    use_container_width: bool = True,
):
    """
    Mixed graph-objects figure: grouped bars (production) + lines (demand).
    Uses the same color mapping rules so categories stay consistent.
    """
    # Build a simple lookup function for colors
    def _color_for(lbl: str) -> Optional[str]:
        if isinstance(colors, dict):
            if lbl in colors:
                return colors[lbl]
            norm = _normalize(lbl)
            return colors.get(norm)
        return None  # GO will use default if None

    fig = go.Figure()

    # Bar traces (production)
    for cat in prod_df[category_col].dropna().astype(str).unique():
        sub = prod_df[prod_df[category_col] == cat]
        fig.add_trace(go.Bar(
            x=sub[x_col], y=sub[y_col], name=cat,
            marker_color=_color_for(cat)
        ))

    # Line traces (demand)
    for cat in demand_df[category_col].dropna().astype(str).unique():
        sub = demand_df[demand_df[category_col] == cat]
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col],
            name=f"{cat} (scenario)",
            mode="lines+markers",
            line=dict(color=_color_for(cat), width=2)
        ))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=height,
        margin=dict(t=60, r=10, b=10, l=10),
        legend_title_text=category_col,
    )
    st.plotly_chart(fig, use_container_width=use_container_width)
