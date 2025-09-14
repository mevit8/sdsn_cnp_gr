from __future__ import annotations # views/charts.py
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import theme
from typing import Optional, List, Literal
import pandas as pd
from pathlib import Path

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
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """
    Renders a bar chart with optional tick_every_years.
    By default, uses theme.CHART_WIDTH/HEIGHT and disables container_width.
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

    # Optional tick logic (if you still use it for years)
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

    # Apply explicit size
    fig.update_layout(
        width=width or getattr(theme, "CHART_WIDTH", 800),
        height=height or getattr(theme, "CHART_HEIGHT", 500),
    )

    if plot:
        st.plotly_chart(fig, use_container_width=False)  # key change: disable full width

    return fig


def render_line_chart(
    df,
    x,
    y,
    color,
    title,
    tick_every_years: Optional[int] = None,
    start_year: Optional[int] = None,
    plot: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    fig = px.line(df, x=x, y=y, color=color, title=title)

    if tick_every_years and x.lower() == "year":
        try:
            y0 = int(df[x].min()) if start_year is None else int(start_year)
            y0 -= y0 % tick_every_years
            fig.update_xaxes(tick0=y0, dtick=tick_every_years)
        except Exception:
            pass

    fig.update_layout(
        width=width or getattr(theme, "CHART_WIDTH", 800),
        height=height or getattr(theme, "CHART_HEIGHT", 500),
    )

    if plot:
        st.plotly_chart(fig, use_container_width=False)

    return fig


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

def _wrap_text(s: str, max_len: int = 14) -> str:
    """
    Insert <br> into long labels so they wrap nicely in Sankey.
    Tries to break at spaces closest to multiples of max_len.
    """
    if not isinstance(s, str) or len(s) <= max_len:
        return s
    parts, line, count = [], [], 0
    for word in s.split():
        if count + len(word) + (1 if line else 0) > max_len:
            parts.append(" ".join(line))
            line, count = [word], len(word)
        else:
            line.append(word)
            count += len(word) + (1 if line[:-1] else 0)
    if line:
        parts.append(" ".join(line))
    return "<br>".join(parts)

def _wrap_text(s: str, max_len: int = 14) -> str:
    if not isinstance(s, str) or len(s) <= max_len:
        return s
    parts, line, count = [], [], 0
    for word in s.split():
        new_len = count + len(word) + (1 if line else 0)
        if new_len > max_len:
            parts.append(" ".join(line)); line, count = [word], len(word)
        else:
            line.append(word); count = new_len
    if line: parts.append(" ".join(line))
    return "<br>".join(parts)

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c*2 for c in hex_color)
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

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
    node_colors: Optional[Dict[str, str]] = None,   # e.g. {"Electricity":"#1f77b4", ...}
    link_alpha: float = 0.35,
):
    # Validate
    needed = {"source", "target", "value"}
    if not needed.issubset(links_df.columns):
        raise ValueError(f"Sankey links_df missing columns: {needed - set(links_df.columns)}")

    # Node order
    if node_order:
        nodes = node_order
    else:
        nodes = []
        for n in links_df["source"].tolist() + links_df["target"].tolist():
            if n not in nodes:
                nodes.append(n)
    idx = {n: i for i, n in enumerate(nodes)}

    # Labels (built-in; wrap for readability)
    raw_labels = [label_map.get(n, n) if label_map else n for n in nodes]
    wrapped_labels = [_wrap_text(s, max_len=label_wrap) for s in raw_labels]

    # Node colors (opaque)
    default_node_color = "#e5e7eb"
    node_color_list = [(node_colors.get(n) if node_colors else default_node_color) or default_node_color for n in nodes]

    # Link colors from SOURCE node color with alpha
    link_colors = []
    for s in links_df["source"].map(idx).tolist():
        c = node_color_list[s]
        link_colors.append(_hex_to_rgba(c, link_alpha))

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            label=wrapped_labels,     # 👈 use built-in labels (aligned with nodes)
            pad=node_pad,
            thickness=node_thickness,
            line=dict(width=0.6, color="rgba(0,0,0,0.25)"),
            color=node_color_list,
        ),
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

    # Layout (respect full_width & height)
    layout_kwargs = dict(
        title=title,
        height=height or getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Arial, Helvetica, sans-serif", color="#111"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if not full_width:
        layout_kwargs["width"] = width or getattr(theme, "CHART_WIDTH", 800)
    fig.update_layout(**layout_kwargs)

    if plot:
        st.plotly_chart(fig, use_container_width=full_width)
    return fig

def biofuels_demand_potential_chart(
    scenario: Literal["BAU","NCNC"] = "BAU",
    path: str | Path = "data/LEAP_Biofuels.xlsx"
):
    """
    Build 'Biofuels demand and potential supply [ktoe]'
    from LEAP_Biofuels.xlsx (sheet: LEAPs output).

    scenario:
      - "BAU"  -> Baseline demand series
      - "NCNC" -> NECP demand series
    """
    df = pd.read_excel(path, sheet_name="LEAPs output")

    # Locate header row with "Fuel"
    hdr_idx = df.index[df.iloc[:,0].astype(str).str.strip().eq("Fuel")][0]
    header = df.loc[hdr_idx]

    # Collect year columns (Baseline block)
    baseline_cols = []
    for c in df.columns[2:]:
        val = header[c]
        if pd.api.types.is_number(val):
            baseline_cols.append(c)
        elif isinstance(val,str) and val.strip().lower()=="total":
            break
    years = [int(header[c]) for c in baseline_cols]

    # Collect year cols for NECP (second block, after first 'Total')
    start_necp = df.columns.get_loc(baseline_cols[-1]) + 2
    necp_cols = []
    for c in df.columns[start_necp:]:
        val = header[c]
        if pd.api.types.is_number(val):
            necp_cols.append(c)
        elif isinstance(val,str) and val.strip().lower()=="total":
            break

    # Demand row
    demand_row = df[df.iloc[:,0].astype(str).str.strip().eq("Total Biomass required")].iloc[0]
    demand_baseline = [float(demand_row[c]) for c in baseline_cols]
    demand_necp     = [float(demand_row[c]) for c in necp_cols]

    # MIN/MAX rows
    min_row = df[df.iloc[:,1].astype(str).str.strip().eq("MIN")].iloc[0]
    max_row = df[df.iloc[:,1].astype(str).str.strip().eq("MAX")].iloc[0]
    min_prod = [float(min_row[c]) for c in baseline_cols]
    max_prod = [float(max_row[c]) for c in baseline_cols]

    # Pick demand by scenario
    scenario = (scenario or "BAU").upper()
    demand = demand_baseline if scenario=="BAU" else demand_necp
    demand_label = "Baseline" if scenario=="BAU" else "NECP"

    # Build figure
    fig = go.Figure()
    fig.add_bar(x=years,y=min_prod,name="Minimum Production Potential [ktoe]")
    fig.add_bar(x=years,y=max_prod,name="Maximum Production Potential [ktoe]")
    fig.add_scatter(x=years,y=demand,mode="lines+markers",
                    name=f"Biofuel Demand {demand_label} [ktoe]")

    fig.update_layout(
        title="Biofuels demand and potential supply [ktoe]",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="ktoe",
        legend_title="Series",
        margin=dict(l=10,r=10,t=40,b=10)
    )
    return fig
def render_biofuels_demand_and_potential(
    df_biofuels: pd.DataFrame,
    scenario: Literal["BAU","NCNC"] = "BAU",
    title: str = "Biofuels demand and potential supply [ktoe]"
):
    """
    Renders Chart 1 using the already-loaded LEAP_Biofuels.xlsx dataframe.
    Expects a wide table: first column = row label, other columns = years.
    """
    if df_biofuels.empty:
        st.info("No data found in LEAP_Biofuels.xlsx.")
        return

    # Ensure first column is 'Metric'
    first_col = df_biofuels.columns[0]
    if first_col != "Metric":
        df_biofuels = df_biofuels.rename(columns={first_col: "Metric"})

    # Make sure year columns are strings we can convert to int later
    # (load_and_prepare_excel typically keeps them as numeric/strings; be permissive)
    year_cols = [c for c in df_biofuels.columns if c != "Metric"]

    # Row labels we need
    BAR_ROWS = [
        "Minimum Production Potential [ktoe]",
        "Maximum Production Potential [ktoe]",
    ]
    LINE_ROWS = {
        "BAU":  "Biofuel Demand Baseline scenario [ktoe]",
        "NCNC": "Biofuel Demand NECP [ktoe]",
    }
    line_row = LINE_ROWS.get((scenario or "BAU").upper(), LINE_ROWS["BAU"])

    # Bars (Min/Max)
    bar_src = df_biofuels[df_biofuels["Metric"].isin(BAR_ROWS)].copy()
    bar_long = bar_src.melt(id_vars="Metric", value_vars=year_cols,
                            var_name="Year", value_name="Value")
    # Lines (selected scenario only)
    line_src = df_biofuels[df_biofuels["Metric"] == line_row].copy()
    line_long = line_src.melt(id_vars="Metric", value_vars=year_cols,
                              var_name="Year", value_name="Value")

    # Clean types
    # Coerce year to int when possible
    def _to_int_safe(x):
        try:
            return int(float(x))
        except Exception:
            return x
    bar_long["Year"] = bar_long["Year"].map(_to_int_safe)
    line_long["Year"] = line_long["Year"].map(_to_int_safe)

    # Rename for the generic renderer
    bar_long = bar_long.rename(columns={"Metric": "Component"})
    line_long = line_long.rename(columns={"Metric": "Component"})

    # Sort x if they are numeric years
    if pd.api.types.is_numeric_dtype(pd.Series([y for y in bar_long["Year"] if isinstance(y, (int, float))])):
        bar_long = bar_long.sort_values("Year")
        line_long = line_long.sort_values("Year")

    # Use your shared renderer (grouped bars + single line)
    render_grouped_bar_and_line(
        prod_df=bar_long,
        demand_df=line_long,
        x_col="Year",
        y_col="Value",
        category_col="Component",
        title=title,
    )
 # --- append near your other chart renderers ---


import pandas as pd
import plotly.express as px
from config import theme

def render_ships_stock(df_base: pd.DataFrame):
    """
    Stock Ships [number], stacked by ship type.
    Expects columns: 'Year' + any that start with 'Stock_Ships_' (e.g., 'Stock_Ships_B').
    """
    if df_base.empty:
        return px.bar(title="Stock Ships — no data")

    stock_cols = [c for c in df_base.columns if c.startswith("Stock_Ships_")]
    if "Year" not in df_base.columns or not stock_cols:
        return px.bar(title="Stock Ships — expected 'Year' and 'Stock_Ships_*' columns")

    d_long = (
        df_base[["Year"] + stock_cols]
        .melt(id_vars="Year", var_name="col", value_name="value")
        .assign(type=lambda x: x["col"].str.replace("^Stock_Ships_", "", regex=True))
    )

    pref = ["C", "T", "B", "G", "O"]
    seen = list(dict.fromkeys([t for t in pref if t in d_long["type"].unique()]))
    rest = [t for t in sorted(d_long["type"].unique()) if t not in seen]
    order = seen + rest
    d_long["type"] = pd.Categorical(d_long["type"], categories=order, ordered=True)

    fig = px.bar(
        d_long.sort_values(["Year", "type"]),
        x="Year", y="value", color="type",
        title="Stock Ships [number]",
        barmode="stack",
    )
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),   # match agri_costs
        height=getattr(theme, "CHART_HEIGHT", 500), # match agri_costs
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Year",
        yaxis_title="Number of Stock Ships",
        legend_title=None,
    )
    fig.update_xaxes(ticks="outside", showline=True)
    fig.update_yaxes(ticks="outside", showline=True, zeroline=True)
    return fig


def render_ships_new(df_base: pd.DataFrame):
    """
    New Ships [number], stacked by ship type.
    Selects columns that start with 'New_Ships_' (case-insensitive).
    Falls back to any 'new_' prefix if needed.
    """
    if df_base.empty:
        return px.bar(title="New Ships — no data")

    cols = [c for c in df_base.columns if c.lower().startswith("new_ships_")]
    if not cols:
        cols = [c for c in df_base.columns if c.lower().startswith("new_")]

    if "Year" not in df_base.columns or not cols:
        return px.bar(title="New Ships — expected 'Year' and 'New_Ships_*' columns")

    d_long = (
        df_base[["Year"] + cols]
        .melt(id_vars="Year", var_name="col", value_name="value")
    )

    # Extract ship type from column suffix
    def _extract_type(col: str) -> str:
        cl = col.lower()
        if cl.startswith("new_ships_"):
            return col.split("_", 2)[-1]
        if cl.startswith("new_"):
            return col.split("_", 1)[-1]
        return col

    d_long["type"] = d_long["col"].map(_extract_type)

    # Stable legend order if present
    pref = ["C", "T", "B", "G", "O"]
    seen = list(dict.fromkeys([t for t in pref if t in d_long["type"].unique()]))
    rest = [t for t in sorted(d_long["type"].unique()) if t not in seen]
    order = seen + rest
    d_long["type"] = pd.Categorical(d_long["type"], categories=order, ordered=True)

    fig = px.bar(
        d_long.sort_values(["Year", "type"]),
        x="Year",
        y="value",
        color="type",
        title="New Ships [number]",
        barmode="stack",
    )
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="Year",
        yaxis_title="Number of New Ships",
        legend_title=None,
    )
    fig.update_xaxes(ticks="outside", showline=True)
    fig.update_yaxes(ticks="outside", showline=True, zeroline=True)
    return fig

