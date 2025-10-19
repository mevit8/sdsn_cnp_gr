# views/charts.py
from __future__ import annotations
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import theme
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots

ColorSpec = Union[Dict[str, str], List[str], None]

# -----------------------------
# Helpers
# -----------------------------
def _normalize(label: str) -> str:
    if hasattr(theme, "normalize"):
        return theme.normalize(label)
    return " ".join(str(label).split())


def _build_px_colors(df, color_col: str, colors: ColorSpec):
    color_discrete_map: Optional[Dict[str, str]] = None
    color_discrete_sequence: Optional[List[str]] = None

    if isinstance(colors, dict):
        uniques = df[color_col].dropna().astype(str).unique().tolist()
        color_discrete_map = {}
        for lbl in uniques:
            if lbl in colors:
                color_discrete_map[lbl] = colors[lbl]
                continue
            norm = _normalize(lbl)
            if norm in colors:
                color_discrete_map[lbl] = colors[norm]
        color_discrete_sequence = getattr(theme, "DEFAULT_PALETTE", None)

    elif isinstance(colors, list):
        color_discrete_sequence = colors

    return color_discrete_map, color_discrete_sequence


def _resolve_ci(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Case-insensitive column resolver."""
    low = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    return None


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
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ------------------------------------------------------------
# BIOFUELS – Shared Excel Parser (new helper)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_biofuels_data(path: str = "data/LEAP_Biofuels.xlsx") -> dict[str, pd.DataFrame]:
    """
    Load and split LEAP_biofuels.xlsx into its main logical blocks.
    Returns:
        {
            "production": DataFrame of min/max + demand series,
            "exports":    DataFrame of export potential series,
            "combos":     DataFrame of all option combinations (a-a-b, etc.)
        }
    """
    xl = pd.ExcelFile(path)
    sheet_name = xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # --- locate key anchors ---
    prod_start = df.index[df.iloc[:, 0].astype(str)
                          .str.contains("Minimum Production", case=False, na=False)][0]
    exp_start = df.index[df.iloc[:, 8:].apply(
        lambda s: s.astype(str).str.contains("Minimum export", case=False, na=False)
    ).any(axis=1)][0]
    combo_start = df.index[df.iloc[:, 0].astype(str)
                           .str.contains("All other options", case=False, na=False)][0] + 2

    # --- column years (consistent across both) ---
    years = [2022, 2025, 2030, 2035, 2040, 2045, 2050]

    # --- production & demand block (left side) ---
    left_block = df.iloc[prod_start:prod_start + 4, :1 + len(years)].copy()
    left_block.columns = ["Label"] + [str(y) for y in years]
    prod = (
        left_block.melt(id_vars="Label", var_name="Year", value_name="Value")
        .pivot_table(index="Year", columns="Label", values="Value", aggfunc="first")
        .reset_index()
    )

    # --- export potentials (right side) ---
    right_block = df.iloc[exp_start:exp_start + 4, 8:8 + 1 + len(years)].copy()
    right_block.columns = ["Label"] + [str(y) for y in years]
    exports = (
        right_block.melt(id_vars="Label", var_name="Year", value_name="Value")
        .pivot_table(index="Year", columns="Label", values="Value", aggfunc="first")
        .reset_index()
    )

    # --- all combinations (bottom block) ---
    combos = df.iloc[combo_start:, :4].dropna(how="all").copy()
    combos.columns = [
        "Code", "Year",
        "Selected Production Potential (ktoe)",
        "Selected Export Potential (ktoe)"
    ]
    for c in ["Year", "Selected Production Potential (ktoe)", "Selected Export Potential (ktoe)"]:
        combos[c] = pd.to_numeric(combos[c], errors="coerce")

    return {"production": prod, "exports": exports, "combos": combos}

# -----------------------------
# Generic Charts
# -----------------------------
def render_bar_chart(df, x_col, y_col, category_col, title,
                     x_order=None, colors=None, y_label=None, key=None):
    """Generic bar chart renderer respecting explicit stack orders from theme."""

    df = df.copy()

    # --- Determine base category order ---
    if pd.api.types.is_categorical_dtype(df[category_col]):
        cat_order = list(df[category_col].cat.categories)
    else:
        cat_order = list(df[category_col].unique())

    # --- Apply theme-based canonical order if known ---
    explicit_order = None
    if colors is theme.COST_COLORS:
        explicit_order = getattr(theme, "COST_ORDER", None)
    elif colors is theme.EMISSIONS_COLORS:
        explicit_order = getattr(theme, "EMISSIONS_ORDER", None)
    elif colors is theme.LANDUSE_COLORS:
        explicit_order = getattr(theme, "LANDUSE_ORDER", None)

    if explicit_order:
        # keep only categories that exist in df, preserve order
        cat_order = [c for c in explicit_order if c in cat_order]

    # --- Set categorical and sort ---
    df[category_col] = pd.Categorical(df[category_col], categories=cat_order, ordered=True)
    df = df.sort_values([x_col, category_col])

    # --- Build Plotly bar ---
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=category_col,
        category_orders={category_col: cat_order},
        color_discrete_map=colors or {},
        color_discrete_sequence=None,
    )

    # --- Fix stacking order (bottom→top according to cat_order) ---
    sorted_traces = sorted(
        fig.data,
        key=lambda t: cat_order.index(t.name) if t.name in cat_order else 999
    )
    fig.data = tuple(reversed(sorted_traces))  # Plotly draws in reverse order

    # --- Layout style ---
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_label if y_label else y_col,
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
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=category_col,
        color_discrete_map=colors or {}
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_label if y_label else y_col
    )
    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_line")

def render_grouped_bar_and_line(prod_df, demand_df, x_col, y_col,
                                category_col, title, height=None,
                                colors=None, y_label=None, key=None):
    fig = px.bar(
        prod_df,
        x=x_col,
        y=y_col,
        color=category_col,
        color_discrete_map=colors or {}
    )

    # Overlay line if demand_df is provided
    if demand_df is not None and not demand_df.empty:
        line_name = demand_df[category_col].iloc[0] if category_col in demand_df else "Line"
        line_color = None
        if colors and isinstance(colors, dict):
            line_color = colors.get(line_name)

        fig.add_trace(
            go.Scatter(
                x=demand_df[x_col],
                y=demand_df[y_col],
                mode="lines+markers",
                name=line_name,
                line=dict(color=line_color or "#2563eb", width=2)
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="",
        yaxis_title=y_label if y_label else y_col
    )
    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_grouped")


# -----------------------------
# Sankey
# -----------------------------
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
        raise ValueError(f"Sankey links_df missing columns: {needed - set(links_df.columns)}")

    if node_order:
        nodes = node_order
    else:
        nodes = []
        for n in links_df["source"].tolist() + links_df["target"].tolist():
            if n not in nodes:
                nodes.append(n)
    idx = {n: i for i, n in enumerate(nodes)}

    raw_labels = [label_map.get(n, n) if label_map else n for n in nodes]
    wrapped_labels = [_wrap_text(s, max_len=label_wrap) for s in raw_labels]

    default_node_color = "#e5e7eb"
    node_color_list = [(node_colors.get(n) if node_colors else default_node_color) or default_node_color for n in nodes]

    link_colors = []
    for s in links_df["source"].map(idx).tolist():
        c = node_color_list[s]
        link_colors.append(_hex_to_rgba(c, link_alpha))

    sankey = go.Sankey(
        arrangement="snap",
        node=dict(
            label=wrapped_labels,
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
        st.plotly_chart(fig, use_container_width=full_width, key=key or f"{title}_sankey")
    return fig


# -----------------------------
# Ships Charts
# -----------------------------
SHIP_TYPE_LABELS = {
    "C": "C: Container",
    "T": "T: Tanker",
    "B": "B: Bulk",
    "G": "G: Cargo",
    "O": "O: Other",
}

def render_ships_stock(df_base: pd.DataFrame, y_label: str = "Number of Stock Ships"):
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

    d_long["type"] = d_long["type"].replace(SHIP_TYPE_LABELS)

    pref = ["C: Container", "T: Tanker", "B: Bulk", "G: Cargo", "O: Other"]
    seen = list(dict.fromkeys([t for t in pref if t in d_long["type"].unique()]))
    rest = [t for t in sorted(d_long["type"].unique()) if t not in seen]
    order = seen + rest
    d_long["type"] = pd.Categorical(d_long["type"], categories=order, ordered=True)

    fig = px.bar(
        d_long.sort_values(["Year", "type"]),
        x="Year", y="value", color="type",
        title="Stock Ships",
        barmode="stack",
    )
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig

def render_ships_new(df_base: pd.DataFrame, y_label: str = "Number of New Ships"):
    if df_base.empty:
        return px.bar(title="New Ships — no data")

    new_cols = [c for c in df_base.columns if c.startswith("New_Ships_")]
    if "Year" not in df_base.columns or not new_cols:
        return px.bar(title="New Ships — expected 'Year' and 'New_Ships_*' columns")

    d_long = (
        df_base[["Year"] + new_cols]
        .melt(id_vars="Year", var_name="col", value_name="value")
        .assign(type=lambda x: x["col"].str.replace("^New_Ships_", "", regex=True))
    )

    d_long["type"] = d_long["type"].replace(SHIP_TYPE_LABELS)

    pref = ["C: Container", "T: Tanker", "B: Bulk", "G: Cargo", "O: Other"]
    seen = list(dict.fromkeys([t for t in pref if t in d_long["type"].unique()]))
    rest = [t for t in sorted(d_long["type"].unique()) if t not in seen]
    order = seen + rest
    d_long["type"] = pd.Categorical(d_long["type"], categories=order, ordered=True)

    fig = px.bar(
        d_long.sort_values(["Year", "type"]),
        x="Year", y="value", color="type",
        title="New Ships",
        barmode="stack",
    )
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig

def render_ships_investment_cost(df_base: pd.DataFrame, y_label: str = "Costs (M€)"):
    if df_base.empty or "Year" not in df_base.columns or "Investment_Cost" not in df_base.columns:
        return px.line(title="Investment Costs — data missing")

    d = df_base[["Year", "Investment_Cost"]].copy()
    fig = px.line(d, x="Year", y="Investment_Cost", title="Investment Costs")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig


def render_ships_operational_cost(df_base: pd.DataFrame, y_label: str = "Costs (M€)"):
    if df_base.empty or "Year" not in df_base.columns or "Operational_Cost" not in df_base.columns:
        return px.line(title="Operational Costs — data missing")

    d = df_base[["Year", "Operational_Cost"]].copy()
    fig = px.line(d, x="Year", y="Operational_Cost", title="Operational Costs")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig


def render_ships_fuel_demand(df_base: pd.DataFrame, y_label: str = "Fuel Demand [tonnes]"):
    if df_base.empty or "Year" not in df_base.columns:
        return px.bar(title="Fuel Demand — data missing")

    fuel_cols = [c for c in df_base.columns if c.startswith("Fuel_Demand_")]
    if not fuel_cols:
        return px.bar(title="Fuel Demand — no Fuel_Demand_* columns found")

    d_long = (
        df_base[["Year"] + fuel_cols]
        .melt(id_vars="Year", var_name="col", value_name="value")
        .assign(fuel=lambda x: x["col"].str.replace("^Fuel_Demand_", "", regex=True))
    )

    fig = px.bar(
        d_long.sort_values(["Year", "fuel"]),
        x="Year", y="value", color="fuel",
        title="Fuel Demand",
        barmode="stack",
    )
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig


def render_ships_fuel_cost(df_base: pd.DataFrame, y_label: str = "Costs (M€)"):
    if df_base.empty or "Year" not in df_base.columns or "Fuel_Cost" not in df_base.columns:
        return px.line(title="Fuel Costs — data missing")

    d = df_base[["Year", "Fuel_Cost"]].copy()
    fig = px.line(d, x="Year", y="Fuel_Cost", title="Fuel Costs")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig


def render_ships_emissions_and_cap(
    df_base: pd.DataFrame,
    cap_df: pd.DataFrame | None = None,
    cap_year_col: str = "Year",
    cap_value_col: str = "CO2_Cap",
    scale_emissions: float = 1e-6,
    scale_cap: float = 1e-6,
    scale_excess: float = 1e-6,
    y_label: str = "CO₂ Emissions (MtCO₂e)",
):
    if df_base.empty or "Year" not in df_base.columns:
        fig = go.Figure(); fig.update_layout(title="CO₂ Emissions & Cap — no data"); return fig

    em_col = _resolve_ci(df_base, ["CO2_Emissions", "CO2 Emissions"])
    ex_col = _resolve_ci(df_base, ["Excess_Emissions", "Excess Emissions"])
    if not em_col:
        fig = go.Figure(); fig.update_layout(title="CO₂ Emissions & Cap — missing emissions col"); return fig

    d = df_base[["Year", em_col] + ([ex_col] if ex_col else [])].copy()
    d = d.rename(columns={em_col: "Emissions"})
    if ex_col: d = d.rename(columns={ex_col: "Excess"})

    if cap_df is not None and not cap_df.empty:
        cap_work = cap_df.rename(columns={cap_year_col: "Year", cap_value_col: "Cap"})[["Year", "Cap"]].copy()
        d = d.merge(cap_work, on="Year", how="left")
    else:
        d["Cap"] = d["Emissions"] - d.get("Excess", 0)

    d["Emissions"] = pd.to_numeric(d["Emissions"], errors="coerce") * scale_emissions
    d["Cap"]       = pd.to_numeric(d["Cap"], errors="coerce") * scale_cap
    if "Excess" in d: d["Excess"] = pd.to_numeric(d["Excess"], errors="coerce") * scale_excess

    d = d.sort_values("Year"); x = d["Year"]
    fig = go.Figure()

    if d["Cap"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=d["Cap"], name="CO₂ Cap", mode="lines", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=d["Emissions"], name="CO₂ Emissions", mode="lines+markers", line=dict(width=2)))

    if "Excess" in d and d["Excess"].fillna(0).abs().sum() > 0:
        cap_series = d["Cap"]
        top  = d["Emissions"].where(d["Excess"] > 0)
        base = cap_series.where(d["Excess"] > 0)
        fig.add_trace(go.Scatter(x=x, y=base, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=top, name="Excess Emissions", mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(220,38,38,0.3)"))

    fig.update_layout(
        title="CO₂ Emissions and Cap",
        xaxis_title="",
        yaxis_title=y_label,
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10,r=10,t=60,b=10), legend_title=None,
    )
    return fig


def render_ships_ets_penalty(df_base: pd.DataFrame, y_label: str = "Costs (M€)"):
    if df_base.empty or "Year" not in df_base.columns:
        return px.line(title="ETS Penalty — no data")

    penalty_col = _resolve_ci(df_base, ["ETS_Penalty", "ets_penalty", "ETS penalty", "eth_penalty"])
    if not penalty_col:
        return px.line(title="ETS Penalty — 'ETS_Penalty' column not found")

    d = df_base[["Year", penalty_col]].rename(columns={penalty_col: "ETS_Penalty"}).copy()
    d["ETS_Penalty"] = pd.to_numeric(d["ETS_Penalty"], errors="coerce")

    fig = px.line(d, x="Year", y="ETS_Penalty", title="ETS Penalty")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        width=getattr(theme, "CHART_WIDTH", 800),
        height=getattr(theme, "CHART_HEIGHT", 500),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title="",
        yaxis_title=y_label,
        legend_title=None,
    )
    return fig


# -----------------------------
# Water Charts
# -----------------------------
def render_water_band(
    df: pd.DataFrame,
    title: str,
    year_col: str = "Year",
    avg_col_candidates: list[str] = None,
    min_col_candidates: list[str] = None,
    max_col_candidates: list[str] = None,
    y_label: str = "Water Requirements [hm³]",
    key=None,
):
    avg_col_candidates = avg_col_candidates or ["Average", f"{title} Average", f"{title} Avg", "Avg"]
    min_col_candidates = min_col_candidates or ["Min", f"{title} Min", "Minimum"]
    max_col_candidates = max_col_candidates or ["Max", f"{title} Max", "Maximum"]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — no data")
        return fig

    ycol = _resolve_ci(df, [year_col, "Year"])
    ac   = _resolve_ci(df, avg_col_candidates)
    mic  = _resolve_ci(df, min_col_candidates)
    mac  = _resolve_ci(df, max_col_candidates)

    if not (ycol and ac and mic and mac):
        fig = go.Figure()
        fig.update_layout(title=f"{title} — missing required cols")
        return fig

    d = df[[ycol, ac, mic, mac]].rename(columns={ycol:"Year", ac:"Avg", mic:"Min", mac:"Max"}).copy()
    for c in ("Avg","Min","Max"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("Year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Min"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Max"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.20)", name="Range (Min–Max)"))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Avg"], mode="lines+markers", name="Average", line=dict(width=2)))

    fig.update_layout(title=title, xaxis_title="", yaxis_title=y_label)
    return fig


def render_water_monthly_band(
    df: pd.DataFrame,
    title: str = "Monthly Water Requirements (2020)",
    month_col_candidates: list[str] = None,
    avg_col_candidates: list[str] = None,
    min_col_candidates: list[str] = None,
    max_col_candidates: list[str] = None,
    y_label: str = "Water Requirements [hm³]",
    key=None,
):
    month_col_candidates = month_col_candidates or ["Month", "Months", "month", "months"]
    avg_col_candidates   = avg_col_candidates or ["Average", "Avg", f"{title} Average"]
    min_col_candidates   = min_col_candidates or ["Min", "Minimum", f"{title} Min"]
    max_col_candidates   = max_col_candidates or ["Max", "Maximum", f"{title} Max"]

    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — no data")
        return fig

    mcol = _resolve_ci(df, month_col_candidates)
    ac   = _resolve_ci(df, avg_col_candidates)
    mic  = _resolve_ci(df, min_col_candidates)
    mac  = _resolve_ci(df, max_col_candidates)

    if not (mcol and ac and mic and mac):
        fig = go.Figure()
        fig.update_layout(title=f"{title} — missing required cols")
        return fig

    d = df[[mcol, ac, mic, mac]].rename(columns={mcol:"Month", ac:"Avg", mic:"Min", mac:"Max"}).copy()
    for c in ("Avg","Min","Max"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    if d["Month"].dtype == object:
        d["Month"] = d["Month"].astype(str).str.strip().str.upper()
        d["Month"] = pd.Categorical(d["Month"], categories=month_order, ordered=True)
    d = d.sort_values("Month")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Min"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Max"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.20)", name="Range (Min–Max)"))
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Avg"], mode="lines+markers", name="Average", line=dict(width=2)))

    fig.update_layout(title=title, xaxis_title="Months", yaxis_title=y_label)
    return fig


# ------------------------------------------------------------
# Interactive Scenarios - FOOD/LAND
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_interactive_data(
    path: str = "data/Fable_results_combos.xlsx",
    sheet_name: str = "Custom combinations from user"
) -> pd.DataFrame:
    """Parse the interactive Excel file into tidy form."""
    df_raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    option_rows = df_raw.index[
        df_raw.iloc[:, 0].astype(str).str.contains("Option", na=False)
    ].tolist()
    if not option_rows:
        st.error("No 'Option' labels found in the sheet.")
        return pd.DataFrame()

    blocks = []
    for i, start_row in enumerate(option_rows):
        label = str(df_raw.iloc[start_row, 0]).strip()
        next_row = option_rows[i + 1] if i + 1 < len(option_rows) else len(df_raw)
        header_row = start_row + 1
        if header_row >= len(df_raw):
            continue

        headers = df_raw.iloc[header_row].tolist()
        data_start = header_row + 1
        block = df_raw.iloc[data_start:next_row].copy()
        block = block.dropna(how="all")
        if block.empty:
            continue

        block.columns = [str(h).strip() for h in headers]
        block["ScenarioOption"] = label
        blocks.append(block)

    if not blocks:
        st.warning("No scenario blocks parsed.")
        return pd.DataFrame()

    df = pd.concat(blocks, ignore_index=True)

    year_col = _resolve_ci(df, ["Year"])
    if year_col:
        df["Year"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["Year"] = None

    # Extract identifiers (A/B/C)
    import re
    pat = r"Option\s*([A-C])\-([A-C])\-([A-C])"
    df[["PopOpt", "DietOpt", "ProdOpt"]] = df["ScenarioOption"].str.extract(pat)
    df["ScenarioOption"] = df["ScenarioOption"].str.replace(r"Option\s*", "", regex=True)

    # Normalize and convert numeric columns
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if c not in ["ScenarioOption", "PopOpt", "DietOpt", "ProdOpt", "Year"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def render_interactive_controls(tab_name: str):
    st.subheader(f"{tab_name} – Interactive Scenario Builder")

    with st.expander("ℹ️ About the FABLE Calculator"):
        st.markdown("""
        **FABLE Calculator** uses CORINE national land cover baseline, FAOSTAT crop yields (historical & trend),
        livestock numbers, and food demand projections, at an annual time-step, to estimate food-land system pathways to 2050.

        Its large scenario explorer leverages **national population and GDP projections** based on
        the *Shared Socioeconomic Pathways (SSPs)*.  
        Dietary choices imply uptake in ingredients/products defined by **SSPs**, **EAT-Lancet Diet**, or
        **custom scenarios**, while default rates of **crop and livestock productivity** (low, medium, high)
        can be adjusted.

        The FABLE Calculator offers a portfolio of **more than 1.5 billion pathways**, reflecting variations in climate,
        economics, agricultural policy, regulation, and demographics.

        Here, you can explore the **key demand-side drivers** (Population, Diet, Productivity) that shape these pathways.
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        pop = st.selectbox(
            "Population & GDP projection",
            options=["A", "B", "C"],
            format_func=lambda x: {"A": "SSP2 (BAU)", "B": "SSP1 (NCNC)", "C": "SSP5"}[x],
            help=(
                "Population and GDP projections to 2050:\n\n"
                "**Option A (SSP2 – BAU)**: Moderate population decline (~24% by 2100), GDP growth 1.9–2.2%.\n\n"
                "**Option B (SSP1 – NCNC)**: Sustainability-oriented, smaller decline (~14%) and higher GDP growth (2.0–2.5%).\n\n"
                "**Option C (SSP5)**: Rapid economic growth, strong tech progress, decline up to 40% by 2100."
            ),
        )

    with col2:
        diet = st.selectbox(
            "Dietary choice",
            options=["A", "B", "C"],
            format_func=lambda x: {"A": "Baseline (BAU)", "B": "EAT-Lancet (NCNC)", "C": "FatDiet"}[x],
            help=(
                "Dietary choices affecting agricultural land:\n\n"
                "**Option A (Baseline)**: FAO baseline, 2010–2020 diet, limited land expansion.\n\n"
                "**Option B (EAT-Lancet)**: Healthier diet, limited land expansion, low deforestation.\n\n"
                "**Option C (FatDiet)**: High-fat, sugar- and meat-rich diet, greater land expansion potential."
            ),
        )

    with col3:
        prod = st.selectbox(
            "Crop & livestock productivity",
            options=["A", "B", "C"],
            format_func=lambda x: {"A": "Baseline", "B": "High growth", "C": "Low growth"}[x],
            help=(
                "Crop and livestock productivity assumptions:\n\n"
                "**Option A (Baseline)**: No change from 2010–2020 levels.\n\n"
                "**Option B (High growth)**: Closes ~50–80% of yield gaps (FAOSTAT).\n\n"
                "**Option C (Low growth)**: Closes ~30–40% of yield gaps."
            ),
        )

    render_interactive_charts(tab_name, pop, diet, prod)

    st.markdown("---")
    st.subheader("📈 Sensitivity Summary")

    sensitivity_image_path = Path("content/fable_sensitivity.png")

    if sensitivity_image_path.exists():
        with st.expander("📊 Show sensitivity summary figure", expanded=False):
            st.image(str(sensitivity_image_path), width=600)
            st.caption("Relative sensitivity of key indicators under different scenario combinations.")
    else:
        st.info("Sensitivity summary image not found.")


def render_interactive_charts(tab_name: str, pop: str, diet: str, prod: str):
    """Filter interactive Excel data and render charts matching BAU/NCNC layout & styling."""
    df_all = load_interactive_data()
    if df_all.empty:
        st.warning("Interactive data could not be loaded or parsed.")
        return

    mask = (
        (df_all["PopOpt"] == pop)
        & (df_all["DietOpt"] == diet)
        & (df_all["ProdOpt"] == prod)
    )
    df = df_all.loc[mask].copy()
    if df.empty:
        st.info(f"No data found for combination {pop}-{diet}-{prod}.")
        return

    st.markdown(f"### Results for Option {pop}-{diet}-{prod}")
    st.caption("Below are indicative charts based on your selected combination.")

    # Detect relevant column groups
    ghg_cols = [c for c in df.columns if "GHG" in c or "CO2" in c or "Emissions" in c]
    cost_cols = [c for c in df.columns if "Cost" in c]
    land_cols = [c for c in df.columns if any(x in c for x in ["Land", "Cropland", "Forest", "Pasture", "Area"])]

    col1, col2 = st.columns(2)

    # Emissions chart
    # Emissions chart
    with col1:
        if ghg_cols:
            ghg_long = df.melt(
                id_vars=["Year"],
                value_vars=ghg_cols,
                var_name="Component",
                value_name="Value"
            )

            # --- Normalize component names to match theme keys
            name_map = {
                "GHG Crop": "Crops",
                "GHG Livestock": "Livestock",
                "GHG LUC": "Land-use",
                "GHG Total": "Total emissions",
                "Total CO₂e": "Total emissions",
            }
            ghg_long["Component"] = ghg_long["Component"].replace(name_map)

            # --- Apply categorical order from theme
            if hasattr(theme, "EMISSIONS_ORDER"):
                ghg_long["Component"] = pd.Categorical(
                    ghg_long["Component"],
                    categories=theme.EMISSIONS_ORDER,
                    ordered=True,
                )
                ghg_long = ghg_long.sort_values(["Year", "Component"])

            # --- Extract total line (if exists)
            total_candidates = [c for c in ghg_cols if "Total" in c]
            if total_candidates:
                total_df = (
                    df[["Year", total_candidates[0]]]
                    .rename(columns={total_candidates[0]: "Value"})
                    .assign(Component="Total emissions")
                )
            else:
                total_df = pd.DataFrame()

            # --- Render using same theme-based settings
            render_grouped_bar_and_line(
                prod_df=ghg_long,
                demand_df=total_df if not total_df.empty else None,
                x_col="Year",
                y_col="Value",
                category_col="Component",
                title="Production-based agricultural emissions",
                y_label="Mt CO₂e",
                colors=getattr(theme, "EMISSIONS_COLORS", {}),
                key=f"ghg_interactive_{pop}{diet}{prod}",
            )


    # Costs chart
    with col2:
        if cost_cols:
            cost_long = df.melt(
                id_vars=["Year"],
                value_vars=cost_cols,
                var_name="Component",
                value_name="Value"
            )
            cost_long["Component"] = pd.Categorical(
                cost_long["Component"],
                categories=list(theme.COST_COLORS.keys()),
                ordered=True
            )
            render_bar_chart(
                cost_long,
                "Year", "Value", "Component",
                "Agricultural production cost",
                [str(y) for y in sorted(df["Year"].unique())],
                colors=theme.COST_COLORS,
                y_label="M€",
                key=f"costs_interactive_{pop}{diet}{prod}"
            )
    # Land use chart
    col3, = st.columns(1)
    with col3:
        if land_cols:
            land_long = df.melt(id_vars=["Year"], value_vars=land_cols, var_name="Component", value_name="Value")
            render_line_chart(
                land_long,
                x_col="Year", y_col="Value", category_col="Component",
                title="Land uses evolution",
                y_label="1000 km²",
                colors=getattr(theme, "LANDUSE_COLORS", {}),
                key=f"land_interactive_{pop}{diet}{prod}"
            )


# ------------------------------------------------------------
# Energy Tab Interactive Scenarios
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_energy_interactive_data(scenario_code: str):
    """Load and filter interactive energy datasets for a given scenario code (AA, BB, etc.)."""
    from models.data_loader import load_and_prepare_excel

    # Include scenario_code in cache key so different selections reload fresh data
    cache_key = f"energy_data_{scenario_code}"

    df_cons = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
    df_emis = load_and_prepare_excel("data/LEAP_Demand_Emissions.xlsx")
    df_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
    df_supply_emis = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")

    def filt(df, code):
        if "Scenario" in df.columns:
            mask = df["Scenario"].astype(str).str.strip().eq(code)
            return df.loc[mask].copy()
        return df.copy()

    # Return as a tuple so caching recognizes scenario_code as unique
    return (
        filt(df_cons, scenario_code),
        filt(df_emis, scenario_code),
        filt(df_supply, scenario_code),
        filt(df_supply_emis, scenario_code),
    )

def render_energy_interactive_controls(tab_name: str):
    """Interactive scenario UI for Energy tab with rich help boxes."""
    st.subheader("⚡ Interactive Scenario – select your data and explore the results!")

    with st.expander("ℹ️ About the Energy–Emissions Explorer"):
        st.markdown("""
        **LEAP (Low Emissions Analysis Platform)** is the core model of our framework, integrating
        all energy demand and supply sectors, and simulating energy flows, fuel generation, and
        resulting emissions.

        It is a **scenario-based tool**, where the user defines pathways through key assumptions:
        - **Population & GDP projections** based on *Shared Socioeconomic Pathways (SSPs)*,
          harmonized across all models.
        - **Renewables uptake** accounting for autonomous trends such as technology cost declines,
          electrification, or other ongoing shifts toward cleaner energy.

        For Greece, according to its *National Energy and Climate Plan (NECP)*, hydropower generation
        is expected to remain roughly constant by 2050 due to water scarcity. Thus, **solar and wind**
        are the main drivers of renewables uptake.
        """)

    col1, col2 = st.columns(2)

    with col1:
        ssp = st.selectbox(
            "Population and GDP projections to 2050",
            options=["A", "B", "C"],
            format_func=lambda x: f"Option {x}: " + {
                "A": "SSP1 (NCNC)",
                "B": "SSP2 (BAU)",
                "C": "SSP5",
            }[x],
            help=(
                "**Option A – SSP1 (NCNC):**\n"
                "Under the SSP1 ('strong societal sustainability and environmental consciousness') "
                "scenario, Greece's population is projected to decline by ~14% by 2050 relative to 2021. "
                "GDP grows by **2.0–2.5% per year**.\n\n"
                "**Option B – SSP2 (BAU):**\n"
                "Represents a 'middle of the road' case. Population declines more steeply (≈24% by 2100), "
                "with moderate fertility, mortality, and migration trends. GDP growth averages **1.9–2.2%**.\n\n"
                "**Option C – SSP5:**\n"
                "An 'extreme rapid development' case: high tech progress, global integration, and GDP growth "
                "of **~2.3–2.5%**, while population declines up to 40% by 2100."
            ),
        )

    with col2:
        renew = st.selectbox(
            "Renewables uptake to 2050",
            options=["A", "B", "C"],
            format_func=lambda x: f"Option {x}: " + {
                "A": "Conservative baseline",
                "B": "Central (plausible market-driven baseline)",
                "C": "Optimistic baseline",
            }[x],
            help=(
                "**Option A – Conservative baseline:** Slower technology and market uptake; "
                "limited by grid integration and permitting. Solar +0.4 %/yr, wind +0.2 %/yr.\n\n"
                "**Option B – Central baseline:** Matches Greece's rapid PV rollout (as of 2023) and "
                "moderate global cost declines. Solar +0.6 %/yr, wind +0.4 %/yr; hydropower steady. "
                "Aligned with NECP baseline.\n\n"
                "**Option C – Optimistic baseline:** Fast technology adoption with fewer practical constraints "
                "(grid, permitting, storage). Solar +0.8 %/yr, wind +0.6 %/yr, faster conversion to delivered share."
            ),
        )

    st.markdown(
        "In all cases, **renewables shares follow NECP projections:** electrification in residential "
        "(+15% by 2050), transport (~10%), and decreasing oil refining activity."
    )
    st.markdown(f"**Selected configuration:** Option {ssp}–{renew}")

    render_energy_interactive_charts(tab_name, ssp, renew)

    st.markdown("---")
    st.subheader("📈 Sensitivity Summary")
    sens_img = Path("content/leap_sensitivity.png")
    if sens_img.exists():
        with st.expander("Show sensitivity summary", expanded=False):
            st.image(str(sens_img), width=600)
            st.caption("Relative contribution of SSP and renewables uptake assumptions.")
    else:
        st.info("Sensitivity summary image not found.")


import streamlit as st
import pandas as pd
from models.data_loader import load_and_prepare_excel, prepare_stacked_data, aggregate_to_periods
from config import theme

@st.cache_data(show_spinner=False)
def _load_energy_interactive_data(scenario_code: str):
    """Load and filter interactive energy datasets for a given scenario."""
    df_cons = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
    df_emis = load_and_prepare_excel("data/LEAP_Demand_Emissions.xlsx")
    df_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
    df_supply_emis = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")

    def filter_df(df, code):
        if "Scenario" in df.columns:
            mask = df["Scenario"].astype(str).str.strip() == code
            return df.loc[mask].copy()
        return df.copy()

    return (
        filter_df(df_cons, scenario_code),
        filter_df(df_emis, scenario_code),
        filter_df(df_supply, scenario_code),
        filter_df(df_supply_emis, scenario_code),
    )

# ------------------------------------------------------------
# Energy Tab Interactive Scenarios (fixed)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_energy_interactive_data(scenario_code: str):
    """Load and filter interactive energy datasets for a given scenario code (AA, BB, etc.)."""
    from models.data_loader import load_and_prepare_excel

    df_cons = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
    df_emis = load_and_prepare_excel("data/LEAP_Demand_Emissions.xlsx")
    df_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
    df_supply_emis = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")

    def filt(df, code):
        if "Scenario" in df.columns:
            mask = df["Scenario"].astype(str).str.strip() == code
            return df.loc[mask].copy()
        return df.copy()

    return (
        filt(df_cons, scenario_code),
        filt(df_emis, scenario_code),
        filt(df_supply, scenario_code),
        filt(df_supply_emis, scenario_code),
    )

def render_energy_interactive_controls(tab_name: str):
    """Interactive scenario UI for Energy tab with concise help tooltips (❓ icons)."""
    st.subheader("⚡ Interactive Energy–Emissions Explorer")

    with st.expander("ℹ️ About the Energy–Emissions Explorer"):
        st.markdown("""
        **LEAP (Low Emissions Analysis Platform)** integrates all energy demand and supply sectors,
        simulating energy flows, fuel generation, and resulting emissions.
        
        Use the dropdowns below to explore how **population, GDP growth, and renewables uptake** 
        affect Greece's energy and emissions pathways to 2050.
        """)

    col1, col2 = st.columns(2)

    # -------------------------------------------------------------------------
    # Population & GDP projections dropdown
    # -------------------------------------------------------------------------
    with col1:
        ssp = st.selectbox(
            "Population & GDP projections to 2050:",
            options=["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – SSP1 (NCNC)",
                "B": "Option B – SSP2 (BAU)",
                "C": "Option C – SSP5",
            }[x],
            help=(
                "**Population and GDP projections to 2050:**\n\n"
                "**Option A – SSP1 (NCNC):**\n"
                "Under the SSP1 (“strong societal sustainability and environmental consciousness”) scenario, "
                "Greece's population is projected to decline substantially (~14% by 2050 vs. 2021). "
                "GDP growth averages **2.0–2.5%/yr**.\n\n"
                "**Option B – SSP2 (BAU):**\n"
                "Under the SSP2 ('middle of the road') scenario, Greece's population declines more sharply "
                "(~24% by 2100), with moderate fertility, mortality, and migration. "
                "GDP grows **1.9–2.2%/yr**.\n\n"
                "**Option C – SSP5:**\n"
                "An “extreme rapid development” case: high tech progress and global integration, "
                "but population declines up to 40% by 2100. GDP growth **2.3–2.5%/yr**."
            ),
            key="ssp_select",
        )

    # -------------------------------------------------------------------------
    # Renewables uptake dropdown
    # -------------------------------------------------------------------------
    with col2:
        renew = st.selectbox(
            "Renewables uptake to 2050:",
            options=["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – Conservative baseline",
                "B": "Option B – Central (market-driven)",
                "C": "Option C – Optimistic (fast cost declines)",
            }[x],
            help=(
                "**Renewables uptake to 2050:**\n\n"
                "**Option A – Conservative baseline:**\n"
                "Slower market and technology uptake due to grid integration and permitting limits. "
                "Solar +0.4 %/yr, wind +0.2 %/yr.\n\n"
                "**Option B – Central (plausible market-driven baseline):**\n"
                "Reflects Greece’s rapid PV rollout (up to end of 2023) and moderate global cost declines. "
                "Aligned with NECP baseline: solar +0.6 %/yr, wind +0.4 %/yr, hydropower steady.\n\n"
                "**Option C – Optimistic baseline:**\n"
                "Fast uptake with fewer constraints (grid, permitting, storage). "
                "Solar +0.8 %/yr, wind +0.6 %/yr."
            ),
            key="renew_select",
        )

    # -------------------------------------------------------------------------
    # Baseline note (shared context)
    # -------------------------------------------------------------------------
    st.markdown("""
    In all cases, renewables shares follow NECP projections:
    - **Residential sector:** electrification +15 % by 2050  
    - **Transport sector:** ~10 % by 2050  
    - **Oil refining activity:** decreasing by 2050  

    The **NCNC (SSP1)** pathway includes additional sustainability measures from the NECP.
    """)

    st.markdown(f"**Selected configuration:** Option {ssp}–{renew}")

    # --- Render charts dynamically
    render_energy_interactive_charts(tab_name, ssp, renew)

    # --- Sensitivity Summary (unchanged)
    st.markdown("---")
    st.subheader("📈 Sensitivity Summary")
    sens_img = Path("content/leap_sensitivity.png")
    if sens_img.exists():
        with st.expander("Show sensitivity summary", expanded=False):
            st.image(str(sens_img), width=600)
            st.caption("Relative contribution of SSP and renewables uptake assumptions.")
    else:
        st.info("Sensitivity summary image not found.")


    # ------------------------------------------------------------------
    # 📈 Sensitivity Summary (identical structure to Food–Land)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Sensitivity Summary")

    sensitivity_image_path = Path("content/leap_sensitivity.png")

    if sensitivity_image_path.exists():
        with st.expander("📊 Show sensitivity summary figure", expanded=False):
            st.image(str(sensitivity_image_path), use_container_width=True)
            st.caption("Sensitivity summary for LEAP Energy interactive scenarios (Sheet2 of LEAP Excel files).")
    else:
        st.info("Sensitivity summary image not found. Please add 'content/leap_sensitivity.png'.")



def render_energy_interactive_charts(tab_name: str, ssp: str, renew: str):
    """Render four Energy charts matching BAU/NCNC layout."""
    from models.data_loader import aggregate_to_periods
    scenario_code = f"{ssp}{renew}"

    df_energy, df_demand_emissions, df_energy_supply, df_supply_emissions = _load_energy_interactive_data(scenario_code)

    if all(df.empty for df in [df_energy, df_demand_emissions, df_energy_supply, df_supply_emissions]):
        st.warning(f"No data found for scenario {scenario_code}.")
        return

    st.markdown(f"### Results for Option {ssp}–{renew}")
    st.caption("Charts update automatically based on your selections.")

    # --- 1️⃣ Energy consumption & emissions per sector ---
    cols_sectors = [
        "Residential", "Agriculture", "Industry", "Energy Products",
        "Passenger Transportation", "Freight Transportation", "Maritime", "Services",
    ]

    c1, c2 = st.columns(2)
    with c1:
        if not df_energy.empty:
            melted = df_energy.melt(id_vars=["Year"], value_vars=cols_sectors,
                                    var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(
                grouped, "PeriodStr", "Value", "Component",
                "Total energy consumption per sector",
                order,
                y_label="ktoe",
                key=f"int_energy_cons_{scenario_code}"
            )
    with c2:
        if not df_demand_emissions.empty:
            melted = df_demand_emissions.melt(id_vars=["Year"], value_vars=cols_sectors,
                                              var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(
                grouped, "PeriodStr", "Value", "Component",
                "Emissions from energy consumption by sector",
                order,
                y_label="MtCO₂e",
                key=f"int_energy_emis_{scenario_code}"
            )

    # --- 2️⃣ Generation and fuel emissions ---
    cols_fuels = ["Hydrogen Generation", "Electricity Generation", "Heat Generation", "Oil Refining"]
    c3, c4 = st.columns(2)
    with c3:
        if not df_energy_supply.empty:
            melted = df_energy_supply.melt(id_vars=["Year"], value_vars=cols_fuels,
                                           var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(
                grouped, "PeriodStr", "Value", "Component",
                "Generated energy per fuel type",
                order,
                colors=theme.FUEL_COLORS,
                y_label="ktoe",
                key=f"int_energy_fuel_{scenario_code}"
            )
    with c4:
        if not df_supply_emissions.empty:
            melted = df_supply_emissions.melt(
                id_vars=["Year"],
                value_vars=["Electricity Generation", "Heat Generation", "Oil Refining"],
                var_name="Component", value_name="Value",
            )
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(
                grouped, "PeriodStr", "Value", "Component",
                "Emissions per fuel type",
                order,
                colors=theme.FUEL_COLORS,
                y_label="MtCO₂e",
                key=f"int_energy_fuel_emis_{scenario_code}"
            )
# ------------------------------------------------------------
# Energy Sensitivity Summary (Sheet2)
# ------------------------------------------------------------
import plotly.express as px

@st.cache_data(show_spinner=False)
def _load_energy_sensitivity_sheets() -> dict[str, pd.DataFrame]:
    """Load Sheet2 from the four LEAP Energy Excel files."""
    paths = {
        "cons": "data/LEAP_Demand_Cons.xlsx",
        "emis": "data/LEAP_Demand_Emissions.xlsx",
        "supply": "data/LEAP_Supply.xlsx",
        "supply_emis": "data/LEAP_Supply_Emissions.xlsx",
    }
    out = {}
    for key, p in paths.items():
        try:
            df = pd.read_excel(p, sheet_name="Sheet2")
            df.columns = df.columns.map(str).str.strip()
            if "Scenario" in df.columns:
                df["Scenario"] = df["Scenario"].astype(str).str.strip().str.upper()
            out[key] = df
        except Exception as e:
            st.warning(f"⚠️ Could not load Sheet2 from {p}: {e}")
            out[key] = pd.DataFrame()
    return out


def render_energy_sensitivity_summary():
    """Display collapsible sensitivity summary image (bottom of Energy tab)."""
    st.markdown("---")
    st.subheader("📈 Sensitivity Summary")

    img_path = Path("content/leap_sensitivity.png")

    if img_path.exists():
        with st.expander("📊 Show sensitivity summary figure", expanded=False):
            st.image(str(img_path), width=800, caption="LEAP Energy sensitivity summary (Sheet2 of LEAP Excel files).")
    else:
        st.info("Sensitivity summary image not found. Please add 'content/leap_sensitivity.png'.")

# --- BIOFUELS INTERACTIVE ---
@st.cache_data(show_spinner=False)
def load_biofuels_data(path: str = "data/LEAP_biofuels.xlsx") -> dict[str, pd.DataFrame]:
    """
    Parse the LEAP_biofuels.xlsx file (single sheet) into structured blocks:
    - production:  min/max + demand (wide → tidy)
    - exports:     min/max + demand (wide → tidy)
    - combos:      scenario combinations (A–A–A, etc.)
    """
    df = pd.read_excel(path, sheet_name="Custom combinations from user", header=None)

    # --- locate block starts ---
    prod_start = df.index[df.iloc[:, 0].astype(str)
                          .str.contains("Minimum Production", case=False, na=False)][0]
    exp_start = df.index[df.iloc[:, 9:].apply(
        lambda s: s.astype(str).str.contains("Minimum export", case=False, na=False)
    ).any(axis=1)][0]
    combo_start = df.index[df.iloc[:, 0].astype(str)
                           .str.contains("All other options", case=False, na=False)][0] + 2

    years = [2022, 2025, 2030, 2035, 2040, 2045, 2050]

    # --- production block (left, rows 3–6) ---
    left_block = df.iloc[prod_start:prod_start + 4, 0:1 + len(years)].copy()
    left_block.columns = ["Label"] + [str(y) for y in years]
    prod = (
        left_block.melt(id_vars="Label", var_name="Year", value_name="Value")
        .pivot(index="Year", columns="Label", values="Value")
        .reset_index()
    )

    # --- export potentials (right side) ---
    right_block = df.iloc[exp_start:exp_start + 4, 9:9 + 1 + len(years)].copy()
    right_block.columns = ["Label"] + [str(y) for y in years]
    exports = (
        right_block.melt(id_vars="Label", var_name="Year", value_name="Value")
        .pivot(index="Year", columns="Label", values="Value")
        .reset_index()
    )

    # --- combos block (bottom of sheet) ---
    combos = df.iloc[combo_start:, :4].dropna(how="all").copy()
    combos.columns = [
        "Code", "Year",
        "Selected Production Potential (ktoe)",
        "Selected Export Potential (ktoe)"
    ]
    for c in ["Year", "Selected Production Potential (ktoe)", "Selected Export Potential (ktoe)"]:
        combos[c] = pd.to_numeric(combos[c], errors="coerce")

    # --- rename columns consistently for app ---
    prod = prod.rename(columns={
        "Minimum Production Potential [ktoe]": "MinProd_ktoe",
        "Maximum Production Potential [ktoe]": "MaxProd_ktoe",
        "Biofuel Demand Baseline scenario [ktoe]": "Demand_BAU_ktoe",
        "Biofuel Demand NECP [ktoe]": "Demand_NCNC_ktoe",
    })

    exports = exports.rename(columns={
        "Minimum export potential, Baseline scenario [ktoe]": "MinExport_BAU_ktoe",
        "Maximum export potential, Baseline scenario [ktoe]": "MaxExport_BAU_ktoe",
        "Minimum export potential, NECP [ktoe]": "MinExport_NCNC_ktoe",
        "Maximum export potential, NECP [ktoe]": "MaxExport_NCNC_ktoe",
    })

    return {"production": prod, "exports": exports, "combos": combos}
# ---------------------------------------------------------------------
# Interactive Biofuels helper dictionaries
# ---------------------------------------------------------------------
def get_biofuels_option_sets():
    """Return mapping dictionaries and descriptions for interactive selectors."""
    sets = {
        "residual": {
            "title": "– Residual Availability",
            "description": (
                "This is the amount of production residuals (corn, sugarbeets, sunflowers, olives, wheat) "
                "that can be used for biofuels production, without affecting agricultural production. "
                "Primarily derived from the FABLE Calculator."
            ),
            "options": {
                "A": "Option A – minimum (30%)",
                "B": "Option B – average (35%)",
                "C": "Option C – maximum (40%)",
            },
        },
        "coefficient": {
            "title": "– Biofuel Production Coefficient [L/t]",
            "description": (
                "Liters of biofuel produced per ton of crop, based on empirical data and studies."
            ),
            "options": {
                "A": "Option A – minimum (340–380)",
                "B": "Option B – average (380–450)",
                "C": "Option C – maximum (450–520)",
            },
        },
        "technology": {
            "title": "– Technology Adoption Rate",
            "description": (
                "Reflects the mandated fuels blending uptake per national commitments, "
                "driving bioethanol and biodiesel use by 2050. Derived from LEAP outputs."
            ),
            "options": {
                "A": "Option A – slow",
                "B": "Option B – moderate",
                "C": "Option C – fast",
            },
        },
    }
    return sets

def render_biofuels_interactive_controls(tab_name: str):
    from views.charts import load_biofuels_data, _render_biofuels_base_chart
    st.header(f"{tab_name} — Interactive Biofuels Scenarios")

    with st.expander("ℹ️ About the Biofuels Explorer"):
        st.markdown("""
        Explore sensitivities in potential **production** and **export** of biofuels under different
        assumptions for **residual availability**, **conversion efficiency**, and **technology adoption rates**.
        """)

    sets = get_biofuels_option_sets()
    col1, col2, col3 = st.columns(3)

    with col1:
        res_opt = st.selectbox(
            "Residual Availability",
            options=list(sets["residual"]["options"].keys()),
            format_func=lambda o: sets["residual"]["options"][o],
            help=(
                f"**{sets['residual']['title']}**\n\n"
                + sets['residual']['description']
                + "\n\n"
                + "\n".join([f"- {v}" for v in sets['residual']['options'].values()])
            ),
        )

    with col2:
        coef_opt = st.selectbox(
            "Biofuel Production Coefficient [L/t]",
            options=list(sets["coefficient"]["options"].keys()),
            format_func=lambda o: sets["coefficient"]["options"][o],
            help=(
                f"**{sets['coefficient']['title']}**\n\n"
                + sets['coefficient']['description']
                + "\n\n"
                + "\n".join([f"- {v}" for v in sets['coefficient']['options'].values()])
            ),
        )

    with col3:
        tech_opt = st.selectbox(
            "Technology Adoption Rate",
            options=list(sets["technology"]["options"].keys()),
            format_func=lambda o: sets["technology"]["options"][o],
            help=(
                f"**{sets['technology']['title']}**\n\n"
                + sets['technology']['description']
                + "\n\n"
                + "\n".join([f"- {v}" for v in sets['technology']['options'].values()])
            ),
        )

    combo_code = f"{res_opt}-{coef_opt}-{tech_opt}"
    st.markdown(f"**Selected combination:** {combo_code}**")

    # --- Load and plot
    data = load_biofuels_data()
    prod, exports, combos = data["production"], data["exports"], data["combos"]

    def _normalize_code(s):
        return str(s).strip().lower().replace("–", "-").replace("—", "-").replace(" ", "")

    combo_code_clean = _normalize_code(combo_code)
    combos["CodeClean"] = combos["Code"].apply(_normalize_code)

    combo = combos[combos["CodeClean"] == combo_code_clean]
    if combo.empty:
        partial_matches = [c for c in combos["CodeClean"] if c.startswith(combo_code_clean[:-1])]
        if partial_matches:
            fallback = partial_matches[0]
            st.info(f"No exact match for {combo_code_clean}; using nearest {fallback.upper()} instead.")
            combo = combos[combos["CodeClean"] == fallback]
        else:
            st.warning(f"No data found for combination {combo_code}.")
            return

    combo_line = combo[["Year", "Selected Production Potential (ktoe)"]].rename(
        columns={"Selected Production Potential (ktoe)": "Value"}
    )
    combo_line["Component"] = f"Biofuel Production ({combo_code})"

    _render_biofuels_base_chart(prod, exports, scen_key="BAU", line_override=combo_line)

        
def _render_biofuels_base_chart(prod: pd.DataFrame, exports: pd.DataFrame,
                                scen_key: str, line_override: pd.DataFrame | None = None):
    """Render the standard Biofuels charts (bars + line overlay) reused by BAU/NCNC and interactive tabs."""
    import plotly.express as px
    from config import theme
    import streamlit as st

    col1, col2 = st.columns(2)

    # --- Demand vs Potential Supply ---
    with col1:
        demand_col = f"Demand_{scen_key}_ktoe"
        bars = prod.melt(
            id_vars=["Year"],
            value_vars=["MinProd_ktoe", "MaxProd_ktoe"],
            var_name="Component",
            value_name="Value"
        )
        bars["Component"] = bars["Component"].replace({
            "MinProd_ktoe": "Minimum Production Potential [ktoe]",
            "MaxProd_ktoe": "Maximum Production Potential [ktoe]",
        })

        line = prod[["Year", demand_col]].rename(columns={demand_col: "Value"})
        line["Component"] = "Biofuel Demand [ktoe]"

        # use override if provided
        if line_override is not None:
            line = line_override

        from views.charts import render_grouped_bar_and_line
        render_grouped_bar_and_line(
            prod_df=bars,
            demand_df=line,
            x_col="Year",
            y_col="Value",
            category_col="Component",
            title=f"Biofuels Demand vs Potential Supply ({scen_key})",
            height=theme.CHART_HEIGHT,
            y_label="ktoe",
            key=f"biofuels_demand_supply_{scen_key.lower()}",
        )

    # --- Potential for Export ---
    with col2:
        exp_min_col = f"MinExport_{scen_key}_ktoe"
        exp_max_col = f"MaxExport_{scen_key}_ktoe"

        exp_df = exports.melt(
            id_vars=["Year"],
            value_vars=[exp_min_col, exp_max_col],
            var_name="Component",
            value_name="Value"
        )
        exp_df["Component"] = exp_df["Component"].replace({
            exp_min_col: "Min export potential [ktoe]",
            exp_max_col: "Max export potential [ktoe]",
        })

        fig2 = px.bar(
            exp_df,
            x="Year",
            y="Value",
            color="Component",
            color_discrete_map={
                "Min export potential [ktoe]": "#86efac",
                "Max export potential [ktoe]": "#22c55e",
            },
            title=f"Potential for Biofuels Export ({scen_key})",
            barmode="group",
            width=theme.CHART_WIDTH,
            height=theme.CHART_HEIGHT,
        )
        st.plotly_chart(fig2, use_container_width=False, key=f"biofuels_export_{scen_key.lower()}")
