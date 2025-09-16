# views/charts.py
from __future__ import annotations
from typing import Dict, List, Optional, Union
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import theme
import pandas as pd

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
# -----------------------------
# Generic Charts
# -----------------------------
def render_bar_chart(df, x_col, y_col, category_col, title,
                     x_order=None, colors=None, y_label=None, key=None):
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=category_col,
        category_orders={x_col: x_order} if x_order else None,
        color_discrete_map=colors or {}
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_label if y_label else y_col
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

    # Overlay line
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
# Sankey (no y-axis)
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
# -----------------------------
# Ships Charts (refactored)
# -----------------------------
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

    pref = ["C", "T", "B", "G", "O"]
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

    cols = [c for c in df_base.columns if c.lower().startswith("new_ships_")]
    if not cols:
        cols = [c for c in df_base.columns if c.lower().startswith("new_")]

    if "Year" not in df_base.columns or not cols:
        return px.bar(title="New Ships — expected 'Year' and 'New_Ships_*' columns")

    d_long = (
        df_base[["Year"] + cols]
        .melt(id_vars="Year", var_name="col", value_name="value")
    )
    def _extract_type(col: str) -> str:
        cl = col.lower()
        if cl.startswith("new_ships_"):
            return col.split("_", 2)[-1]
        if cl.startswith("new_"):
            return col.split("_", 1)[-1]
        return col
    d_long["type"] = d_long["col"].map(_extract_type)

    pref = ["C", "T", "B", "G", "O"]
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

    def _resolve(df: pd.DataFrame, names: list[str]) -> str | None:
        lower = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lower: return lower[n.lower()]
        return None

    em_col = _resolve(df_base, ["CO2_Emissions", "CO2 Emissions"])
    ex_col = _resolve(df_base, ["Excess_Emissions", "Excess Emissions"])
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

    penalty_col = None
    lower_map = {c.lower(): c for c in df_base.columns}
    for key in ["ets_penalty", "ets penalty", "eth_penalty"]:
        if key in lower_map:
            penalty_col = lower_map[key]; break

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

    if df_base.empty or "Year" not in df_base.columns:
        return px.line(title="ETS Penalty — no data")
    penalty_col = _resolve_ci(df_base, ["ETS_Penalty", "ets_penalty", "ETS penalty", "eth_penalty"])
    if not penalty_col:
        return px.line(title="ETS Penalty — 'ETS_Penalty' column not found")
    d = df_base[["Year", penalty_col]].rename(columns={penalty_col: "ETS_Penalty"}).copy()
    d["ETS_Penalty"] = pd.to_numeric(d["ETS_Penalty"], errors="coerce")
    fig = px.line(d, x="Year", y="ETS_Penalty", title="ETS Penalty [M€]")
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True, key=key or "ships_ets_penalty")
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
        fig = go.Figure(); fig.update_layout(title=f"{title} — no data"); return fig
    ycol = _resolve_ci(df, [year_col, "Year"])
    ac   = _resolve_ci(df, avg_col_candidates)
    mic  = _resolve_ci(df, min_col_candidates)
    mac  = _resolve_ci(df, max_col_candidates)
    if not (ycol and ac and mic and mac):
        fig = go.Figure(); fig.update_layout(title=f"{title} — missing required cols"); return fig
    d = df[[ycol, ac, mic, mac]].rename(columns={ycol:"Year", ac:"Avg", mic:"Min", mac:"Max"}).copy()
    for c in ("Avg","Min","Max"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values("Year")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Min"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Max"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.20)", name="Range (Min–Max)"))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Avg"], mode="lines+markers", name="Average", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="", yaxis_title=y_label)
    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_water_band")
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
        fig = go.Figure(); fig.update_layout(title=f"{title} — no data"); return fig
    mcol = _resolve_ci(df, month_col_candidates)
    ac   = _resolve_ci(df, avg_col_candidates)
    mic  = _resolve_ci(df, min_col_candidates)
    mac  = _resolve_ci(df, max_col_candidates)
    if not (mcol and ac and mic and mac):
        fig = go.Figure(); fig.update_layout(title=f"{title} — missing required cols"); return fig
    d = df[[mcol, ac, mic, mac]].rename(columns={mcol:"Month", ac:"Avg", mic:"Min", mac:"Max"}).copy()
    for c in ("Avg","Min","Max"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    if d["Month"].dtype == object:
        d["Month"] = d["Month"].astype(str).str.strip().str.upper()
        d["Month"] = pd.Categorical(d["Month"], categories=month_order, ordered=True)
    d = d.sort_values("Month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Min"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Max"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.20)", name="Range (Min–Max)"))
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Avg"], mode="lines+markers", name="Average", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Months", yaxis_title=y_label)
    st.plotly_chart(fig, use_container_width=True, key=key or f"{title}_water_monthly")
    return fig
