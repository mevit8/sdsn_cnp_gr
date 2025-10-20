from __future__ import annotations
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import theme
from .helpers import _resolve_ci

SHIP_TYPE_LABELS = {"C": "C: Container", "T": "T: Tanker", "B": "B: Bulk", "G": "G: Cargo", "O": "O: Other"}

def _long_by_prefix(df: pd.DataFrame, prefix: str, new_col: str):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return (df[["Year"] + cols]
            .melt(id_vars="Year", var_name="col", value_name="value")
            .assign(**{new_col: lambda x: x["col"].str.replace(f"^{prefix}", "", regex=True)}))

def render_ships_stock(df_base: pd.DataFrame, y_label: str = "Number of Stock Ships"):
    if df_base.empty:
        return px.bar(title="Stock Ships — no data")
    d = _long_by_prefix(df_base, "Stock_Ships_", "type")
    d["type"] = d["type"].replace(SHIP_TYPE_LABELS)
    pref = ["C: Container", "T: Tanker", "B: Bulk", "G: Cargo", "O: Other"]
    d["type"] = pd.Categorical(d["type"], categories=[t for t in pref if t in d["type"].unique()], ordered=True)
    fig = px.bar(d, x="Year", y="value", color="type", title="Stock Ships", barmode="stack")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT,
                      margin=dict(l=10,r=10,t=60,b=10), xaxis_title="", yaxis_title=y_label, legend_title=None)
    return fig

def render_ships_new(df_base: pd.DataFrame, y_label: str = "Number of New Ships"):
    if df_base.empty:
        return px.bar(title="New Ships — no data")
    d = _long_by_prefix(df_base, "New_Ships_", "type")
    d["type"] = d["type"].replace(SHIP_TYPE_LABELS)
    pref = ["C: Container", "T: Tanker", "B: Bulk", "G: Cargo", "O: Other"]
    d["type"] = pd.Categorical(d["type"], categories=[t for t in pref if t in d["type"].unique()], ordered=True)
    fig = px.bar(d, x="Year", y="value", color="type", title="New Ships", barmode="stack")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT,
                      margin=dict(l=10,r=10,t=60,b=10), xaxis_title="", yaxis_title=y_label, legend_title=None)
    return fig

def render_ships_investment_cost(df_base: pd.DataFrame, y_label="Costs (M€)"):
    if df_base.empty or "Investment_Cost" not in df_base:
        return px.line(title="Investment Costs — data missing")
    fig = px.line(df_base, x="Year", y="Investment_Cost", title="Investment Costs")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT, margin=dict(l=10,r=10,t=60,b=10),
                      xaxis_title="", yaxis_title=y_label)
    return fig

def render_ships_operational_cost(df_base: pd.DataFrame, y_label="Costs (M€)"):
    if df_base.empty or "Operational_Cost" not in df_base:
        return px.line(title="Operational Costs — data missing")
    fig = px.line(df_base, x="Year", y="Operational_Cost", title="Operational Costs")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT, margin=dict(l=10,r=10,t=60,b=10),
                      xaxis_title="", yaxis_title=y_label)
    return fig

def render_ships_fuel_demand(df_base: pd.DataFrame, y_label="Fuel Demand [tonnes]"):
    if df_base.empty:
        return px.bar(title="Fuel Demand — no data")
    cols = [c for c in df_base.columns if c.startswith("Fuel_Demand_")]
    if not cols:
        return px.bar(title="Fuel Demand — no Fuel_Demand_* columns found")
    d = (df_base[["Year"] + cols]
         .melt(id_vars="Year", var_name="col", value_name="value")
         .assign(fuel=lambda x: x["col"].str.replace("^Fuel_Demand_", "", regex=True)))
    fig = px.bar(d, x="Year", y="value", color="fuel", title="Fuel Demand", barmode="stack")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT, margin=dict(l=10,r=10,t=60,b=10),
                      xaxis_title="", yaxis_title=y_label, legend_title=None)
    return fig

def render_ships_fuel_cost(df_base: pd.DataFrame, y_label="Costs (M€)"):
    if df_base.empty or "Fuel_Cost" not in df_base:
        return px.line(title="Fuel Costs — data missing")
    fig = px.line(df_base, x="Year", y="Fuel_Cost", title="Fuel Costs")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT,
                      margin=dict(l=10,r=10,t=60,b=10), xaxis_title="", yaxis_title=y_label)
    return fig

def render_ships_emissions_and_cap(df_base: pd.DataFrame,
                                   cap_df: pd.DataFrame | None = None,
                                   cap_year_col="Year",
                                   cap_value_col="CO2_Cap",
                                   scale_emissions=1e-6,
                                   scale_cap=1e-6,
                                   scale_excess=1e-6,
                                   y_label="CO₂ Emissions (MtCO₂e)"):
    if df_base.empty or "Year" not in df_base:
        return go.Figure().update_layout(title="CO₂ Emissions & Cap — no data")
    em_col = _resolve_ci(df_base, ["CO2_Emissions", "CO2 Emissions"])
    ex_col = _resolve_ci(df_base, ["Excess_Emissions", "Excess Emissions"])
    if not em_col:
        return go.Figure().update_layout(title="CO₂ Emissions & Cap — missing emissions col")

    d = df_base[["Year", em_col] + ([ex_col] if ex_col else [])].rename(columns={em_col: "Emissions"})
    if ex_col: d = d.rename(columns={ex_col: "Excess"})
    if cap_df is not None and not cap_df.empty:
        cap = cap_df.rename(columns={cap_year_col: "Year", cap_value_col: "Cap"})[["Year","Cap"]]
        d = d.merge(cap, on="Year", how="left")
    else:
        d["Cap"] = d["Emissions"] - d.get("Excess", 0)

    for c in ("Emissions","Cap","Excess"):
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Emissions"] *= scale_emissions; d["Cap"] *= scale_cap
    if "Excess" in d: d["Excess"] *= scale_excess

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Cap"], name="CO₂ Cap", mode="lines", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Emissions"], name="CO₂ Emissions", mode="lines+markers", line=dict(width=2)))
    if "Excess" in d and d["Excess"].fillna(0).abs().sum() > 0:
        cap_series = d["Cap"]
        fig.add_trace(go.Scatter(x=d["Year"], y=cap_series.where(d["Excess"] > 0), mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=d["Year"], y=d["Emissions"].where(d["Excess"] > 0), name="Excess Emissions",
                                 mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(220,38,38,0.3)"))
    fig.update_layout(title="CO₂ Emissions and Cap", xaxis_title="", yaxis_title=y_label,
                      width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT, margin=dict(l=10,r=10,t=60,b=10))
    return fig

def render_ships_ets_penalty(df_base: pd.DataFrame, y_label="Costs (M€)"):
    if df_base.empty:
        return px.line(title="ETS Penalty — no data")
    col = _resolve_ci(df_base, ["ETS_Penalty", "ETS penalty"])
    if not col:
        return px.line(title="ETS Penalty — column not found")
    fig = px.line(df_base, x="Year", y=col, title="ETS Penalty")
    fig.update_traces(mode="lines+markers")
    fig.update_layout(width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT,
                      margin=dict(l=10,r=10,t=60,b=10), xaxis_title="", yaxis_title=y_label)
    return fig
