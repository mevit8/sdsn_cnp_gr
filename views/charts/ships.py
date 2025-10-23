from __future__ import annotations
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
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


# ---------------------------------------------------------------------
# Interactive mode for Shipping tab
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_maritime_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Load a specific sheet from the Maritime Excel file."""
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        for c in df.columns:
            if c != "Year":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Could not load sheet '{sheet}': {e}")
        return pd.DataFrame()


def render_ships_interactive_controls(section_title: str):
    """
    Render interactive controls and charts for the Shipping (Maritime) tab.
    Allows users to select from 20 different scenarios and see all 8 charts update dynamically.
    """
    st.subheader(f"{section_title} – Interactive Scenario Builder")

    explainer_path = Path("content/ships_explainer_INTERACTIVE.md")
    if explainer_path.exists():
        with st.expander("ℹ️ About the Maritime Interactive Explorer", expanded=False):
            st.markdown(explainer_path.read_text(encoding="utf-8"), unsafe_allow_html=True)
    
    # Define all 20 scenarios with descriptions
    SCENARIOS = {
        "base": "Base (Business-as-usual - BAU)",
        "bau_fuel_meoh": "BAU + Methanol shift",
        "bau_fuel_h2": "BAU + Hydrogen shift",
        "bau_ssp5": "BAU + High demand (SSP5)",
        "bau_ssp1": "BAU + Low demand (SSP1)",
        "tech_ccs": "Technology: Carbon Capture & Storage",
        "tech_hull": "Technology: Hull cleaning / drag reduction",
        "tech_eng_opt": "Technology: Engine optimization",
        "tech_port_call": "Technology: Port call optimization",
        "tech_route_opt": "Technology: Route optimization",
        "tech_propul": "Technology: Propulsion system upgrades",
        "techcomb": "Technology: Combined package",
        "fuel_cost_high": "High fuel cost trajectory",
        "fuel_cost_low": "Low fuel cost trajectory",
        "co2_cap_pess": "Strict CO₂ cap (pessimistic)",
        "co2_cap_opt": "Loose CO₂ cap (optimistic)",
        "ets_price_no": "No ETS penalty",
        "ets_price_strict": "High/strict ETS price",
        "fuel_cons_fast": "Fast fuel transition",
        "fuel_cons_slow": "Slow fuel transition",
    }
    
    # Dropdown control
    selected_sheet = st.selectbox(
        "Select scenario:",
        options=list(SCENARIOS.keys()),
        format_func=lambda x: SCENARIOS[x],
        index=0,  # default to "base"
        key="ships_scenario_selector"
    )
    
    # Load data from selected sheet
    data_path = "data/Maritime_results_all_scenarios.xlsx"
    df = _load_maritime_sheet(data_path, selected_sheet)
    
    if df.empty:
        st.error(f"❌ No data available for scenario: {SCENARIOS[selected_sheet]}")
        return
    
    # Render all 8 charts in 2×4 grid layout (same as NCNC mode)
    col1, col2 = st.columns(2)
    with col1:
        fig = render_ships_stock(df, y_label="Number of Stock Ships")
        st.plotly_chart(fig, use_container_width=False, key=f"ships_stock_{selected_sheet}")
    with col2:
        fig_new = render_ships_new(df, y_label="Number of New Ships")
        st.plotly_chart(fig_new, use_container_width=False, key=f"ships_new_{selected_sheet}")
    
    col3, col4 = st.columns(2)
    with col3:
        fig_inv = render_ships_investment_cost(df, y_label="Costs (M€)")
        st.plotly_chart(fig_inv, use_container_width=False, key=f"ships_investment_{selected_sheet}")
    with col4:
        fig_op = render_ships_operational_cost(df, y_label="Costs (M€)")
        st.plotly_chart(fig_op, use_container_width=False, key=f"ships_operational_{selected_sheet}")
    
    col5, col6 = st.columns(2)
    with col5:
        fig_fd = render_ships_fuel_demand(df, y_label="Fuel Demand [tonnes]")
        st.plotly_chart(fig_fd, use_container_width=False, key=f"ships_fuel_demand_{selected_sheet}")
    with col6:
        fig_fc = render_ships_fuel_cost(df, y_label="Costs (M€)")
        st.plotly_chart(fig_fc, use_container_width=False, key=f"ships_fuel_cost_{selected_sheet}")
    
    col7, col8 = st.columns(2)
    with col7:
        fig_emcap = render_ships_emissions_and_cap(df, cap_df=None, y_label="CO₂ Emissions (MtCO₂e)")
        st.plotly_chart(fig_emcap, use_container_width=False, key=f"ships_emissions_{selected_sheet}")
    with col8:
        fig_penalty = render_ships_ets_penalty(df, y_label="Costs (M€)")
        st.plotly_chart(fig_penalty, use_container_width=False, key=f"ships_ets_{selected_sheet}")