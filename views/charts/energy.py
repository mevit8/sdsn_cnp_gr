from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path
from config import theme
from models.data_loader import load_and_prepare_excel, aggregate_to_periods
from .generic import render_bar_chart

@st.cache_data(show_spinner=False)
def load_energy_interactive_data(scenario_code: str):
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
    st.subheader("⚡ Interactive Energy–Emissions Explorer")

    with st.expander("ℹ️ About the Energy–Emissions Explorer"):
        st.markdown("""
**LEAP** integrates all energy demand and supply sectors.
Use the dropdowns to explore how **population/GDP** and **renewables uptake**
affect Greece's energy and emissions pathways to 2050.
""")

    col1, col2 = st.columns(2)

    with col1:
        ssp = st.selectbox(
            "Population & GDP projections to 2050:",
            ["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – SSP1 (NCNC)",
                "B": "Option B – SSP2 (BAU)",
                "C": "Option C – SSP5",
            }[x],
            help=(
                "**Option A – SSP1 (NCNC):** ~14% population decline by 2050 vs 2021; GDP +2.0–2.5%/yr.\n\n"
                "**Option B – SSP2 (BAU):** Stronger decline by 2100 (~24%); GDP +1.9–2.2%/yr.\n\n"
                "**Option C – SSP5:** High tech progress; population down up to 40% by 2100; GDP +2.3–2.5%/yr."
            ),
            key="ssp_select",
        )

    with col2:
        renew = st.selectbox(
            "Renewables uptake to 2050:",
            ["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – Conservative",
                "B": "Option B – Central (NECP-aligned)",
                "C": "Option C – Optimistic",
            }[x],
            help=(
                "**A:** Slower uptake; grid/permitting constraints. Solar +0.4 %/yr, wind +0.2 %/yr.\n\n"
                "**B:** Matches recent PV rollout; hydropower steady. Solar +0.6 %/yr, wind +0.4 %/yr.\n\n"
                "**C:** Faster rollout with fewer constraints. Solar +0.8 %/yr, wind +0.6 %/yr."
            ),
            key="renew_select",
        )

    st.caption(
        "NECP baselines: Residential electrification (+15% by 2050), "
        "Transport electrification (~10%), decreasing oil refining."
    )

    render_energy_interactive_charts(tab_name, ssp, renew)

    st.markdown("---")
    render_energy_sensitivity_summary()

def render_energy_interactive_charts(tab_name: str, ssp: str, renew: str):
    scenario_code = f"{ssp}{renew}"
    df_energy, df_demand_emissions, df_energy_supply, df_supply_emissions = load_energy_interactive_data(scenario_code)

    if all(df.empty for df in [df_energy, df_demand_emissions, df_energy_supply, df_supply_emissions]):
        st.warning(f"No data found for scenario {scenario_code}.")
        return

    st.markdown(f"### Results for Option {ssp}–{renew}")
    st.caption("Charts update automatically based on your selections.")

    cols_sectors = [
        "Residential", "Agriculture", "Industry", "Energy Products",
        "Passenger Transportation", "Freight Transportation", "Maritime", "Services",
    ]
    c1, c2 = st.columns(2)
    with c1:
        if not df_energy.empty:
            melted = df_energy.melt(id_vars=["Year"], value_vars=cols_sectors, var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(grouped, "PeriodStr", "Value", "Component",
                             "Total energy consumption per sector", order, y_label="ktoe",
                             key=f"int_energy_cons_{scenario_code}")
    with c2:
        if not df_demand_emissions.empty:
            melted = df_demand_emissions.melt(id_vars=["Year"], value_vars=cols_sectors, var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(grouped, "PeriodStr", "Value", "Component",
                             "Emissions from energy consumption by sector", order, y_label="MtCO₂e",
                             key=f"int_energy_emis_{scenario_code}")

    cols_fuels = ["Hydrogen Generation", "Electricity Generation", "Heat Generation", "Oil Refining"]
    c3, c4 = st.columns(2)
    with c3:
        if not df_energy_supply.empty:
            melted = df_energy_supply.melt(id_vars=["Year"], value_vars=cols_fuels, var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(grouped, "PeriodStr", "Value", "Component",
                             "Generated energy per fuel type", order, colors=theme.FUEL_COLORS,
                             y_label="ktoe", key=f"int_energy_fuel_{scenario_code}")
    with c4:
        if not df_supply_emissions.empty:
            melted = df_supply_emissions.melt(id_vars=["Year"], value_vars=["Electricity Generation","Heat Generation","Oil Refining"],
                                              var_name="Component", value_name="Value")
            grouped, order = aggregate_to_periods(melted, "Year", "Value", "Component", 4, "mean", "range")
            render_bar_chart(grouped, "PeriodStr", "Value", "Component",
                             "Emissions per fuel type", order, colors=theme.FUEL_COLORS,
                             y_label="MtCO₂e", key=f"int_energy_fuel_emis_{scenario_code}")

@st.cache_data(show_spinner=False)
def _load_energy_sensitivity_sheets() -> dict[str, pd.DataFrame]:
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
        except Exception:
            out[key] = pd.DataFrame()
    return out

def render_energy_sensitivity_summary():
    st.subheader("📈 Sensitivity Summary")
    img = Path("content/leap_sensitivity.png")
    if img.exists():
        with st.expander("Show sensitivity summary", expanded=False):
            st.image(str(img), width=600)
            st.caption("Relative contribution of SSP and renewables uptake assumptions.")
    else:
        st.info("Sensitivity summary image not found.")
