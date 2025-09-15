from __future__ import annotations
import streamlit as st
from config import theme
import pandas as pd
from models.data_loader import load_and_prepare_excel, prepare_stacked_data
from views.charts import (
    render_bar_chart,
    render_line_chart,
    render_grouped_bar_and_line,
    render_sankey,
    render_ships_stock,
    render_ships_new,
    render_ships_investment_cost,
    render_ships_operational_cost,
    render_ships_fuel_demand,
    render_ships_fuel_cost,
    render_ships_emissions_and_cap,
    render_ships_ets_penalty,
)

from PIL import Image
from pathlib import Path
import plotly.graph_objects as go


st.set_page_config(page_title="SDSN GCH Scenarios", layout="wide")

import pandas as pd

@st.cache_data(show_spinner=False)
def load_energy_balance(path: str = "data/LEAP_Energy_Balance.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    c0, c1 = df.columns[:2]
    df = df.rename(columns={c0: "Scenario", c1: "Flow"})
    for col in df.columns:
        if col not in ("Scenario", "Flow"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def aggregate_to_periods(
    df: pd.DataFrame,
    year_col: str = "Year",
    value_col: str = "Value",
    component_col: str = "Component",
    period_years: int = 4,
    agg: str = "mean",   # or "sum"
    label_mode: str = "range",  # "range" -> "2000–2003", "start" -> "2000"
):
    """
    Bin annual rows into N-year periods and aggregate by Component.
    Returns (df_period, period_order_str_list).
    """
    df = df.copy()
    # base = 0 (calendar aligned). If you want to align to a specific start, change the modulo base.
    start = (df[year_col].min() // period_years) * period_years
    # Compute period start
    df["PeriodStart"] = ((df[year_col] - start) // period_years) * period_years + start
    df["PeriodEnd"] = df["PeriodStart"] + (period_years - 1)

    if label_mode == "range":
        df["PeriodStr"] = df["PeriodStart"].astype(str) + "–" + df["PeriodEnd"].astype(str)
    else:
        df["PeriodStr"] = df["PeriodStart"].astype(str)

    # Aggregate within each period by component
    if agg == "sum":
        grouped = df.groupby(["PeriodStart", "PeriodStr", component_col], as_index=False)[value_col].sum()
    else:
        grouped = df.groupby(["PeriodStart", "PeriodStr", component_col], as_index=False)[value_col].mean()

    # Build category order for x-axis
    period_order = grouped.drop_duplicates(subset=["PeriodStart", "PeriodStr"]) \
                          .sort_values("PeriodStart")["PeriodStr"].tolist()

    return grouped, period_order


# Resolve paths relative to this file
BASE_DIR = Path(__file__).parent
CSS_PATH = BASE_DIR / "static" / "style.css"
LOGO_PATH = BASE_DIR / "static" / "logo.png"

def load_css(path: Path = CSS_PATH):
    """Inject CSS from a file once per run."""
    if path.exists():
        css = path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.sidebar.info(f"CSS not found at: {path}")

@st.cache_resource(show_spinner=False)
def load_logo(path: Path = LOGO_PATH):
    """Open the logo image once and reuse the PIL object."""
    return Image.open(path)

# Call once at startup
load_css()

# Sidebar logo (cached) with a safe fallback
if LOGO_PATH.exists():
    try:
        st.sidebar.image(load_logo(), width=160)
    except Exception:
        st.sidebar.image(str(LOGO_PATH), width=160)
else:
    st.sidebar.info(f"Logo not found at: {LOGO_PATH}")

@st.cache_data(show_spinner=False)
def load_biofuels_simple(path: str = "data/LEAP_Biofuels.xlsx", sheet: str = "Biofuels") -> pd.DataFrame:
    import pandas as pd
    df = pd.read_excel(path, sheet_name=sheet)

    # Normalize column names (strip spaces)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    required = ["Year", "MinProd_ktoe", "MaxProd_ktoe", "Demand_BAU_ktoe", "Demand_NCNC_ktoe"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in '{sheet}': {missing}")

    # Numbers
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    for c in df.columns:
        if c != "Year":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@st.cache_data(show_spinner=False)
def load_maritime_base(path: str = "data/Maritime_results_all_scenarios.xlsx", sheet: str = "base") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    for c in df.columns:
        if c != "Year":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_cap_series(path: str) -> pd.DataFrame:
    import pandas as pd
    # Try common separators
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep)
            break
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        return df

    # Normalize and detect columns
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    year_col = lower.get("year")
    cap_col = lower.get("co2_cap") or lower.get("cap")
    if not year_col or not cap_col:
        return pd.DataFrame()

    out = df[[year_col, cap_col]].rename(columns={year_col: "Year", cap_col: "CO2_Cap"}).copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["CO2_Cap"] = pd.to_numeric(out["CO2_Cap"], errors="coerce")
    out = out.dropna(subset=["Year", "CO2_Cap"]).sort_values("Year")
    return out


# Load data
df_costs = load_and_prepare_excel("data/Fable_46_Agricultural.xlsx")
df_emissions = load_and_prepare_excel("data/Fable_46_GHG.xlsx")
df_land = load_and_prepare_excel("data/Fable_46_Land.xlsx")
df_energy = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
df_energy_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
df_supply_emissions = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")
df_energy_balance = load_energy_balance("data/LEAP_Energy_Balance.xlsx")
df_biofuels = load_biofuels_simple("data/LEAP_Biofuels.xlsx")



# Shared scenario selector
scenarios = sorted(set(df_costs["Scenario"]).intersection(
    df_emissions["Scenario"], df_land["Scenario"], df_energy["Scenario"]
))
selected_scenario = st.sidebar.selectbox("🎯 Select Scenario", scenarios)

# Define which rows act as Sources / Converters / Sinks (as seen in your static Sankey)
SOURCES = [
    "Coal Lignite Production", "Coal Lignite Imports",
    "Wind Production", "Hydro Production", "Solar Production", "Geothermal Production",
    "Crude Oil Imports", "Refinery Feedstocks Imports", "Petroleum Coke Imports",
    "Natural Gas Imports", "Hydrogen Imports", "Biogas Imports", "CNG Imports",
    "Coal Unspecified Imports", "Biomass Production", "Biomass Imports", "Biomass Supply",
]
CONVERTERS = [
    "Electricity Generation", "Heat Generation", "Oil Refining",
    "Synthetic Fuels Module", "Transmission and Distribution",
]
SINKS = [
    "Residential", "Industry", "Agriculture", "Service Tertiary Sector",
    "Passenger Transportation", "Freight Transportation", "Maritime",
    "Energy Product Industry", "Hydrogen Generation",
    "Losses", "Exports", "Waste",
]

# app.py (helper)
def build_sankey_from_balance(df: pd.DataFrame, scenario: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Build links in three steps:
      1) Sources (rows) -> Carriers (columns)     [positive cells]
      2) Carrier <-> Converter rows               [negative -> into converter; positive -> out of converter]
      3) Carriers (columns) -> Sinks (rows)       [positive cells]
    If `scenario` is provided and a 'Scenario' column exists, the table is filtered to that scenario first.
    """
    df = df.copy()
    if "Flow" not in df.columns:
        raise ValueError("Energy balance must have a 'Flow' column (first column).")

    # Optional scenario filter (if present)
    if scenario and "Scenario" in df.columns:
        df = df[df["Scenario"] == scenario].copy()
        # if multiple rows per Flow, aggregate them
        df = df.groupby("Flow", as_index=False).sum(numeric_only=True)
    carriers = [c for c in df.columns if c not in ("Flow", "Total", "Scenario")]

    links = []

    # 1) Sources -> Carriers
    src_df = df[df["Flow"].isin(SOURCES)]
    for _, row in src_df.iterrows():
        src = row["Flow"]
        for fuel in carriers:
            val = row.get(fuel, 0.0)
            if pd.notna(val) and float(val) > 0:
                links.append({"source": src, "target": fuel, "value": float(val)})

    # 2) Converters <-> Carriers
    conv_df = df[df["Flow"].isin(CONVERTERS)]
    for _, row in conv_df.iterrows():
        conv = row["Flow"]
        for fuel in carriers:
            v = float(row.get(fuel, 0.0) or 0.0)
            if v < 0:
                links.append({"source": fuel, "target": conv, "value": abs(v)})
            elif v > 0:
                links.append({"source": conv, "target": fuel, "value": v})

    # 3) Carriers -> Sinks
    sink_df = df[df["Flow"].isin(SINKS)]
    for _, row in sink_df.iterrows():
        sink = row["Flow"]
        for fuel in carriers:
            val = row.get(fuel, 0.0)
            if pd.notna(val) and float(val) > 0:
                links.append({"source": fuel, "target": sink, "value": float(val)})

    links_df = pd.DataFrame(links)

    present = lambda items: [i for i in items if i in set(df["Flow"]).union(carriers)]
    node_order = (
        present(SOURCES) +
        present(CONVERTERS) +
        [f for f in carriers if f in links_df["source"].tolist() or f in links_df["target"].tolist()] +
        present(SINKS)
    )
    return links_df, node_order


# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(theme.TAB_TITLES)

with tab1:
    cols = ["FertilizerCost", "LabourCost", "MachineryRunningCost", "DieselCost", "PesticideCost"]
    melted, years = prepare_stacked_data(df_costs, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Production-based agricultural emissions", [str(y) for y in years])

with tab2:
    cols = ["CropCO2e", "LiveCO2e", "LandCO2", "FAOTotalCO2e"]
    melted, years = prepare_stacked_data(df_emissions, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Agricultural production cost", [str(y) for y in years])

with tab3:
    cols = ["FAOCropland", "FAOHarvArea", "FAOPasture", "FAOUrban", "FAOForest", "FAOOtherLand"]
    df_filtered = df_land[df_land["Scenario"] == selected_scenario].copy()
    df_filtered[cols] = df_filtered[cols].fillna(0)
    melted = df_filtered.melt(id_vars=["Year"], value_vars=cols, var_name="Component", value_name="Value")
    render_line_chart(melted, "Year", "Value", "Component", "Land uses evolution")

with tab4:
    cols = ["Residential", "Agriculture", "Industry", "Energy Products",
            "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
    melted, years = prepare_stacked_data(df_energy, selected_scenario, "Year", cols)
    
    period_df, period_order = aggregate_to_periods(
        melted,
        year_col="Year",
        value_col="Value",
        component_col="Component",
        period_years=4,
        agg="mean",         # or "sum"
        label_mode="range"  # "range" -> "2000–2003"; "start" -> "2000"
    )
    render_bar_chart(
        period_df, "PeriodStr", "Value", "Component",
        "Total energy consumption per sector",
        period_order,
        tick_every_years=None  # not needed; x is now periods
    )


with tab5:
    cols = ["Residential", "Agriculture", "Industry", "Energy Products",
            "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
    melted, years = prepare_stacked_data(df_energy, selected_scenario, "Year", cols)
    period_df, period_order = aggregate_to_periods(
        melted, year_col="Year", value_col="Value", component_col="Component",
        period_years=4, agg="mean", label_mode="range"
    )

    render_bar_chart(
        period_df, "PeriodStr", "Value", "Component",
        "Emissions energy consumption by sector",
        period_order
    )

with tab6:
    cols = ["Hydrogen Generation", "Electricity Generation", "Heat Generation", "Oil Refining"]
    melted, years = prepare_stacked_data(df_energy_supply, selected_scenario, "Year", cols)
    
    period_df, period_order = aggregate_to_periods(
        melted, year_col="Year", value_col="Value", component_col="Component",
        period_years=4, agg="mean", label_mode="range"
    )

    render_bar_chart(
        period_df, "PeriodStr", "Value", "Component",
        "Generated energy per fuel type",
        period_order,
        colors=theme.FUEL_COLORS
    )

with tab7:
    cols = ["Electricity Generation", "Heat Generation", "Oil Refining"]
    melted, years = prepare_stacked_data(df_supply_emissions, selected_scenario, "Year", cols)
    period_df, period_order = aggregate_to_periods(
        melted, year_col="Year", value_col="Value", component_col="Component",
        period_years=4, agg="mean", label_mode="range"
    )

    render_bar_chart(
        period_df, "PeriodStr", "Value", "Component",
        "Generated energy per fuel type",
        period_order,
        colors=theme.FUEL_COLORS
    )

SANKEY_LABEL_MAP = {
    "Service Tertiary Sector": "Service<br>Tertiary",
    "Passenger Transportation": "Passenger<br>Transport",
    "Freight Transportation": "Freight<br>Transport",
    "Energy Product Industry": "Energy Product<br>Industry",
    "Coal Lignite Production": "Coal/Lignite<br>Production",
    "Coal Lignite Imports": "Coal/Lignite<br>Imports",
    "Coal Unspecified Imports": "Coal (unspec.)<br>Imports",
    "Refinery Feedstocks Imports": "Refinery Feedstocks<br>Imports",
    "Synthetic Fuels Module": "Synthetic<br>Fuels Module",
    "Transmission and Distribution": "Transmission &<br>Distribution",
}

with tab8:
    SANKEY_NODE_COLORS = {
    # fuels/carriers
    "Electricity": "#2563eb",
    "Natural Gas": "#0ea5e9",
    "Heat": "#ea580c",
    "Crude Oil": "#f59e0b",
    "Biomass": "#16a34a",
    "Solar": "#fbbf24",
    "Hydrogen": "#a855f7",
    "Ethanol": "#22c55e",
    "Synthetic Fuels": "#06b6d4",
    # converters
    "Electricity Generation": "#93c5fd",
    "Heat Generation": "#fdba74",
    "Oil Refining": "#fcd34d",
    "Synthetic Fuels Module": "#99f6e4",
    "Transmission and Distribution": "#cbd5e1",
    # sinks
    "Residential": "#111827",
    "Industry": "#374151",
    "Agriculture": "#4b5563",
    "Service Tertiary Sector": "#6b7280",
    "Passenger Transportation": "#9ca3af",
    "Freight Transportation": "#9ca3af",
    "Maritime": "#9ca3af",
    "Energy Product Industry": "#6b7280",
    "Hydrogen Generation": "#a78bfa",
    "Losses": "#d1d5db",
    "Exports": "#d1d5db",
    "Waste": "#d1d5db",
}

    st.subheader("⚡ Energy Generation ↔ Consumption (Balance)")
    try:
        links_df, node_order = build_sankey_from_balance(df_energy_balance, scenario=selected_scenario)
        if links_df.empty:
            st.info(f"No links could be derived for scenario: {selected_scenario}.")
        else:
            render_sankey(
                links_df,
                title=f"{selected_scenario} – Energy Generation → Consumption",
                node_order=node_order,
                full_width=True,
                height=720,
                label_wrap=14,
                node_colors=SANKEY_NODE_COLORS,  # optional
            )
    except Exception as e:
        st.error(f"Failed to build Sankey: {e}")
with tab9:
    st.subheader("🌿 Biofuels")

    scen = (selected_scenario or "").strip().upper()
    scen_key = "BAU" if scen == "BAU" else "NCNC"  # default to NCNC for anything else

    # -----------------------------
    # a) Demand vs Potential Supply
    # -----------------------------
    st.markdown("**a) Biofuels demand and potential supply [ktoe]**")

    # Bars: Min/Max production potential
    bar_long = (
        df_biofuels
        .melt(id_vars=["Year"], value_vars=["MinProd_ktoe", "MaxProd_ktoe"],
              var_name="Component", value_name="Value")
        .replace({"Component": {
            "MinProd_ktoe": "Minimum Production Potential [ktoe]",
            "MaxProd_ktoe": "Maximum Production Potential [ktoe]",
        }})
    )

    # Line: chosen demand series
    demand_col = "Demand_BAU_ktoe" if scen_key == "BAU" else "Demand_NCNC_ktoe"
    line_long = (
        df_biofuels[["Year", demand_col]]
        .rename(columns={demand_col: "Value"})
        .assign(Component=f"Demand ({'Baseline' if scen_key=='BAU' else 'NECP'}) [ktoe]")
    )

    render_grouped_bar_and_line(
        prod_df=bar_long,
        demand_df=line_long,
        x_col="Year",
        y_col="Value",
        category_col="Component",
        title=f"Biofuels demand vs potential supply ({scen_key})",
    )

    # -----------------------------
    # b) Potential for Biofuels Export
    # -----------------------------
    st.markdown("**b) Potential for Biofuels Export [ktoe]**")

    # Prefer explicit export cols; otherwise compute from potential − demand
    if scen_key == "BAU":
        col_min_exp, col_max_exp = "ExportMin_BAU_ktoe", "ExportMax_BAU_ktoe"
    else:
        col_min_exp, col_max_exp = "ExportMin_NCNC_ktoe", "ExportMax_NCNC_ktoe"

    have_explicit = (col_min_exp in df_biofuels.columns) and (col_max_exp in df_biofuels.columns)
    if have_explicit and (df_biofuels[[col_min_exp, col_max_exp]].notna().any().any()):
        min_export = df_biofuels[col_min_exp].fillna(0)
        max_export = df_biofuels[col_max_exp].fillna(0)
    else:
        dem = df_biofuels[demand_col].fillna(0)
        min_export = (df_biofuels["MinProd_ktoe"].fillna(0) - dem).clip(lower=0)
        max_export = (df_biofuels["MaxProd_ktoe"].fillna(0) - dem).clip(lower=0)

    export_long = (
        pd.DataFrame({
            "Year": df_biofuels["Year"].astype(int),
            "Min export potential [ktoe]": min_export,
            "Max export potential [ktoe]": max_export,
        })
        .melt(id_vars=["Year"], var_name="Component", value_name="Value")
    )

    # Use go.Figure for grouped bars (your render_bar_chart defaults to relative)
    import plotly.graph_objects as go
    fig = go.Figure()
    for comp, color in [
        ("Min export potential [ktoe]", "#86efac"),
        ("Max export potential [ktoe]", "#22c55e"),
    ]:
        sub = export_long[export_long["Component"] == comp]
        fig.add_trace(go.Bar(x=sub["Year"], y=sub["Value"], name=comp, marker_color=color))

    fig.update_layout(
        title=f"Export potential ({scen_key})",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="ktoe",
        height=getattr(theme, "CHART_HEIGHT", 500),
        width=getattr(theme, "CHART_WIDTH", 800),
        margin=dict(t=60, r=10, b=10, l=10),
        legend_title_text="Series",
    )
    st.plotly_chart(fig, use_container_width=False)
    
with tab10:
    st.subheader("🚢 Ships")

    try:
        base_df = load_maritime_base()
    except Exception as e:
        st.warning(f"Could not load maritime data: {e}")
    else:
        scen = (selected_scenario or "").strip().upper()

        # Special rule: BAU shows ONE number (KPI), not the 8 charts
        if scen == "BAU":
            # BAU shows a single KPI (not charts)
            st.metric(label="BAU – Total Emissions", value="99.68 MtCO₂e")
            st.caption("Currently, the Greek fleet is estimated to emit 99.68MtCO2e, which is well above the European regulatory threshold of 97.9MtCO2e.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = render_ships_stock(base_df)
                st.plotly_chart(fig, use_container_width=False)
            with col2:
                fig_new = render_ships_new(base_df)
                st.plotly_chart(fig_new, use_container_width=False)
            col3, col4 = st.columns(2)
            with col3:
                fig_inv = render_ships_investment_cost(base_df)
                st.plotly_chart(fig_inv, use_container_width=False)
            with col4:
                fig_op = render_ships_operational_cost(base_df)
                st.plotly_chart(fig_op, use_container_width=False)
            col5, col6 = st.columns(2)
            with col5:
                fig_fd = render_ships_fuel_demand(base_df)
                st.plotly_chart(fig_fd, use_container_width=False)
            with col6:
                fig_fc = render_ships_fuel_cost(base_df)
                st.plotly_chart(fig_fc, use_container_width=False)
            col7, col8 = st.columns(2)
            with col7:
                # Load preferred cap file (default to 'real' if available)
                cap_candidates = [
                    Path("data/co2_cap_real.csv"),
                    Path("data/co2_cap_opt.csv"),
                    Path("data/co2_cap_pess.csv"),
                    Path("data/co2_cap_no.csv"),
                ]
                available_caps = [p for p in cap_candidates if p.exists()]
                cap_df_to_use = None

                if available_caps:
                    cap_choice = st.sidebar.selectbox(
                        "CO₂ Cap series (CSV)",
                        [p.name for p in available_caps],
                        index=0,
                    )
                    chosen = next(p for p in available_caps if p.name == cap_choice)
                    tmp_cap = load_cap_series(str(chosen))
                    if tmp_cap.empty:
                        st.info(f"{chosen.name} is missing 'Year' and 'CO2_Cap' (or 'Cap'). Using reconstructed cap.")
                    else:
                        cap_df_to_use = tmp_cap
                else:
                    st.info("No CO₂ Cap file found → reconstructing from Emissions − Excess.")

                fig_emcap = render_ships_emissions_and_cap(base_df, cap_df=cap_df_to_use)
                st.plotly_chart(fig_emcap, use_container_width=False)

            with col8:
                fig_penalty = render_ships_ets_penalty(base_df)
                st.plotly_chart(fig_penalty, use_container_width=False)






