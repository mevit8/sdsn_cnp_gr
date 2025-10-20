# app.py
# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

# --- Config & Theme
from config import theme
from pathlib import Path
BASE_DIR = Path(__file__).parent


# --- Data Models & Utilities
from models.data_loader import (
    load_and_prepare_excel,
    prepare_stacked_data,
    aggregate_to_periods,
)

# --- Visualization Modules (refactored)
from views.charts import (
    # Generic plotting
    render_bar_chart,
    render_line_chart,
    render_grouped_bar_and_line,
    render_sankey,

    # Ships (Maritime)
    render_ships_stock,
    render_ships_new,
    render_ships_investment_cost,
    render_ships_operational_cost,
    render_ships_fuel_demand,
    render_ships_fuel_cost,
    render_ships_emissions_and_cap,
    render_ships_ets_penalty,
    render_ships_interactive_controls, 

    # Water & FABLE (Food–Land)
    render_water_band,
    render_water_monthly_band,
    render_fable_interactive_controls,
    load_water_requirements,

    # Energy
    render_energy_interactive_controls,

    # Biofuels
    render_biofuels_interactive_controls,
    load_biofuels_data,
)


# ---------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="SDSN GCH - GR Climate Neutrality", layout="wide")

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_scenario_md(base_name: str, scenario: str) -> str | None:
    """Return scenario-specific or fallback Markdown text."""
    scen = (scenario or "").strip().upper()
    candidates = [
        BASE_DIR / "content" / f"{base_name}_{scen}.md",
        BASE_DIR / "content" / f"{base_name}.md",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


@st.cache_data(show_spinner=False)
def load_energy_balance(path: str = "data/LEAP_Energy_Balance.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    c0, c1 = df.columns[:2]
    df = df.rename(columns={c0: "Scenario", c1: "Flow"})
    for col in df.columns:
        if col not in ("Scenario", "Flow"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
# ---------------------------------------------------------------------
# Paths and styling
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
CSS_PATH = BASE_DIR / "static" / "style.css"
LOGO_PATH = BASE_DIR / "static" / "logo.png"
DATA_DIR = BASE_DIR / "data"


def load_css(path: Path = CSS_PATH):
    """Inject CSS from file once per run."""
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_logo(path: Path = LOGO_PATH):
    return Image.open(path)


# Apply CSS and logo
load_css()
if LOGO_PATH.exists():
    try:
        st.sidebar.image(load_logo(), width=160)
    except Exception:
        st.sidebar.image(str(LOGO_PATH), width=160)

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

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
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep)
            break
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    year_col = lower.get("year")
    cap_col = lower.get("co2_cap") or lower.get("cap")
    if not year_col or not cap_col:
        return pd.DataFrame()
    out = df[[year_col, cap_col]].rename(columns={year_col: "Year", cap_col: "CO2_Cap"}).copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["CO2_Cap"] = pd.to_numeric(out["CO2_Cap"], errors="coerce")
    return out.dropna(subset=["Year", "CO2_Cap"]).sort_values("Year")

# ---------------------------------------------------------------------
# Load all datasets
# ---------------------------------------------------------------------
df_costs = load_and_prepare_excel("data/Fable_46_Agricultural.xlsx")
df_emissions = load_and_prepare_excel("data/Fable_46_GHG.xlsx")
df_land = load_and_prepare_excel("data/Fable_46_Land.xlsx")
df_energy = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
df_demand_emissions = load_and_prepare_excel("data/LEAP_Demand_Emissions.xlsx")
df_energy_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
df_supply_emissions = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")
df_energy_balance = load_energy_balance("data/LEAP_Energy_Balance.xlsx")
df_biofuels = load_and_prepare_excel("data/LEAP_Biofuels.xlsx")

# ---------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------
st.sidebar.title("🎯 Scenario Selection")
scenarios = ["BAU", "NCNC", "Interactive"]
selected_scenario = st.sidebar.radio("Choose scenario:", scenarios, index=0, horizontal=True, key="selected_scenario")
st.sidebar.markdown(
    f"<div class='scenario-active'>Active Scenario: <b>{selected_scenario}</b></div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("""
**Scenario definitions:**

- **BAU (Business-as-usual):** continues current trends.  
- **NCNC (Near Carbon Neutral):** follows climate neutrality pathways.  
- **Interactive:** allows dynamic sensitivity analysis.
""")

# ---------------------------------------------------------------------
# Energy flow structure (Sankey)
# ---------------------------------------------------------------------
SOURCES = [
    "Coal Lignite Production", "Coal Lignite Imports", "Wind Production", "Hydro Production",
    "Solar Production", "Geothermal Production", "Crude Oil Imports", "Refinery Feedstocks Imports",
    "Petroleum Coke Imports", "Natural Gas Imports", "Hydrogen Imports", "Biogas Imports",
    "CNG Imports", "Coal Unspecified Imports", "Biomass Production", "Biomass Imports", "Biomass Supply",
]
CONVERTERS = ["Electricity Generation", "Heat Generation", "Oil Refining", "Synthetic Fuels Module", "Transmission and Distribution"]
SINKS = [
    "Residential", "Industry", "Agriculture", "Service Tertiary Sector", "Passenger Transportation",
    "Freight Transportation", "Maritime", "Energy Product Industry", "Hydrogen Generation",
    "Losses", "Exports", "Waste",
]

def build_sankey_from_balance(df: pd.DataFrame, scenario: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Construct Sankey links for a given scenario."""
    df = df.copy()
    if "Flow" not in df.columns:
        raise ValueError("Energy balance must have a 'Flow' column.")
    if scenario and "Scenario" in df.columns:
        df = df[df["Scenario"] == scenario].copy()
        df = df.groupby("Flow", as_index=False).sum(numeric_only=True)
    carriers = [c for c in df.columns if c not in ("Flow", "Total", "Scenario")]
    links = []
    for _, row in df[df["Flow"].isin(SOURCES)].iterrows():
        for fuel in carriers:
            val = row.get(fuel, 0.0)
            if pd.notna(val) and val > 0:
                links.append({"source": row["Flow"], "target": fuel, "value": float(val)})
    for _, row in df[df["Flow"].isin(CONVERTERS)].iterrows():
        for fuel in carriers:
            v = float(row.get(fuel, 0.0) or 0.0)
            if v < 0:
                links.append({"source": fuel, "target": row["Flow"], "value": abs(v)})
            elif v > 0:
                links.append({"source": row["Flow"], "target": fuel, "value": v})
    for _, row in df[df["Flow"].isin(SINKS)].iterrows():
        for fuel in carriers:
            val = row.get(fuel, 0.0)
            if pd.notna(val) and val > 0:
                links.append({"source": fuel, "target": row["Flow"], "value": float(val)})
    links_df = pd.DataFrame(links)
    present = lambda items: [i for i in items if i in set(df["Flow"]).union(carriers)]
    node_order = present(SOURCES) + present(CONVERTERS) + [f for f in carriers if f in links_df["source"].tolist() or f in links_df["target"].tolist()] + present(SINKS)
    return links_df, node_order
# Tabs
tab_overview, tab_food, tab_energy, tab_biofuels, tab_shipping, tab_water, tab_about, tab_custom = st.tabs(theme.TAB_TITLES)

# Overview Tab
with tab_overview:
    st.header("Overview")

    intro_path = Path("content/introduction.md")
    if intro_path.exists():
        st.markdown(intro_path.read_text(), unsafe_allow_html=True)
    else:
        st.warning("⚠️ content/introduction.md not found.")
with tab_food:
    if selected_scenario == "Interactive":
        from views.charts import render_fable_interactive_controls
        render_fable_interactive_controls("Food–Land")
    else:
        scen = (selected_scenario or "").strip().upper()
        # Scenario-specific explainer first
        scen_file = BASE_DIR / "content" / f"food_land_explainer_{scen}.md"
        default_file = BASE_DIR / "content" / "food_land_explainer.md"

        if scen_file.exists():
            text = scen_file.read_text(encoding="utf-8")
            if text.strip():
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.warning(f"{scen_file.name} is empty.")
        elif default_file.exists():
            text = default_file.read_text(encoding="utf-8")
            if text.strip():
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.warning("food_land_explainer.md is empty.")
        else:
            st.warning(f"No food explainer found for scenario {scen} or default.")

        # --- Layout for charts ---
        col1, col2 = st.columns(2)

        # --- Emissions (Mt CO₂e) ---
        with col1:
            # --- Select relevant columns
            cols = ["CropCO2e", "LiveCO2e", "LandCO2"]
            melted, years = prepare_stacked_data(df_emissions, selected_scenario, "Year", cols)

            # --- Normalize component names to match theme keys
            name_map = {
                "CropCO2e": "Crops",
                "LiveCO2e": "Livestock",
                "LandCO2": "Land-use",
                "FAOTotalCO2e": "Total emissions",
                "Total CO₂e": "Total emissions",
            }
            melted["Component"] = melted["Component"].replace(name_map)

            # --- Apply categorical order from theme
            if hasattr(theme, "EMISSIONS_ORDER"):
                melted["Component"] = pd.Categorical(
                    melted["Component"],
                    categories=theme.EMISSIONS_ORDER,
                    ordered=True,
                )
                melted = melted.sort_values(["Year", "Component"])

            # --- Prepare total line
            total_df = (
                df_emissions[df_emissions["Scenario"] == selected_scenario][["Year", "FAOTotalCO2e"]]
                .rename(columns={"FAOTotalCO2e": "Value"})
                .assign(Component="Total emissions")
            )

            # --- Render chart using theme colors and line style
            render_grouped_bar_and_line(
                prod_df=melted,
                demand_df=total_df,
                x_col="Year",
                y_col="Value",
                category_col="Component",
                title="Production-based agricultural emissions",
                colors=getattr(theme, "EMISSIONS_COLORS", {}),
                y_label="Mt CO₂e",
                key=f"food_emissions_{selected_scenario.lower()}",
            )
        # --- Costs (M€) ---
        with col2:
            cols = ["FertilizerCost", "LabourCost", "MachineryRunningCost", "DieselCost", "PesticideCost"]
            melted, years = prepare_stacked_data(df_costs, selected_scenario, "Year", cols)

            # ✅ Use the same color palette as the Interactive chart
            render_bar_chart(
                melted,
                "Year", "Value", "Component",
                "Agricultural production cost",
                [str(y) for y in years],
                colors=theme.COST_COLORS,   # ← add this line
                y_label="M€",
                key="food_costs"
            )


        # --- Land use (1000 km²) ---
        col3, = st.columns(1)
        with col3:
            cols = ["FAOCropland", "FAOHarvArea", "FAOPasture", "FAOUrban", "FAOForest", "FAOOtherLand"]
            df_filtered = df_land[df_land["Scenario"] == selected_scenario].copy()
            df_filtered[cols] = df_filtered[cols].fillna(0)
            melted = df_filtered.melt(
                id_vars=["Year"], value_vars=cols,
                var_name="Component", value_name="Value"
            )
            render_line_chart(
                melted,
                "Year", "Value", "Component",
                "Land uses evolution",
                y_label="1000 km²",
                colors=theme.LANDUSE_COLORS,   # ✅ ensure consistent colors for BAU/NCNC
                key="food_landuse"
            )


with tab_energy:
    scen = (selected_scenario or "").strip().upper()

    # --- INTERACTIVE scenario: render ONLY interactive charts and exit this tab section
    if scen == "INTERACTIVE":
        from views.charts import render_energy_interactive_controls
        render_energy_interactive_controls("Energy Emissions")
    else:
        # ---------------------------------------------------------------------
        # SCENARIOS: BAU / NCNC (existing charts)
        # ---------------------------------------------------------------------

        # Scenario-specific explainer first
        scen_file = BASE_DIR / "content" / f"energy_emissions_explainer_{scen}.md"
        default_file = BASE_DIR / "content" / "energy_emissions_explainer.md"

        if scen_file.exists():
            text = scen_file.read_text(encoding="utf-8")
            if text.strip():
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.warning(f"{scen_file.name} is empty.")
        elif default_file.exists():
            text = default_file.read_text(encoding="utf-8")
            if text.strip():
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.warning("energy_emissions_explainer.md is empty.")
        else:
            st.warning(f"No energy explainer found for scenario {scen} or default.")

        # ---------------------------------------------------------------------
        # First row: Energy consumption & Emissions
        # ---------------------------------------------------------------------
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            cols = ["Residential", "Agriculture", "Industry", "Energy Products",
                    "Passenger Transportation", "Freight Transportation", "Maritime", "Services"]
            melted, years = prepare_stacked_data(df_energy, selected_scenario, "Year", cols)
            period_df, period_order = aggregate_to_periods(
                melted, year_col="Year", value_col="Value", component_col="Component",
                period_years=4, agg="mean", label_mode="range"
            )
            render_bar_chart(
                period_df, "PeriodStr", "Value", "Component",
                "Total energy consumption per sector",
                period_order,
                y_label="ktoe",
                key="energy_consumption"
            )

        with row1_col2:
            cols = ["Residential", "Agriculture", "Industry", "Energy Products",
                    "Passenger Transportation", "Freight Transportation", "Maritime", "Services"]
            melted, years = prepare_stacked_data(df_demand_emissions, selected_scenario, "Year", cols)
            period_df, period_order = aggregate_to_periods(
                melted, year_col="Year", value_col="Value", component_col="Component",
                period_years=4, agg="mean", label_mode="range"
            )
            render_bar_chart(
                period_df, "PeriodStr", "Value", "Component",
                "Emissions from energy consumption by sector",
                period_order,
                y_label="MtCO₂e",
                key="energy_demand_emissions"
            )

        # ---------------------------------------------------------------------
        # Second row: Energy per fuel & Fuel emissions
        # ---------------------------------------------------------------------
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
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
                colors=theme.FUEL_COLORS,
                y_label="ktoe",
                key="energy_generated_fuel"
            )

        with row2_col2:
            cols = ["Electricity Generation", "Heat Generation", "Oil Refining"]
            melted, years = prepare_stacked_data(df_supply_emissions, selected_scenario, "Year", cols)
            period_df, period_order = aggregate_to_periods(
                melted, year_col="Year", value_col="Value", component_col="Component",
                period_years=4, agg="mean", label_mode="range"
            )
            render_bar_chart(
                period_df, "PeriodStr", "Value", "Component",
                "Emissions per fuel type",
                period_order,
                colors=theme.FUEL_COLORS,
                y_label="MtCO₂e",
                key="emissions_per_fuel"
            )

        st.markdown("---")  # visual separator

        # ---------------------------------------------------------------------
        # Energy Balance (Sankey) — only for BAU/NCNC
        # ---------------------------------------------------------------------
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

        scen_file = BASE_DIR / "content" / f"energy_balance_explainer_{scen}.md"
        default_file = BASE_DIR / "content" / "energy_balance_explainer.md"

        if scen_file.exists():
            text = scen_file.read_text(encoding="utf-8")
            if text.strip():
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.warning(f"{scen_file.name} is empty.")
        elif default_file.exists():
            text = default_file.read_text(encoding="utf-8")
            if text.strip():
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.warning("energy_balance_explainer.md is empty.")
        else:
            st.warning(f"No energy balance explainer found for scenario {scen} or default.")

        SANKEY_NODE_COLORS = {
            "Electricity": "#2563eb", "Natural Gas": "#0ea5e9", "Heat": "#ea580c",
            "Crude Oil": "#f59e0b", "Biomass": "#16a34a", "Solar": "#fbbf24",
            "Hydrogen": "#a855f7", "Ethanol": "#22c55e", "Synthetic Fuels": "#06b6d4",
            "Electricity Generation": "#93c5fd", "Heat Generation": "#fdba74",
            "Oil Refining": "#fcd34d", "Synthetic Fuels Module": "#99f6e4",
            "Transmission and Distribution": "#cbd5e1",
            "Residential": "#111827", "Industry": "#374151", "Agriculture": "#4b5563",
            "Service Tertiary Sector": "#6b7280", "Passenger Transportation": "#9ca3af",
            "Freight Transportation": "#9ca3af", "Maritime": "#9ca3af",
            "Energy Product Industry": "#6b7280", "Hydrogen Generation": "#a78bfa",
            "Losses": "#d1d5db", "Exports": "#d1d5db", "Waste": "#d1d5db",
        }

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
                    node_colors=SANKEY_NODE_COLORS,
                )
        except Exception as e:
            st.error(f"Failed to build Sankey: {e}")

with tab_biofuels:
    scen = (selected_scenario or "").strip().upper()
    from views.charts import (
        load_biofuels_data,
        render_biofuels_base_charts,
        render_biofuels_interactive_controls,
    )

    # --- Markdown explainer ---
    text = load_scenario_md("biofuels_explainer", scen)
    if text:
        st.markdown(text, unsafe_allow_html=True)

    # --- INTERACTIVE MODE ---
    if scen == "INTERACTIVE":
        st.subheader("Biofuels – Interactive Scenario Builder")

        # Render all interactive controls (first)
        try:
            render_biofuels_interactive_controls("Biofuels")
        except Exception as e:
            st.error(f"❌ Could not render Biofuels interactive charts: {e}")

        st.markdown("---")

        # ✅ Sensitivity Summary LAST (like Energy tab)
        with st.expander("📉 Sensitivity Summary", expanded=False):
            img_path = BASE_DIR / "content" / "bio_sensitivity.png"
            if img_path.exists():
                st.image(
                    str(img_path),
                    width=800,
                    caption="Relative contribution of SSP and renewables uptake assumptions."
                )
            else:
                st.warning("⚠️ bio_sensitivity.png not found in content/ folder.")

    # --- NON-INTERACTIVE MODES (BAU / NCNC) ---
    else:
        scen_key = "BAU" if scen == "BAU" else "NCNC"

        try:
            data = load_biofuels_data()
            prod, exports = data.get("production"), data.get("exports")

            if prod is not None and exports is not None:
                render_biofuels_base_charts(prod, exports, scen_key)
            else:
                st.warning("⚠️ Missing Biofuels data (production or exports).")
        except Exception as e:
            st.error(f"❌ Failed to load Biofuels data: {e}")
       
with tab_shipping:
    scen = (selected_scenario or "").strip().upper()
    
    # Load scenario-specific explainer
    text = load_scenario_md("ships_explainer", selected_scenario)
    if text:
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.warning("No Ships explainer found for this scenario.")
    
    # --- INTERACTIVE MODE ---
    if scen == "INTERACTIVE":
        from views.charts import render_ships_interactive_controls
        render_ships_interactive_controls("Shipping (Maritime)")
    
    # --- BAU MODE (single KPI) ---
    elif scen == "BAU":
        try:
            base_df = load_maritime_base()
        except Exception as e:
            st.warning(f"Could not load maritime data: {e}")
        else:
            import plotly.graph_objects as go

            fig_bau = go.Figure()

            fig_bau.add_trace(go.Indicator(
                mode="number",
                value=99.68,  # Replace with dynamic value if available
                number={"suffix": " MtCO₂e", "font": {"size": 80}},
                title={"text": "BAU – Total Emissions", "font": {"size": 24}},
            ))

            fig_bau.update_layout(
                width=600,
                height=400,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            st.plotly_chart(fig_bau, use_container_width=False, key="bau_total_emissions")
    
    # --- NCNC MODE (8 charts in 2×4 grid) ---
    else:
        try:
            base_df = load_maritime_base()
        except Exception as e:
            st.warning(f"Could not load maritime data: {e}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = render_ships_stock(base_df, y_label="Number of Stock Ships")
                st.plotly_chart(fig, use_container_width=False, key="ships_stock")
            with col2:
                fig_new = render_ships_new(base_df, y_label="Number of New Ships")
                st.plotly_chart(fig_new, use_container_width=False, key="ships_new")

            col3, col4 = st.columns(2)
            with col3:
                fig_inv = render_ships_investment_cost(base_df, y_label="Costs (M€)")
                st.plotly_chart(fig_inv, use_container_width=False, key="ships_investment_cost")
            with col4:
                fig_op = render_ships_operational_cost(base_df, y_label="Costs (M€)")
                st.plotly_chart(fig_op, use_container_width=False, key="ships_operational_cost")

            col5, col6 = st.columns(2)
            with col5:
                fig_fd = render_ships_fuel_demand(base_df, y_label="Fuel Demand [tonnes]")
                st.plotly_chart(fig_fd, use_container_width=False, key="ships_fuel_demand")
            with col6:
                fig_fc = render_ships_fuel_cost(base_df, y_label="Costs (M€)")
                st.plotly_chart(fig_fc, use_container_width=False, key="ships_fuel_cost")

            col7, col8 = st.columns(2)
            with col7:
                cap_df_to_use = None
                fig_emcap = render_ships_emissions_and_cap(
                    base_df, cap_df=cap_df_to_use, y_label="CO₂ Emissions (MtCO₂e)"
                )
                st.plotly_chart(fig_emcap, use_container_width=False, key="ships_emissions_and_cap")
            with col8:
                fig_penalty = render_ships_ets_penalty(base_df, y_label="Costs (M€)")
                st.plotly_chart(fig_penalty, use_container_width=False, key="ships_ets_penalty")
with tab_water:
    scen = (selected_scenario or "").strip().upper()

    # Load explainer
    text = load_scenario_md("water_explainer", scen)
    if text:
        st.markdown(text, unsafe_allow_html=True)
    else:
        st.warning("No Water explainer found for this scenario.")

    # --- INTERACTIVE MODE ---
    if scen == "INTERACTIVE":
        from views.charts import render_land_water_interactive_controls
        render_land_water_interactive_controls("Land & Water Requirements")
    
    # --- BAU / NCNC MODES (existing charts) ---
    else:
        try:
            water = load_water_requirements()
        except Exception as e:
            st.warning(f"Could not load water data: {e}")
            water = {}

        # --- Urban | Agriculture ---
        wcol1, wcol2 = st.columns(2)

        with wcol1:
            df_u = water.get("urban")
            fig_u = render_water_band(
                df_u if df_u is not None else pd.DataFrame(),
                title="Urban Water Requirements",
                y_label="Water Requirements [hm³]",
            )
            st.plotly_chart(fig_u, use_container_width=True, key="urban_water")

        with wcol2:
            df_a = water.get("agriculture")
            fig_a = render_water_band(
                df_a if df_a is not None else pd.DataFrame(),
                title="Agriculture Water Requirements",
                y_label="Water Requirements [hm³]",
            )
            st.plotly_chart(fig_a, use_container_width=True, key="agri_water")

        # --- Industrial | Monthly ---
        wcol3, wcol4 = st.columns(2)

        with wcol3:
            df_i = water.get("industrial")
            fig_i = render_water_band(
                df_i if df_i is not None else pd.DataFrame(),
                title="Industrial Water Requirements",
                y_label="Water Requirements [hm³]",
            )
            st.plotly_chart(fig_i, use_container_width=True, key="ind_water")

        with wcol4:
            df_m = water.get("monthly")
            fig_m = render_water_monthly_band(
                df_m if df_m is not None else pd.DataFrame(),
                title="Monthly Water Requirements (2020)",
                y_label="Water Requirements [hm³]",
            )
            st.plotly_chart(fig_m, use_container_width=True, key="monthly_water")

with tab_about:
    st.header("About SDSN Greece")

    # Load Markdown from file
    sdsn_file = BASE_DIR / "content" / "sdsn_explainer.md"

    if sdsn_file.exists():
        text = sdsn_file.read_text(encoding="utf-8")
        if text.strip():
            st.markdown(text, unsafe_allow_html=True)
        else:
            st.warning("sdsn_explainer.md exists but is empty.")
    else:
        st.warning("⚠️ content/sdsn.md not found.")
        
with tab_custom:
    st.header("Custom")
    
    # Load Markdown from file
    custom_file = BASE_DIR / "content" / "custom.md"
    
    if custom_file.exists():
        text = custom_file.read_text(encoding="utf-8")
        if text.strip():
            st.markdown(text, unsafe_allow_html=True)
        else:
            st.warning("custom.md exists but is empty.")
    else:
        st.warning("⚠️ content/custom.md not found.")