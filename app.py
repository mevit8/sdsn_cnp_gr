import streamlit as st
from config import theme
from models.data_loader import load_and_prepare_excel, prepare_stacked_data
from views.charts import render_bar_chart, render_line_chart, render_grouped_bar_and_line
from PIL import Image
import os
from pathlib import Path

st.set_page_config(page_title="SDSN GCH Scenarios", layout="wide")

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


# Load data
df_costs = load_and_prepare_excel("data/Fable_46_Agricultural.xlsx")
df_emissions = load_and_prepare_excel("data/Fable_46_GHG.xlsx")
df_land = load_and_prepare_excel("data/Fable_46_Land.xlsx")
df_energy = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
df_energy_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
df_supply_emissions = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")
df_biofuels = load_and_prepare_excel("data/LEAP_Biofuels_Demand.xlsx", year_col="Year")



# Shared scenario selector
scenarios = sorted(set(df_costs["Scenario"]).intersection(
    df_emissions["Scenario"], df_land["Scenario"], df_energy["Scenario"]
))
selected_scenario = st.sidebar.selectbox("🎯 Select Scenario", scenarios)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(theme.TAB_TITLES)

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
    render_bar_chart(melted, "YearStr", "Value", "Component", "Total energy consumption per sector", [str(y) for y in years])

with tab5:
    cols = ["Residential", "Agriculture", "Industry", "Energy Products",
            "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
    melted, years = prepare_stacked_data(df_energy, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Emissions energy consumption by sector", [str(y) for y in years])

with tab6:
    cols = ["Hydrogen Generation", "Electricity Generation", "Heat Generation", "Oil Refining"]
    melted, years = prepare_stacked_data(df_energy_supply, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Generated energy per fuel type", [str(y) for y in years])

with tab7:
    cols = ["Electricity Generation", "Heat Generation", "Oil Refining"]
    melted, years = prepare_stacked_data(df_supply_emissions, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Emissions from energy generation", [str(y) for y in years])



