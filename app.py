import streamlit as st
from config import theme
from models.data_loader import load_and_prepare_excel, prepare_stacked_data
from views.charts import render_bar_chart, render_line_chart
from PIL import Image
import os

st.set_page_config(page_title=theme.APP_TITLE, layout="wide")

# Load custom CSS
with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar logo
if os.path.exists(theme.LOGO_PATH):
    st.sidebar.image(Image.open(theme.LOGO_PATH), width=theme.LOGO_WIDTH)
st.sidebar.markdown(theme.SIDEBAR_TITLE)

# Load data
df_costs = load_and_prepare_excel("data/Fable_46_Agricultural.xlsx")
df_emissions = load_and_prepare_excel("data/Fable_46_GHG.xlsx")
df_land = load_and_prepare_excel("data/Fable_46_Land.xlsx")
df_energy = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")

# Shared scenario selector
scenarios = sorted(set(df_costs["Scenario"]).intersection(
    df_emissions["Scenario"], df_land["Scenario"], df_energy["Scenario"]
))
selected_scenario = st.sidebar.selectbox("🎯 Select Scenario", scenarios)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(theme.TAB_TITLES)

with tab1:
    cols = ["FertilizerCost", "LabourCost", "MachineryRunningCost", "DieselCost", "PesticideCost"]
    melted, years = prepare_stacked_data(df_costs, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Costs by Component", [str(y) for y in years])

with tab2:
    cols = ["CropCO2e", "LiveCO2e", "LandCO2", "FAOTotalCO2e"]
    melted, years = prepare_stacked_data(df_emissions, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Emissions by Component", [str(y) for y in years])

with tab3:
    cols = ["FAOCropland", "FAOHarvArea", "FAOPasture", "FAOUrban", "FAOForest", "FAOOtherLand"]
    df_filtered = df_land[df_land["Scenario"] == selected_scenario].copy()
    df_filtered[cols] = df_filtered[cols].fillna(0)
    melted = df_filtered.melt(id_vars=["Year"], value_vars=cols, var_name="Component", value_name="Value")
    render_line_chart(melted, "Year", "Value", "Component", "Land Area by Type")

with tab4:
    cols = ["Residential", "Agriculture", "Industry", "Energy Products",
            "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
    melted, years = prepare_stacked_data(df_energy, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Total Energy Consumption per Sector", [str(y) for y in years])
    
with tab5:
    cols = ["Residential", "Agriculture", "Industry", "Energy Products",
            "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
    melted, years = prepare_stacked_data(df_energy, selected_scenario, "Year", cols)
    render_bar_chart(melted, "YearStr", "Value", "Component", "Final Energy Consumption by Sector", [str(y) for y in years])
