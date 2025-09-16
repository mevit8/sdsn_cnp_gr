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
    render_water_band,
    render_water_monthly_band,
)
from PIL import Image
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="SDSN GCH - GR Climate Neutrality", layout="wide")

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
DATA_DIR = BASE_DIR / "data"

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
@st.cache_data(show_spinner=False)
def load_water_requirements(
    uses_path: str | Path | None = None,
    month_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load Urban/Agriculture/Industrial from Greeceplots_uses.xlsx and Monthly from Greeceplots_month.xlsx."""
    import pandas as pd

    uses_path  = uses_path  or (DATA_DIR / "Greeceplots_uses.xlsx")
    month_path = month_path or (DATA_DIR / "Greeceplots_month.xlsx")
    out: dict[str, pd.DataFrame | None] = {"urban": None, "agriculture": None, "industrial": None, "monthly": None}

    # ---------- uses (wide 'in' sheet with *.min/avr/max) ----------
    try:
        xls = pd.ExcelFile(uses_path)
        sheet = "in" if "in" in xls.sheet_names else xls.sheet_names[0]
        df = xls.parse(sheet)
        df.columns = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in df.columns}

        def pick(prefixes: list[str]) -> pd.DataFrame | None:
            year = lower.get("year")
            avg  = next((lower.get(f"{p}avr") or lower.get(f"{p}avg") or lower.get(f"{p}average") for p in prefixes), None)
            mn   = next((lower.get(f"{p}min") for p in prefixes), None)
            mx   = next((lower.get(f"{p}max") for p in prefixes), None)
            if year and avg and mn and mx:
                g = df[[year, avg, mn, mx]].rename(columns={year: "Year", avg: "Average", mn: "Min", mx: "Max"}).copy()
                for c in ("Average", "Min", "Max"):
                    g[c] = pd.to_numeric(g[c], errors="coerce")
                return g
            return None

        out["urban"]       = pick(["urban.", "urban_", "urban "])
        out["agriculture"] = pick(["agr.", "agri.", "agriculture.", "agriculture_", "agr_"])
        out["industrial"]  = pick(["ind.", "industrial.", "ind_", "industrial_"])
    except Exception as e:
        st.info(f"Water (uses) not loaded from {uses_path}: {e}")

    # ---------- monthly (various layouts) ----------
    def norm_months(x: pd.Series) -> pd.Series:
        names = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        s = x.astype(str).str.strip().str.upper()

        def conv(v: str):
            try:
                n = int(float(v))
                if 1 <= n <= 12: return names[n-1]
            except Exception:
                pass
            gr = {"ΙΑΝ":"JAN","ΦΕΒ":"FEB","ΜΑΡ":"MAR","ΑΠΡ":"APR","ΜΑΙ":"MAY","ΙΟΥΝ":"JUN",
                  "ΙΟΥΛ":"JUL","ΑΥΓ":"AUG","ΣΕΠ":"SEP","ΟΚΤ":"OCT","ΝΟΕ":"NOV","ΔΕΚ":"DEC"}
            return gr.get(v, v)

        cat = pd.Categorical(s.map(conv), categories=names, ordered=True)
        return cat

    def try_monthly(dfm: pd.DataFrame) -> pd.DataFrame | None:
        if dfm is None or dfm.empty: return None
        dfm = dfm.copy()
        dfm.columns = [str(c).strip() for c in dfm.columns]
        lower = [c.lower() for c in dfm.columns]

        # A) Columns: Month, Average/Avg/Mean, (Min), (Max)
        if any(c in lower for c in ("month","months")) and any(c in lower for c in ("average","avg","mean","avr")):
            mcol = next((c for c in dfm.columns if c.lower() in ("month","months")), None)
            acol = next((c for c in dfm.columns if c.lower() in ("average","avg","mean","avr")), None)
            mn   = next((c for c in dfm.columns if c.lower() in ("min","minimum","lower")), None)
            mx   = next((c for c in dfm.columns if c.lower() in ("max","maximum","upper")), None)
            sub = dfm[[mcol, acol] + ([mn] if mn else []) + ([mx] if mx else [])].rename(
                columns={mcol:"Month", acol:"Average", **({mn:"Min"} if mn else {}), **({mx:"Max"} if mx else {})}
            )
            sub["Month"] = norm_months(sub["Month"])
            for c in [c for c in ("Average","Min","Max") if c in sub.columns]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
            return sub.sort_values("Month")

        # B) Rows: first col labels (min/avr/max), columns = JAN..DEC (or 1..12)
        month_like = [c for c in dfm.columns if str(c).strip().upper() in
                      ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC",
                       "1","2","3","4","5","6","7","8","9","10","11","12"]]
        if len(month_like) >= 12:
            idx_avg = idx_min = idx_max = None
            for i in range(len(dfm)):
                tag = str(dfm.iloc[i, 0]).strip().lower()
                if tag in ("average","avg","mean","avr"): idx_avg = i
                elif tag in ("min","minimum","lower"):    idx_min = i
                elif tag in ("max","maximum","upper"):    idx_max = i
            if idx_avg is not None:
                months = [str(c).strip().upper() for c in month_like[:12]]
                data = {"Month": months, "Average": pd.to_numeric(dfm.loc[idx_avg, month_like].values[:12], errors="coerce")}
                if idx_min is not None: data["Min"] = pd.to_numeric(dfm.loc[idx_min, month_like].values[:12], errors="coerce")
                if idx_max is not None: data["Max"] = pd.to_numeric(dfm.loc[idx_max, month_like].values[:12], errors="coerce")
                sub = pd.DataFrame(data)
                sub["Month"] = norm_months(sub["Month"])
                return sub.sort_values("Month")

        return None

    try:
        mxls = pd.ExcelFile(month_path)
        for s in mxls.sheet_names:
            cand = try_monthly(mxls.parse(s))
            if cand is not None and not cand.empty:
                out["monthly"] = cand
                break
        if out["monthly"] is None:
            st.info(f"Could not detect a monthly table in {month_path}.")
    except Exception as e:
        st.info(f"Water (monthly) not loaded from {month_path}: {e}")

    return out  # {'urban','agriculture','industrial','monthly'}

# Load data
df_costs = load_and_prepare_excel("data/Fable_46_Agricultural.xlsx")
df_emissions = load_and_prepare_excel("data/Fable_46_GHG.xlsx")
df_land = load_and_prepare_excel("data/Fable_46_Land.xlsx")
df_energy = load_and_prepare_excel("data/LEAP_Demand_Cons.xlsx")
df_demand_emissions = load_and_prepare_excel("data/LEAP_Demand_Emissions.xlsx")
df_energy_supply = load_and_prepare_excel("data/LEAP_Supply.xlsx")
df_supply_emissions = load_and_prepare_excel("data/LEAP_Supply_Emissions.xlsx")
df_energy_balance = load_energy_balance("data/LEAP_Energy_Balance.xlsx")
df_biofuels = load_biofuels_simple("data/LEAP_Biofuels.xlsx")



# Shared scenario selector
scenarios = sorted(set(df_costs["Scenario"]).intersection(
    df_emissions["Scenario"], df_land["Scenario"], df_energy["Scenario"]
))
selected_scenario = st.sidebar.selectbox("🎯 Select Scenario", scenarios)

# Explain to users what the dropdown does
st.sidebar.markdown(
    """
    ℹ️ **How to use the selector**

    Choose between scenarios to update all charts and results:

    - **BAU (Business-as-usual):** projects Greece’s future based on current trends without additional climate measures.  
    - **NCNC:** applies the policies and measures described in the main sectoral climate neutrality pathways for Greece.
    """,
    unsafe_allow_html=False
)


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
tab_overview, tab_food, tab_energy, tab8, tab9, tab10, tab11 = st.tabs(theme.TAB_TITLES)

# Overview Tab
with tab_overview:
    st.header("Overview")

    intro_path = Path("content/introduction.md")
    if intro_path.exists():
        st.markdown(intro_path.read_text(), unsafe_allow_html=True)
    else:
        st.warning("⚠️ content/introduction.md not found.")

with tab_food:
    explainer_path = BASE_DIR / "content" / "food_land_explainer.md"
    if explainer_path.exists():
        text = explainer_path.read_text(encoding="utf-8")
        if text.strip():
            st.markdown(text, unsafe_allow_html=True)
        else:
            st.warning("food_land_explainer.md is empty.")
    else:
        st.warning(f"Explanation file not found at {explainer_path}")

    col1, col2 = st.columns(2)

    # --- Emissions (Mt CO₂e) ---
    with col1:
        cols = ["CropCO2e", "LiveCO2e", "LandCO2"]
        melted, years = prepare_stacked_data(df_emissions, selected_scenario, "Year", cols)

        total_df = df_emissions[df_emissions["Scenario"] == selected_scenario][["Year", "FAOTotalCO2e"]].copy()
        total_df = total_df.rename(columns={"FAOTotalCO2e": "Value"})
        total_df["Component"] = "Total CO₂e"

        render_grouped_bar_and_line(
            prod_df=melted,
            demand_df=total_df,
            x_col="Year",
            y_col="Value",
            category_col="Component",
            title="Production-based agricultural emissions",
            colors=theme.EMISSIONS_COLORS,
            y_label="Mt CO₂e",
            key="food_emissions"
        )

    # --- Costs (M€) ---
    with col2:
        cols = ["FertilizerCost", "LabourCost", "MachineryRunningCost", "DieselCost", "PesticideCost"]
        melted, years = prepare_stacked_data(df_costs, selected_scenario, "Year", cols)
        render_bar_chart(
            melted,
            "Year", "Value", "Component",
            "Agricultural production cost",
            [str(y) for y in years],
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
            key="food_landuse"
        )

with tab_energy:
    explainer_path = BASE_DIR / "content" / "energy_emissions_explainer.md"
    try:
        text = explainer_path.read_text(encoding="utf-8")
        st.markdown(text)
    except Exception as e:
        st.warning(f"Explanation file not found at {explainer_path}: {e}")

    # First row: Energy consumption & Energy consumption emissions
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        cols = ["Residential", "Agriculture", "Industry", "Energy Products",
                "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
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
                "Terrestrial Transportation", "Aviation", "Maritime", "Services"]
        melted, years = prepare_stacked_data(df_demand_emissions, selected_scenario, "Year", cols)
        period_df, period_order = aggregate_to_periods(
            melted, year_col="Year", value_col="Value", component_col="Component",
            period_years=4, agg="mean", label_mode="range"
        )
        render_bar_chart(
            period_df, "PeriodStr", "Value", "Component",
            "Emissions energy consumption by sector",
            period_order,
            y_label="MtCO₂e",
            key="energy_demand_emissions"
        )

    # Second row: Energy per fuel & Fuel emissions
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
    explainer_path = BASE_DIR / "content" / "energy_balance_explainer.md"
    if explainer_path.exists():
        st.markdown(explainer_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Explanation file not found at {explainer_path}")

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

    # st.subheader("⚡ Energy Generation ↔ Consumption (Balance)")
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

with tab9:
    explainer_path = BASE_DIR / "content" / "biofuels_explainer.md"
    if explainer_path.exists():
        st.markdown(explainer_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Explanation file not found at {explainer_path}")

    scen = (selected_scenario or "").strip().upper()
    scen_key = "BAU" if scen == "BAU" else "NCNC"  # default to NCNC for anything else

    # -----------------------------
    # Side-by-side layout for biofuels charts
    # -----------------------------
    col1, col2 = st.columns(2)

    # a) Demand vs Potential Supply
    with col1:
        # st.markdown("**Biofuels demand and potential supply [ktoe]**")

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
            height=theme.CHART_HEIGHT,
            y_label="ktoe",
            key="biofuels_demand_supply"
        )

    # b) Potential for Biofuels Export
    with col2:
        # st.markdown("**Potential for Biofuels Export [ktoe]**")

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

        fig = go.Figure()
        for comp, color in [
            ("Min export potential [ktoe]", "#86efac"),
            ("Max export potential [ktoe]", "#22c55e"),
        ]:
            sub = export_long[export_long["Component"] == comp]
            fig.add_trace(go.Bar(x=sub["Year"], y=sub["Value"], name=comp, marker_color=color))

        fig.update_layout(
            title=f"Potential for Biofuels Export ({scen_key})",
            barmode="group",
            xaxis_title="Year",
            yaxis_title="ktoe",
            width=theme.CHART_WIDTH,
            height=theme.CHART_HEIGHT,
            margin=dict(t=60, r=10, b=10, l=10),
            legend_title_text="Series",
        )
        st.plotly_chart(fig, use_container_width=False, key="biofuels_export")

with tab10:
    explainer_path = BASE_DIR / "content" / "ships_explainer.md"
    if explainer_path.exists():
        st.markdown(explainer_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Explanation file not found at {explainer_path}")

    try:
        base_df = load_maritime_base()
    except Exception as e:
        st.warning(f"Could not load maritime data: {e}")
    else:
        scen = (selected_scenario or "").strip().upper()

        # Special rule: BAU shows ONE number (KPI), not the 8 charts
        if scen == "BAU":
            st.metric(label="BAU – Total Emissions", value="99.68 MtCO₂e")
            st.caption(
                "Currently, the Greek fleet is estimated to emit 99.68 MtCO₂e, "
                "which is well above the European regulatory threshold of 97.9 MtCO₂e."
            )
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
                # 🔒 CO₂ Cap selection hidden — default to None
                cap_df_to_use = None
                fig_emcap = render_ships_emissions_and_cap(base_df, cap_df=cap_df_to_use, y_label="CO₂ Emissions (MtCO₂e)")
                st.plotly_chart(fig_emcap, use_container_width=False, key="ships_emissions_and_cap")
            with col8:
                fig_penalty = render_ships_ets_penalty(base_df, y_label="Costs (M€)")
                st.plotly_chart(fig_penalty, use_container_width=False, key="ships_ets_penalty")

with tab11:
    explainer_path = BASE_DIR / "content" / "water_explainer.md"
    if explainer_path.exists():
        st.markdown(explainer_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Explanation file not found at {explainer_path}")

    # st.subheader("💧 Water Requirements")

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
            avg_col_candidates=["Average"],
            min_col_candidates=["Min"],
            max_col_candidates=["Max"],
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_u, use_container_width=False)

    with wcol2:
        df_a = water.get("agriculture")
        fig_a = render_water_band(
            df_a if df_a is not None else pd.DataFrame(),
            title="Agriculture Water Requirements",
            avg_col_candidates=["Average"],
            min_col_candidates=["Min"],
            max_col_candidates=["Max"],
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_a, use_container_width=False)

    # --- Industrial | Monthly ---
    wcol3, wcol4 = st.columns(2)

    with wcol3:
        df_i = water.get("industrial")
        fig_i = render_water_band(
            df_i if df_i is not None else pd.DataFrame(),
            title="Industrial Water Requirements",
            avg_col_candidates=["Average"],
            min_col_candidates=["Min"],
            max_col_candidates=["Max"],
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_i, use_container_width=False)

    with wcol4:
        df_m = water.get("monthly")
        fig_m = render_water_monthly_band(
            df_m if df_m is not None else pd.DataFrame(),
            title="Monthly Water Requirements (2020)",
            month_col_candidates=["Month", "Months"],
            avg_col_candidates=["Average", "Avg", "Mean"],
            min_col_candidates=["Min"],
            max_col_candidates=["Max"],
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_m, use_container_width=False)











