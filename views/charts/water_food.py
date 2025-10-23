from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from config import theme
from .helpers import _resolve_ci
from .generic import render_grouped_bar_and_line, render_bar_chart, render_line_chart

import streamlit as st
import pandas as pd
import plotly.express as px
from config import theme

import os
import pandas as pd
import streamlit as st

import pandas as pd
from pathlib import Path
import streamlit as st

@st.cache_data(show_spinner=False)
def load_water_requirements(
    path_uses: str = "data/Greeceplots_uses.xlsx",
    path_month: str = "data/Greeceplots_month.xlsx"
) -> dict[str, pd.DataFrame]:
    """Load and structure water requirement data for urban, agriculture, industry, and monthly."""
    data = {}

    # ------------------------------------------------------------
    # 1️⃣ Annual water use file (Urban / Agriculture / Industry)
    # ------------------------------------------------------------
    try:
        xl_uses = pd.ExcelFile(path_uses)
        sheet_uses = xl_uses.sheet_names[0]
        df_uses = pd.read_excel(xl_uses, sheet_name=sheet_uses)

        # Normalize column names
        df_uses.columns = [str(c).strip() for c in df_uses.columns]
        if "Year" not in df_uses.columns:
            raise ValueError("Missing 'Year' column in Greeceplots_uses.xlsx")

        # Helper to extract one sector
        def extract_sector(df, prefix):
            cols = [c for c in df.columns if c.lower().startswith(prefix.lower())]
            if len(cols) < 3:
                return pd.DataFrame()
            out = df[["Year"] + cols].copy()
            out.columns = ["Year", "Min", "Average", "Max"]
            return out

        data["urban"] = extract_sector(df_uses, "Urban")
        data["agriculture"] = extract_sector(df_uses, "Agr")
        data["industrial"] = extract_sector(df_uses, "Ind")

    except Exception as e:
        st.warning(f"❌ Could not load {path_uses}: {e}")
        data["urban"] = data["agriculture"] = data["industrial"] = pd.DataFrame()

    # ------------------------------------------------------------
    # 2️⃣ Monthly file (2020)
    # ------------------------------------------------------------
    try:
        xl_month = pd.ExcelFile(path_month)
        sheet_month = xl_month.sheet_names[0]
        df_m = pd.read_excel(xl_month, sheet_name=sheet_month, header=None)

        # Expect structure: first column is min/avr/max, next 12 are JAN–DEC
        df_m.columns = ["Type"] + [str(c).strip() for c in df_m.iloc[0, 1:].tolist()]
        df_m = df_m.iloc[1:].reset_index(drop=True)

        df_m = df_m.melt(id_vars=["Type"], var_name="Month", value_name="Value")
        df_m["Type"] = df_m["Type"].str.lower().map({"min": "Min", "avr": "Average", "max": "Max"})
        df_m = df_m.pivot_table(index="Month", columns="Type", values="Value").reset_index()

        data["monthly"] = df_m
    except Exception as e:
        st.warning(f"⚠️ Could not load {path_month}: {e}")
        data["monthly"] = pd.DataFrame()

    return data

@st.cache_data(show_spinner=False)
def render_water_band(df: pd.DataFrame, title: str, y_label="Water Requirements [hm³]"):
    if df.empty: return go.Figure().update_layout(title=f"{title} — no data")
    y, a, mi, ma = (_resolve_ci(df, cands) for cands in (["Year"], ["Average"], ["Min"], ["Max"]))
    if not (y and a and mi and ma): return go.Figure().update_layout(title=f"{title} — missing columns")
    d = df[[y, a, mi, ma]].rename(columns={y: "Year", a: "Avg", mi: "Min", ma: "Max"})
    for c in ("Avg","Min","Max"): d[c] = pd.to_numeric(d[c], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Min"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Max"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.20)", name="Range"))
    fig.add_trace(go.Scatter(x=d["Year"], y=d["Avg"], mode="lines+markers", name="Average", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="", yaxis_title=y_label)
    return fig

@st.cache_data(show_spinner=False)
def render_water_monthly_band(df: pd.DataFrame, title="Monthly Water Requirements (2020)", y_label="Water Requirements [hm³]"):
    if df.empty: return go.Figure().update_layout(title=f"{title} — no data")
    m, a, mi, ma = (_resolve_ci(df, cands) for cands in (["Month"], ["Average"], ["Min"], ["Max"]))
    if not (m and a and mi and ma): return go.Figure().update_layout(title=f"{title} — missing columns")
    d = df[[m, a, mi, ma]].rename(columns={m: "Month", a: "Avg", mi: "Min", ma: "Max"})
    for c in ("Avg","Min","Max"): d[c] = pd.to_numeric(d[c], errors="coerce")
    order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    d["Month"] = pd.Categorical(d["Month"].astype(str).str.upper(), categories=order, ordered=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Min"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Max"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(59,130,246,0.20)", name="Range"))
    fig.add_trace(go.Scatter(x=d["Month"], y=d["Avg"], mode="lines+markers", name="Average", line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title="Month", yaxis_title=y_label)
    return fig

# ---- FABLE Interactive ----
def load_fable_combos(path: str = "data/Fable_results_combos.xlsx",
                      sheet: str = "Custom combinations from user") -> pd.DataFrame:
    """Parse the 'Custom combinations from user' sheet into tidy rows with Pop/Diet/Prod codes."""
    df_raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # rows that start a block (contain "Option X-Y-Z")
    opt_rows = df_raw.index[df_raw.iloc[:, 0].astype(str).str.contains(r"Option\s*[A-C]-[A-C]-[A-C]", na=False)].tolist()
    if not opt_rows:
        st.warning("No 'Option A-B-C' headers found in FABLE combos sheet.")
        return pd.DataFrame()

    blocks = []
    for i, start in enumerate(opt_rows):
        end = opt_rows[i + 1] if (i + 1) < len(opt_rows) else len(df_raw)
        header_row = start + 1
        data_start = header_row + 1
        # sanity checks
        if header_row >= len(df_raw):
            continue
        headers = [str(h).strip() for h in df_raw.iloc[header_row].tolist()]
        block = df_raw.iloc[data_start:end].dropna(how="all").copy()
        if block.empty:
            continue
        block.columns = headers
        label = str(df_raw.iloc[start, 0]).strip()
        block["ScenarioOption"] = label
        blocks.append(block)

    if not blocks:
        return pd.DataFrame()

    df = pd.concat(blocks, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    # numeric conversions
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # extract A/B/C codes
    df[["PopOpt", "DietOpt", "ProdOpt"]] = df["ScenarioOption"].str.extract(r"Option\s*([A-C])\-([A-C])\-([A-C])")

    # make other columns numeric where possible
    for c in df.columns:
        if c not in ["ScenarioOption", "PopOpt", "DietOpt", "ProdOpt", "Year"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def render_fable_interactive_controls(tab_name="Food & Land"):
    st.subheader(f"{tab_name} – Interactive Scenario Builder")

    with st.expander("ℹ️ About the FABLE Calculator"):
        st.markdown("""Here, the users can explore different data and variables, and interactively see the results from our models.  
**FABLE Calculator** uses CORINE national land cover baseline, FAOSTAT crop yields (historical & trend), livestock numbers, food demand projections, at an annual time-step, to estimate food-land systems pathways to 2050. Its huge scenario explorer uses national population and GDP projections based on the Shared **Socioeconomic Pathways** (SSPs); Dietary choices imply a basic uptake in ingredients/products which are defined by SSPs, EAT Lancet Diet, or other custom scenarios; Different default rates of crop, livestock productivity (low, middle, high), are included. The FABLE Calculator offers a portfolio of more than 1.5 billion pathways (a combination of in-build scenarios through changing different assumption variables on climate conditions, economic and agricultural policy, regulation and demographics). Here, the key driving factors of the demand (since the tool is demand-based), can be explored:
        """)

    col1, col2, col3 = st.columns(3)

    # -----------------------------
    # 1️⃣ Population & GDP projections
    # -----------------------------
    with col1:
        pop = st.selectbox(
            "Population & GDP projections to 2050:",
            ["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – SSP2 (BAU)",
                "B": "Option B – SSP1 (NCNC)",
                "C": "Option C – SSP5",
            }[x],
            key="fable_pop_select",
            help="""**Option A – SSP2 (BAU):** ~24% population decline by 2100 vs 2021; GDP +1.9–2.2%/yr.  
    **Option B – SSP1 (NCNC):** ~14% population decline by 2050 vs 2021; GDP +2.0–2.5%/yr.  
    **Option C – SSP5:** High-tech progress; population down up to 40% by 2100; GDP +2.3–2.5%/yr."""
        )

    # -----------------------------
    # 2️⃣ Dietary choices
    # -----------------------------
    with col2:
        diet = st.selectbox(
            "Dietary choices, with agricultural land implications:",
            ["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – Baseline (BAU)",
                "B": "Option B – EAT-Lancet (NCNC)",
                "C": "Option C – FatDiet",
            }[x],
            key="fable_diet_select",
            help="""**Option A – Baseline (BAU):** FAO baseline diet; similar to 2010–2020; limited land expansion.  
    **Option B – EAT-Lancet (NCNC):** Sustainable average diet; minimal deforestation beyond 2030.  
    **Option C – FatDiet:** High-fat, high-meat diet; potential free expansion of productive land."""
        )

    # -----------------------------
    # 3️⃣ Crop & livestock productivity
    # -----------------------------
    with col3:
        prod = st.selectbox(
            "Crop & livestock productivity:",
            ["A", "B", "C"],
            format_func=lambda x: {
                "A": "Option A – Baseline (BAU)",
                "B": "Option B – High growth (NCNC)",
                "C": "Option C – Low growth",
            }[x],
            key="fable_prod_select",
            help="""**Option A – Baseline (BAU):** FAOSTAT baseline; similar to 2010–2020 yields.  
    **Option B – High growth (NCNC):** +50–80% yield gap closure; strong productivity gains.  
    **Option C – Low growth:** +30–40% yield gap closure; lower productivity increase."""
        )

    try:
        render_fable_interactive_charts(pop, diet, prod)
    except Exception as e:
        st.error(f"❌ Could not render charts for {pop}-{diet}-{prod}: {e}")

    # ============================================================================
    # SENSITIVITY ANALYSIS SECTION
    # ============================================================================
    st.markdown("---")

    with st.expander("📊 Sensitivity Analysis by Main Demand Drivers", expanded=False):
        df_sensitivity = load_fable_sensitivity()

        metrics = [
            ('GHG_total', 'GHG total (MtCO2e)', 'GHG total (MtCO2e)'),
            ('TotalCost', 'Total Cost (M€)', 'Total Cost (M€)'),
            ('Cropland', 'Cropland (kha)', 'Cropland (kha)'),
            ('Pasture', 'Pasture (kha)', 'Pasture (kha)'),
            ('OtherLand', 'Other Land (kha)', 'Other Land (kha)')
        ]

        col1, col2 = st.columns(2)

        for idx, (metric, ylabel, title) in enumerate(metrics):
            fig = create_sensitivity_boxplot(df_sensitivity, metric, ylabel, title)
            
            if idx % 2 == 0:
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col2:
                    st.plotly_chart(fig, use_container_width=True)

def render_fable_interactive_charts(pop: str, diet: str, prod: str):
    """Filter the combos sheet for the chosen Option (A/B/C) and render 3 charts."""
    df_all = load_fable_combos()
    if df_all.empty:
        st.warning("FABLE combos could not be loaded or parsed.")
        return

    # Filter rows for the exact selection
    mask = (df_all["PopOpt"] == pop) & (df_all["DietOpt"] == diet) & (df_all["ProdOpt"] == prod)
    df = df_all.loc[mask].copy()
    if df.empty:
        st.info(f"No data found for Option {pop}-{diet}-{prod}.")
        return

    # ✅ Normalize all column names
    df.columns = [str(c).strip() for c in df.columns]
    # ✅ Fix 'Year' column variations
    if "year" in [c.lower() for c in df.columns]:
        df.rename(columns={c: "Year" for c in df.columns if c.lower() == "year"}, inplace=True)

    # ---- Column resolution helpers (case-insensitive) ----
    def _pick(df_, candidates):
        low = {c.lower(): c for c in df_.columns}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    # ---------- Emissions ----------
    crop_col = _pick(df, ["CropCO2e", "GHG Crop", "Crop GHG", "Crops"])
    live_col = _pick(df, ["LiveCO2e", "GHG Livestock", "Livestock GHG", "Livestock"])
    land_col = _pick(df, ["LandCO2", "GHG LUC", "Land-use", "Land use", "LUC"])
    totl_col = _pick(df, ["FAOTotalCO2e", "GHG Total", "Total CO2e", "Total CO₂e", "Total emissions"])

    # ---------- Costs ----------
    cost_cols_map = {
        "FertilizerCost": ["FertilizerCost", "Fertiliser Cost", "Fertilizer Cost"],
        "LabourCost": ["LabourCost", "Labor Cost", "Labour Cost"],
        "MachineryRunningCost": ["MachineryRunningCost", "Machinery Cost", "Machinery Running Cost"],
        "DieselCost": ["DieselCost", "Diesel Cost"],
        "PesticideCost": ["PesticideCost", "Pesticides Cost", "Pesticide Cost"],
    }
    resolved_costs = {k: _pick(df, v) for k, v in cost_cols_map.items()}

    # ---------- Build tidy frames ----------
    em_parts = []
    if crop_col: em_parts.append(("Crops", crop_col))
    if live_col: em_parts.append(("Livestock", live_col))
    if land_col: em_parts.append(("Land-use", land_col))

    em_long = pd.concat(
        [
            df[["Year", col]].rename(columns={col: "Value"}).assign(Component=label)
            for label, col in em_parts
            if "Year" in df.columns and col in df.columns
        ],
        ignore_index=True,
    ) if em_parts else pd.DataFrame(columns=["Year", "Value", "Component"])

    # Optional total line
    if totl_col and totl_col in df.columns and "Year" in df.columns:
        total_df = df[["Year", totl_col]].rename(columns={totl_col: "Value"}).assign(Component="Total emissions")
    else:
        total_df = pd.DataFrame()

    # ---------- Costs ----------
    cost_parts = [(label, col) for label, col in resolved_costs.items() if col and col in df.columns]
    cost_long = pd.concat(
        [
            df[["Year", col]].rename(columns={col: "Value"}).assign(Component=label)
            for label, col in cost_parts
            if "Year" in df.columns
        ],
        ignore_index=True,
    ) if cost_parts else pd.DataFrame(columns=["Year", "Value", "Component"])

    # ---------- Land Use (with corrected labels) ----------
    land_long = pd.DataFrame(columns=["Year", "Value", "Component"])
    year_col = next((c for c in df.columns if str(c).strip().lower() == "year"), None)
    if year_col:
        land_aliases = {
            "FAOCropland": ["FAOCropland", "Cropland"],
            "FAOHarvArea": ["FAOHarvArea", "Total Harvested Area", "Harvested Area", "HarvArea"],
            "FAOPasture": ["FAOPasture", "Pasture"],
            "FAOUrban": ["FAOUrban", "Urban"],
            "FAOForest": ["FAOForest", "Forest"],
            "FAOOtherLand": ["FAOOtherLand", "Other Land", "OtherLand"],
        }

        present = {}
        for pretty, candidates in land_aliases.items():
            for c in candidates:
                if c in df.columns:
                    present[pretty] = c
                    break

        if present:
            value_cols = list(present.values())
            land_long_raw = df[[year_col] + value_cols].copy()
            land_long = land_long_raw.melt(
                id_vars=[year_col],
                value_vars=value_cols,
                var_name="Component",
                value_name="Value",
            )
            reverse_map = {v: k for k, v in present.items()}
            land_long["Component"] = land_long["Component"].map(reverse_map).fillna(land_long["Component"])
            land_long["Value"] = pd.to_numeric(land_long["Value"], errors="coerce")
            land_long = land_long.dropna(subset=["Value"])
            land_long = land_long.rename(columns={year_col: "Year"})

    # ---------- Plot ----------
    col1, col2 = st.columns(2)
    with col1:
        render_grouped_bar_and_line(
            prod_df=em_long,
            demand_df=total_df if not total_df.empty else None,
            x_col="Year",
            y_col="Value",
            category_col="Component",
            title="Agricultural GHG emissions",
            colors=getattr(theme, "EMISSIONS_COLORS", {}),
            y_label="Mt CO₂e",
            key=f"fable_ghg_{st.session_state.get('fable_selection','')}",
        )

    with col2:
        if not cost_long.empty:
            render_bar_chart(
                cost_long,
                "Year", "Value", "Component",
                "Agricultural Costs",
                colors=getattr(theme, "COST_COLORS", {}),
                y_label="M€",
                key=f"fable_costs_{st.session_state.get('fable_selection','')}",
            )
        else:
            st.info("No cost components available for this option.")

    col3, = st.columns(1)
    with col3:
        if not land_long.empty:
            # --- Debug: check what Plotly is actually seeing ---
            if not land_long.empty:
                render_line_chart(
                    land_long,
                    "Year", "Value", "Component",
                    "Land Use Change",
                    y_label="1000 km²",
                    colors=getattr(theme, "LANDUSE_COLORS", {}),
                    key=f"fable_land_{st.session_state.get('fable_selection','')}",
                )
            else:
                st.info("No land components available for this option.")


# ---------------------------------------------------------------------
# Interactive mode for Land & Water Requirements tab
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_water_scenario(path: str, sheet: str) -> dict[str, pd.DataFrame]:
    """Load water data for a specific SSP scenario (Option A/B/C)."""
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        
        if "Year" not in df.columns:
            st.warning(f"⚠️ 'Year' column missing in {sheet}")
            return {"urban": pd.DataFrame(), "agriculture": pd.DataFrame(), "industrial": pd.DataFrame()}
        
        # Extract sectors with Min/Average/Max columns
        urban_cols = [c for c in df.columns if c.startswith("Urban_")]
        irr_cols = [c for c in df.columns if c.startswith("Irr_")]
        ind_cols = [c for c in df.columns if c.startswith("Ind_")]
        
        urban_df = df[["Year"] + urban_cols].copy()
        urban_df.columns = ["Year", "Min", "Average", "Max"]
        
        agri_df = df[["Year"] + irr_cols].copy()
        agri_df.columns = ["Year", "Min", "Average", "Max"]
        
        ind_df = df[["Year"] + ind_cols].copy()
        ind_df.columns = ["Year", "Min", "Average", "Max"]
        
        return {
            "urban": urban_df,
            "agriculture": agri_df,
            "industrial": ind_df,
        }
    except Exception as e:
        st.warning(f"❌ Could not load water scenario {sheet}: {e}")
        return {"urban": pd.DataFrame(), "agriculture": pd.DataFrame(), "industrial": pd.DataFrame()}
   
def render_land_water_interactive_controls(section_title: str):
    """
    Render interactive controls for Land & Water Requirements tab.
    Two sections:
    1. Land requirements for renewable energy (solar/wind)
    2. Water requirements by SSP scenario
    """
    st.subheader(f"{section_title}")

    from pathlib import Path
    explainer_path = Path("content/water_explainer_INTERACTIVE.md")
    if explainer_path.exists():
        with st.expander("ℹ️ About the Land & Water Requirements Explorer", expanded=False):
            st.markdown(explainer_path.read_text(encoding="utf-8"), unsafe_allow_html=True)
    
    # =========================================================================
    # SECTION 1: LAND REQUIREMENTS FOR RENEWABLE ENERGY
    # =========================================================================
    st.markdown("### Land Requirements for Renewable Energy Expansion")
    
    col_solar, col_wind = st.columns(2)
    
    with col_solar:
        solar_mw = st.number_input(
            "Required Additional Capacity for Solar Power (MW):",
            min_value=0,
            value=28051,
            step=100,
            key="land_solar_capacity"
        )
    
    with col_wind:
        wind_mw = st.number_input(
            "Required Additional Capacity for Wind Farms (MW):",
            min_value=0,
            value=8540,
            step=100,
            key="land_wind_capacity"
        )
    
    # Calculate results
    solar_land_min = solar_mw * 0.0239
    solar_land_avg = solar_mw * 0.0301
    solar_land_max = solar_mw * 0.0364
    solar_cost_min = solar_land_min * 1e6
    solar_cost_avg = solar_land_avg * 1e6
    solar_cost_max = solar_land_max * 1e6
    
    wind_land_min = wind_mw * 0.0022
    wind_land_avg = wind_mw * 0.0030
    wind_land_max = wind_mw * 0.0041
    wind_cost_min = wind_land_min * 1.5e6
    wind_cost_avg = wind_land_avg * 1.5e6
    wind_cost_max = wind_land_max * 1.5e6
    
    # Display results in tables
    st.markdown("#### 📊 Estimated Land Requirements and Installation Costs")
    
    col_table1, col_table2 = st.columns(2)
    
    with col_table1:
        st.markdown("**Solar PVs**")
        solar_results = pd.DataFrame({
            "Scenario": ["Min", "Avg", "Max"],
            "Land (km²)": [f"{solar_land_min:.2f}", f"{solar_land_avg:.2f}", f"{solar_land_max:.2f}"],
            "Installation Cost (M€)": [f"{solar_cost_min/1e6:.2f}", f"{solar_cost_avg/1e6:.2f}", f"{solar_cost_max/1e6:.2f}"],
        })
        st.dataframe(solar_results, hide_index=True, use_container_width=True)
    
    with col_table2:
        st.markdown("**Wind Farms (Onshore)**")
        wind_results = pd.DataFrame({
            "Scenario": ["Min", "Avg", "Max"],
            "Land (km²)": [f"{wind_land_min:.2f}", f"{wind_land_avg:.2f}", f"{wind_land_max:.2f}"],
            "Installation Cost (M€)": [f"{wind_cost_min/1e6:.2f}", f"{wind_cost_avg/1e6:.2f}", f"{wind_cost_max/1e6:.2f}"],
        })
        st.dataframe(wind_results, hide_index=True, use_container_width=True)
    
    # Land sensitivity summary
    with st.expander("📉 Land Requirements - Sensitivity Summary", expanded=False):
        from pathlib import Path
        img_path = Path("content/land_sensitivity.png")
        if img_path.exists():
            st.image(str(img_path), width=800, caption="Sensitivity of land requirements to capacity assumptions.")
        else:
            st.info("⚠️ land_sensitivity.png not found in content/ folder.")
    
    st.markdown("---")
    
    # =========================================================================
    # SECTION 2: WATER REQUIREMENTS BY SSP SCENARIO
    # =========================================================================
    st.markdown("### 💧 Water Requirements by SSP Scenario")
    
    # SSP scenario selector
    ssp_option = st.selectbox(
        "Select SSP Scenario (Population, GDP, and technology projections to 2050):",
        options=["A", "B", "C"],
        format_func=lambda x: {
            "A": "Option A – SSP2 (BAU): Moderate population decline, GDP +1.9–2.2%/yr",
            "B": "Option B – SSP1: Strong sustainability, GDP +2.0–2.5%/yr, reduced water losses",
            "C": "Option C – SSP5: Rapid economic development, GDP +2.3–2.5%/yr, high water losses",
        }[x],
        index=0,
        key="water_ssp_selector",
        help="""**Option A (SSP2):** Middle-of-the-road scenario with moderate growth and network losses (30-40%).
**Option B (SSP1):** Sustainable development with green industry and improved water infrastructure (15-25% losses).
**Option C (SSP5):** High growth scenario with rapid industrial expansion and high water losses (40-50%)."""
    )
    
    # Load water data for selected scenario
    data_path = "data/Water_scenarios.xlsx"
    sheet_map = {"A": "Option_A", "B": "Option_B", "C": "Option_C"}
    water_data = _load_water_scenario(data_path, sheet_map[ssp_option])
    
    # Load monthly data (static, doesn't change with SSP)
    try:
        monthly_df = load_water_requirements()["monthly"]
    except:
        monthly_df = pd.DataFrame()
    
    # Render the 4 water charts in 2×2 grid
    wcol1, wcol2 = st.columns(2)
    
    with wcol1:
        df_u = water_data.get("urban", pd.DataFrame())
        fig_u = render_water_band(
            df_u,
            title="Urban Water Requirements",
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_u, use_container_width=True, key=f"urban_water_{ssp_option}")
    
    with wcol2:
        df_a = water_data.get("agriculture", pd.DataFrame())
        fig_a = render_water_band(
            df_a,
            title="Agriculture Water Requirements",
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_a, use_container_width=True, key=f"agri_water_{ssp_option}")
    
    wcol3, wcol4 = st.columns(2)
    
    with wcol3:
        df_i = water_data.get("industrial", pd.DataFrame())
        fig_i = render_water_band(
            df_i,
            title="Industrial Water Requirements",
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_i, use_container_width=True, key=f"ind_water_{ssp_option}")
    
    with wcol4:
        fig_m = render_water_monthly_band(
            monthly_df,
            title="Monthly Water Requirements (2020)",
            y_label="Water Requirements [hm³]",
        )
        st.plotly_chart(fig_m, use_container_width=True, key=f"monthly_water_{ssp_option}")
    
    # Water sensitivity summary
    with st.expander("📉 Water Requirements - Sensitivity Summary", expanded=False):
        from pathlib import Path
        img_path = Path("content/water_sensitivity.png")
        if img_path.exists():
            st.image(str(img_path), width=800, caption="Sensitivity of water requirements to SSP assumptions.")
        else:
            st.info("⚠️ water_sensitivity.png not found in content/ folder.")

@st.cache_data(show_spinner=False)
def load_fable_sensitivity():
    """Load FABLE sensitivity analysis data from Excel"""
    df = pd.read_excel(
        'data/Fable_results_combos.xlsx',
        sheet_name='Sensitivity2',
        header=0
    )
    
    # Rename columns to match code (remove units from headers)
    df.columns = ['InputSet', 'Option', 'GHG_total', 'TotalCost', 
                'Cropland', 'Pasture', 'OtherLand']
    
    # Rename for consistency
    df['Driver'] = df['InputSet']
    
    return df


def create_sensitivity_boxplot(df, metric, ylabel, title):
    """Create box plot for sensitivity analysis"""
    fig = go.Figure()
    
    drivers = ['Population/GDP', 'Diets', 'Productivity']
    
    for driver in drivers:
        df_driver = df[df['Driver'] == driver]
        values = df_driver[metric].values
        mean_val = values.mean()
        
        fig.add_trace(go.Box(
            y=values,
            name=driver,
            boxmean=True,
            marker=dict(
                color='rgba(0,0,0,0)',
                line=dict(color='black', width=1)
            ),
            line=dict(color='black'),
            fillcolor='white',
            showlegend=False
        ))
        
        # Add mean point (orange)
        fig.add_trace(go.Scatter(
            x=[driver],
            y=[mean_val],
            mode='markers',
            marker=dict(color='orange', size=8),
            showlegend=False,
            hovertemplate=f'Mean: {mean_val:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=12)),
        yaxis_title=ylabel,
        xaxis_title='',
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig


