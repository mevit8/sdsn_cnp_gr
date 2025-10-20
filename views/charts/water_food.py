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
        st.markdown("""
        Explore sensitivities for **Population**, **Diet**, and **Productivity** assumptions.
        Use the dropdowns below to explore how these drivers affect food demand, land use,
        and agricultural emissions.
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


    st.markdown(f"**Selected combination:** `{pop}-{diet}-{prod}`")

    try:
        render_fable_interactive_charts(pop, diet, prod)
    except Exception as e:
        st.error(f"❌ Could not render charts for {pop}-{diet}-{prod}: {e}")

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

    st.markdown(f"### Results for Option {pop}-{diet}-{prod}")

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




