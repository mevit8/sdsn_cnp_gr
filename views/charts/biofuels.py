from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from config import theme
from .generic import render_grouped_bar_and_line

@st.cache_data(show_spinner=False)
def load_biofuels_data(path: str = "data/LEAP_Biofuels.xlsx") -> dict[str, pd.DataFrame]:
    """Load biofuels data from the cleaned multi-sheet Excel file."""
    prod = pd.read_excel(path, sheet_name="Production")
    exports = pd.read_excel(path, sheet_name="Export")
    combos = pd.read_excel(path, sheet_name="Combos")

    # Fix common typo in the Export sheet
    if "Yeaer" in exports.columns:
        exports = exports.rename(columns={"Yeaer": "Year"})

    # Rename for internal consistency
    prod = prod.rename(columns={
        "Minimum Production Potential [ktoe]": "MinProd_ktoe",
        "Maximum Production Potential [ktoe]": "MaxProd_ktoe",
        "Biofuel Demand Baseline scenario [ktoe]": "Demand_BAU_ktoe",
        "Biofuel Demand NECP [ktoe]": "Demand_NCNC_ktoe",
    })

    exports = exports.rename(columns={
        "Minimum export potential, Baseline scenario [ktoe]": "MinExport_BAU_ktoe",
        "Maximum export potential, Baseline scenario [ktoe]": "MaxExport_BAU_ktoe",
        "Minimum export potential, NECP [ktoe]": "MinExport_NCNC_ktoe",
        "Maximum export potential, NECP [ktoe]": "MaxExport_NCNC_ktoe",
    })

    # Convert numeric columns safely
    for df in [prod, exports, combos]:
        for col in df.columns:
            if col != "Code":
                df[col] = pd.to_numeric(df[col], errors="ignore")

    return {"production": prod, "exports": exports, "combos": combos}

def get_biofuels_option_sets():
    return {
        "residual": {
            "title": "Residual Availability",
            "description": (
                "Share of production residuals (corn, sugarbeets, sunflowers, olives, wheat) "
                "usable for biofuels without affecting agricultural production (FABLE)."
            ),
            "options": {"A": "Option A – 30%", "B": "Option B – 35%", "C": "Option C – 40%"},
        },
        "coefficient": {
            "title": "Biofuel Production Coefficient [L/t]",
            "description": "Liters of biofuel per ton of crop (empirical data and studies).",
            "options": {"A": "Option A – 340–380", "B": "Option B – 380–450", "C": "Option C – 450–520"},
        },
        "technology": {
            "title": "Technology Adoption Rate",
            "description": "Mandated blending uptake per national commitments (LEAP).",
            "options": {"A": "Option A – slow", "B": "Option B – moderate", "C": "Option C – fast"},
        },
    }

def render_biofuels_base_charts(prod: pd.DataFrame, exports: pd.DataFrame, scen_key: str,
                                line_override: pd.DataFrame | None = None):
    c1, c2 = st.columns(2)
    with c1:
        demand_col = f"Demand_{scen_key}_ktoe"
        bars = prod.melt(id_vars=["Year"], value_vars=["MinProd_ktoe", "MaxProd_ktoe"],
                         var_name="Component", value_name="Value")
        bars["Component"] = bars["Component"].replace({
            "MinProd_ktoe": "Minimum Production Potential [ktoe]",
            "MaxProd_ktoe": "Maximum Production Potential [ktoe]",
        })
        line = prod[["Year", demand_col]].rename(columns={demand_col: "Value"})
        line["Component"] = "Biofuel Demand [ktoe]"
        if line_override is not None:
            line = line_override
        render_grouped_bar_and_line(
            prod_df=bars, demand_df=line, x_col="Year", y_col="Value", category_col="Component",
            title=f"Biofuels Demand vs Potential Supply ({scen_key})",
            height=theme.CHART_HEIGHT, y_label="ktoe",
            key=f"biofuels_demand_supply_{scen_key.lower()}",
        )

    with c2:
        exp_min_col = f"MinExport_{scen_key}_ktoe"
        exp_max_col = f"MaxExport_{scen_key}_ktoe"
        exp_df = exports.melt(id_vars=["Year"], value_vars=[exp_min_col, exp_max_col],
                              var_name="Component", value_name="Value")
        exp_df["Component"] = exp_df["Component"].replace({
            exp_min_col: "Min export potential [ktoe]",
            exp_max_col: "Max export potential [ktoe]",
        })
        fig2 = px.bar(
            exp_df, x="Year", y="Value", color="Component",
            color_discrete_map={"Min export potential [ktoe]": "#86efac", "Max export potential [ktoe]": "#22c55e"},
            title=f"Potential for Biofuels Export ({scen_key})", barmode="group",
            width=theme.CHART_WIDTH, height=theme.CHART_HEIGHT,
        )
        st.plotly_chart(fig2, use_container_width=False, key=f"biofuels_export_{scen_key.lower()}")

def render_biofuels_interactive_controls(tab_name: str):
    # st.header(f"{tab_name} — Interactive Biofuels Scenarios")

    with st.expander("ℹ️ About the Biofuels Explorer"):
        st.markdown("Explore potential **production** and **export** under different assumptions.")

    sets = get_biofuels_option_sets()
    col1, col2, col3 = st.columns(3)
    with col1:
        res_opt = st.selectbox(
            "Residual Availability",
            options=list(sets["residual"]["options"].keys()),
            format_func=lambda o: sets["residual"]["options"][o],
            help=(f"**{sets['residual']['title']}**\n\n" + sets['residual']['description'] +
                  "\n\n" + "\n".join([f"- {v}" for v in sets['residual']['options'].values()])),
        )
    with col2:
        coef_opt = st.selectbox(
            "Biofuel Production Coefficient [L/t]",
            options=list(sets["coefficient"]["options"].keys()),
            format_func=lambda o: sets["coefficient"]["options"][o],
            help=(f"**{sets['coefficient']['title']}**\n\n" + sets['coefficient']['description'] +
                  "\n\n" + "\n".join([f"- {v}" for v in sets['coefficient']['options'].values()])),
        )
    with col3:
        tech_opt = st.selectbox(
            "Technology Adoption Rate",
            options=list(sets["technology"]["options"].keys()),
            format_func=lambda o: sets["technology"]["options"][o],
            help=(f"**{sets['technology']['title']}**\n\n" + sets['technology']['description'] +
                  "\n\n" + "\n".join([f"- {v}" for v in sets['technology']['options'].values()])),
        )

    combo_code = f"{res_opt}-{coef_opt}-{tech_opt}"

    data = load_biofuels_data()
    prod, exports, combos = data["production"], data["exports"], data["combos"]

    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("–", "-").replace("—", "-").replace(" ", "")

    combos["CodeClean"] = combos["Code"].apply(_norm)
    cc = _norm(combo_code)
    combo = combos[combos["CodeClean"] == cc]
    if combo.empty:
        starts = [c for c in combos["CodeClean"] if c.startswith(cc[:-1])]
        if starts:
            combo = combos[combos["CodeClean"] == starts[0]]
        else:
            # Render default BAU/NCNC without override if nothing matches.
            render_biofuels_base_charts(prod, exports, scen_key="BAU")
            return

    combo_line = combo[["Year", "Selected Production Potential (ktoe)"]].rename(
        columns={"Selected Production Potential (ktoe)": "Value"}
    )
    combo_line["Component"] = f"Biofuel Production ({combo_code})"
    render_biofuels_base_charts(prod, exports, scen_key="BAU", line_override=combo_line)

