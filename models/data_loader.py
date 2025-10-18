from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Optional, List
from pathlib import Path 

DATA_DIR = Path(__file__).parent.parent / "data"

# ==============================================================
#  Aggregate annual data into multi-year periods
# ==============================================================

def aggregate_to_periods(
    df: pd.DataFrame,
    year_col: str = "Year",
    value_col: str = "Value",
    component_col: str = "Component",
    period_years: int = 4,
    agg: str = "mean",
    label_mode: str = "range",
):
    """Aggregate annual data into multi-year periods."""
    df = df.copy()
    start = (df[year_col].min() // period_years) * period_years
    df["PeriodStart"] = ((df[year_col] - start) // period_years) * period_years + start
    df["PeriodEnd"] = df["PeriodStart"] + (period_years - 1)

    if label_mode == "range":
        df["PeriodStr"] = df["PeriodStart"].astype(str) + "–" + df["PeriodEnd"].astype(str)
    else:
        df["PeriodStr"] = df["PeriodStart"].astype(str)

    grouped = (
        df.groupby(["PeriodStart", "PeriodStr", component_col], as_index=False)[value_col]
        .agg(agg)
    )
    period_order = (
        grouped.drop_duplicates(subset=["PeriodStart", "PeriodStr"])
        .sort_values("PeriodStart")["PeriodStr"]
        .tolist()
    )
    return grouped, period_order


# ==============================================================
#  Load Excel file and standardize structure
# ==============================================================

def load_and_prepare_excel(path, year_col="YEAR"):
    """Load an Excel file and normalize column names for consistent access."""
    df = pd.read_excel(path)

    # --- Clean up column names (preserve CamelCase but remove bad spaces) ---
    df.columns = (
        df.columns.astype(str)
        .str.replace("\xa0", " ", regex=False)           # remove non-breaking spaces
        .str.replace("\u202f", " ", regex=False)         # remove narrow no-break spaces
        .str.replace(r"[\n\r\t]", " ", regex=True)       # remove hidden newlines/tabs
        .str.replace(r"\s+", " ", regex=True)            # collapse multiple spaces
        .str.strip()
    )

    # --- Drop merged header rows like "GHG Emissions / Costs / Land use" ---
    if df.iloc[0].astype(str).str.contains("GHG|Cost|Land", case=False).any():
        df = df.iloc[1:].reset_index(drop=True)

    # --- Normalize Year column name ---
    if year_col in df.columns:
        df = df.rename(columns={year_col: "Year"})
    else:
        # Try case-insensitive fallback
        for c in df.columns:
            if c.strip().lower() == year_col.lower():
                df = df.rename(columns={c: "Year"})
                break

    # --- Normalize Scenario column ---
    if "Scenario" in df.columns:
        df["Scenario"] = df["Scenario"].astype(str).str.strip().str.upper()
    else:
        df["Scenario"] = "BAU"
    return df


# ==============================================================
#  Prepare melted / stacked data for bar or line charts
# ==============================================================

def prepare_stacked_data(df: pd.DataFrame, scenario: Optional[str], year_col: str, cols: List[str]):
    """Prepare melted stacked data robustly, tolerant of column header differences.

    - If `scenario` is None, skip filtering and per-column warnings (used by interactive tabs).
    - Otherwise, filter case-insensitively by Scenario column.
    """
    import streamlit as st

    # --- Normalize column names ---
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\n\r\t]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # --- Scenario filtering ---
    if scenario is not None and "Scenario" in df.columns:
        mask = df["Scenario"].astype(str).str.strip().str.upper() == str(scenario).strip().upper()
        df = df.loc[mask].copy()
        if df.empty:
            st.info(f"ℹ️ No data found for scenario '{scenario}'.")
    elif "Scenario" not in df.columns:
        # Only warn once (not per column)
        st.info("ℹ️ No 'Scenario' column found — using all data (e.g. interactive datasets).")

    # --- Case- and space-insensitive matching ---
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    lower_map = {_norm(c): c for c in df.columns}
    matched_cols = []
    for name in cols:
        key = _norm(name)
        if key in lower_map:
            colname = lower_map[key]
            if scenario is not None:
                # Only check emptiness if scenario-based filtering is active
                if df[colname].notna().sum() == 0:
                    st.info(f"ℹ️ Column '{name}' is present but empty for scenario {scenario}.")
            matched_cols.append(colname)
        else:
            if scenario is not None:
                st.warning(f"⚠️ Column '{name}' not found in data (scenario {scenario}). Skipped.")
            else:
                # For interactive datasets, skip quietly
                pass

    if not matched_cols:
        st.warning(f"No matching data columns found among: {cols}")
        return pd.DataFrame(), []

    # --- Fill NaNs and melt ---
    df[matched_cols] = df[matched_cols].fillna(0)
    melted = df.melt(
        id_vars=[year_col],
        value_vars=matched_cols,
        var_name="Component",
        value_name="Value"
    )
    melted["YearStr"] = melted[year_col].astype(str)
    years = sorted(df[year_col].dropna().unique().tolist())

    return melted, years
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
            gr = {"╬β╬Σ╬ζ":"JAN","╬ο╬Χ╬Τ":"FEB","╬ε╬Σ╬κ":"MAR","╬Σ╬ι╬κ":"APR","╬ε╬Σ╬β":"MAY","╬β╬θ╬ξ╬ζ":"JUN",
                  "╬β╬θ╬ξ╬δ":"JUL","╬Σ╬ξ╬Υ":"AUG","╬μ╬Χ╬ι":"SEP","╬θ╬γ╬ν":"OCT","╬ζ╬θ╬Χ":"NOV","╬Φ╬Χ╬γ":"DEC"}
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