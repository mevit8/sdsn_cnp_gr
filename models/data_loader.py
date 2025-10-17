import pandas as pd

def load_and_prepare_excel(path, year_col="YEAR"):
    # --- Load sheet normally ---
    df = pd.read_excel(path)

    # --- Clean up column names (critical for color consistency) ---
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\n\r\t]", " ", regex=True)   # remove hidden newlines/tabs
        .str.replace(r"\s+", "", regex=True)         # remove all spaces
        .str.strip()
    )

    # --- Drop merged header rows like "GHG Emissions / Costs / Land use" ---
    if df.iloc[0].astype(str).str.contains("GHG|Cost|Land", case=False).any():
        df = df.iloc[1:].reset_index(drop=True)

    # --- Normalize key columns ---
    if year_col in df.columns:
        df = df.rename(columns={year_col: "Year"})
    else:
        # try case-insensitive fallback
        for c in df.columns:
            if c.strip().lower() == year_col.lower():
                df = df.rename(columns={c: "Year"})
                break

    if "Scenario" in df.columns:
        df["Scenario"] = df["Scenario"].astype(str).str.strip().str.upper()
    else:
        df["Scenario"] = "BAU"

    return df

def prepare_stacked_data(df: pd.DataFrame, scenario: str, year_col: str, cols: list[str]):
    """Prepare melted stacked data robustly, tolerant of column header differences."""
    # --- Normalize column names ---
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\n\r\t]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # --- Filter by scenario (case-insensitive) ---
    if "Scenario" in df.columns:
        mask = df["Scenario"].astype(str).str.strip().str.upper() == str(scenario).strip().upper()
        df = df.loc[mask].copy()
    else:
        st.warning("⚠️ No 'Scenario' column in dataframe; skipping filter.")

    # --- Case- and space-insensitive column matching ---
    lower_map = {c.lower().replace(" ", ""): c for c in df.columns}
    matched_cols = []
    for name in cols:
        key = name.lower().replace(" ", "")
        if key in lower_map:
            matched_cols.append(lower_map[key])
        else:
            st.warning(f"⚠️ Column '{name}' not found in data (scenario {scenario}). Skipped.")

    if not matched_cols:
        raise KeyError(f"No matching columns found for any of: {cols}")

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
