import pandas as pd

def load_and_prepare_excel(path, year_col="YEAR"):
    df = pd.read_excel(path)
    df = df.rename(columns={year_col: "Year"})
    df["Scenario"] = df["Scenario"].astype(str).str.strip().str.upper()
    return df

''' def prepare_stacked_data(df, scenario, year_col, value_cols):
    real_years = sorted(df[year_col].dropna().unique())
    five_years = [y for y in real_years if y % 5 == 0]

    df_filtered = df[df["Scenario"] == scenario].copy()
    df_filtered = df_filtered.set_index(year_col).reindex(five_years).reset_index()
    df_filtered[value_cols] = df_filtered[value_cols].fillna(0)

    melted = df_filtered.melt(
        id_vars=[year_col],
        value_vars=value_cols,
        var_name="Component",
        value_name="Value"
    )
    melted["YearStr"] = melted[year_col].astype(str)
    return melted, five_years
'''

def prepare_stacked_data(df: pd.DataFrame, scenario: str, year_col: str, cols: list[str]):
    # Strong, whitespace-insensitive filter
    mask = df["Scenario"].astype(str).str.strip() == str(scenario).strip()
    df = df.loc[mask].copy()

    df[cols] = df[cols].fillna(0)

    melted = df.melt(id_vars=[year_col], value_vars=cols,
                     var_name="Component", value_name="Value")
    melted["YearStr"] = melted[year_col].astype(str)

    years = sorted(df[year_col].dropna().unique().tolist())
    return melted, years

