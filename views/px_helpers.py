# views/px_helpers.py
import plotly.express as px
from typing import Iterable
from config.colors import color_map

def px_bar_consistent(df, x, y, variable_col: str, tab: int, **kwargs):
    cmap = color_map(df[variable_col].dropna().unique().tolist(), tab)
    return px.bar(df, x=x, y=y, color=variable_col, color_discrete_map=cmap, **kwargs)

def px_line_consistent(df, x, y, variable_col: str, tab: int, **kwargs):
    cmap = color_map(df[variable_col].dropna().unique().tolist(), tab)
    return px.line(df, x=x, y=y, color=variable_col, color_discrete_map=cmap, **kwargs)

def px_scatter_consistent(df, x, y, variable_col: str, tab: int, **kwargs):
    cmap = color_map(df[variable_col].dropna().unique().tolist(), tab)
    return px.scatter(df, x=x, y=y, color=variable_col, color_discrete_map=cmap, **kwargs)
