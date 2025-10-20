from typing import Dict, List

APP_TITLE = "📊 SDSN GCH - GR Climate Neutrality"
LOGO_PATH = "static/logo.png"
LOGO_WIDTH = 160
SIDEBAR_TITLE = "### SDSN GCH Scenarios"
TAB_TITLES = [
    "📖 Overview",
    "🌱 Food–Land",
    "⚡ Energy–Emissions",
    "🌿 Biofuels",
    "🚢 Shipping",
    "💧 Land & Water Requirements",
    "🌍 SDSN Pathway",
    "📄 Custom",
]
# Fuel charts palette (black, gold, brown)
# Make sure the keys match  melted 'Component' values exactly.
FUEL_COLORS: Dict[str, str] = {
    "Hydrogen Generation":   "#0EA5E9",
    "Electricity Generation":"#6366F1",
    "Heat Generation":       "#F59E0B",
    "Oil Refining":          "#000000",  # ← black as requested
}

# Optional default palette (used for any categories not in FUEL_COLORS)
DEFAULT_PALETTE: List[str] = [
    "#4F46E5","#059669","#D97706","#DC2626",
    "#0EA5E9","#7C3AED","#16A34A","#8B7061"
]

# Optional: minimal normalization helpers if your labels vary a bit
ALIASES = {
    "oil refining": "Oil Refining",
    "oil-refining": "Oil Refining",
    "oil refinery": "Oil Refining",
}
def normalize(label: str) -> str:
    key = " ".join(str(label).split())
    low = key.casefold()
    return ALIASES.get(low, key)  # return canonical display text if alias found

CHART_WIDTH = 800  # default width for all charts
CHART_HEIGHT = 500  # optional

# theme.py

EMISSIONS_COLORS = {
    "Crops": "#81b29a",           # soft green
    "Livestock": "#e07a5f",       # warm reddish-brown
    "Land-use": "#f2cc8f",        # muted yellow
    "Total emissions": "#3b82f6", # clean blue (for line)
}

COST_COLORS = {
    "LabourCost": "#b56576",          # muted reddish-brown
    "PesticideCost": "#5bc0be",       # turquoise / aqua
    "MachineryRunningCost": "#8ecae6",# light sky blue
    "DieselCost": "#6b7280",          # dark gray
    "FertilizerCost": "#facc15",      # golden yellow
}

LANDUSE_COLORS = {
    "FAOCropland": "#ef4444",    # bright red
    "FAOHarvArea": "#f97316",    # orange
    "FAOPasture": "#facc15",     # yellow
    "FAOUrban": "#9ca3af",       # gray
    "FAOForest": "#10b981",      # teal-green
    "FAOOtherLand": "#374151",   # dark gray
}

# --- Explicit stack orders (bottom → top) to match figure panels ---

COST_ORDER = [
    "LabourCost",             # bottom
    "MachineryRunningCost",
    "PesticideCost",
    "DieselCost",
    "FertilizerCost",         # top
]

EMISSIONS_ORDER = [
    "Crops",                  # bottom
    "Livestock",
    "Land-use",               # top
]

LANDUSE_ORDER = [
    "FAOCropland",            # bottom
    "FAOHarvArea",
    "FAOPasture",
    "FAOForest",
    "FAOUrban",
    "FAOOtherLand",           # top
]

