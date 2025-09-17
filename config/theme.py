from typing import Dict, List

APP_TITLE = "📊 Scenario Dashboard"
LOGO_PATH = "static/logo.png"
LOGO_WIDTH = 160
SIDEBAR_TITLE = "### SDSN GCH Scenarios"
TAB_TITLES = [
    "📖 Overview",
    "🌱🍽️ Food–Land",
    "⚡🌍 Energy–Emissions",
    "🌿 Biofuels",
    "🚢 Shipping",
    "💧 Water Requirements",
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
    "Livestock": "#f87171",   # pink/red
    "Crops": "#34d399",       # green
    "Land-use": "#facc15",    # yellow
    "Total emissions": "#2563eb",  # blue line
}

COST_COLORS = {
    "LabourCost": "#92400e",         # brown
    "PesticideCost": "#22d3ee",      # turquoise
    "MachineryRunningCost": "#60a5fa", # light blue
    "DieselCost": "#9ca3af",         # gray
    "FertilizerCost": "#facc15",     # yellow
}

LANDUSE_COLORS = {
    "FAOCropland": "#ef4444",    # red
    "FAOHarvArea": "#fb923c",    # orange-red
    "FAOPasture": "#facc15",     # yellow
    "FAOUrban": "#9ca3af",       # gray
    "FAOForest": "#22c55e",      # green
    "FAOOtherLand": "#374151",   # dark gray
}

