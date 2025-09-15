from typing import Dict, List

APP_TITLE = "📊 Scenario Dashboard"
LOGO_PATH = "static/logo.png"
LOGO_WIDTH = 160
SIDEBAR_TITLE = "### SDSN GCH Scenarios"
TAB_TITLES = [
    "🌱🍽️ Food–Land system Results",
    "⚡🌍 Energy–Emissions",
    "⚡ Energy Gen ↔ Cons",
    "🌿 Biofuels",
    "🚢 Ships",
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
# CHART_HEIGHT = 500  # optional
