# config/colors.py
from typing import Dict, Iterable, List

# Pin exact labels (after normalization) to specific colors
CATEGORY_COLOR_OVERRIDES: Dict[str, str] = {
    "oil refining": "#000000",  # ← black, your requirement
    # add more fixed mappings here if needed
}

# Optional bases (used only when a category isn't pinned above)
TAB6_PALETTE: List[str] = ["#4F46E5","#059669","#D97706","#DC2626","#0EA5E9","#7C3AED","#16A34A","#EA580C"]
TAB7_PALETTE: List[str] = ["#6366F1","#10B981","#F59E0B","#EF4444","#38BDF8","#8B5CF6","#22C55E","#FB923C"]

# Aliases to collapse small wording differences to one bucket
ALIASES: Dict[str, str] = {
    "oil-refining": "oil refining",
    "oil refining ": "oil refining",
    "oil refinery": "oil refining",
    "refining (oil)": "oil refining",
}

def _norm(s: str) -> str:
    return " ".join(s.split()).casefold()

def _canon(label: str) -> str:
    n = _norm(label)
    return _norm(ALIASES.get(n, n))

def _hash_to_index(label: str, n: int) -> int:
    # deterministic (stable across runs) simple string hash
    h = 0
    for ch in label:
        h = ((h << 5) - h) + ord(ch)
        h &= 0xFFFFFFFF
    return abs(h) % n

def color_for(label: str, tab: int) -> str:
    key = _canon(label)
    if key in CATEGORY_COLOR_OVERRIDES:
        return CATEGORY_COLOR_OVERRIDES[key]
    palette = TAB6_PALETTE if tab == 6 else TAB7_PALETTE
    return palette[_hash_to_index(key, len(palette))]

def color_map(variables: Iterable[str], tab: int) -> Dict[str, str]:
    """Returns {original_label: color} for Plotly Express color_discrete_map."""
    cmap: Dict[str, str] = {}
    for v in variables:
        cmap[v] = color_for(v, tab)
    return cmap
