from __future__ import annotations
from typing import Dict, List, Optional, Union
import pandas as pd
from config import theme

ColorSpec = Union[Dict[str, str], List[str], None]

def _normalize(label: str) -> str:
    if hasattr(theme, "normalize"):
        return theme.normalize(label)
    return " ".join(str(label).split())

def _resolve_ci(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = low.get(name.lower())
        if c:
            return c
    return None

def _wrap_text(s: str, max_len: int = 14) -> str:
    if not isinstance(s, str) or len(s) <= max_len:
        return s
    parts, line, count = [], [], 0
    for word in s.split():
        n = count + len(word) + (1 if line else 0)
        if n > max_len:
            parts.append(" ".join(line)); line, count = [word], len(word)
        else:
            line.append(word); count = n
    if line: parts.append(" ".join(line))
    return "<br>".join(parts)

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3: h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
