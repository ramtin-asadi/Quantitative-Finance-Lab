from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core import InputError

TENOR_PATTERN = re.compile(r"^\d+[MY]$")
DEFAULT_METHODS = ("loglinear", "pchip", "nss", "qp")
DEFAULT_HOLDOUTS = ("6M", "2Y", "7Y", "20Y")
DEFAULT_ISSUE_MATURITIES = (2, 5, 10, 30)

_COLUMN_ALIASES = {
    "date": "date",
    "1 mo": "1M",
    "2 mo": "2M",
    "3 mo": "3M",
    "4 mo": "4M",
    "6 mo": "6M",
    "1 yr": "1Y",
    "2 yr": "2Y",
    "3 yr": "3Y",
    "5 yr": "5Y",
    "7 yr": "7Y",
    "10 yr": "10Y",
    "20 yr": "20Y",
    "30 yr": "30Y",
}

def tenor_to_years(x: str | int | float) -> float:
    """
    Convert tenor labels like '6M', '2Y' to years.
    Also accepts numeric years (int/float).
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().upper()
    if s.endswith("M"):
        return float(int(s[:-1])) / 12.0
    if s.endswith("Y"):
        return float(int(s[:-1]))
    # allow '2' to mean 2Y
    if s.isdigit():
        return float(int(s))
    raise ValueError(f"Unsupported tenor label: {x!r}")

def nearest_tenor_label(
    tenor_labels: list[str] | tuple[str, ...] | pd.Index,
    *,
    target_maturity_years: float,
) -> str:
    labels = [str(x) for x in tenor_labels]
    if not labels:
        raise InputError("tenor_labels is empty.")
    return min(labels, key=lambda c: abs(tenor_to_years(c) - float(target_maturity_years)))

__all__ = [
    "DEFAULT_HOLDOUTS",
    "DEFAULT_ISSUE_MATURITIES",
    "DEFAULT_METHODS",
    "TENOR_PATTERN",
    "_COLUMN_ALIASES",
    "nearest_tenor_label",
    "tenor_to_years",
]
