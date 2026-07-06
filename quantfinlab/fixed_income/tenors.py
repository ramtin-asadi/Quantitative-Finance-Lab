from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..common.errors import InputError

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
    """Convert tenor labels or numeric maturities to years.

    Parameters
    ----------
    x : str, int, or float
        Tenor label such as ``"6M"`` or ``"2Y"``, a numeric year value, or a digit
        string interpreted as years.

    Returns
    -------
    float
        Maturity expressed in years.

    Raises
    ------
    ValueError
        If the label cannot be interpreted as months or years.

    Notes
    -----
    Month labels are converted as ``months / 12``. Numeric inputs are returned as
    floating-point years.
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
    """Find the tenor label closest to a target maturity.

    Parameters
    ----------
    tenor_labels : list of str, tuple of str, or pandas.Index
        Candidate tenor labels.
    target_maturity_years : float
        Target maturity in years.

    Returns
    -------
    str
        Candidate label whose year value is closest to the target.

    Raises
    ------
    InputError
        If no candidate labels are supplied.

    Notes
    -----
    The function compares absolute distance in years after converting each tenor
    label.
    """

    labels = [str(x) for x in tenor_labels]
    if not labels:
        raise InputError("tenor_labels is empty.")
    return min(labels, key=lambda c: abs(tenor_to_years(c) - float(target_maturity_years)))

__all__ = [
    "DEFAULT_HOLDOUTS",
    "DEFAULT_ISSUE_MATURITIES",
    "DEFAULT_METHODS",
    "TENOR_PATTERN",
    "nearest_tenor_label",
    "tenor_to_years",
]
