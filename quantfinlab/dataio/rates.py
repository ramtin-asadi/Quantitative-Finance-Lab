"""Par-yield curve loaders.

A single ``load_par_yield_curve`` handles US Treasury and Japan MOF
files via a registered ``source`` (or an explicit ``column_map``). The
output schema is stable: a sorted, deduplicated ``DatetimeIndex`` with
compact tenor columns ('1M', '2Y', '10Y', ...) and decimal values.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..fixed_income.bootstrap import normalize_par_yields
from ..fixed_income.tenors import TENOR_PATTERN, tenor_to_years
from .schemas import get_rate_source


_TENOR_NORMALIZE_REPLACEMENTS = (
    ("MONTHS", "M"),
    ("MONTH", "M"),
    ("MOS", "M"),
    ("MO", "M"),
    ("YEARS", "Y"),
    ("YEAR", "Y"),
    ("YRS", "Y"),
    ("YR", "Y"),
)


def tenor_label_to_years(label: str | int | float) -> float:
    """Convert a tenor label like '6M' or '2Y' to a float number of years."""
    return tenor_to_years(label)


def _normalize_tenor_name(name: str) -> str:
    s = str(name).strip().upper().replace(" ", "")
    for src, dst in _TENOR_NORMALIZE_REPLACEMENTS:
        s = s.replace(src, dst)
    return s


def load_par_yield_curve(
    path: str | Path,
    *,
    source: str | None = None,
    column_map: dict[str, str] | None = None,
    percent: bool | None = None,
) -> pd.DataFrame:
    """Load a par-yield curve panel into a normalized wide DataFrame.

    Returns a DataFrame indexed by a sorted, deduplicated ``DatetimeIndex``
    with tenor columns drawn from {'1M','2M','3M','4M','6M','1Y','2Y','3Y',
    '5Y','7Y','10Y','20Y','30Y', ...}. Values are in decimals
    (e.g. 4.25% -> 0.0425) when ``percent`` is True or auto-detected.

    Exactly one of ``source`` or ``column_map`` should be supplied for
    sources outside the registry; ``column_map`` (raw -> normalized name)
    overrides ``source`` when both are given.

    Example
    -------
    >>> us = load_par_yield_curve("data/par-yield-curve-rates-1990-2026.csv",
    ...                           source="us_treasury")
    >>> jp = load_par_yield_curve("data/japan_yield_curve_all.csv",
    ...                           source="japan_mof")
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Par-yield file does not exist: {p}")

    cfg: dict[str, object] = {}
    if source is not None:
        cfg = get_rate_source(source)
    elif column_map is None:
        cfg = get_rate_source("us_treasury")

    skip_banner = bool(cfg.get("skip_banner", False))
    na_values = list(cfg.get("na_values", ("",))) if cfg else [""]

    if skip_banner:
        first_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip().lower()
        if first_line.startswith("interest rate") or first_line.startswith("﻿interest rate"):
            raw = pd.read_csv(p, skiprows=1, na_values=na_values, keep_default_na=True)
        else:
            raw = pd.read_csv(p, na_values=na_values, keep_default_na=True)
    else:
        raw = pd.read_csv(p, na_values=na_values, keep_default_na=True)

    raw = raw.rename(columns={c: str(c).strip() for c in raw.columns})

    if column_map:
        raw = raw.rename(columns=column_map)

    raw = raw.rename(columns={c: _normalize_tenor_name(c) if c.lower() != "date" else "date"
                              for c in raw.columns})

    lower_cols = [c.lower() for c in raw.columns]
    if "date" in lower_cols:
        date_col = raw.columns[lower_cols.index("date")]
        if date_col != "date":
            raw = raw.rename(columns={date_col: "date"})
    elif not isinstance(raw.index, pd.DatetimeIndex):
        raw = raw.rename(columns={raw.columns[0]: "date"})

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").set_index("date")
    raw = raw[~raw.index.duplicated(keep="last")]

    tenor_cols = [c for c in raw.columns if re.fullmatch(r"\d+[MY]", str(c))]
    if not tenor_cols:
        raise ValueError(
            f"No tenor columns detected in {p.name}. "
            "Expected labels like '1M', '6M', '2Y', '10Y' after normalization."
        )

    tenor_cols = sorted(dict.fromkeys(tenor_cols), key=tenor_to_years)
    par = raw[tenor_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()

    assume_percent: bool
    if percent is None:
        cfg_percent = cfg.get("percent")
        if cfg_percent is None:
            arr = par.to_numpy(dtype=float)
            assume_percent = bool(np.isfinite(arr).any() and np.nanmedian(np.abs(arr)) > 1.0)
        else:
            assume_percent = bool(cfg_percent)
    else:
        assume_percent = bool(percent)

    out = normalize_par_yields(
        par.reset_index(),
        date_col="date",
        tenor_cols=tenor_cols,
        assume_percent=assume_percent,
    )
    return out


def tenor_first_valid(curve: pd.DataFrame) -> pd.Series:
    """For each tenor column, the first date with a finite value."""
    tenor_cols = [c for c in curve.columns if TENOR_PATTERN.fullmatch(str(c).strip().upper())]
    return curve[tenor_cols].apply(lambda s: s.first_valid_index())


__all__ = [
    "load_par_yield_curve",
    "tenor_first_valid",
    "tenor_label_to_years",
]
