from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_data_file(path: str | Path, filename: str) -> Path:
    p = Path(path)
    return p / filename if p.is_dir() else p


def clean_monthly_index(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    out.index = out.index.to_period("M").to_timestamp("M")
    out = out.groupby(out.index).last()
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_macro_factors(path: str | Path, *, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    data = pd.read_csv(path)
    date_col = "date" if "date" in data.columns else data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    out = data.dropna(subset=[date_col]).set_index(date_col)
    out = clean_monthly_index(out)
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    return out


def load_nfci(path: str | Path, *, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    data = pd.read_csv(_resolve_data_file(path, "nfci.csv"))
    date_col = "Friday_of_Week" if "Friday_of_Week" in data.columns else data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    out = data.dropna(subset=[date_col]).set_index(date_col).sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    monthly = out.groupby(out.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp("M")
    if start is not None:
        monthly = monthly.loc[monthly.index >= pd.Timestamp(start)]
    return monthly


def load_acm_term_premium(
    path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    monthly: bool = False,
) -> pd.DataFrame:
    """Load the NY Fed ACM term-premia file.

    The source-centered CSV preserves the ACM fitted-yield (``ACMY``),
    term-premium (``ACMTP``), and risk-neutral-yield (``ACMRNY``)
    columns where available. Values are read as numeric and indexed by
    date. Set ``monthly=True`` to collapse to month-end observations.
    """
    data = pd.read_csv(_resolve_data_file(path, "acm_term_premium.csv"))
    date_col = "date" if "date" in data.columns else data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    out = data.dropna(subset=[date_col]).set_index(date_col).sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if monthly:
        out = out.groupby(out.index.to_period("M")).last()
        out.index = out.index.to_timestamp("M")
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    return out


def macro_availability_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in data.columns:
        series = data[column]
        rows.append(
            {
                "column": column,
                "first_date": series.first_valid_index(),
                "last_date": series.last_valid_index(),
                "observations": int(series.notna().sum()),
                "available_share": float(series.notna().mean()),
            }
        )
    return pd.DataFrame(rows).set_index("column")


__all__ = [
    "clean_monthly_index",
    "load_acm_term_premium",
    "load_macro_factors",
    "load_nfci",
    "macro_availability_table",
]
