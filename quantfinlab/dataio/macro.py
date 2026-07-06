from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_data_file(path: str | Path, filename: str) -> Path:
    p = Path(path)
    return p / filename if p.is_dir() else p


def clean_monthly_index(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame to a month-end datetime index.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data with a date-like index.

    Returns
    -------
    pandas.DataFrame
        Numeric DataFrame indexed by month-end timestamps. Dates are parsed,
        invalid dates are removed, duplicate dates keep the last observation, and
        multiple observations in the same month collapse to the last available
        observation.

    Notes
    -----
    All values are coerced to numeric and infinite values are replaced by NaN. The
    returned index uses calendar month-end timestamps, not necessarily original
    observation dates.
    """

    out = data.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    out.index = out.index.to_period("M").to_timestamp("M")
    out = out.groupby(out.index).last()
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_macro_factors(path: str | Path, *, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Load macro factor data from a CSV file and normalize it monthly.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file containing a date column and one or more macro variables.
    start : str, pandas.Timestamp, or None, optional
        Optional lower date bound applied after monthly normalization.

    Returns
    -------
    pandas.DataFrame
        Numeric month-end macro factor table indexed by ``DatetimeIndex``.

    Notes
    -----
    The date column is inferred as ``"date"`` when present, otherwise the first
    column is treated as the date column. Duplicate monthly observations collapse
    to the last observation in each month.
    """

    data = pd.read_csv(path)
    date_col = "date" if "date" in data.columns else data.columns[0]
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    out = data.dropna(subset=[date_col]).set_index(date_col)
    out = clean_monthly_index(out)
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    return out


def load_nfci(path: str | Path, *, start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Load and monthly-align a National Financial Conditions Index table.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the NFCI CSV file or a directory containing ``nfci.csv``.
    start : str, pandas.Timestamp, or None, optional
        Optional lower date bound applied after monthly aggregation.

    Returns
    -------
    pandas.DataFrame
        Numeric NFCI table indexed by calendar month-end timestamps.

    Notes
    -----
    Weekly observations are grouped by calendar month and the last available weekly
    observation in each month is retained. Infinite values are replaced by NaN.
    """

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
    """Load an ACM term-premium data file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ACM CSV file or a directory containing
        ``acm_term_premium.csv``.
    start : str, pandas.Timestamp, or None, optional
        Optional lower date bound applied after loading.
    monthly : bool, default False
        If ``True``, collapse daily observations to the last available observation
        in each calendar month.

    Returns
    -------
    pandas.DataFrame
        Numeric ACM table indexed by date. Available fitted-yield, term-premium,
        and risk-neutral-yield columns are preserved.

    Notes
    -----
    The date column is inferred as ``"date"`` when present, otherwise the first
    column is treated as the date column. Values are coerced to numeric and
    infinite values are replaced by NaN.
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
    """Summarize date coverage and missingness for each macro column.

    Parameters
    ----------
    data : pandas.DataFrame
        Time-indexed macro or factor table.

    Returns
    -------
    pandas.DataFrame
        Availability table indexed by column name with ``first_date``,
        ``last_date``, ``observations``, and ``available_share``.

    Notes
    -----
    The function measures non-missing observations column by column and does not
    drop rows globally before computing availability.
    """

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
