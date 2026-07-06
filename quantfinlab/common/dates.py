from __future__ import annotations

import pandas as pd

from .errors import InputError


def yearfrac(t0: pd.Timestamp, t1: pd.Timestamp, basis: str = "ACT/365") -> float:
    """Compute an ACT/365 year fraction between two dates.

    Parameters
    ----------
    t0 : pandas.Timestamp
        Start date.
    t1 : pandas.Timestamp
        End date.
    basis : str, default "ACT/365"
        Day-count basis. Only ``"ACT/365"`` is supported.

    Returns
    -------
    float
        Calendar-day difference divided by 365.

    Raises
    ------
    InputError
        If ``basis`` is not ``"ACT/365"``.

    Notes
    -----
    The calculation uses the integer day difference between pandas timestamps. It
    does not account for business-day calendars, leap-year conventions, or
    30/360-style coupon accrual rules.
    """

    if basis.upper() != "ACT/365":
        raise InputError("Only ACT/365 is supported for Project 1.")
    return float((pd.Timestamp(t1) - pd.Timestamp(t0)).days) / 365.0


def month_end_dates(index: pd.Index) -> pd.DatetimeIndex:
    """Extract the last available observation date in each calendar month.

    Parameters
    ----------
    index : pandas.Index
        Date-like index or collection of date-like values.

    Returns
    -------
    pandas.DatetimeIndex
        Sorted month-end observation dates. If multiple dates occur within the
        same month, the last available date is returned. Empty input returns an
        empty ``DatetimeIndex``.

    Notes
    -----
    Returned dates are the last available dates from the input, not necessarily
    calendar month-end dates if the input has no observation on the actual month
    end.
    """

    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(idx.to_series().resample("ME").last().dropna().values)


def previous_available_date(index: pd.Index, date: pd.Timestamp) -> pd.Timestamp | None:
    """Find the latest available date on or before a target date.

    Parameters
    ----------
    index : pandas.Index
        Available date index.
    date : pandas.Timestamp
        Target date.

    Returns
    -------
    pandas.Timestamp or None
        Exact date if present, otherwise the latest available date before the
        target. Returns ``None`` when the index is empty or all available dates are
        after the target.

    Notes
    -----
    The input index is converted to a sorted ``DatetimeIndex`` before lookup. The
    function is useful for preventing look-ahead bias when mapping market data to
    valuation dates.
    """

    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if len(idx) == 0:
        return None
    d = pd.Timestamp(date)
    if d in idx:
        return d
    pos = idx.searchsorted(d, side="right") - 1
    if pos < 0:
        return None
    return pd.Timestamp(idx[pos])


def align_to_previous_available(index: pd.Index, dates) -> pd.DatetimeIndex:
    """Map a sequence of dates to the previous available dates in an index.

    Parameters
    ----------
    index : pandas.Index
        Available date index.
    dates : array-like
        Target dates to resolve.

    Returns
    -------
    pandas.DatetimeIndex
        Resolved dates, excluding targets for which no previous available date
        exists.

    Notes
    -----
    The output may be shorter than ``dates`` because unresolved dates are dropped.
    Duplicate resolved dates are preserved when multiple targets map to the same
    available observation.
    """

    out = [previous_available_date(index, d) for d in pd.to_datetime(dates)]
    return pd.DatetimeIndex([d for d in out if d is not None])


__all__ = [
    "align_to_previous_available",
    "month_end_dates",
    "previous_available_date",
    "yearfrac",
]
