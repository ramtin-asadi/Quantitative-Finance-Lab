from __future__ import annotations

import pandas as pd

from .errors import InputError


def yearfrac(t0: pd.Timestamp, t1: pd.Timestamp, basis: str = "ACT/365") -> float:
    if basis.upper() != "ACT/365":
        raise InputError("Only ACT/365 is supported for Project 1.")
    return float((pd.Timestamp(t1) - pd.Timestamp(t0)).days) / 365.0


def month_end_dates(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(idx.to_series().resample("ME").last().dropna().values)


def previous_available_date(index: pd.Index, date: pd.Timestamp) -> pd.Timestamp | None:
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
    out = [previous_available_date(index, d) for d in pd.to_datetime(dates)]
    return pd.DatetimeIndex([d for d in out if d is not None])


__all__ = [
    "align_to_previous_available",
    "month_end_dates",
    "previous_available_date",
    "yearfrac",
]
