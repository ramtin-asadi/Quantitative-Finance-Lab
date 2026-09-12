"""Overnight accrual and quarterly reference windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def imm_date(year: int, month: int) -> pd.Timestamp:
    """Third Wednesday of the supplied month."""
    start = pd.Timestamp(year, month, 1)
    return start + pd.Timedelta(days=(2 - start.weekday()) % 7 + 14)


def reference_window(start) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Quarterly IMM window [start, third Wednesday three months later)."""
    start = pd.Timestamp(start)
    end_month = start + pd.DateOffset(months=3)
    return start, imm_date(end_month.year, end_month.month)


def compound_overnight(rates, dates, end, *, day_count: int = 360):
    """Annualized compounded rate in decimals, including weekend/holiday accrual.

    The final array axis matches fixing dates. The caller supplies the actual
    fixing calendar, including the fixing covering the window's first day.
    """
    dates = pd.DatetimeIndex(dates)
    end = pd.Timestamp(end)
    r = np.asarray(rates, dtype=float)
    if len(dates) == 0 or r.shape[-1] != len(dates) or not dates.is_monotonic_increasing:
        raise ValueError("Rates must match a nonempty, ordered fixing calendar.")
    dt = np.diff(np.r_[dates.to_numpy(), end.to_datetime64()]).astype("timedelta64[D]").astype(float)
    if np.any(dt <= 0) or not np.isfinite(r).all() or np.any(1 + r * dt / day_count <= 0):
        raise ValueError("Invalid fixing dates or overnight accumulation factors.")
    return np.expm1(np.log1p(r * dt / day_count).sum(axis=-1)) * day_count / dt.sum()
