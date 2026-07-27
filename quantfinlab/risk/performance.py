from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk.utils import (
    DEFAULT_ANNUALIZATION,
    _coerce_objects,
    _excess_returns,
    _to_numeric_series,
)


def nav_series(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    start_value: float = 1.0,
) -> pd.Series:
    """Convert returns into a NAV series.

    Parameters
    ----------
    returns : array-like
        One-period returns in decimal units.
    start_value : float, default=1.0
        Initial NAV. Must be positive.

    Returns
    -------
    pandas.Series
        Cumulative NAV series.

    Raises
    ------
    InputError
        If ``start_value`` is not positive.
    """

    if start_value <= 0:
        raise InputError("start_value must be positive.")
    r = _to_numeric_series(returns, name="returns").fillna(0.0)
    return float(start_value) * (1.0 + r).cumprod()


def total_return(values: pd.Series | Sequence[float] | np.ndarray) -> float:
    """Total return from a NAV/value series."""
    s = _to_numeric_series(values, name="values").dropna()
    if len(s) < 2 or abs(float(s.iloc[0])) <= 1e-12:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def sortino_ratio(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> float:
    """Compute the annualized Sortino ratio.

    Parameters
    ----------
    returns : array-like
        Return series.
    rf_daily : float or pandas.Series, default=0.0
        One-period risk-free rate. A Series is aligned to the return dates.
    annualization : float, default=252.0
        Annualization factor.

    Returns
    -------
    float
        Annualized Sortino ratio, or ``NaN`` when downside deviation is zero or
        unavailable.
    """

    r = _to_numeric_series(returns, name="returns")
    ex = _excess_returns(r, rf_daily).dropna()
    dn = np.minimum(ex.to_numpy(dtype=float), 0.0)
    den = float(np.sqrt(np.mean(np.square(dn))))
    if den <= 1e-12:
        return float("nan")
    return float((ex.mean() / den) * math.sqrt(float(annualization)))

def performance_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> pd.DataFrame:
    """Build annualized performance metrics for return objects.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects.
    rf_daily : float or pandas.Series, default=0.0
        One-period risk-free rate used for Sharpe and Sortino.
    annualization : float, default=252.0
        Annualization factor.

    Returns
    -------
    pandas.DataFrame
        Table with annualized return, annualized volatility, Sharpe ratio, and
        Sortino ratio.
    """

    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    ann = float(annualization)
    for name, r in obj.items():
        nav = nav_series(r)
        n = len(r)
        years = n / ann if ann > 0 else float("nan")
        ann_ret = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else float("nan")
        dvol = float(r.std(ddof=1)) if n > 1 else float("nan")
        ann_vol = dvol * math.sqrt(ann) if np.isfinite(dvol) else float("nan")
        excess = _excess_returns(r, rf_daily).dropna()
        sharpe_vol = dvol if np.isscalar(rf_daily) else float(excess.std(ddof=1))
        sharpe = (
            float(excess.mean() / sharpe_vol * math.sqrt(ann))
            if np.isfinite(sharpe_vol) and sharpe_vol > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "object": name,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino_ratio(r, rf_daily=rf_daily, annualization=annualization),
            }
        )
    return pd.DataFrame(rows).set_index("object").sort_index()


def make_returns_panel(objects: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned object return panel with object names as columns."""
    obj = _coerce_objects(objects)
    return pd.concat(dict(obj.items()), axis=1).sort_index()


def rolling_volatility(
    returns: pd.Series | Sequence[float] | np.ndarray | Mapping[str, Any] | pd.DataFrame,
    *,
    windows: Sequence[int] = (20, 60, 252),
    annualization: float = DEFAULT_ANNUALIZATION,
) -> pd.DataFrame:
    """Compute annualized rolling volatility.

    Parameters
    ----------
    returns : array-like, mapping, or pandas.DataFrame
        One return series or several return objects.
    windows : sequence of int, default=(20, 60, 252)
        Rolling windows.
    annualization : float, default=252.0
        Annualization factor.

    Returns
    -------
    pandas.DataFrame
        Rolling volatility table. For multiple objects, columns are a MultiIndex of
        ``(object, vol_window)``.

    Raises
    ------
    InputError
        If no valid window greater than one is supplied.
    """

    wlist = [int(w) for w in windows if int(w) > 1]
    if not wlist:
        raise InputError("windows must contain at least one integer > 1.")
    ann = float(annualization)
    if isinstance(returns, (Mapping, pd.DataFrame)):
        obj = _coerce_objects(returns)
        cols: dict[tuple[str, str], pd.Series] = {}
        for name, r in obj.items():
            for w in wlist:
                cols[(str(name), f"vol_{w}")] = r.rolling(w).std(ddof=1) * math.sqrt(ann)
        return pd.concat(cols, axis=1).sort_index()
    r = _to_numeric_series(returns, name="returns")
    return pd.DataFrame({f"vol_{w}": r.rolling(w).std(ddof=1) * math.sqrt(ann) for w in wlist})


__all__ = [
    "make_returns_panel",
    "nav_series",
    "performance_table",
    "rolling_volatility",
    "sortino_ratio",
    "total_return",
]
