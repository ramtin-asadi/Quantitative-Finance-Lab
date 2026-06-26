from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk.utils import DEFAULT_ANNUALIZATION, _coerce_objects, _to_numeric_series


def nav_series(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    start_value: float = 1.0,
) -> pd.Series:
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
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> float:
    r = _to_numeric_series(returns, name="returns")
    ex = r - float(rf_daily)
    dn = np.minimum(ex.to_numpy(dtype=float), 0.0)
    den = float(np.sqrt(np.mean(np.square(dn))))
    if den <= 1e-12:
        return float("nan")
    return float((ex.mean() / den) * math.sqrt(float(annualization)))

def performance_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> pd.DataFrame:
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
        sharpe = (
            float((r.mean() - float(rf_daily)) / dvol * math.sqrt(ann))
            if np.isfinite(dvol) and dvol > 1e-12
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
    """Compute annualized rolling volatility for one series or a panel of objects."""
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
