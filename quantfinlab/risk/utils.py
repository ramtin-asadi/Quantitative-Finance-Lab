from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.core import BacktestResult, InputError, PortfolioState

DEFAULT_ANNUALIZATION = 252.0
VAR_BACKTEST_METHODS = ("hist", "cf", "fhs")

def _to_numeric_series(
    x: pd.Series | Sequence[float] | np.ndarray,
    *,
    name: str = "series",
) -> pd.Series:
    s = x.copy() if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype=float))
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise InputError(f"{name} is empty after numeric cleaning.")
    return s.astype(float)

def _to_datetime_if_possible(s: pd.Series) -> pd.Series:
    idx = pd.Index(s.index)
    if isinstance(idx, pd.DatetimeIndex):
        out = s.copy()
        out = out[~out.index.isna()].sort_index()
        return out
    dt = pd.to_datetime(idx, errors="coerce")
    if dt.notna().all():
        out = s.copy()
        out.index = pd.DatetimeIndex(dt)
        out = out[~out.index.isna()].sort_index()
        return out
    return s

def _coerce_objects(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    dropna: bool = True,
) -> dict[str, pd.Series]:
    if isinstance(objects, pd.DataFrame):
        data = {str(c): objects[c] for c in objects.columns}
    elif isinstance(objects, Mapping):
        data = {str(k): v for k, v in objects.items()}
    else:
        raise InputError("objects must be a mapping of name -> returns series, or a DataFrame.")
    if not data:
        raise InputError("objects is empty.")
    out: dict[str, pd.Series] = {}
    for name, val in data.items():
        s = _to_numeric_series(val, name=f"objects[{name!r}]")
        s = _to_datetime_if_possible(s)
        if dropna:
            s = s.dropna()
        if s.empty:
            continue
        out[name] = s.astype(float)
    if not out:
        raise InputError("No non-empty object series remain after cleaning.")
    return out

def _align_pair(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(z) == 0:
        raise InputError("Series do not overlap after alignment.")
    return z

def _normalize_alpha(alpha: float) -> float:
    a = float(alpha)
    if not (0.0 < a < 0.5):
        raise InputError("alpha must be in (0, 0.5).")
    return a

def _normalize_var_methods(
    *,
    method: str | None = None,
    methods: Sequence[str] | None = None,
) -> list[str]:
    if methods is None:
        base = [method or "hist"]
    else:
        base = list(methods)
        if len(base) == 0:
            raise InputError("methods must contain at least one method.")
    out = [str(m).strip().lower() for m in base]
    valid = set(VAR_BACKTEST_METHODS)
    unknown = [m for m in out if m not in valid]
    if unknown:
        raise InputError(f"Unknown VaR method(s): {unknown}. Valid methods: {sorted(valid)}.")
    return out

def _as_result_mapping(result: BacktestResult | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(result, BacktestResult):
        return result.as_dict()
    if not isinstance(result, Mapping):
        raise InputError("result must be a BacktestResult or dict-like mapping.")
    return result

def _as_state_mapping(state: PortfolioState | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(state, PortfolioState):
        return state.as_dict()
    if isinstance(state, Mapping):
        return state
    raise InputError("state must be PortfolioState or dict-like.")

__all__ = [
    "DEFAULT_ANNUALIZATION",
    "VAR_BACKTEST_METHODS",
    "_align_pair",
    "_as_result_mapping",
    "_as_state_mapping",
    "_coerce_objects",
    "_normalize_alpha",
    "_normalize_var_methods",
    "_to_datetime_if_possible",
    "_to_numeric_series",
]
