from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .errors import InputError


def require_non_empty_frame(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise InputError(f"{name} must be a pandas DataFrame.")
    if df.empty:
        raise InputError(f"{name} is empty.")
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "df") -> pd.DataFrame:
    require_non_empty_frame(df, name=name)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise InputError(f"{name} is missing required columns: {missing}.")
    return df


def require_monotonic_index(
    obj: pd.Series | pd.DataFrame,
    name: str = "obj",
    *,
    increasing: bool = True,
) -> pd.Series | pd.DataFrame:
    idx = obj.index
    ok = idx.is_monotonic_increasing if increasing else idx.is_monotonic_decreasing
    if not ok:
        direction = "increasing" if increasing else "decreasing"
        raise InputError(f"{name}.index must be monotonic {direction}.")
    return obj


def require_finite_array(x, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise InputError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} must contain only finite values.")
    return arr


def normalize_weights(
    weights,
    index: pd.Index | list | tuple | None = None,
    *,
    allow_short: bool = False,
) -> pd.Series:
    if isinstance(weights, pd.Series):
        w = weights.astype(float).copy()
    elif isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
    else:
        if index is None:
            raise InputError("index is required when weights is array-like.")
        w = pd.Series(np.asarray(weights, dtype=float), index=index, dtype=float)

    if index is not None:
        w = w.reindex(index).fillna(0.0)
    if not np.all(np.isfinite(w.to_numpy(dtype=float))):
        raise InputError("weights must be finite.")
    if not allow_short and (w < -1e-12).any():
        raise InputError("weights must be non-negative when allow_short=False.")

    total = float(w.sum())
    if abs(total) <= 1e-12:
        raise InputError("weights sum to zero.")
    return w / total


__all__ = [
    "normalize_weights",
    "require_columns",
    "require_finite_array",
    "require_monotonic_index",
    "require_non_empty_frame",
]
