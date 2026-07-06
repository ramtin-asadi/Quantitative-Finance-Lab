from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .errors import InputError


def require_non_empty_frame(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Validate that an object is a non-empty pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Object to validate.
    name : str, default "df"
        Name used in validation error messages.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame, returned unchanged.

    Raises
    ------
    InputError
        If ``df`` is not a DataFrame or is empty.
    """

    if not isinstance(df, pd.DataFrame):
        raise InputError(f"{name} must be a pandas DataFrame.")
    if df.empty:
        raise InputError(f"{name} is empty.")
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "df") -> pd.DataFrame:
    """Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    columns : Iterable[str]
        Required column names.
    name : str, default "df"
        Name used in validation error messages.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame, returned unchanged.

    Raises
    ------
    InputError
        If ``df`` is empty, is not a DataFrame, or is missing any required column.
    """

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
    """Validate monotonic ordering of a Series or DataFrame index.

    Parameters
    ----------
    obj : pandas.Series or pandas.DataFrame
        Object whose index should be checked.
    name : str, default "obj"
        Name used in validation error messages.
    increasing : bool, default True
        If ``True``, require a monotonically increasing index. If ``False``,
        require a monotonically decreasing index.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The original object, returned unchanged.

    Raises
    ------
    InputError
        If the index is not monotonic in the requested direction.
    """

    idx = obj.index
    ok = idx.is_monotonic_increasing if increasing else idx.is_monotonic_decreasing
    if not ok:
        direction = "increasing" if increasing else "decreasing"
        raise InputError(f"{name}.index must be monotonic {direction}.")
    return obj


def require_finite_array(x, name: str = "x") -> np.ndarray:
    """Convert input to a finite NumPy float array.

    Parameters
    ----------
    x : array-like
        Values to convert and validate.
    name : str, default "x"
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        Float array with the same shape implied by ``numpy.asarray``.

    Raises
    ------
    InputError
        If the array is empty or contains NaN or infinite values.
    """

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
    """Convert portfolio weights to a normalized pandas Series.

    Parameters
    ----------
    weights : pandas.Series, dict, or array-like
        Raw portfolio weights. Series and dictionaries provide their own labels.
        Array-like inputs require ``index``.
    index : pandas.Index, list, tuple, or None, optional
        Target index for the returned weights. Existing Series/dict weights are
        reindexed to this order and missing entries are filled with zero.
    allow_short : bool, default False
        If ``False``, negative weights below numerical tolerance are rejected.

    Returns
    -------
    pandas.Series
        Weight vector normalized to sum to one.

    Raises
    ------
    InputError
        If array-like weights are supplied without an index, weights are non-finite,
        short positions are disallowed but present, or the total weight is zero.

    Notes
    -----
    When ``index`` is provided, assets not present in ``weights`` receive zero
    weight before normalization. This makes the helper useful for aligning user
    weights to a larger universe.
    """

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
