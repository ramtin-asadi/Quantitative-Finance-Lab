from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.risk.utils import _align_pair, _coerce_objects, _to_numeric_series


def corr_matrix(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Compute a correlation matrix across return objects.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects to combine.
    min_periods : int, default=20
        Minimum number of observations required for each pairwise correlation.

    Returns
    -------
    pandas.DataFrame
        Pairwise correlation matrix.
    """

    obj = _coerce_objects(objects)
    mat = pd.concat(dict(obj.items()), axis=1)
    return mat.corr(min_periods=int(min_periods))

def rolling_corr(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    *,
    window: int = 252,
) -> pd.Series:
    """Compute rolling correlation between two series.

    Parameters
    ----------
    x, y : array-like
        Series to align and compare.
    window : int, default=252
        Rolling window. Must be at least two observations.

    Returns
    -------
    pandas.Series
        Rolling correlation series named ``corr_window``.

    Raises
    ------
    ValueError
        If ``window < 2``.
    """

    if int(window) < 2:
        raise ValueError("window must be >= 2.")
    xs = _to_numeric_series(x, name="x")
    ys = _to_numeric_series(y, name="y")
    z = _align_pair(xs, ys)
    out = z["y"].rolling(int(window)).corr(z["x"])
    out.name = f"corr_{int(window)}"
    return out


__all__ = ["corr_matrix", "rolling_corr"]
