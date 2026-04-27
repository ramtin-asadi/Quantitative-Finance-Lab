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
    obj = _coerce_objects(objects)
    mat = pd.concat(dict(obj.items()), axis=1)
    return mat.corr(min_periods=int(min_periods))

def rolling_corr(
    x: pd.Series | Sequence[float] | np.ndarray,
    y: pd.Series | Sequence[float] | np.ndarray,
    *,
    window: int = 252,
) -> pd.Series:
    """Rolling correlation between two return series."""
    if int(window) < 2:
        raise ValueError("window must be >= 2.")
    xs = _to_numeric_series(x, name="x")
    ys = _to_numeric_series(y, name="y")
    z = _align_pair(xs, ys)
    out = z["y"].rolling(int(window)).corr(z["x"])
    out.name = f"corr_{int(window)}"
    return out


__all__ = ["corr_matrix", "rolling_corr"]
