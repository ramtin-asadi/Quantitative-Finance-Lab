from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.risk.utils import _coerce_objects, _to_numeric_series


def tail_shape_table(objects: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Summarize distribution shape and empirical tail behavior.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects to analyze.

    Returns
    -------
    pandas.DataFrame
        Table with skew, excess kurtosis, 95/05 tail ratio, worst one-period
        return, and average worst 5 and 10 one-period returns.
    """

    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        q05 = float(r.quantile(0.05)) if len(r) else float("nan")
        q95 = float(r.quantile(0.95)) if len(r) else float("nan")
        tail_ratio = float(abs(q95 / q05)) if abs(q05) > 1e-12 else float("nan")
        rows.append(
            {
                "object": name,
                "skew": float(r.skew()) if len(r) else float("nan"),
                "excess_kurtosis": float(r.kurt()) if len(r) else float("nan"),
                "tail_ratio_95_05": tail_ratio,
                "worst_1d": float(r.min()) if len(r) else float("nan"),
                "worst_5d_avg": float(r.nsmallest(5).mean()) if len(r) >= 5 else float("nan"),
                "worst_10d_avg": float(r.nsmallest(10).mean()) if len(r) >= 10 else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("object").sort_index()

def tail_ratio(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    upper: float = 0.95,
    lower: float = 0.05,
) -> float:
    """Absolute upper/lower tail quantile ratio."""
    r = _to_numeric_series(returns, name="returns")
    q_low = float(r.quantile(float(lower)))
    q_high = float(r.quantile(float(upper)))
    return float(abs(q_high / q_low)) if abs(q_low) > 1e-12 else float("nan")


def worst_returns_summary(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    counts: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    """Compute average worst returns for several tail counts.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects.
    counts : sequence of int, default=(1, 5, 10)
        Number of worst observations to average.

    Returns
    -------
    pandas.DataFrame
        Table indexed by object with one ``worst_nd_avg`` column per count.
    """

    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        row: dict[str, Any] = {"object": name}
        for n in counts:
            k = int(n)
            row[f"worst_{k}d_avg"] = float(r.nsmallest(k).mean()) if len(r) >= k else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("object").sort_index()


__all__ = ["tail_ratio", "tail_shape_table", "worst_returns_summary"]
