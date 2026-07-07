from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk.performance import nav_series
from quantfinlab.risk.utils import _coerce_objects, _to_datetime_if_possible


def _window_returns(
    series: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    return series.loc[(series.index >= start) & (series.index <= end)].dropna()

def stress_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]],
    worst_only: bool = True,
    worst_by: Literal["cum_return", "max_dd", "worst_day", "worst_week"] = "cum_return",
) -> pd.DataFrame:
    """Evaluate return objects across named stress windows.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects.
    windows : mapping
        Mapping from stress-window name to ``(start, end)`` date tuple.
    worst_only : bool, default=True
        If true, return one worst window per object. If false, return all
        object-window rows.
    worst_by : {"cum_return", "max_dd", "worst_day", "worst_week"}, default="cum_return"
        Metric used to choose the worst row when ``worst_only=True``.

    Returns
    -------
    pandas.DataFrame
        Stress table with cumulative return, max drawdown, worst daily return, and
        worst weekly return.

    Raises
    ------
    InputError
        If windows are empty or ``worst_by`` is invalid.
    """

    if not windows:
        raise InputError("windows cannot be empty.")
    valid_worst_by = {"cum_return", "max_dd", "worst_day", "worst_week"}
    if worst_by not in valid_worst_by:
        raise InputError(f"worst_by must be one of {sorted(valid_worst_by)}.")
    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for wname, (start, end) in windows.items():
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if e < s:
            s, e = e, s
        for name, r0 in obj.items():
            r = _to_datetime_if_possible(r0)
            x = _window_returns(r, start=s, end=e)
            if len(x) == 0:
                continue
            nav = nav_series(x)
            dd = nav / nav.cummax() - 1.0
            has_dates = isinstance(x.index, pd.DatetimeIndex)
            worst_week = x.resample("W-FRI").sum().min() if has_dates and len(x) > 5 else float("nan")
            rows.append(
                {
                    "window": str(wname),
                    "object": str(name),
                    "cum_return": float(nav.iloc[-1] - 1.0),
                    "max_dd": float(dd.min()),
                    "worst_day": float(x.min()),
                    "worst_week": float(worst_week) if np.isfinite(worst_week) else float("nan"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["object", "cum_return", "max_dd", "worst_day", "worst_week"])
    out = pd.DataFrame(rows)
    if not bool(worst_only):
        return out.set_index("window").sort_values(["window", "object"])

    # One worst stress scenario row per object; keep the source window for context.
    out = out.sort_values([worst_by, "window"]).groupby("object", as_index=False).first()
    return out.set_index("object").sort_index()


historical_stress_table = stress_table

__all__ = ["historical_stress_table", "stress_table"]
