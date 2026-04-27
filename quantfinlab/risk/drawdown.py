from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.core import InputError
from quantfinlab.risk.utils import _coerce_objects, _to_numeric_series


def drawdown_series(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> pd.Series:
    x = _to_numeric_series(returns_or_nav, name="returns_or_nav")
    kind = str(input_kind).strip().lower()
    if kind in {"returns", "ret", "r"}:
        nav = (1.0 + x).cumprod()
    elif kind in {"nav", "equity", "value"}:
        nav = x
    else:
        raise InputError("input_kind must be either 'returns' or 'nav'.")
    return nav / nav.cummax() - 1.0

def ulcer_index(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> float:
    dd = drawdown_series(returns_or_nav, input_kind=input_kind)
    return float(np.sqrt(np.mean(np.square(dd.to_numpy(dtype=float))))) if len(dd) else float("nan")

def drawdown_episodes(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> pd.DataFrame:
    dd = drawdown_series(returns_or_nav, input_kind=input_kind)
    if dd.empty:
        return pd.DataFrame(columns=["start", "end", "depth", "duration"])
    in_dd = False
    start_i = 0
    rows: list[tuple[Any, Any, float, int]] = []
    vals = dd.to_numpy(dtype=float)
    for i, v in enumerate(vals):
        if v < 0 and not in_dd:
            in_dd = True
            start_i = i
        if v >= -1e-15 and in_dd:
            seg = dd.iloc[start_i:i]
            if len(seg):
                rows.append((seg.index[0], seg.index[-1], float(seg.min()), len(seg)))
            in_dd = False
    if in_dd:
        seg = dd.iloc[start_i:]
        if len(seg):
            rows.append((seg.index[0], seg.index[-1], float(seg.min()), len(seg)))
    return pd.DataFrame(rows, columns=["start", "end", "depth", "duration"])

def avg_recovery_time(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> float:
    dd = drawdown_series(returns_or_nav, input_kind=input_kind)
    if dd.empty:
        return float("nan")
    rec_times: list[int] = []
    in_dd = False
    start = 0
    for i, v in enumerate(dd.to_numpy(dtype=float)):
        if v < 0 and not in_dd:
            in_dd = True
            start = i
        if v >= -1e-15 and in_dd:
            rec_times.append(i - start)
            in_dd = False
    return float(np.mean(rec_times)) if rec_times else float("nan")

def drawdown_summary_table(objects: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        dd = drawdown_series(r, input_kind="returns")
        ep = drawdown_episodes(r, input_kind="returns")
        rows.append(
            {
                "object": name,
                "max_dd": float(dd.min()) if len(dd) else float("nan"),
                "longest_dd_days": int(ep["duration"].max()) if len(ep) else 0,
                "avg_recovery_days": avg_recovery_time(r, input_kind="returns"),
                "ulcer_index": ulcer_index(r, input_kind="returns"),
            }
        )
    return pd.DataFrame(rows).set_index("object").sort_index()

def drawdown_episodes_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    top_n: int = 1,
) -> pd.DataFrame:
    if top_n <= 0:
        raise InputError("top_n must be positive.")
    obj = _coerce_objects(objects)
    rows: list[pd.DataFrame] = []
    for name, r in obj.items():
        ep = drawdown_episodes(r, input_kind="returns").sort_values("depth")
        ep = ep.head(int(top_n)).copy()
        if ep.empty:
            continue
        ep.insert(0, "object", name)
        rows.append(ep)
    if not rows:
        return pd.DataFrame(columns=["object", "start", "end", "depth", "duration"])
    return pd.concat(rows, axis=0).reset_index(drop=True)

__all__ = [
    "avg_recovery_time",
    "drawdown_episodes",
    "drawdown_episodes_table",
    "drawdown_series",
    "drawdown_summary_table",
    "ulcer_index",
]
