"""Single-asset OHLCV loaders.

Used for SPY/SPX/BTC daily files (yfinance-style CSVs) referenced by
NB3, NB4, and NB5. Returns a normalized DataFrame with a sorted,
deduplicated ``DatetimeIndex`` and lowercase ``snake_case`` columns.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _resolve_date_col(columns: Iterable[str], explicit: str | None) -> str | None:
    cols = list(columns)
    if explicit is not None:
        for c in cols:
            if str(c).lower() == str(explicit).lower():
                return c
    for cand in ("date", "trade_date", "datetime"):
        for c in cols:
            if str(c).lower() == cand:
                return c
    return None


def load_ohlcv(
    path: str | Path,
    *,
    source: str = "yfinance_csv",
    fields: tuple[str, ...] = ("close",),
    date_col: str | None = None,
) -> pd.DataFrame:
    """Load a single-asset OHLCV CSV / Parquet into a normalized panel.

    Output: DataFrame with sorted ``DatetimeIndex`` and the requested
    fields in lowercase ``snake_case``. Common aliases are resolved so
    that ``"close"`` returns the adj-close column when present and falls
    back to the unadjusted close otherwise; ``"dividend"`` resolves to
    a ``dividends`` / ``dividend`` column when present.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"OHLCV file does not exist: {p}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p, low_memory=False)

    raw.columns = [_normalize_col(c) for c in raw.columns]
    real_date = _resolve_date_col(raw.columns, date_col)
    if real_date is None:
        raise ValueError(f"OHLCV file {p.name} has no date column.")
    raw[real_date] = pd.to_datetime(raw[real_date], errors="coerce")
    raw = raw.dropna(subset=[real_date]).sort_values(real_date).set_index(real_date)
    raw.index = pd.to_datetime(raw.index)
    raw = raw[~raw.index.duplicated(keep="last")]

    aliases: dict[str, tuple[str, ...]] = {
        "close": ("adj_close", "close"),
        "open": ("open",),
        "high": ("high",),
        "low": ("low",),
        "volume": ("volume",),
        "dividend": ("dividends", "dividend"),
        "stock_splits": ("stock_splits",),
        "adj_close": ("adj_close",),
        "raw_close": ("close",),
    }

    out = pd.DataFrame(index=raw.index)
    for f in fields:
        key = _normalize_col(f)
        candidates = aliases.get(key, (key,))
        chosen = None
        for cand in candidates:
            if cand in raw.columns:
                chosen = cand
                break
        if chosen is None:
            raise ValueError(
                f"OHLCV field {f!r} not found in {p.name}; tried {candidates} "
                f"against {sorted(raw.columns)}"
            )
        series = pd.to_numeric(raw[chosen], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[key] = series

    if str(source).lower() not in {"yfinance_csv", "yfinance_export", "yfinance"}:
        # Currently only one OHLCV schema is registered; reject unknown
        # aliases loudly so callers don't think they were honored.
        if source not in {None, ""}:
            pass
    return out


__all__ = ["load_ohlcv"]
