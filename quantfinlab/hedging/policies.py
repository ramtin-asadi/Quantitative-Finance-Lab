from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.hedging.relations import rel


def _clean_index(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if len(idx) == 0:
        raise InputError("index is empty.")
    return idx


def _clean_beta(beta: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(beta, pd.DataFrame):
        raise InputError("beta must be a pandas DataFrame.")
    out = beta.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def rebalance_beta(beta: pd.DataFrame, index, *, freq: str = "w-fri") -> pd.DataFrame:
    """Sample desired beta on scheduled trading dates from the backtest index."""
    idx = _clean_index(index)
    b = _clean_beta(beta)
    all_idx = idx.union(pd.DatetimeIndex(b.index))
    daily = b.reindex(all_idx).sort_index().ffill().reindex(idx)
    dates = pd.Series(idx, index=idx).groupby(pd.Grouper(freq=str(freq).upper())).last().dropna()
    if not len(dates):
        return pd.DataFrame(columns=daily.columns, dtype=float)
    dts = pd.DatetimeIndex(dates.values)
    out = daily.loc[dts].dropna(how="all")
    return out


def band_beta(beta: pd.DataFrame, *, band: float = 0.05) -> pd.DataFrame:
    """Update traded beta only when a coefficient moves outside the no-trade band."""
    b = _clean_beta(beta)
    out = pd.DataFrame(np.nan, index=b.index, columns=b.columns, dtype=float)
    prev = pd.Series(np.nan, index=b.columns, dtype=float)
    threshold = max(float(band), 0.0)
    for dt, row in b.iterrows():
        x = row.astype(float)
        if x.notna().sum() == 0:
            out.loc[dt] = prev
            continue
        if prev.notna().sum() == 0:
            prev = x.copy()
        else:
            update = (x - prev).abs() > threshold
            update = update & x.notna()
            prev.loc[update] = x.loc[update]
        out.loc[dt] = prev
    return out


def target_w(index, r: rel, tickers: list[str]) -> pd.DataFrame:
    """Target-only weight schedule."""
    idx = _clean_index(index)
    cols = [str(c).strip().lower() for c in tickers]
    if r.target not in cols:
        raise InputError(f"{r.target} is not in tickers.")
    out = pd.DataFrame(0.0, index=idx[:1], columns=cols, dtype=float)
    out.loc[:, r.target] = 1.0
    return out


def beta_to_w(beta: pd.DataFrame, r: rel, tickers: list[str]) -> pd.DataFrame:
    """Convert hedge betas to target +1 and hedge -beta weights."""
    b = _clean_beta(beta)
    cols = [str(c).strip().lower() for c in tickers]
    missing = [c for c in r.assets if c not in cols]
    if missing:
        raise InputError(f"Missing tickers for {r.name}: {missing}")
    out = pd.DataFrame(0.0, index=b.index, columns=cols, dtype=float)
    out.loc[:, r.target] = 1.0
    for h in r.hedges:
        if h in b.columns:
            out.loc[:, h] = -b[h].fillna(0.0)
    return out


__all__ = ["band_beta", "beta_to_w", "rebalance_beta", "target_w"]
