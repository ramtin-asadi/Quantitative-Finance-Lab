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
    """Sample desired hedge betas on scheduled trading dates.

    The function forward-fills a beta path onto the backtest index and then samples
    the latest beta at each scheduled rebalance date.

    Parameters
    ----------
    beta : pandas.DataFrame
        Estimated hedge-ratio path.
    index : array-like
        Trading index used by the backtest.
    freq : str, default="w-fri"
        Pandas frequency string for scheduled beta updates.

    Returns
    -------
    pandas.DataFrame
        Beta table sampled on scheduled trading dates.

    Notes
    -----
    The function does not estimate betas; it only converts a daily or irregular
    desired-beta path into a tradable schedule.
    """

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
    """Apply a no-trade band to a desired beta path.

    The traded beta is updated only when a coefficient moves more than ``band`` away
    from the previously traded value. This reduces small beta changes that would
    otherwise generate unnecessary turnover.

    Parameters
    ----------
    beta : pandas.DataFrame
        Desired beta path.
    band : float, default=0.05
        Absolute movement threshold required to update a coefficient.

    Returns
    -------
    pandas.DataFrame
        Traded beta path with fewer updates.

    Notes
    -----
    The first non-missing beta row initializes the traded beta. Missing rows carry
    the previous traded value forward.
    """

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
    """Convert hedge betas into target and hedge portfolio weights.

    The target asset receives weight ``+1`` and each hedge asset receives weight
    ``-beta``. The output is useful for diagnostics and direct weight-based
    backtests of hedge books.

    Parameters
    ----------
    beta : pandas.DataFrame
        Hedge beta path with one column per hedge asset.
    r : rel
        Relationship defining target and hedge assets.
    tickers : list of str
        Full set of available tickers.

    Returns
    -------
    pandas.DataFrame
        Weight table indexed like ``beta`` and containing all requested tickers.

    Raises
    ------
    InputError
        If the target or hedge tickers are missing from ``tickers``.
    """

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
