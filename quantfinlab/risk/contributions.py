from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk.utils import _as_result_mapping, _as_state_mapping, _normalize_alpha


def vol_contribution(
    weights: pd.Series | Sequence[float] | np.ndarray,
    cov: np.ndarray,
    *,
    index: Sequence[str] | None = None,
) -> pd.Series:
    w = np.asarray(weights, dtype=float).reshape(-1)
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise InputError("cov must be a square matrix.")
    if S.shape[0] != w.shape[0]:
        raise InputError("weights length must match cov shape.")
    S = 0.5 * (S + S.T)
    m = S @ w
    var = float(w @ m)
    vol = math.sqrt(max(var, 1e-18))
    rc = (w * m) / vol
    labels = [str(i) for i in index] if index is not None else [f"a{i}" for i in range(len(w))]
    return pd.Series(rc, index=labels, dtype=float)

def scenario_es_contribution(
    returns_window: pd.DataFrame,
    weights: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> pd.Series:
    a = _normalize_alpha(alpha)
    if not isinstance(returns_window, pd.DataFrame) or returns_window.empty:
        raise InputError("returns_window must be a non-empty DataFrame.")
    x = (
        returns_window.copy()
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )
    if x.empty:
        raise InputError("returns_window is empty after cleaning.")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x.shape[1] != w.shape[0]:
        raise InputError("returns_window columns must match weights length.")
    rp = x.to_numpy(dtype=float) @ w
    q = float(np.quantile(rp, a))
    mask = rp <= q
    if not np.any(mask):
        mask[np.argmin(rp)] = True
    contrib = -np.mean(x.to_numpy(dtype=float)[mask] * w, axis=0)
    return pd.Series(contrib, index=[str(c) for c in x.columns], dtype=float)

def _resolve_state_date(cache: Mapping[Any, Any], dt: pd.Timestamp) -> pd.Timestamp | None:
    idx = pd.DatetimeIndex(pd.to_datetime(list(cache.keys()))).sort_values().unique()
    if len(idx) == 0:
        return None
    if dt in idx:
        return pd.Timestamp(dt)
    pos = int(idx.searchsorted(dt, side="right")) - 1
    if pos < 0:
        return None
    return pd.Timestamp(idx[pos])

def _weights_state_from_spec(
    spec: Mapping[str, Any],
    *,
    date: pd.Timestamp | None = None,
) -> tuple[pd.Series, Mapping[str, Any], pd.Timestamp]:
    result = spec.get("backtest", spec.get("result"))
    cache = spec.get("state_cache", spec.get("cache"))
    if result is None or cache is None:
        raise InputError("Portfolio spec requires 'backtest' (or 'result') and 'state_cache' (or 'cache').")
    res_map = _as_result_mapping(result)
    wdf = pd.DataFrame(res_map.get("weights"))
    if wdf.empty:
        raise InputError("Portfolio result has empty weights.")
    wdf.index = pd.to_datetime(wdf.index)
    dt = pd.Timestamp(date) if date is not None else pd.Timestamp(wdf.index[-1])
    st_dt = _resolve_state_date(cache, dt)
    if st_dt is None:
        raise InputError("Could not resolve state date from state cache.")
    state = _as_state_mapping(cache[st_dt])
    tickers = [str(t) for t in state.get("tickers", [])]
    if not tickers:
        raise InputError("State is missing tickers.")
    if dt not in wdf.index:
        pos = int(wdf.index.searchsorted(dt, side="right")) - 1
        if pos < 0:
            raise InputError("No weights available on or before requested date.")
        dt = pd.Timestamp(wdf.index[pos])
    w = wdf.loc[dt].reindex(tickers).fillna(0.0).astype(float)
    s = float(w.sum())
    if not np.isfinite(s) or abs(s) <= 1e-12:
        raise InputError("Resolved weights sum to zero.")
    w = w / s
    return w, state, pd.Timestamp(dt)

def portfolio_contribution_snapshot(
    portfolio_spec: Mapping[str, Any],
    *,
    cov_key: str | None = None,
    es_alpha: float = 0.05,
    date: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Return (volatility contribution, scenario-ES contribution) for one portfolio snapshot.

    If a returns window is unavailable in the state cache, ES contribution is returned
    as an all-NaN series (same index as vol contribution) instead of raising.
    """
    w, state, _ = _weights_state_from_spec(portfolio_spec, date=date)
    ck = str(cov_key or portfolio_spec.get("cov_key", "ledoitwolf"))
    cov_map = state.get("cov_ann_map", {})
    if ck not in cov_map:
        low = {str(k).lower(): k for k in cov_map}
        if ck.lower() in low:
            ck = low[ck.lower()]
        else:
            raise InputError(f"cov_key {ck!r} not found in state covariance map.")
    cov = np.asarray(cov_map[ck], dtype=float)
    vol_rc = vol_contribution(w.to_numpy(dtype=float), cov, index=w.index).sort_values(ascending=False)

    window = state.get("R_cov", state.get("window"))
    if window is None:
        window = state.get("R_mu")
    if window is None:
        meta = state.get("metadata")
        if isinstance(meta, Mapping):
            window = meta.get("R_cov", meta.get("window"))
            if window is None:
                window = meta.get("R_mu")
    if window is None:
        es_rc = pd.Series(np.nan, index=vol_rc.index, dtype=float)
        return vol_rc, es_rc
    if not isinstance(window, pd.DataFrame):
        window = pd.DataFrame(window, columns=w.index)
    x = window.reindex(columns=w.index)
    try:
        es_rc = scenario_es_contribution(x, w.to_numpy(dtype=float), alpha=es_alpha).sort_values(ascending=False)
    except Exception:
        es_rc = pd.Series(np.nan, index=vol_rc.index, dtype=float)
    return vol_rc, es_rc

def attribution_tables(
    portfolios: Mapping[str, Any],
    *,
    es_alpha: float = 0.05,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not portfolios:
        raise InputError("portfolios cannot be empty.")
    if top_k <= 0:
        raise InputError("top_k must be positive.")
    vol_map: dict[str, pd.Series] = {}
    es_map: dict[str, pd.Series] = {}
    overlap_rows: list[dict[str, Any]] = []
    for pname, raw in portfolios.items():
        if isinstance(raw, Mapping):
            spec = raw
        elif isinstance(raw, tuple) and len(raw) >= 2:
            spec = {"backtest": raw[0], "state_cache": raw[1], "cov_key": raw[2] if len(raw) > 2 else None}
        else:
            raise InputError("Each portfolio entry must be a mapping or tuple.")
        vol_rc, es_rc = portfolio_contribution_snapshot(spec, es_alpha=es_alpha)
        vol_map[str(pname)] = vol_rc
        es_map[str(pname)] = es_rc
        es_rank = es_rc.dropna()
        overlap = len(set(vol_rc.head(int(top_k)).index).intersection(set(es_rank.head(int(top_k)).index)))
        overlap_rows.append({"portfolio": str(pname), f"top{int(top_k)}_overlap_count": int(overlap)})
    vol_tbl = pd.DataFrame.from_dict(vol_map, orient="index").sort_index(axis=0)
    es_tbl = pd.DataFrame.from_dict(es_map, orient="index").sort_index(axis=0)
    overlap_tbl = pd.DataFrame(overlap_rows).set_index("portfolio").sort_index()
    return vol_tbl, es_tbl, overlap_tbl

__all__ = [
    "attribution_tables",
    "portfolio_contribution_snapshot",
    "scenario_es_contribution",
    "vol_contribution",
]
