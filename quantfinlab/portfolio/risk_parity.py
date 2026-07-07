from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import optimize

from quantfinlab.portfolio.covariance import make_psd


def _cap_series(index, w_max):
    idx = pd.Index(index)
    if isinstance(w_max, (pd.Series, Mapping)):
        caps = pd.Series(w_max, dtype=float).reindex(idx).fillna(1.0)
    else:
        caps = pd.Series(float(w_max), index=idx, dtype=float)
    if float(caps.sum()) < 1.0 - 1e-12:
        caps = pd.Series(max(float(caps.max()), 1.0 / len(idx)), index=idx, dtype=float)
    return caps.clip(lower=0.0)


def _clean_weights(values, index, *, w_min=0.0, w_max=1.0):
    idx = pd.Index(index)
    caps = _cap_series(idx, w_max)
    w = pd.Series(values, index=idx, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=float(w_min), upper=caps)
    if float(w.sum()) <= 1e-12:
        w = pd.Series(1.0 / len(idx), index=idx, dtype=float)
    else:
        w = w / float(w.sum())
    for _ in range(50):
        over = w > caps + 1e-12
        if not bool(over.any()):
            break
        extra = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        room = (caps[~over] - w[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        w.loc[room.index] += extra * room / float(room.sum())
    w = w.clip(lower=float(w_min), upper=caps)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(idx), index=idx)


def risk_contributions(weights, cov_ann):
    """Compute volatility risk contributions for a weighted portfolio.

    Parameters
    ----------
    weights : pandas.Series or mapping
        Portfolio weights indexed by asset.
    cov_ann : pandas.DataFrame or array-like
        Annualized covariance matrix aligned to the weight labels.

    Returns
    -------
    pandas.Series
        Volatility contribution of each asset.

    Notes
    -----
    The sum of the returned contributions is portfolio volatility. Divide by the
    sum to obtain fractional risk contributions.
    """

    w = pd.Series(weights, dtype=float)
    labels = w.index
    cov = pd.DataFrame(cov_ann, index=labels, columns=labels).reindex(index=labels, columns=labels).to_numpy(dtype=float)
    cov = make_psd(cov, eps=1e-10)
    vals = w.to_numpy(dtype=float)
    sigma_w = cov @ vals
    port_vol = math.sqrt(max(float(vals @ sigma_w), 1e-18))
    return pd.Series(vals * sigma_w / port_vol, index=labels, dtype=float)


def risk_contribution_table(weights, cov_ann):
    w = pd.Series(weights, dtype=float)
    rc = risk_contributions(w, cov_ann)
    total = float(rc.sum())
    return pd.DataFrame(
        {
            "weight": w.reindex(rc.index).values,
            "risk_contribution": rc.values,
            "percent_risk_contribution": rc.values / total if abs(total) > 1e-12 else np.nan,
        },
        index=rc.index,
    )


def equal_risk_contribution_weights(cov_ann, *, tickers=None, w_min=0.0, w_max=0.40):
    """Compute equal-risk-contribution portfolio weights.

    The optimizer minimizes squared deviations between asset risk contributions
    and equal risk budgets, subject to full investment and per-asset bounds.

    Parameters
    ----------
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    tickers : sequence of str, optional
        Asset labels.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.Series
        Long-only risk-parity weights indexed by asset.

    Notes
    -----
    If SLSQP fails, the function returns a cleaned equal-weight-like initial
    allocation under the same bounds.
    """

    cov = make_psd(np.asarray(cov_ann, dtype=float), eps=1e-10)
    n = cov.shape[0]
    labels = pd.Index(tickers if tickers is not None else [f"a{i}" for i in range(n)])
    caps = _cap_series(labels, w_max)
    x0 = _clean_weights(np.ones(n), labels, w_min=w_min, w_max=w_max).to_numpy(dtype=float)
    bounds = [(float(w_min), float(caps.iloc[i])) for i in range(n)]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}

    def objective(x):
        sigma_x = cov @ x
        vol = math.sqrt(max(float(x @ sigma_x), 1e-18))
        rc = x * sigma_x / vol
        return float(np.sum((rc - vol / n) ** 2))

    res = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 700, "ftol": 1e-12})
    if not res.success or not np.all(np.isfinite(res.x)):
        return pd.Series(x0, index=labels, dtype=float)
    return _clean_weights(res.x, labels, w_min=w_min, w_max=w_max)


def risk_parity_weight_frame(cache, rebalance_dates, *, cov_model, w_min=0.0, w_max=0.40):
    """Build a rebalance-date panel of equal-risk-contribution weights.

    Parameters
    ----------
    cache : mapping
        Rebalance-state cache containing tickers and covariance matrices.
    rebalance_dates : sequence
        Candidate rebalance dates.
    cov_model : str
        Covariance model key to use from each cached state.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.DataFrame
        Risk-parity weight frame indexed by rebalance date.

    Notes
    -----
    Dates missing from the cache are skipped.
    """

    rows = []
    for dt in [pd.Timestamp(x) for x in rebalance_dates if pd.Timestamp(x) in cache][:-1]:
        state = cache[pd.Timestamp(dt)]
        tickers = list(state.get("tickers", state["R_cov"].columns))
        cov = pd.DataFrame(state["cov_ann_map"][cov_model], index=tickers, columns=tickers).reindex(index=tickers, columns=tickers)
        w = equal_risk_contribution_weights(cov, tickers=tickers, w_min=w_min, w_max=w_max)
        rows.append(w.rename(pd.Timestamp(dt)))
    return pd.DataFrame(rows).fillna(0.0)


__all__ = [
    "equal_risk_contribution_weights",
    "risk_contribution_table",
    "risk_contributions",
    "risk_parity_weight_frame",
]
