from __future__ import annotations

from collections.abc import Mapping

import cvxpy as cp
import numpy as np
import pandas as pd


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
    floor = pd.Series(float(w_min), index=idx, dtype=float).clip(lower=0.0)
    w = pd.Series(values, index=idx, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = w.clip(lower=floor, upper=caps)
    if float(w.sum()) <= 1e-12:
        room = (caps - floor).clip(lower=0.0)
        w = floor + (1.0 - float(floor.sum())) * room / float(room.sum())
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
    w = w.clip(lower=floor, upper=caps)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(idx), index=idx)


def _solve(problem):
    installed = set(cp.installed_solvers())
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in installed:
            continue
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in {"optimal", "optimal_inaccurate"}:
                return problem.status
        except Exception:
            pass
    return str(problem.status)


def _dates(cache, rebalance_dates):
    return [pd.Timestamp(dt) for dt in rebalance_dates if pd.Timestamp(dt) in cache][:-1]


def historical_cvar_loss(returns, *, alpha=0.95):
    r = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return float("nan")
    losses = -r.to_numpy(dtype=float)
    var = float(np.quantile(losses, float(alpha)))
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else var


def portfolio_cvar_loss(returns, weights, *, alpha=0.95):
    r = pd.DataFrame(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if r.empty:
        return float("nan")
    w = pd.Series(weights, dtype=float).reindex(r.columns).fillna(0.0)
    return historical_cvar_loss(r @ w, alpha=alpha)


def min_cvar_weights(returns, *, alpha=0.95, w_min=0.0, w_max=0.40):
    r = pd.DataFrame(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    tickers = list(r.columns)
    t, n = r.shape
    caps = _cap_series(tickers, w_max).to_numpy(dtype=float)
    w = cp.Variable(n)
    eta = cp.Variable()
    u = cp.Variable(t)
    losses = -r.to_numpy(dtype=float) @ w
    cvar = eta + cp.sum(u) / ((1.0 - float(alpha)) * t)
    problem = cp.Problem(cp.Minimize(cvar), [cp.sum(w) == 1.0, w >= float(w_min), w <= caps, u >= 0.0, u >= losses - eta])
    status = _solve(problem)
    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return _clean_weights(np.ones(n), tickers, w_min=w_min, w_max=w_max)
    return _clean_weights(w.value, tickers, w_min=w_min, w_max=w_max)


def mean_cvar_weights(
    returns,
    mu_ann,
    *,
    reference="equal",
    cvar_budget=None,
    budget_scale=0.90,
    relax_scales=(1.00, 1.10),
    alpha=0.95,
    w_min=0.0,
    w_max=0.40,
):
    r = pd.DataFrame(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    tickers = list(r.columns)
    mu = pd.Series(mu_ann, dtype=float).reindex(tickers).fillna(0.0)
    if isinstance(reference, str) and reference == "equal":
        ref_w = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    else:
        ref_w = _clean_weights(reference, tickers, w_min=w_min, w_max=w_max)
    base_budget = portfolio_cvar_loss(r, ref_w, alpha=alpha) if cvar_budget is None else float(cvar_budget)
    scales = [float(budget_scale), *[float(x) for x in relax_scales]]
    caps = _cap_series(tickers, w_max).to_numpy(dtype=float)
    arr = r.to_numpy(dtype=float)
    t, n = arr.shape
    for scale in scales:
        budget = base_budget * scale if cvar_budget is None else base_budget
        w = cp.Variable(n)
        eta = cp.Variable()
        u = cp.Variable(t)
        losses = -arr @ w
        cvar = eta + cp.sum(u) / ((1.0 - float(alpha)) * t)
        problem = cp.Problem(cp.Maximize(mu.to_numpy(dtype=float) @ w), [cp.sum(w) == 1.0, w >= float(w_min), w <= caps, u >= 0.0, u >= losses - eta, cvar <= budget])
        status = _solve(problem)
        if w.value is not None and status in {"optimal", "optimal_inaccurate"}:
            return _clean_weights(w.value, tickers, w_min=w_min, w_max=w_max)
    return _clean_weights(ref_w, tickers, w_min=w_min, w_max=w_max)


def cvar_budget_path(returns, mu_ann, *, budget_scales=(0.80, 0.90, 1.00, 1.10, 1.25), alpha=0.95, w_min=0.0, w_max=0.40):
    r = pd.DataFrame(returns).dropna(how="any")
    tickers = list(r.columns)
    ref_w = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    base = portfolio_cvar_loss(r, ref_w, alpha=alpha)
    rows = []
    for scale in budget_scales:
        w = mean_cvar_weights(r, mu_ann, cvar_budget=float(scale) * base, budget_scale=1.0, relax_scales=(), alpha=alpha, w_min=w_min, w_max=w_max)
        rows.append(
            {
                "budget_scale": float(scale),
                "cvar_budget": float(scale) * base,
                "expected_return": float(pd.Series(mu_ann).reindex(tickers).fillna(0.0).to_numpy(dtype=float) @ w.reindex(tickers).values),
                "cvar_loss": portfolio_cvar_loss(r, w, alpha=alpha),
                "effective_n": float(1.0 / np.square(w.to_numpy(dtype=float)).sum()),
                "max_weight": float(w.max()),
            }
        )
    return pd.DataFrame(rows)


def min_cvar_weight_frame(cache, rebalance_dates, *, cov_model=None, alpha=0.95, w_min=0.0, w_max=0.40):
    rows = []
    prev = None
    for dt in _dates(cache, rebalance_dates):
        state = cache[pd.Timestamp(dt)]
        tickers = list(state.get("tickers", state["R_cov"].columns))
        r = state["R_cov"].reindex(columns=tickers).dropna(how="any")
        w = min_cvar_weights(r, alpha=alpha, w_min=w_min, w_max=w_max) if len(r) else prev
        if w is None:
            w = pd.Series(1.0 / len(tickers), index=tickers)
        w = _clean_weights(w, tickers, w_min=w_min, w_max=w_max)
        rows.append(w.rename(pd.Timestamp(dt)))
        prev = w
    return pd.DataFrame(rows).fillna(0.0)


def mean_cvar_weight_frame(cache, rebalance_dates, *, cov_model, mu_model, reference="equal", alpha=0.95, budget_scale=0.90, w_min=0.0, w_max=0.40):
    rows = []
    prev = None
    for dt in _dates(cache, rebalance_dates):
        state = cache[pd.Timestamp(dt)]
        tickers = list(state.get("tickers", state["R_cov"].columns))
        r = state["R_cov"].reindex(columns=tickers).dropna(how="any")
        mu = state["mu_ann_map"][cov_model][mu_model].reindex(tickers).fillna(0.0)
        ref = prev if not (isinstance(reference, str) and reference == "equal") and prev is not None else reference
        w = mean_cvar_weights(r, mu, reference=ref, budget_scale=budget_scale, alpha=alpha, w_min=w_min, w_max=w_max) if len(r) else prev
        if w is None:
            w = pd.Series(1.0 / len(tickers), index=tickers)
        w = _clean_weights(w, tickers, w_min=w_min, w_max=w_max)
        rows.append(w.rename(pd.Timestamp(dt)))
        prev = w
    return pd.DataFrame(rows).fillna(0.0)


__all__ = [
    "cvar_budget_path",
    "historical_cvar_loss",
    "mean_cvar_weight_frame",
    "mean_cvar_weights",
    "min_cvar_weight_frame",
    "min_cvar_weights",
    "portfolio_cvar_loss",
]
