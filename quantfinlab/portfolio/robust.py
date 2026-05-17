from __future__ import annotations

from collections.abc import Mapping
import math

import cvxpy as cp
import numpy as np
import pandas as pd

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


def psd_sqrt(mat, eps=1e-10):
    arr = make_psd(np.asarray(mat, dtype=float), eps=eps)
    vals, vecs = np.linalg.eigh(arr)
    vals = np.maximum(vals, eps)
    out = (vecs * np.sqrt(vals)) @ vecs.T
    return 0.5 * (out + out.T)


def _inputs(mu_ann, cov_ann):
    mu = pd.Series(mu_ann, dtype=float)
    labels = pd.Index(mu.index)
    cov = pd.DataFrame(cov_ann, index=labels, columns=labels).reindex(index=labels, columns=labels)
    return labels, mu, make_psd(cov.to_numpy(dtype=float), eps=1e-10)


def box_robust_mv_weights(mu_ann, cov_ann, *, n_mu_obs, radius=0.25, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    labels, mu, cov = _inputs(mu_ann, cov_ann)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0)) / math.sqrt(max(int(n_mu_obs), 1))
    robust_mu = mu.to_numpy(dtype=float) - float(radius) * se
    caps = _cap_series(labels, w_max).to_numpy(dtype=float)
    w = cp.Variable(len(labels))
    problem = cp.Problem(cp.Maximize(robust_mu @ w - 0.5 * float(mv_lambda) * cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1.0, w >= float(w_min), w <= caps])
    status = _solve(problem)
    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return _clean_weights(np.ones(len(labels)), labels, w_min=w_min, w_max=w_max)
    return _clean_weights(w.value, labels, w_min=w_min, w_max=w_max)


def ellipsoid_robust_mv_weights(mu_ann, cov_ann, *, n_mu_obs, radius=0.25, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    labels, mu, cov = _inputs(mu_ann, cov_ann)
    omega_sqrt = psd_sqrt(cov / max(int(n_mu_obs), 1))
    caps = _cap_series(labels, w_max).to_numpy(dtype=float)
    w = cp.Variable(len(labels))
    penalty = float(radius) * cp.norm(omega_sqrt @ w, 2)
    problem = cp.Problem(cp.Maximize(mu.to_numpy(dtype=float) @ w - penalty - 0.5 * float(mv_lambda) * cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1.0, w >= float(w_min), w <= caps])
    status = _solve(problem)
    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return _clean_weights(np.ones(len(labels)), labels, w_min=w_min, w_max=w_max)
    return _clean_weights(w.value, labels, w_min=w_min, w_max=w_max)


def wasserstein_drmv_weights(mu_ann, cov_ann, *, n_mu_obs, radius=0.10, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    labels, mu, cov = _inputs(mu_ann, cov_ann)
    n = len(labels)
    sqrt_delta = float(radius) * math.sqrt(float(np.trace(cov)) / n) / math.sqrt(max(int(n_mu_obs), 1))
    caps = _cap_series(labels, w_max).to_numpy(dtype=float)
    w = cp.Variable(n)
    penalty = sqrt_delta * cp.norm(w, 2)
    problem = cp.Problem(cp.Maximize(mu.to_numpy(dtype=float) @ w - penalty - 0.5 * float(mv_lambda) * cp.quad_form(w, cp.psd_wrap(cov))), [cp.sum(w) == 1.0, w >= float(w_min), w <= caps])
    status = _solve(problem)
    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return _clean_weights(np.ones(n), labels, w_min=w_min, w_max=w_max)
    return _clean_weights(w.value, labels, w_min=w_min, w_max=w_max)


def robust_radius_path(model, mu_ann, cov_ann, *, n_mu_obs, radii, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    labels, mu, cov = _inputs(mu_ann, cov_ann)
    mu_arr = mu.to_numpy(dtype=float)
    omega_sqrt = psd_sqrt(cov / max(int(n_mu_obs), 1))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0)) / math.sqrt(max(int(n_mu_obs), 1))
    rows = []
    for radius in radii:
        model_key = str(model).lower()
        if model_key == "box":
            w = box_robust_mv_weights(mu_ann, cov_ann, n_mu_obs=n_mu_obs, radius=radius, mv_lambda=mv_lambda, w_min=w_min, w_max=w_max)
        elif model_key == "ellipsoid":
            w = ellipsoid_robust_mv_weights(mu_ann, cov_ann, n_mu_obs=n_mu_obs, radius=radius, mv_lambda=mv_lambda, w_min=w_min, w_max=w_max)
        elif model_key == "wasserstein":
            w = wasserstein_drmv_weights(mu_ann, cov_ann, n_mu_obs=n_mu_obs, radius=radius, mv_lambda=mv_lambda, w_min=w_min, w_max=w_max)
        else:
            raise ValueError("model must be 'box', 'ellipsoid', or 'wasserstein'.")
        w_arr = pd.Series(w, dtype=float).reindex(labels).fillna(0.0).to_numpy(dtype=float)
        if model_key == "ellipsoid":
            penalty = float(radius) * float(np.linalg.norm(omega_sqrt @ w_arr))
        elif model_key == "wasserstein":
            penalty = float(radius) * math.sqrt(float(np.trace(cov)) / len(labels)) / math.sqrt(max(int(n_mu_obs), 1)) * float(np.linalg.norm(w_arr))
        else:
            penalty = float(radius) * float(se @ w_arr)
        empirical_return = float(mu_arr @ w_arr)
        volatility = float(math.sqrt(max(w_arr @ cov @ w_arr, 0.0)))
        row = {
            "empirical_return": empirical_return,
            "penalty": penalty,
            "robust_return": empirical_return - penalty,
            "volatility": volatility,
            "effective_n": float(1.0 / np.square(w_arr).sum()) if np.square(w_arr).sum() > 0 else np.nan,
        }
        row["radius"] = float(radius)
        row["max_weight"] = float(w.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _frame(cache, rebalance_dates, *, cov_model, mu_model, func, kwargs):
    rows = []
    for dt in [pd.Timestamp(x) for x in rebalance_dates if pd.Timestamp(x) in cache][:-1]:
        state = cache[pd.Timestamp(dt)]
        tickers = list(state.get("tickers", state["R_cov"].columns))
        mu = state["mu_ann_map"][cov_model][mu_model].reindex(tickers).fillna(0.0)
        cov = pd.DataFrame(state["cov_ann_map"][cov_model], index=tickers, columns=tickers)
        w = func(mu, cov, n_mu_obs=len(state["R_mu"]), **kwargs)
        rows.append(w.rename(pd.Timestamp(dt)))
    return pd.DataFrame(rows).fillna(0.0)


def box_robust_weight_frame(cache, rebalance_dates, *, cov_model, mu_model, radius=0.25, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    return _frame(cache, rebalance_dates, cov_model=cov_model, mu_model=mu_model, func=box_robust_mv_weights, kwargs={"radius": radius, "mv_lambda": mv_lambda, "w_min": w_min, "w_max": w_max})


def ellipsoid_robust_weight_frame(cache, rebalance_dates, *, cov_model, mu_model, radius=0.25, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    return _frame(cache, rebalance_dates, cov_model=cov_model, mu_model=mu_model, func=ellipsoid_robust_mv_weights, kwargs={"radius": radius, "mv_lambda": mv_lambda, "w_min": w_min, "w_max": w_max})


def wasserstein_weight_frame(cache, rebalance_dates, *, cov_model, mu_model, radius=0.10, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    return _frame(cache, rebalance_dates, cov_model=cov_model, mu_model=mu_model, func=wasserstein_drmv_weights, kwargs={"radius": radius, "mv_lambda": mv_lambda, "w_min": w_min, "w_max": w_max})


__all__ = [
    "box_robust_mv_weights",
    "box_robust_weight_frame",
    "ellipsoid_robust_mv_weights",
    "ellipsoid_robust_weight_frame",
    "psd_sqrt",
    "robust_radius_path",
    "wasserstein_drmv_weights",
    "wasserstein_weight_frame",
]
