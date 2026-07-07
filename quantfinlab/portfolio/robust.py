from __future__ import annotations

import math
from collections.abc import Mapping

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


def _wasserstein_epsilon(cov, *, n_mu_obs, radius, radius_scale="avg_vol"):
    cov_arr = np.asarray(cov, dtype=float)
    n = max(int(cov_arr.shape[0]), 1)
    trace = max(float(np.trace(cov_arr)), 0.0)
    key = str(radius_scale).lower().replace("-", "_").replace(" ", "_")
    if key in {"avg", "avg_vol", "average_vol", "per_asset"}:
        scale = math.sqrt(trace / n)
    elif key in {"trace", "total", "total_vol"}:
        scale = math.sqrt(trace)
    elif key in {"unit", "raw"}:
        scale = 1.0
    else:
        raise ValueError("radius_scale must be 'avg_vol', 'trace', or 'unit'.")
    return float(radius) * scale / math.sqrt(max(int(n_mu_obs), 1))


def box_robust_mv_weights(mu_ann, cov_ann, *, n_mu_obs, radius=0.25, mv_lambda=3.0, w_min=0.0, w_max=0.40):
    """Compute box-uncertainty robust mean-variance weights.

    Expected returns are adjusted downward by a multiple of their standard error,
    then a constrained mean-variance problem is solved using the conservative
    return vector.

    Parameters
    ----------
    mu_ann : array-like or pandas.Series
        Annualized expected returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    n_mu_obs : int
        Number of observations used to estimate expected returns.
    radius : float, default=0.25
        Size of the box uncertainty set in standard-error units.
    mv_lambda : float, default=3.0
        Risk-aversion parameter.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float or mapping, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.Series
        Robust mean-variance weights.

    Notes
    -----
    If the optimization fails, a cleaned equal-weight portfolio is returned.
    """

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


def wasserstein_drmv_weights(
    mu_ann,
    cov_ann,
    *,
    n_mu_obs,
    radius=1.0,
    mv_lambda=1.5,
    radius_scale="avg_vol",
    worst_case_variance=True,
    w_min=0.0,
    w_max=0.40,
):
    """Compute Wasserstein distributionally robust mean-variance weights.

    The objective penalizes expected return by a Wasserstein uncertainty radius
    and can optionally inflate variance using a worst-case volatility term.

    Parameters
    ----------
    mu_ann : array-like or pandas.Series
        Annualized expected returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    n_mu_obs : int
        Number of observations used to estimate expected returns.
    radius : float, default=1.0
        Distributional uncertainty radius.
    mv_lambda : float, default=1.5
        Risk-aversion parameter.
    radius_scale : str, default="avg_vol"
        Rule used to scale the Wasserstein radius.
    worst_case_variance : bool, default=True
        Whether to use worst-case volatility in the risk penalty.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float or mapping, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.Series
        Distributionally robust portfolio weights.

    Notes
    -----
    The formulation discourages large fragile exposures by taxing the L2 norm of
    the weight vector under mean uncertainty.
    """

    labels, mu, cov = _inputs(mu_ann, cov_ann)
    n = len(labels)
    epsilon = _wasserstein_epsilon(cov, n_mu_obs=n_mu_obs, radius=radius, radius_scale=radius_scale)
    caps = _cap_series(labels, w_max).to_numpy(dtype=float)
    w = cp.Variable(n)
    mean_tax = epsilon * cp.norm(w, 2)
    if worst_case_variance:
        empirical_vol = cp.norm(psd_sqrt(cov) @ w, 2)
        risk_penalty = 0.5 * float(mv_lambda) * cp.square(empirical_vol + mean_tax)
    else:
        risk_penalty = 0.5 * float(mv_lambda) * cp.quad_form(w, cp.psd_wrap(cov))
    problem = cp.Problem(cp.Maximize(mu.to_numpy(dtype=float) @ w - mean_tax - risk_penalty), [cp.sum(w) == 1.0, w >= float(w_min), w <= caps])
    status = _solve(problem)
    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return _clean_weights(np.ones(n), labels, w_min=w_min, w_max=w_max)
    return _clean_weights(w.value, labels, w_min=w_min, w_max=w_max)


def robust_radius_path(
    model,
    mu_ann,
    cov_ann,
    *,
    n_mu_obs,
    radii,
    mv_lambda=3.0,
    radius_scale="avg_vol",
    worst_case_variance=True,
    w_min=0.0,
    w_max=0.40,
):
    """Evaluate robust-portfolio behavior across uncertainty radii.

    For each radius, the function builds robust weights using the selected model
    and reports empirical return, robust penalty, robust return, volatility,
    risk penalty, objective value, effective number of holdings, and maximum
    weight.

    Parameters
    ----------
    model : {"box", "ellipsoid", "wasserstein"}
        Robust optimization model.
    mu_ann : array-like or pandas.Series
        Annualized expected returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    n_mu_obs : int
        Number of observations used to estimate expected returns.
    radii : sequence of float
        Uncertainty radii to evaluate.
    mv_lambda : float, default=3.0
        Risk-aversion parameter.
    radius_scale : str, default="avg_vol"
        Scaling rule for Wasserstein radius.
    worst_case_variance : bool, default=True
        Whether Wasserstein portfolios use worst-case volatility.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float or mapping, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.DataFrame
        Radius-path diagnostic table.

    Raises
    ------
    ValueError
        If ``model`` is not one of the supported robust models.
    """

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
            w = wasserstein_drmv_weights(mu_ann, cov_ann, n_mu_obs=n_mu_obs, radius=radius, mv_lambda=mv_lambda, radius_scale=radius_scale, worst_case_variance=worst_case_variance, w_min=w_min, w_max=w_max)
        else:
            raise ValueError("model must be 'box', 'ellipsoid', or 'wasserstein'.")
        w_arr = pd.Series(w, dtype=float).reindex(labels).fillna(0.0).to_numpy(dtype=float)
        if model_key == "ellipsoid":
            penalty = float(radius) * float(np.linalg.norm(omega_sqrt @ w_arr))
        elif model_key == "wasserstein":
            epsilon = _wasserstein_epsilon(cov, n_mu_obs=n_mu_obs, radius=radius, radius_scale=radius_scale)
            penalty = epsilon * float(np.linalg.norm(w_arr))
        else:
            penalty = float(radius) * float(se @ w_arr)
        empirical_return = float(mu_arr @ w_arr)
        volatility = float(math.sqrt(max(w_arr @ cov @ w_arr, 0.0)))
        robust_volatility = volatility + penalty if model_key == "wasserstein" else np.nan
        risk_penalty = 0.5 * float(mv_lambda) * ((robust_volatility if worst_case_variance and model_key == "wasserstein" else volatility) ** 2)
        row = {
            "empirical_return": empirical_return,
            "penalty": penalty,
            "robust_return": empirical_return - penalty,
            "volatility": volatility,
            "robust_volatility": robust_volatility,
            "risk_penalty": risk_penalty,
            "objective": empirical_return - penalty - risk_penalty,
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


def wasserstein_weight_frame(cache, rebalance_dates, *, cov_model, mu_model, radius=1.0, mv_lambda=1.5, radius_scale="avg_vol", worst_case_variance=True, w_min=0.0, w_max=0.40):
    return _frame(cache, rebalance_dates, cov_model=cov_model, mu_model=mu_model, func=wasserstein_drmv_weights, kwargs={"radius": radius, "mv_lambda": mv_lambda, "radius_scale": radius_scale, "worst_case_variance": worst_case_variance, "w_min": w_min, "w_max": w_max})


def robust_weight_frames(
    cache,
    rebalance_dates,
    *,
    cov_model="LedoitWolf",
    mu_model="Momentum",
    box_radius=0.25,
    ellipsoid_radius=0.10,
    wasserstein_radius=1.0,
    mv_lambda=2.0,
    wasserstein_cov_model=None,
    wasserstein_mu_model=None,
    wasserstein_mv_lambda=1.5,
    wasserstein_radius_scale="avg_vol",
    wasserstein_worst_case_variance=True,
    w_min=0.0,
    w_max=0.40,
):
    """Build robust mean-variance weight frames for multiple robust models.

    The function constructs Box Robust MV, Ellipsoid Robust MV, and Wasserstein
    DRMV weight frames from a rebalance cache using explicit covariance and
    expected-return model choices.

    Parameters
    ----------
    cache : mapping
        Rebalance-state cache.
    rebalance_dates : sequence
        Candidate rebalance dates.
    cov_model : str, default="LedoitWolf"
        Covariance model used for box and ellipsoid robust portfolios.
    mu_model : str, default="Momentum"
        Expected-return model used for box and ellipsoid robust portfolios.
    box_radius : float, default=0.25
        Box uncertainty radius.
    ellipsoid_radius : float, default=0.10
        Ellipsoid uncertainty radius.
    wasserstein_radius : float, default=1.0
        Wasserstein uncertainty radius.
    mv_lambda : float, default=2.0
        Risk-aversion parameter for box and ellipsoid portfolios.
    wasserstein_cov_model : str, optional
        Covariance model override for Wasserstein DRMV.
    wasserstein_mu_model : str, optional
        Expected-return model override for Wasserstein DRMV.
    wasserstein_mv_lambda : float, optional
        Risk-aversion override for Wasserstein DRMV.
    wasserstein_radius_scale : str, default="avg_vol"
        Scaling rule for Wasserstein radius.
    wasserstein_worst_case_variance : bool, default=True
        Whether Wasserstein DRMV uses worst-case variance.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    dict
        Mapping from strategy name to weight frame.

    Notes
    -----
    This helper is designed for side-by-side robust-portfolio comparisons.
    """

    w_cov_model = cov_model if wasserstein_cov_model is None else wasserstein_cov_model
    w_mu_model = mu_model if wasserstein_mu_model is None else wasserstein_mu_model
    w_lambda = mv_lambda if wasserstein_mv_lambda is None else wasserstein_mv_lambda
    return {
        "Box Robust MV": box_robust_weight_frame(
            cache,
            rebalance_dates,
            cov_model=cov_model,
            mu_model=mu_model,
            radius=box_radius,
            mv_lambda=mv_lambda,
            w_min=w_min,
            w_max=w_max,
        ),
        "Ellipsoid Robust MV": ellipsoid_robust_weight_frame(
            cache,
            rebalance_dates,
            cov_model=cov_model,
            mu_model=mu_model,
            radius=ellipsoid_radius,
            mv_lambda=mv_lambda,
            w_min=w_min,
            w_max=w_max,
        ),
        "Wasserstein DRMV": wasserstein_weight_frame(
            cache,
            rebalance_dates,
            cov_model=w_cov_model,
            mu_model=w_mu_model,
            radius=wasserstein_radius,
            mv_lambda=w_lambda,
            radius_scale=wasserstein_radius_scale,
            worst_case_variance=wasserstein_worst_case_variance,
            w_min=w_min,
            w_max=w_max,
        ),
    }


__all__ = [
    "box_robust_mv_weights",
    "box_robust_weight_frame",
    "ellipsoid_robust_mv_weights",
    "ellipsoid_robust_weight_frame",
    "psd_sqrt",
    "robust_radius_path",
    "robust_weight_frames",
    "wasserstein_drmv_weights",
    "wasserstein_weight_frame",
]
