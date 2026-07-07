from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError, ModelError
from quantfinlab.portfolio.constraints import (
    coerce_prev_weights,
    constraints_feasible,
    normalize_weights,
)
from quantfinlab.portfolio.covariance import make_psd

DEFAULT_SOLVER_ORDER = ("OSQP", "ECOS", "SCS")


def _as_square_cov(cov_ann: np.ndarray | pd.DataFrame) -> np.ndarray:
    S = np.asarray(cov_ann, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise InputError("cov_ann must be a square matrix.")
    return 0.5 * (S + S.T)


def _normalize_solver_order(order: Sequence[str] | None) -> list[str]:
    if order is None:
        return list(DEFAULT_SOLVER_ORDER)
    vals = [str(x).strip().upper() for x in order if str(x).strip()]
    if not vals:
        return list(DEFAULT_SOLVER_ORDER)
    return list(dict.fromkeys(vals))


def _solve_with_solvers(prob, var, solver_order: Sequence[str]) -> np.ndarray | None:
    for solver in solver_order:
        try:
            kwargs: dict[str, Any] = {"warm_start": True}
            if solver == "OSQP":
                kwargs["max_iter"] = 8000
            elif solver in {"ECOS", "SCS"}:
                kwargs["max_iters"] = 10000
            prob.solve(solver=solver, **kwargs)
            if var.value is None:
                continue
            w = np.asarray(var.value, dtype=float).reshape(-1)
            if np.all(np.isfinite(w)):
                return w
        except Exception:
            continue
    return None


def _turnover_kappa(turnover_penalty_bps: float, kappa_target_annual: float | None) -> float:
    if kappa_target_annual is not None:
        return float(kappa_target_annual)
    return float(turnover_penalty_bps) / 10000.0


def _normalize_solution(w, *, w_min: float | None, w_max: float | None, long_only: bool):
    return normalize_weights(w, w_min=w_min, w_max=w_max, long_only=long_only, as_series=False)


def equal_weight(
    assets: Sequence[str] | int,
    *,
    w_min: float | None = None,
    w_max: float | None = None,
    long_only: bool = True,
    as_series: bool = False,
) -> np.ndarray | pd.Series:
    """Build an equal-weight portfolio under simple box constraints.

    Parameters
    ----------
    assets : sequence of str or int
        Asset labels, or the number of assets.
    w_min : float, optional
        Minimum per-asset weight.
    w_max : float, optional
        Maximum per-asset weight.
    long_only : bool, default=True
        Whether weights must be non-negative.
    as_series : bool, default=False
        If True, return a Series indexed by asset label.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Equal weights satisfying the requested bounds.

    Raises
    ------
    InputError
        If the asset set is empty or the constraints are infeasible.
    ModelError
        If the equal-weight vector cannot be normalized under the constraints.

    Notes
    -----
    This function is used both as a benchmark allocator and as a fallback when
    more complex optimizers fail.
    """

    if isinstance(assets, int):
        n_assets = int(assets)
        labels = [f"a{i}" for i in range(n_assets)]
    else:
        labels = [str(x) for x in assets]
        n_assets = len(labels)
    if n_assets <= 0:
        raise InputError("assets must contain at least one asset.")
    if not constraints_feasible(n_assets, w_min=w_min, w_max=w_max, long_only=long_only):
        raise InputError("Constraints are infeasible for equal weights.")
    w = np.ones(n_assets, dtype=float) / n_assets
    wn = _normalize_solution(w, w_min=w_min, w_max=w_max, long_only=long_only)
    if wn is None:
        raise ModelError("Failed to normalize equal weights under constraints.")
    if as_series:
        return pd.Series(wn, index=labels, dtype=float)
    return np.asarray(wn, dtype=float)


def minimum_variance(
    *,
    cov_ann: np.ndarray | pd.DataFrame,
    w_prev: np.ndarray | None = None,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    kappa_target_annual: float | None = None,
    ridge: float = 1e-8,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """Solve a constrained minimum-variance portfolio problem.

    The optimizer minimizes annualized portfolio variance plus optional turnover
    and ridge penalties, subject to full investment, long-only and box
    constraints.

    Parameters
    ----------
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    w_prev : array-like, optional
        Previous portfolio weights used for turnover penalization.
    w_min : float, optional
        Minimum per-asset weight.
    w_max : float, optional
        Maximum per-asset weight.
    long_only : bool, default=True
        Whether negative weights are disallowed.
    turnover_penalty_bps : float, default=10.0
        Turnover penalty expressed in basis points.
    kappa_target_annual : float, optional
        Annualized turnover penalty override.
    ridge : float, default=1e-8
        L2 ridge penalty added to stabilize weights.
    solver_order : sequence of str, optional
        CVXPY solver preference order.
    raise_on_fail : bool, default=False
        If True, raise ``ModelError`` when constraints are infeasible or the
        solver fails; otherwise return ``None``.

    Returns
    -------
    numpy.ndarray or None
        Optimized weights, or ``None`` when no feasible solution is found and
        ``raise_on_fail=False``.

    Raises
    ------
    InputError
        If fewer than two assets are supplied.
    ImportError
        If CVXPY is not installed.
    ModelError
        If optimization fails and ``raise_on_fail=True``.
    """

    S = _as_square_cov(cov_ann)
    n = S.shape[0]
    if n < 2:
        raise InputError("Need at least two assets for optimization.")
    if not constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None

    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise ImportError("cvxpy is required for portfolio optimization.") from exc

    wprev = coerce_prev_weights(w_prev, n)
    kappa = _turnover_kappa(turnover_penalty_bps, kappa_target_annual)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if w_min is not None:
        cons.append(w >= float(w_min))
    if w_max is not None:
        cons.append(w <= float(w_max))
    obj = cp.Minimize(
        cp.quad_form(w, cp.psd_wrap(make_psd(S, eps=1e-12)))
        + 0.5 * kappa * cp.norm1(w - wprev)
        + 0.5 * float(ridge) * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, cons)
    sol = _solve_with_solvers(prob, w, _normalize_solver_order(solver_order))
    if sol is None:
        if raise_on_fail:
            raise ModelError("MinVar solver failed to produce a feasible solution.")
        return None
    return _normalize_solution(sol, w_min=w_min, w_max=w_max, long_only=long_only)


def mean_variance(
    *,
    mu_excess_ann: np.ndarray | pd.Series,
    cov_ann: np.ndarray | pd.DataFrame,
    w_prev: np.ndarray | None = None,
    mv_lambda: float = 6.0,
    kappa_target_annual: float | None = None,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    ridge: float = 1e-8,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """Solve a constrained mean-variance utility portfolio problem.

    The objective maximizes expected excess return minus covariance risk,
    turnover penalty, and ridge penalty, subject to full investment and
    per-asset bounds.

    Parameters
    ----------
    mu_excess_ann : array-like or pandas.Series
        Annualized expected excess returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    w_prev : array-like, optional
        Previous weights used for turnover penalization.
    mv_lambda : float, default=6.0
        Risk-aversion parameter applied to portfolio variance.
    kappa_target_annual : float, optional
        Annualized turnover-penalty override.
    w_min : float, optional
        Minimum per-asset weight.
    w_max : float, optional
        Maximum per-asset weight.
    long_only : bool, default=True
        Whether negative weights are disallowed.
    turnover_penalty_bps : float, default=10.0
        Turnover penalty expressed in basis points.
    ridge : float, default=1e-8
        L2 ridge penalty for numerical stability.
    solver_order : sequence of str, optional
        CVXPY solver preference order.
    raise_on_fail : bool, default=False
        If True, raise ``ModelError`` on infeasibility or solver failure.

    Returns
    -------
    numpy.ndarray or None
        Optimized weights, or ``None`` if optimization fails and
        ``raise_on_fail=False``.

    Raises
    ------
    InputError
        If the covariance and expected-return dimensions do not match.
    ImportError
        If CVXPY is not installed.
    ModelError
        If optimization fails and ``raise_on_fail=True``.
    """

    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    S = _as_square_cov(cov_ann)
    if S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_excess_ann length.")
    n = len(mu)
    if not constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None
    if np.any(~np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise ImportError("cvxpy is required for portfolio optimization.") from exc

    wprev = coerce_prev_weights(w_prev, n)
    kappa = _turnover_kappa(turnover_penalty_bps, kappa_target_annual)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if w_min is not None:
        cons.append(w >= float(w_min))
    if w_max is not None:
        cons.append(w <= float(w_max))

    obj = cp.Maximize(
        mu @ w
        - 0.5 * float(mv_lambda) * cp.quad_form(w, cp.psd_wrap(make_psd(S, eps=1e-12)))
        - 0.5 * kappa * cp.norm1(w - wprev)
        - 0.5 * float(ridge) * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, cons)
    sol = _solve_with_solvers(prob, w, _normalize_solver_order(solver_order))
    if sol is None:
        if raise_on_fail:
            raise ModelError("MV solver failed to produce a feasible solution.")
        return None
    return _normalize_solution(sol, w_min=w_min, w_max=w_max, long_only=long_only)


def ridge_mean_variance(
    *,
    mu_excess_ann: np.ndarray | pd.Series,
    cov_ann: np.ndarray | pd.DataFrame,
    w_prev: np.ndarray | None = None,
    ridge: float = 1e-4,
    gamma_l2: float = 12.0,
    mv_lambda: float = 6.0,
    kappa_target_annual: float | None = None,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """Solve the ridge-regularized mean-variance allocation used in the research grid.

    This is a convenience wrapper around ``mean_variance`` that adds an
    asset-count-scaled L2 penalty through ``gamma_l2``. The penalty discourages
    unstable concentrated allocations when expected returns are noisy.

    Parameters
    ----------
    mu_excess_ann : array-like or pandas.Series
        Annualized expected excess returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    w_prev : array-like, optional
        Previous weights used for turnover penalization.
    ridge : float, default=1e-4
        Base ridge penalty.
    gamma_l2 : float, default=12.0
        Additional L2 penalty divided by the number of assets.
    mv_lambda : float, default=6.0
        Mean-variance risk-aversion parameter.
    kappa_target_annual : float, optional
        Annualized turnover-penalty override.
    w_min, w_max : float, optional
        Per-asset lower and upper bounds.
    long_only : bool, default=True
        Whether negative weights are disallowed.
    turnover_penalty_bps : float, default=10.0
        Turnover penalty in basis points.
    solver_order : sequence of str, optional
        CVXPY solver preference order.
    raise_on_fail : bool, default=False
        If True, raise on optimization failure.

    Returns
    -------
    numpy.ndarray or None
        Optimized weights, or ``None`` when optimization fails and
        ``raise_on_fail=False``.
    """

    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    gamma_per_asset = float(gamma_l2) / max(len(mu), 1)
    return mean_variance(
        mu_excess_ann=mu,
        cov_ann=cov_ann,
        w_prev=w_prev,
        mv_lambda=mv_lambda,
        kappa_target_annual=kappa_target_annual,
        w_min=w_min,
        w_max=w_max,
        long_only=long_only,
        turnover_penalty_bps=turnover_penalty_bps,
        ridge=float(ridge) + gamma_per_asset,
        solver_order=solver_order,
        raise_on_fail=raise_on_fail,
    )


def _sharpe_from_weights(mu: np.ndarray, cov: np.ndarray, w: np.ndarray) -> float:
    ww = np.asarray(w, dtype=float).reshape(-1)
    r = float(np.dot(mu, ww))
    v2 = float(ww @ cov @ ww)
    if not np.isfinite(v2) or v2 <= 1e-18:
        return -np.inf
    return r / math.sqrt(v2)


def max_sharpe_slsqp(
    *,
    mu_excess_ann: np.ndarray | pd.Series,
    cov_ann: np.ndarray | pd.DataFrame,
    w_prev: np.ndarray | None = None,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    kappa_target_annual: float | None = None,
    ridge: float = 1e-8,
    maxiter: int = 8000,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """Maximize a penalized Sharpe-ratio objective using SLSQP.

    The function maximizes expected return divided by volatility while applying
    turnover and ridge penalties. If the expected-return vector is effectively
    zero, it falls back to minimum-variance optimization.

    Parameters
    ----------
    mu_excess_ann : array-like or pandas.Series
        Annualized expected excess returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    w_prev : array-like, optional
        Previous weights used for turnover penalization.
    w_min, w_max : float, optional
        Per-asset lower and upper bounds.
    long_only : bool, default=True
        Whether negative weights are disallowed.
    turnover_penalty_bps : float, default=10.0
        Turnover penalty in basis points.
    kappa_target_annual : float, optional
        Annualized turnover penalty override.
    ridge : float, default=1e-8
        L2 ridge penalty.
    maxiter : int, default=8000
        Maximum SLSQP iterations.
    raise_on_fail : bool, default=False
        If True, raise ``ModelError`` when optimization fails.

    Returns
    -------
    numpy.ndarray or None
        Optimized weights, or ``None`` on failure when ``raise_on_fail=False``.

    Raises
    ------
    InputError
        If dimensions are incompatible or too few assets are supplied.
    ImportError
        If SciPy is not installed.
    ModelError
        If optimization fails and ``raise_on_fail=True``.

    Notes
    -----
    The objective is non-convex because it uses the Sharpe ratio directly.
    Results can be more sensitive than convex mean-variance optimizers.
    """

    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    S = _as_square_cov(cov_ann)
    if S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_excess_ann length.")
    n = len(mu)
    if n < 2:
        raise InputError("Need at least two assets for optimization.")
    if not constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None
    if np.any(~np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(np.abs(mu) < 1e-12):
        return minimum_variance(
            cov_ann=S,
            w_prev=w_prev,
            w_min=w_min,
            w_max=w_max,
            long_only=long_only,
            turnover_penalty_bps=turnover_penalty_bps,
            kappa_target_annual=kappa_target_annual,
            ridge=ridge,
            raise_on_fail=raise_on_fail,
        )

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for SLSQP max-Sharpe optimization.") from exc

    wprev = coerce_prev_weights(w_prev, n)
    cov_psd = make_psd(S, eps=1e-12)
    kappa = _turnover_kappa(turnover_penalty_bps, kappa_target_annual)

    lo = 0.0 if long_only else (-1.0 if w_min is None else float(w_min))
    hi = 1.0 if w_max is None else float(w_max)
    bounds = [(lo, hi) for _ in range(n)]
    x0 = _normalize_solution(wprev, w_min=w_min, w_max=w_max, long_only=long_only)
    if x0 is None:
        x0 = np.ones(n, dtype=float) / n

    def obj(w: np.ndarray) -> float:
        ww = np.asarray(w, dtype=float).reshape(-1)
        if ww.size != n or np.any(~np.isfinite(ww)):
            return 1e12
        ret = float(mu @ ww)
        vol = math.sqrt(max(float(ww @ cov_psd @ ww), 1e-18))
        if vol <= 1e-12:
            return 1e12
        penalty = 0.5 * kappa * float(np.sum(np.abs(ww - wprev))) + 0.5 * float(ridge) * float(np.sum(ww**2))
        return -(ret / vol) + penalty

    result = minimize(
        obj,
        np.asarray(x0, dtype=float),
        method="SLSQP",
        bounds=bounds,
        constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
        options={"maxiter": int(maxiter)},
    )
    if (not bool(result.success)) or result.x is None:
        if raise_on_fail:
            raise ModelError(f"SLSQP max-Sharpe failed: {getattr(result, 'message', 'unknown error')}.")
        return None
    wn = _normalize_solution(np.asarray(result.x, dtype=float), w_min=w_min, w_max=w_max, long_only=long_only)
    if wn is None and raise_on_fail:
        raise ModelError("SLSQP max-Sharpe returned invalid/infeasible weights.")
    return wn


def _greedy_max_return_weight(
    mu: np.ndarray,
    *,
    w_max: float | None,
    w_min: float | None,
    long_only: bool,
) -> np.ndarray | None:
    if not long_only:
        return None
    n = len(mu)
    cap = np.inf if w_max is None else float(w_max)
    floor = 0.0 if w_min is None else float(w_min)
    if cap <= 0:
        return None
    order = np.argsort(mu)[::-1]
    w = np.full(n, floor, dtype=float)
    rem = 1.0 - float(np.sum(w))
    if rem < -1e-10:
        return None
    rem = max(rem, 0.0)
    for i in order:
        if rem <= 1e-12:
            break
        room = cap - w[i]
        if room <= 0:
            continue
        add = min(room, rem)
        w[i] += add
        rem -= add
    if rem > 1e-8:
        return None
    return w


def max_sharpe_frontier_grid(
    *,
    mu_excess_ann: np.ndarray | pd.Series,
    cov_ann: np.ndarray | pd.DataFrame,
    w_prev: np.ndarray | None = None,
    grid_n: int = 25,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    kappa_target_annual: float | None = None,
    ridge: float = 1e-8,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """Approximate maximum-Sharpe allocation by scanning efficient-frontier targets.

    The function builds a minimum-variance anchor and a greedy maximum-return
    anchor, then solves a sequence of convex target-return variance problems and
    selects the feasible solution with the highest Sharpe ratio.

    Parameters
    ----------
    mu_excess_ann : array-like or pandas.Series
        Annualized expected excess returns.
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    w_prev : array-like, optional
        Previous weights used for turnover penalization.
    grid_n : int, default=25
        Number of target-return grid points.
    w_min, w_max : float, optional
        Per-asset lower and upper bounds.
    long_only : bool, default=True
        Whether negative weights are disallowed.
    turnover_penalty_bps : float, default=10.0
        Turnover penalty in basis points.
    kappa_target_annual : float, optional
        Annualized turnover penalty override.
    ridge : float, default=1e-8
        L2 ridge penalty.
    solver_order : sequence of str, optional
        CVXPY solver preference order.
    raise_on_fail : bool, default=False
        If True, raise when no feasible frontier point is found.

    Returns
    -------
    numpy.ndarray or None
        Approximate maximum-Sharpe weights, or ``None`` on failure when
        ``raise_on_fail=False``.

    Raises
    ------
    InputError
        If dimensions are incompatible or ``grid_n < 2``.
    ImportError
        If CVXPY is not installed.
    ModelError
        If the frontier is degenerate or no feasible solution is found and
        ``raise_on_fail=True``.

    Notes
    -----
    This method is slower than direct SLSQP but uses convex subproblems and can
    be more stable under tight constraints.
    """

    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    S = _as_square_cov(cov_ann)
    if S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_excess_ann length.")
    n = len(mu)
    if n < 2:
        raise InputError("Need at least two assets for optimization.")
    if grid_n < 2:
        raise InputError("grid_n must be at least 2.")
    if not constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None
    if np.any(~np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(np.abs(mu) < 1e-12):
        return minimum_variance(
            cov_ann=S,
            w_prev=w_prev,
            w_min=w_min,
            w_max=w_max,
            long_only=long_only,
            turnover_penalty_bps=turnover_penalty_bps,
            kappa_target_annual=kappa_target_annual,
            ridge=ridge,
            solver_order=solver_order,
            raise_on_fail=raise_on_fail,
        )

    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise ImportError("cvxpy is required for portfolio optimization.") from exc

    wprev = coerce_prev_weights(w_prev, n)
    w_minv = minimum_variance(
        cov_ann=S,
        w_prev=wprev,
        w_min=w_min,
        w_max=w_max,
        long_only=long_only,
        turnover_penalty_bps=turnover_penalty_bps,
        kappa_target_annual=kappa_target_annual,
        ridge=ridge,
        solver_order=solver_order,
        raise_on_fail=False,
    )
    if w_minv is None:
        w_minv = np.ones(n, dtype=float) / n
    w_maxr = _greedy_max_return_weight(mu, w_max=w_max, w_min=w_min, long_only=long_only)
    if w_maxr is None:
        if raise_on_fail:
            raise ModelError("Could not build max-return anchor under constraints.")
        return None
    r_lo = float(np.dot(mu, w_minv))
    r_hi = float(np.dot(mu, w_maxr))
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo + 1e-12:
        if raise_on_fail:
            raise ModelError("Degenerate frontier return range.")
        return None

    w = cp.Variable(n)
    r_target = cp.Parameter()
    cons = [cp.sum(w) == 1, mu @ w >= r_target]
    if long_only:
        cons.append(w >= 0)
    if w_min is not None:
        cons.append(w >= float(w_min))
    if w_max is not None:
        cons.append(w <= float(w_max))
    kappa = _turnover_kappa(turnover_penalty_bps, kappa_target_annual)
    obj = cp.Minimize(
        cp.quad_form(w, cp.psd_wrap(make_psd(S, eps=1e-12)))
        + kappa * cp.norm1(w - wprev)
        + 0.5 * float(ridge) * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, cons)
    solver_list = _normalize_solver_order(solver_order)

    best_w: np.ndarray | None = None
    best_s = -np.inf
    for rt in np.linspace(r_lo, r_hi, int(grid_n)):
        r_target.value = float(rt)
        sol = _solve_with_solvers(prob, w, solver_list)
        if sol is None:
            continue
        wn = _normalize_solution(sol, w_min=w_min, w_max=w_max, long_only=long_only)
        if wn is None:
            continue
        sh = _sharpe_from_weights(mu, S, np.asarray(wn, dtype=float))
        if sh > best_s:
            best_s = sh
            best_w = np.asarray(wn, dtype=float)
    if best_w is None and raise_on_fail:
        raise ModelError("Frontier-grid max-Sharpe solver did not find a feasible solution.")
    return best_w


weights_equal = equal_weight
weights_minvar = minimum_variance
weights_mv = mean_variance
weights_ridge_mv = ridge_mean_variance
weights_maxsharpe_slsqp = max_sharpe_slsqp
weights_maxsharpe_frontier_grid = max_sharpe_frontier_grid


__all__ = [
    "DEFAULT_SOLVER_ORDER",
    "equal_weight",
    "max_sharpe_frontier_grid",
    "max_sharpe_slsqp",
    "mean_variance",
    "minimum_variance",
    "ridge_mean_variance",
    "weights_equal",
    "weights_maxsharpe_frontier_grid",
    "weights_maxsharpe_slsqp",
    "weights_minvar",
    "weights_mv",
    "weights_ridge_mv",
]
