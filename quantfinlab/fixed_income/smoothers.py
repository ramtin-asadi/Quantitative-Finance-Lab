from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ..common.contracts import Curve, CurvePillars
from .discounting import par_from_df
from .tenors import tenor_to_years


def _curve_grid(T_min: float, T_max: float = 30.0, n: int = 1000) -> np.ndarray:
    return np.linspace(max(1 / 12, T_min), T_max, n)


def _curve_from_df_func(method: str, name: str, df_func: Callable[[np.ndarray], np.ndarray], *, T_min: float) -> Curve:
    grid = _curve_grid(T_min)
    df_grid = df_func(grid)
    df_grid = np.clip(df_grid, 1e-16, None)
    z_grid = -np.log(df_grid) / grid
    fwd_grid = -np.gradient(np.log(df_grid), grid)
    return Curve(method=method, name=name, grid=grid, df_grid=df_grid, z_grid=z_grid, fwd_grid=fwd_grid, df=df_func)


def fit_curves(
    pillars: CurvePillars,
    *,
    methods: Iterable[str] = ("loglinear", "pchip", "nss", "qp"),
    freq: int = 2,
    min_df: float = 1e-12,
) -> dict[str, Curve]:
    """
    Build multiple curves from pillars using selected methods.
    Returns dict: method -> Curve
    """
    methods = list(methods)
    T = pillars.T
    par = pillars.par
    labels = pillars.labels
    dfs = pillars.dfs

    out: dict[str, Curve] = {}
    for m in methods:
        mm = m.lower().strip()
        if mm == "loglinear":
            out["loglinear"] = _loglinear_curve(T, dfs, min_df=min_df)
        elif mm == "pchip":
            out["pchip"] = _pchip_curve(T, dfs, min_df=min_df)
        elif mm == "nss":
            out["nss"] = _nss_curve(T, par, min_df=min_df, freq=freq)
        elif mm == "qp":
            out["qp"] = _qp_curve(labels, par, freq=freq, min_df=min_df)
        else:
            raise ValueError(f"Unknown curve method: {m!r}")
    return out


def _loglinear_curve(T: np.ndarray, dfs: np.ndarray, *, min_df: float) -> Curve:
    log_dfs = np.log(np.clip(dfs, min_df, None))

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        log_df = np.interp(tt, T, log_dfs, left=log_dfs[0], right=log_dfs[-1])
        return np.exp(log_df)

    return _curve_from_df_func("loglinear", "Log-linear DF", df_func, T_min=float(T.min()))


def _pchip_curve(T: np.ndarray, dfs: np.ndarray, *, min_df: float) -> Curve:
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception as e:  # pragma: no cover
        raise ImportError("PCHIP requires scipy. Install scipy to use method='pchip'.") from e

    zeros = -np.log(np.clip(dfs, min_df, None)) / T
    z_spline = PchipInterpolator(T, zeros, extrapolate=True)

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        z = z_spline(tt)
        return np.exp(-z * tt)

    return _curve_from_df_func("pchip", "PCHIP zero", df_func, T_min=float(T.min()))


def _nss_zero(T: np.ndarray, b0: float, b1: float, b2: float, b3: float, tau1: float, tau2: float) -> np.ndarray:
    T = np.array(T, dtype=float)
    T = np.clip(T, 1e-12, None)

    x1 = T / max(tau1, 1e-12)
    x2 = T / max(tau2, 1e-12)

    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    g1 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)

    return b0 + b1 * f1 + b2 * f2 + b3 * g1


def _nss_curve(T: np.ndarray, par: np.ndarray, *, min_df: float, freq: int) -> Curve:
    try:
        from scipy.optimize import minimize
    except Exception as e:  # pragma: no cover
        raise ImportError("NSS requires scipy. Install scipy to use method='nss'.") from e

    T = np.array(T, dtype=float)
    par = np.array(par, dtype=float)

    def obj(theta: np.ndarray) -> float:
        b0, b1, b2, b3, tau1, tau2 = theta
        z = _nss_zero(T, b0, b1, b2, b3, tau1, tau2)
        dfs_p = np.exp(-z * T)
        log_d = np.log(np.clip(dfs_p, min_df, None))

        def df_func_p(tt: np.ndarray) -> np.ndarray:
            ttt = np.array(tt, dtype=float)
            log_df = np.interp(ttt, T, log_d, left=log_d[0], right=log_d[-1])
            return np.exp(log_df)

        par_fit = par_from_df(df_func_p, T, freq=freq)
        err = par_fit - par
        return float(np.mean(err**2))

    b0_0 = float(np.nanmedian(par[-3:])) if len(par) >= 3 else float(np.nanmedian(par))
    x0 = np.array([b0_0, -0.02, 0.02, 0.01, 1.5, 5.0], dtype=float)

    res = minimize(obj, x0, method="L-BFGS-B")
    b0, b1, b2, b3, tau1, tau2 = res.x

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        z = _nss_zero(tt, b0, b1, b2, b3, tau1, tau2)
        return np.exp(-z * tt)

    return _curve_from_df_func("nss", "NSS", df_func, T_min=float(T.min()))


def _qp_curve(labels: list[str], par_mkt: np.ndarray, *, freq: int, min_df: float) -> Curve:
    """
    QP smoothing approach (matches your notebook design):
    - decision variables are discount factors on a time grid (coupon grid + observed maturities)
    - constraints enforce exact par pricing at observed maturities (linear constraints)
    - objective smooths second differences of DFs plus a small pull to a prior curve
    """
    try:
        import cvxpy  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("QP method requires cvxpy. Install cvxpy to use method='qp'.") from e

    T_obs = np.array([tenor_to_years(label) for label in labels], dtype=float)
    par_mkt = np.array(par_mkt, dtype=float)

    idx = np.argsort(T_obs)
    T_obs = T_obs[idx]
    par_mkt = par_mkt[idx]

    t_grid, grid_index = _qp_build_t_grid(T_obs, freq=freq)
    d, constraints = _qp_build_constraints(t_grid, grid_index, T_obs, par_mkt, freq=freq, min_df=min_df)
    d_sol, status = _qp_solve(t_grid, d, constraints, par_mkt, freq=freq, min_df=min_df)

    if status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"QP solve failed with status={status!r}")

    log_d = np.log(np.clip(d_sol, min_df, None))

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        log_df = np.interp(tt, t_grid, log_d, left=log_d[0], right=log_d[-1])
        return np.exp(log_df)

    return _curve_from_df_func("qp", "QP DF", df_func, T_min=float(t_grid.min()))


def _qp_build_t_grid(T_obs: np.ndarray, *, freq: int) -> tuple[np.ndarray, dict[float, int]]:
    T_max = float(np.max(T_obs))
    n_grid = round(T_max * freq)
    base = np.array([i / freq for i in range(1, n_grid + 1)], dtype=float)
    t_grid = np.unique(np.concatenate([base, T_obs]))
    t_grid = np.array(sorted(t_grid), dtype=float)
    grid_index = {float(np.round(t, 10)): i for i, t in enumerate(t_grid)}
    return t_grid, grid_index


def _qp_build_constraints(
    t_grid: np.ndarray,
    grid_index: dict[float, int],
    T_obs: np.ndarray,
    par_mkt: np.ndarray,
    *,
    freq: int,
    min_df: float,
):
    import cvxpy as cp

    d = cp.Variable(len(t_grid))
    constraints = [d >= min_df, d[1:] <= d[:-1]]

    # Pin short-end nodes to exp(-y*T) when T<1
    for Tk, yk in zip(T_obs, par_mkt, strict=True):
        if Tk < 1.0:
            key = float(np.round(Tk, 10))
            if key in grid_index:
                i = grid_index[key]
                constraints.append(d[i] == float(np.exp(-yk * Tk)))

    # Par-bond constraints (linear): sum coupons + principal = 1
    for Tk, ck in zip(T_obs, par_mkt, strict=True):
        if Tk < 1.0:
            continue
        keyT = float(np.round(Tk, 10))
        if keyT not in grid_index:
            continue
        iT = grid_index[keyT]
        n = round(Tk * freq)

        coupon_idx = []
        for j in range(1, n + 1):
            key = float(np.round(j / freq, 10))
            coupon_idx.append(grid_index[key])

        constraints.append(cp.sum((ck / freq) * d[coupon_idx]) + d[iT] == 1.0)

    return d, constraints


def _qp_solve(
    t_grid: np.ndarray,
    d,
    constraints,
    par_mkt: np.ndarray,
    *,
    freq: int,
    min_df: float,
) -> tuple[np.ndarray, str]:
    import cvxpy as cp

    lam = 1e4
    eps = 1e-4
    prior_rate = float(np.nanmedian(par_mkt[-3:])) if len(par_mkt) >= 3 else float(np.nanmedian(par_mkt))
    d_prior = np.exp(-prior_rate * t_grid)

    d2 = d[2:] - 2 * d[1:-1] + d[:-2]
    obj = cp.Minimize(lam * cp.sum_squares(d2) + eps * cp.sum_squares(d - d_prior))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP)

    d_sol = np.array(d.value).astype(float)
    d_sol = np.clip(d_sol, min_df, None)
    return d_sol, str(prob.status)

__all__ = [
    "fit_curves",
]
