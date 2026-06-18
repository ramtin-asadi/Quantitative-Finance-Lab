from __future__ import annotations

import numpy as np


def pde_grid(spot: float, strike: float, *, s_steps: int = 160, s_max_mult: float = 3.0) -> np.ndarray:
    s_max = max(float(s_max_mult) * max(float(spot), float(strike)), 1.5 * max(float(spot), float(strike)))
    return np.linspace(0.0, s_max, int(max(20, s_steps)) + 1)


def payoff_values(s_grid, strike: float, option_type: str | int = "put") -> np.ndarray:
    s = np.asarray(s_grid, dtype=float)
    is_call = int(option_type) > 0 if isinstance(option_type, (int, np.integer)) else str(option_type).lower().startswith("c")
    if is_call:
        return np.maximum(s - float(strike), 0.0)
    return np.maximum(float(strike) - s, 0.0)


def explicit_step(v, lower, diag, upper):
    v = np.asarray(v, dtype=float)
    out = v.copy()
    out[1:-1] = lower[1:-1] * v[:-2] + diag[1:-1] * v[1:-1] + upper[1:-1] * v[2:]
    return out


def implicit_step(rhs, lower, diag, upper):
    a = np.asarray(lower, dtype=float).copy()
    b = np.asarray(diag, dtype=float).copy()
    c = np.asarray(upper, dtype=float).copy()
    d = np.asarray(rhs, dtype=float).copy()
    n = len(d)
    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    out = np.empty(n, dtype=float)
    out[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        out[i] = (d[i] - c[i] * out[i + 1]) / b[i]
    return out


def crank_nicolson_step(v, lower_i, diag_i, upper_i, lower_e, diag_e, upper_e):
    rhs = explicit_step(v, lower_e, diag_e, upper_e)
    rhs[0] = v[0]
    rhs[-1] = v[-1]
    return implicit_step(rhs, lower_i, diag_i, upper_i)


def psor_solve(lower, diag, upper, rhs, payoff, *, omega: float = 1.35, tol: float = 1e-8, max_iter: int = 5000):
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    payoff = np.asarray(payoff, dtype=float)
    v = np.maximum(rhs.copy(), payoff)
    residual = np.inf
    it = 0
    for iteration in range(1, int(max_iter) + 1):
        it = iteration
        old = v.copy()
        for i in range(1, len(v) - 1):
            y = (rhs[i] - lower[i] * v[i - 1] - upper[i] * v[i + 1]) / diag[i]
            v[i] = max(payoff[i], v[i] + float(omega) * (y - v[i]))
        residual = float(np.max(np.abs(v - old)))
        if residual < tol:
            break
    return v, residual, it


def implicit_psor(lower, diag, upper, rhs, payoff, *, omega: float = 1.35, tol: float = 1e-8, max_iter: int = 5000):
    return psor_solve(lower, diag, upper, rhs, payoff, omega=omega, tol=tol, max_iter=max_iter)


def crank_nicolson_psor(v, lower_i, diag_i, upper_i, lower_e, diag_e, upper_e, payoff, *, omega: float = 1.35, tol: float = 1e-8, max_iter: int = 5000):
    rhs = explicit_step(v, lower_e, diag_e, upper_e)
    rhs[0] = v[0]
    rhs[-1] = v[-1]
    return psor_solve(lower_i, diag_i, upper_i, rhs, payoff, omega=omega, tol=tol, max_iter=max_iter)


def rannacher_steps(v, lower, diag, upper, payoff, steps: int = 2, **kwargs):
    out = np.asarray(v, dtype=float).copy()
    residuals = []
    for _ in range(int(steps)):
        out, residual, _ = psor_solve(lower, diag, upper, out, payoff, **kwargs)
        residuals.append(residual)
    return out, np.asarray(residuals)


def penalty_solve(lower, diag, upper, rhs, payoff, *, penalty: float = 1e4, tol: float = 1e-8, max_iter: int = 100):
    v = np.maximum(np.asarray(rhs, dtype=float), payoff)
    payoff = np.asarray(payoff, dtype=float)
    for _ in range(int(max_iter)):
        active = (v < payoff).astype(float)
        d = np.asarray(diag, dtype=float) + penalty * active
        b = np.asarray(rhs, dtype=float) + penalty * active * payoff
        new_v = implicit_step(b, lower, d, upper)
        if np.max(np.abs(new_v - v)) < tol:
            v = new_v
            break
        v = new_v
    return np.maximum(v, payoff)


def extract_boundary(s_grid, value, payoff, option_type: str | int = "put", *, tol: float = 1e-5) -> float:
    grid = np.asarray(s_grid, dtype=float)
    val = np.asarray(value, dtype=float)
    pay = np.asarray(payoff, dtype=float)
    is_call = int(option_type) > 0 if isinstance(option_type, (int, np.integer)) else str(option_type).lower().startswith("c")
    bind = np.isfinite(val) & np.isfinite(pay) & (pay > 0.0) & ((val - pay) <= float(tol))
    if not bind.any():
        return np.nan
    idx = np.flatnonzero(bind)
    return float(grid[idx[0]] if is_call else grid[idx[-1]])


def complementarity_error(value, payoff, residual=None) -> float:
    val = np.asarray(value, dtype=float)
    pay = np.asarray(payoff, dtype=float)
    violation = np.nanmax(np.maximum(pay - val, 0.0)) if val.size else np.nan
    if residual is None:
        return float(violation)
    res = np.asarray(residual, dtype=float)
    return float(max(violation, np.nanmax(np.abs(res)) if res.size else 0.0))


def pde_diagnostics(values, payoff, residuals, iterations=None) -> dict:
    arr = np.asarray(values, dtype=float)
    pay = np.asarray(payoff, dtype=float)
    res = np.asarray(residuals, dtype=float)
    out = {
        "max_residual": float(np.nanmax(np.abs(res))) if res.size else np.nan,
        "median_residual": float(np.nanmedian(np.abs(res))) if res.size else np.nan,
        "complementarity_error": complementarity_error(arr, pay, res),
    }
    if iterations is not None:
        it = np.asarray(iterations, dtype=float)
        out["max_iterations"] = float(np.nanmax(it)) if it.size else np.nan
        out["median_iterations"] = float(np.nanmedian(it)) if it.size else np.nan
    return out


__all__ = [
    "crank_nicolson_step",
    "crank_nicolson_psor",
    "complementarity_error",
    "explicit_step",
    "extract_boundary",
    "implicit_step",
    "implicit_psor",
    "penalty_solve",
    "payoff_values",
    "pde_diagnostics",
    "pde_grid",
    "psor_solve",
    "rannacher_steps",
]
