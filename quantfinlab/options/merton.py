from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import optimize

from quantfinlab.options.bsm import black76_price

try:
    from numba import njit

    _has_numba = True
except Exception:
    njit = None
    _has_numba = False


def _engine_name(engine: str = "auto") -> str:
    if str(engine).lower() == "auto":
        return "numba" if _has_numba else "numpy"
    if str(engine).lower() == "numba" and not _has_numba:
        return "numpy"
    return str(engine).lower()


if _has_numba:

    @njit(cache=True)
    def _poisson_weights(lam_tau, n_max):
        out = np.empty(n_max + 1)
        out[0] = np.exp(-lam_tau)
        for n in range(1, n_max + 1):
            out[n] = out[n - 1] * lam_tau / n
        return out

else:
    _poisson_weights = None


def _weights(lam_tau: float, n_max: int) -> np.ndarray:
    if _poisson_weights is not None:
        return _poisson_weights(float(lam_tau), int(n_max))
    out = np.empty(int(n_max) + 1, dtype=float)
    out[0] = math.exp(-float(lam_tau))
    for n in range(1, int(n_max) + 1):
        out[n] = out[n - 1] * float(lam_tau) / n
    return out


def merton_cf(u, spot, rate, dividend_yield, tau, sigma, lambda_jump, mu_jump, sigma_jump):
    """Merton jump-diffusion characteristic function of log spot at expiry."""
    u_arr = np.asarray(u, dtype=complex)
    sigma = np.asarray(sigma, dtype=float)
    lam = np.asarray(lambda_jump, dtype=float)
    mu = np.asarray(mu_jump, dtype=float)
    sj = np.asarray(sigma_jump, dtype=float)
    tau = np.asarray(tau, dtype=float)
    omega = -lam * (np.exp(mu + 0.5 * sj * sj) - 1.0)
    drift = np.log(np.asarray(spot, dtype=float)) + (
        np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float) + omega - 0.5 * sigma * sigma
    ) * tau
    return np.exp(
        1j * u_arr * drift
        - 0.5 * sigma * sigma * u_arr * u_arr * tau
        + lam * tau * (np.exp(1j * u_arr * mu - 0.5 * sj * sj * u_arr * u_arr) - 1.0)
    )


def merton_price(option_type, forward, strike, tau, discount_factor, sigma, lambda_jump, mu_jump, sigma_jump, n_max: int = 40, engine: str = "auto"):
    option_type = np.asarray(option_type)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    df = np.asarray(discount_factor, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lambda_jump = np.asarray(lambda_jump, dtype=float)
    mu_jump = np.asarray(mu_jump, dtype=float)
    sigma_jump = np.asarray(sigma_jump, dtype=float)
    option_type, forward, strike, tau, df, sigma, lambda_jump, mu_jump, sigma_jump = np.broadcast_arrays(
        option_type,
        forward,
        strike,
        tau,
        df,
        sigma,
        lambda_jump,
        mu_jump,
        sigma_jump,
    )
    out = np.zeros(forward.shape, dtype=float)
    flat = out.ravel()
    for i in range(flat.size):
        ti = max(float(tau.ravel()[i]), 1e-12)
        sigma_i = max(float(sigma.ravel()[i]), 1e-8)
        lam = max(float(lambda_jump.ravel()[i]), 0.0)
        mu = float(mu_jump.ravel()[i])
        sj = max(float(sigma_jump.ravel()[i]), 1e-8)
        jump_comp = math.exp(mu + 0.5 * sj * sj) - 1.0
        wi = _weights(lam * ti, int(n_max))
        px = 0.0
        for n, wn in enumerate(wi):
            f_n = forward.ravel()[i] * math.exp(-lam * jump_comp * ti + n * mu + 0.5 * n * sj * sj)
            sig_n = math.sqrt(max(sigma_i * sigma_i + n * sj * sj / ti, 1e-12))
            px += wn * float(black76_price(option_type.ravel()[i], f_n, strike.ravel()[i], ti, sig_n, df.ravel()[i]))
        flat[i] = px
    return out


def _scale(q: pd.DataFrame, weight_col: str | None) -> np.ndarray:
    if "calib_scale_px" in q.columns:
        s = pd.to_numeric(q["calib_scale_px"], errors="coerce").to_numpy(dtype=float)
    elif "half_spread" in q.columns:
        s = pd.to_numeric(q["half_spread"], errors="coerce").to_numpy(dtype=float)
    else:
        s = np.full(len(q), np.nanmedian(q["mid"]) * 0.02 if len(q) else 1.0)
    s = np.where(np.isfinite(s) & (s > 0), s, np.nanmedian(s[np.isfinite(s) & (s > 0)]) if np.any(np.isfinite(s) & (s > 0)) else 1.0)
    if weight_col and weight_col in q.columns:
        w = pd.to_numeric(q[weight_col], errors="coerce").to_numpy(dtype=float)
        s = s / np.sqrt(np.where(np.isfinite(w) & (w > 0), w, 1.0))
    return np.maximum(s, 1e-6)


def _fit_prices(q: pd.DataFrame, p: np.ndarray, engine: str) -> pd.DataFrame:
    out = q.copy()
    out["model_price"] = merton_price(out["option_type"].to_numpy(), out["forward"], out["strike"], out["tau"], out["discount_factor"], p[0], p[1], p[2], p[3], engine=engine)
    out["price_residual"] = out["model_price"] - out["mid"]
    out["model_iv"] = np.nan
    out["iv_residual"] = np.nan
    return out


def _fit_prices_from_params(q: pd.DataFrame, params: pd.DataFrame, engine: str) -> pd.DataFrame:
    if q.empty or params.empty:
        return q.head(0).copy()
    keys = ["date", "expiry"] if {"date", "expiry"}.issubset(params.columns) and {"date", "expiry"}.issubset(q.columns) else ["expiry"] if "expiry" in params.columns and "expiry" in q.columns else []
    out = q.copy()
    for col in ["sigma", "lambda_jump", "mu_jump", "sigma_jump"]:
        if col in out.columns and col not in keys:
            out = out.drop(columns=col)
    if keys:
        out = out.merge(params[keys + ["sigma", "lambda_jump", "mu_jump", "sigma_jump"]], on=keys, how="inner")
    else:
        p = params.iloc[0]
        for col in ["sigma", "lambda_jump", "mu_jump", "sigma_jump"]:
            out[col] = p[col]
    out["model_price"] = merton_price(
        out["option_type"].to_numpy(),
        out["forward"],
        out["strike"],
        out["tau"],
        out["discount_factor"],
        out["sigma"],
        out["lambda_jump"],
        out["mu_jump"],
        out["sigma_jump"],
        engine=engine,
    )
    out["price_residual"] = out["model_price"] - out["mid"]
    out["model_iv"] = np.nan
    out["iv_residual"] = np.nan
    return out


def _fit_one_merton(q: pd.DataFrame, weight_col: str, engine: str, max_nfev: int, start: np.ndarray | None = None):
    target = q["mid"].to_numpy(dtype=float)
    scale = _scale(q, weight_col)
    atm_vol = float(np.nanmedian(q["iv_mid"])) if "iv_mid" in q.columns else 0.7
    starts = [
        np.array([atm_vol, 0.5, -0.04, 0.18]),
        np.array([0.8 * atm_vol, 1.2, -0.08, 0.28]),
        np.array([1.1 * atm_vol, 0.15, 0.02, 0.12]),
        np.array([atm_vol, 2.5, -0.12, 0.35]),
    ]
    if start is not None and np.all(np.isfinite(start)):
        starts.insert(0, np.asarray(start, dtype=float))
    bounds = ([0.03, 0.0, -0.95, 0.005], [3.50, 12.0, 0.80, 2.00])

    def residual(p):
        px = merton_price(q["option_type"].to_numpy(), q["forward"], q["strike"], q["tau"], q["discount_factor"], p[0], p[1], p[2], p[3], engine=engine)
        return (px - target) / scale

    best = None
    for s in starts:
        res = optimize.least_squares(residual, np.clip(s, bounds[0], bounds[1]), bounds=bounds, max_nfev=int(max_nfev), xtol=1e-8, ftol=1e-8, gtol=1e-8)
        val = float(np.nanmean(res.fun**2))
        if best is None or val < best[0]:
            best = (val, res)
    usable_fit = np.isfinite(best[0]) and np.sqrt(max(best[0], 0.0)) < 5.0
    return best[1].x, best[0], bool(best[1].success or usable_fit), int(best[1].nfev), scale


def fit_merton_jump_diffusion(quotes: pd.DataFrame, weight_col: str = "obs_weight", engine: str = "auto", max_nfev: int = 180, fit_by_expiry: bool = True) -> dict:
    t0 = time.perf_counter()
    engine_used = _engine_name(engine)
    q = quotes.copy()
    if q.empty:
        return {"model": "merton", "params": pd.DataFrame(), "fit": pd.DataFrame(), "diag": pd.DataFrame(), "elapsed_sec": 0.0, "engine": engine_used}
    group_cols = ["date", "expiry"] if fit_by_expiry and {"date", "expiry"}.issubset(q.columns) else ["expiry"] if fit_by_expiry and "expiry" in q.columns else []
    rows = []
    last = None
    groups = q.groupby(group_cols, sort=True) if group_cols else [(None, q)]
    for key, g in groups:
        if len(g) < 5:
            continue
        p, loss, ok, nfev, _ = _fit_one_merton(g, weight_col, engine_used, max_nfev=max(20, int(max_nfev)), start=last)
        if np.all(np.isfinite(p)):
            last = p
        values = key if isinstance(key, tuple) else (key,) if group_cols else ()
        row = dict(zip(group_cols, values, strict=False))
        row.update({"sigma": p[0], "lambda_jump": p[1], "mu_jump": p[2], "sigma_jump": p[3], "loss": loss, "success": ok, "nfev": nfev, "quotes": len(g), "dte_days": float(np.nanmedian(g.get("dte_days", g["tau"] * 365.25)))})
        rows.append(row)
    params = pd.DataFrame(rows)
    fit = _fit_prices_from_params(q, params, engine_used)
    scale = _scale(fit, weight_col)
    tail = fit[np.abs(fit.get("k", 0.0)) >= 0.14]
    diag = pd.DataFrame([{
        "model": "merton",
        "quotes": len(fit),
        "weighted_price_rmse": float(np.sqrt(np.nanmean(((fit["model_price"] - fit["mid"]) / scale) ** 2))),
        "median_abs_price_error": float(np.nanmedian(np.abs(fit["price_residual"]))),
        "tail_error": float(np.nanmedian(np.abs(tail["price_residual"]))) if not tail.empty else np.nan,
        "tail_count": int(len(tail)),
    }])
    return {"model": "merton", "params": params, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": engine_used}


def merton_prices(quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    params = fit.get("params", pd.DataFrame())
    if quotes.empty or params.empty:
        return quotes.head(0).copy()
    return _fit_prices_from_params(quotes, params, _engine_name(engine))


__all__ = ["fit_merton_jump_diffusion", "merton_cf", "merton_price", "merton_prices"]
