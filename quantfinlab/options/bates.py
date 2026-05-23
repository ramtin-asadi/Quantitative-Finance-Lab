from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import optimize

from quantfinlab.options.heston import _rng_shocks, _scale, heston_cf

try:
    from numba import njit

    _has_numba = True
except Exception:
    njit = None
    _has_numba = False


def _engine_name(engine: str) -> str:
    if str(engine).lower() == "auto":
        return "numba" if _has_numba else "numpy"
    if str(engine).lower() == "numba" and not _has_numba:
        return "numpy"
    return str(engine).lower()


def _jump_shocks(n_path: int, n_step: int, random_state: int = 7):
    n_half = max(int(n_path) // 2, 2)
    rng = np.random.default_rng(int(random_state))
    zj = rng.normal(size=(n_half, n_step))
    uj = rng.uniform(size=(n_half, n_step))
    return np.vstack([zj, -zj]).astype(float), np.vstack([uj, uj]).astype(float)


def bates_cf(u, spot, rate, dividend_yield, tau, v0, kappa, theta, sigma_v, rho, lambda_jump, mu_jump, sigma_jump):
    """Bates characteristic function: Heston times compensated Merton jumps."""
    u_arr = np.asarray(u, dtype=complex)
    lam = np.asarray(lambda_jump, dtype=float)
    mu = np.asarray(mu_jump, dtype=float)
    sj = np.asarray(sigma_jump, dtype=float)
    omega = -lam * (np.exp(mu + 0.5 * sj * sj) - 1.0)
    h = heston_cf(u_arr, spot, np.asarray(rate, dtype=float) + omega, dividend_yield, tau, v0, kappa, theta, sigma_v, rho)
    jump = np.exp(lam * np.asarray(tau, dtype=float) * (np.exp(1j * u_arr * mu - 0.5 * sj * sj * u_arr * u_arr) - 1.0))
    return h * jump


if _has_numba:

    @njit(cache=True)
    def _bates_mc_price_numba(is_call, forward, strike, tau, df, v0, kappa, theta, xi, rho, lambda_jump, mu_jump, sigma_jump, steps_per_year, z_s, z_v, z_j, u_j):
        n_quote = forward.shape[0]
        n_path = z_s.shape[0]
        prices = np.empty(n_quote)
        ses = np.empty(n_quote)
        rho_c = min(max(rho, -0.999), 0.999)
        rho_scale = np.sqrt(1.0 - rho_c * rho_c)
        lam = max(lambda_jump, 0.0)
        sj = max(sigma_jump, 1e-8)
        jump_comp = np.exp(mu_jump + 0.5 * sj * sj) - 1.0
        for i in range(n_quote):
            steps = int(max(2.0, np.ceil(tau[i] * steps_per_year)))
            dt = tau[i] / steps
            payoff_sum = 0.0
            payoff_sq = 0.0
            for pth in range(n_path):
                v = max(v0, 1e-8)
                log_rel = 0.0
                for j in range(steps):
                    vp = max(v, 0.0)
                    zs = z_s[pth, j]
                    zv = rho_c * zs + rho_scale * z_v[pth, j]
                    jump = 0.0
                    if u_j[pth, j] < lam * dt:
                        jump = mu_jump + sj * z_j[pth, j]
                    log_rel += (-0.5 * vp - lam * jump_comp) * dt + np.sqrt(vp * dt) * zs + jump
                    v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * zv
                    v = max(v, 0.0)
                ft = forward[i] * np.exp(log_rel)
                payoff = max(ft - strike[i], 0.0) if is_call[i] else max(strike[i] - ft, 0.0)
                payoff_sum += payoff
                payoff_sq += payoff * payoff
            mean = payoff_sum / n_path
            var = max(payoff_sq / n_path - mean * mean, 0.0)
            prices[i] = df[i] * mean
            ses[i] = df[i] * np.sqrt(var / max(n_path - 1, 1))
        return prices, ses


def bates_mc_price(
    option_type,
    forward,
    strike,
    tau,
    discount_factor,
    v0,
    kappa,
    theta,
    xi,
    rho,
    lambda_jump,
    mu_jump,
    sigma_jump,
    steps_per_year: int = 52,
    z_s=None,
    z_v=None,
    z_j=None,
    u_j=None,
    paths: int = 2048,
    random_state: int = 7,
    engine: str = "auto",
):
    option_type = np.asarray(option_type).astype(str)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    tau = np.asarray(tau, dtype=float)
    df = np.asarray(discount_factor, dtype=float)
    max_step = int(max(2, np.ceil(float(np.nanmax(tau)) * int(steps_per_year)) + 2))
    if z_s is None or z_v is None or z_s.shape[1] < max_step:
        z_s, z_v = _rng_shocks(paths, max_step, random_state=random_state)
    if z_j is None or u_j is None or z_j.shape[1] < max_step:
        z_j, u_j = _jump_shocks(z_s.shape[0], max_step, random_state=random_state + 101)
    if _engine_name(engine) == "numba":
        is_call = np.char.lower(option_type.astype(str)).astype(str)
        is_call = np.array([str(x).startswith("c") for x in is_call], dtype=np.bool_)
        return _bates_mc_price_numba(is_call, forward, strike, tau, df, float(v0), float(kappa), float(theta), float(xi), float(rho), float(lambda_jump), float(mu_jump), float(sigma_jump), int(steps_per_year), z_s, z_v, z_j, u_j)
    n_path = z_s.shape[0]
    prices = np.empty(len(forward), dtype=float)
    ses = np.empty(len(forward), dtype=float)
    rho = float(np.clip(rho, -0.999, 0.999))
    lam = max(float(lambda_jump), 0.0)
    mu = float(mu_jump)
    sj = max(float(sigma_jump), 1e-8)
    jump_comp = math.exp(mu + 0.5 * sj * sj) - 1.0
    for i in range(len(forward)):
        steps = int(max(2, np.ceil(float(tau[i]) * int(steps_per_year))))
        dt = float(tau[i]) / steps
        v = np.full(n_path, max(float(v0), 1e-8), dtype=float)
        log_rel = np.zeros(n_path, dtype=float)
        for j in range(steps):
            zv = rho * z_s[:, j] + np.sqrt(1.0 - rho * rho) * z_v[:, j]
            vp = np.maximum(v, 0.0)
            jump = (u_j[:, j] < lam * dt).astype(float) * (mu + sj * z_j[:, j])
            log_rel += (-0.5 * vp - lam * jump_comp) * dt + np.sqrt(vp * dt) * z_s[:, j] + jump
            v = v + float(kappa) * (float(theta) - vp) * dt + float(xi) * np.sqrt(vp * dt) * zv
            v = np.maximum(v, 0.0)
        f_t = float(forward[i]) * np.exp(log_rel)
        if option_type[i].lower().startswith("c"):
            payoff = np.maximum(f_t - float(strike[i]), 0.0)
        else:
            payoff = np.maximum(float(strike[i]) - f_t, 0.0)
        disc = float(df[i])
        prices[i] = disc * float(np.mean(payoff))
        ses[i] = disc * float(np.std(payoff, ddof=1)) / np.sqrt(n_path) if n_path > 1 else np.nan
    return prices, ses


def _fit_prices(q: pd.DataFrame, p: np.ndarray, steps_per_year: int, z_s, z_v, z_j, u_j, paths: int, random_state: int, engine: str = "auto") -> pd.DataFrame:
    out = q.copy()
    px, se = bates_mc_price(out["option_type"].to_numpy(), out["forward"], out["strike"], out["tau"], out["discount_factor"], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], steps_per_year=steps_per_year, z_s=z_s, z_v=z_v, z_j=z_j, u_j=u_j, paths=paths, random_state=random_state, engine=engine)
    out["model_price"] = px
    out["mc_standard_error"] = se
    out["price_residual"] = out["model_price"] - out["mid"]
    out["model_iv"] = np.nan
    out["iv_residual"] = np.nan
    return out


def fit_bates_mc(
    quotes: pd.DataFrame,
    heston_start: dict | None = None,
    jump_start: dict | None = None,
    paths_opt: int = 2048,
    paths_final: int = 8192,
    steps_per_year: int = 52,
    random_method: str = "antithetic",
    common_random_numbers: bool = True,
    engine: str = "auto",
    random_state: int = 7,
    weight_col: str = "obs_weight",
    max_nfev: int = 45,
) -> dict:
    t0 = time.perf_counter()
    q = quotes.copy()
    engine_used = _engine_name(engine)
    if q.empty:
        return {"model": "bates", "params": pd.DataFrame(), "fit": pd.DataFrame(), "diag": pd.DataFrame(), "elapsed_sec": 0.0, "engine": engine_used}
    target = q["mid"].to_numpy(dtype=float)
    scale = _scale(q, weight_col)
    max_step = int(max(2, np.ceil(float(q["tau"].max()) * int(steps_per_year)) + 2))
    z_s_opt, z_v_opt = _rng_shocks(paths_opt, max_step, random_state=random_state + 211)
    z_j_opt, u_j_opt = _jump_shocks(paths_opt, max_step, random_state=random_state + 223)
    z_s_final, z_v_final = _rng_shocks(paths_final, max_step, random_state=random_state + 229)
    z_j_final, u_j_final = _jump_shocks(paths_final, max_step, random_state=random_state + 233)
    atm_vol = float(np.nanmedian(q["iv_mid"])) if "iv_mid" in q.columns else 0.7
    base_h = np.array([atm_vol**2, 2.0, atm_vol**2, 0.7, -0.45], dtype=float)
    if heston_start and not heston_start.get("params", pd.DataFrame()).empty:
        hp = heston_start["params"].iloc[0]
        base_h = np.array([hp["v0"], hp["kappa"], hp["theta"], hp["xi"], hp["rho"]], dtype=float)
    base_j = np.array([0.8, -0.05, 0.20], dtype=float)
    if jump_start and not jump_start.get("params", pd.DataFrame()).empty:
        jp = jump_start["params"].iloc[0]
        base_j = np.array([jp["lambda_jump"], jp["mu_jump"], jp["sigma_jump"]], dtype=float)
    start = np.r_[base_h, base_j]
    bounds = ([1e-5, 0.05, 1e-5, 0.02, -0.999, 0.0, -0.80, 0.01], [9.0, 12.0, 9.0, 5.0, 0.999, 8.0, 0.80, 1.50])

    def residual(p):
        px, _ = bates_mc_price(q["option_type"].to_numpy(), q["forward"], q["strike"], q["tau"], q["discount_factor"], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], steps_per_year=steps_per_year, z_s=z_s_opt, z_v=z_v_opt, z_j=z_j_opt, u_j=u_j_opt, engine=engine_used)
        reg = np.array([0.10 * (p[5] - base_j[0]), 0.20 * (p[6] - base_j[1]), 0.10 * (p[7] - base_j[2])])
        return np.r_[(px - target) / scale, reg]

    res = optimize.least_squares(residual, np.clip(start, bounds[0], bounds[1]), bounds=bounds, max_nfev=int(max_nfev), xtol=1e-7, ftol=1e-7, gtol=1e-7)
    p = res.x
    fit = _fit_prices(q, p, steps_per_year, z_s_final, z_v_final, z_j_final, u_j_final, paths_final, random_state, engine=engine_used)
    params = pd.DataFrame([{"v0": p[0], "kappa": p[1], "theta": p[2], "xi": p[3], "rho": p[4], "lambda_jump": p[5], "mu_jump": p[6], "sigma_jump": p[7], "loss": float(np.nanmean(res.fun**2)), "success": bool(res.success), "nfev": int(res.nfev)}])
    tail = fit[np.abs(fit.get("k", 0.0)) >= 0.14]
    diag = pd.DataFrame([{
        "model": "bates",
        "quotes": len(fit),
        "weighted_price_rmse": float(np.sqrt(np.nanmean(((fit["model_price"] - fit["mid"]) / scale) ** 2))),
        "median_abs_price_error": float(np.nanmedian(np.abs(fit["price_residual"]))),
        "tail_error": float(np.nanmedian(np.abs(tail["price_residual"]))) if not tail.empty else np.nan,
        "median_mc_standard_error": float(np.nanmedian(fit["mc_standard_error"])),
    }])
    return {
        "model": "bates",
        "params": params,
        "fit": fit,
        "diag": diag,
        "elapsed_sec": time.perf_counter() - t0,
        "engine": engine_used,
        "steps_per_year": int(steps_per_year),
        "paths_final": int(paths_final),
        "random_state": int(random_state),
    }


def bates_prices(quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    params = fit.get("params", pd.DataFrame())
    if quotes.empty or params.empty:
        return quotes.head(0).copy()
    p = params.iloc[0]
    arr = np.array([p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"], p["lambda_jump"], p["mu_jump"], p["sigma_jump"]], dtype=float)
    steps = int(fit.get("steps_per_year", 52))
    paths = int(fit.get("paths_final", 4096))
    state = int(fit.get("random_state", 7))
    return _fit_prices(quotes, arr, steps, None, None, None, None, paths, state, engine=engine)


__all__ = ["bates_cf", "fit_bates_mc", "bates_mc_price", "bates_prices"]
