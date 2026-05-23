from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import optimize

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


def _rng_shocks(n_path: int, n_step: int, random_state: int = 7):
    n_half = max(int(n_path) // 2, 2)
    rng = np.random.default_rng(int(random_state))
    z1 = rng.normal(size=(n_half, n_step))
    z2 = rng.normal(size=(n_half, n_step))
    return np.vstack([z1, -z1]).astype(float), np.vstack([z2, -z2]).astype(float)


def heston_cf(u, spot, rate, dividend_yield, tau, v0, kappa, theta, sigma_v, rho):
    """Stable little-Heston-trap characteristic function of log spot."""
    u = np.asarray(u, dtype=complex)
    spot = np.asarray(spot, dtype=float)
    tau = np.asarray(tau, dtype=float)
    v0 = np.maximum(np.asarray(v0, dtype=float), 1e-10)
    kappa = np.maximum(np.asarray(kappa, dtype=float), 1e-10)
    theta = np.maximum(np.asarray(theta, dtype=float), 1e-10)
    sigma_v = np.maximum(np.asarray(sigma_v, dtype=float), 1e-10)
    rho = np.clip(np.asarray(rho, dtype=float), -0.999, 0.999)
    i = 1j
    d = np.sqrt((rho * sigma_v * i * u - kappa) ** 2 + sigma_v**2 * (i * u + u * u))
    g = (kappa - rho * sigma_v * i * u - d) / (kappa - rho * sigma_v * i * u + d)
    exp_dt = np.exp(-d * tau)
    c = (
        i * u * (np.log(spot) + (np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float)) * tau)
        + (kappa * theta / sigma_v**2)
        * ((kappa - rho * sigma_v * i * u - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    )
    dcoef = ((kappa - rho * sigma_v * i * u - d) / sigma_v**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(c + dcoef * v0)


def _scale(q: pd.DataFrame, weight_col: str | None) -> np.ndarray:
    if "calib_scale_px" in q.columns:
        s = pd.to_numeric(q["calib_scale_px"], errors="coerce").to_numpy(dtype=float)
    elif "half_spread" in q.columns:
        s = pd.to_numeric(q["half_spread"], errors="coerce").to_numpy(dtype=float)
    else:
        s = np.full(len(q), np.nanmedian(q["mid"]) * 0.02 if len(q) else 1.0)
    good = np.isfinite(s) & (s > 0)
    fill = float(np.nanmedian(s[good])) if good.any() else 1.0
    s = np.where(good, s, fill)
    if weight_col and weight_col in q.columns:
        w = pd.to_numeric(q[weight_col], errors="coerce").to_numpy(dtype=float)
        s = s / np.sqrt(np.where(np.isfinite(w) & (w > 0), w, 1.0))
    return np.maximum(s, 1e-6)


if _has_numba:

    @njit(cache=True)
    def _heston_mc_price_numba(is_call, forward, strike, tau, df, v0, kappa, theta, xi, rho, steps_per_year, z_s, z_v):
        n_quote = forward.shape[0]
        n_path = z_s.shape[0]
        prices = np.empty(n_quote)
        ses = np.empty(n_quote)
        rho_c = min(max(rho, -0.999), 0.999)
        rho_scale = np.sqrt(1.0 - rho_c * rho_c)
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
                    log_rel += -0.5 * vp * dt + np.sqrt(vp * dt) * zs
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


def heston_mc_price(
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
    steps_per_year: int = 52,
    z_s=None,
    z_v=None,
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
    if _engine_name(engine) == "numba":
        is_call = np.char.lower(option_type.astype(str)).astype(str)
        is_call = np.array([str(x).startswith("c") for x in is_call], dtype=np.bool_)
        return _heston_mc_price_numba(is_call, forward, strike, tau, df, float(v0), float(kappa), float(theta), float(xi), float(rho), int(steps_per_year), z_s, z_v)
    n_path = z_s.shape[0]
    prices = np.empty(len(forward), dtype=float)
    ses = np.empty(len(forward), dtype=float)
    rho = float(np.clip(rho, -0.999, 0.999))
    for i in range(len(forward)):
        steps = int(max(2, np.ceil(float(tau[i]) * int(steps_per_year))))
        dt = float(tau[i]) / steps
        v = np.full(n_path, max(float(v0), 1e-8), dtype=float)
        log_rel = np.zeros(n_path, dtype=float)
        for j in range(steps):
            zv = rho * z_s[:, j] + np.sqrt(1.0 - rho * rho) * z_v[:, j]
            vp = np.maximum(v, 0.0)
            log_rel += -0.5 * vp * dt + np.sqrt(vp * dt) * z_s[:, j]
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


def _fit_prices(q: pd.DataFrame, p: np.ndarray, steps_per_year: int, z_s, z_v, paths: int, random_state: int, engine: str = "auto") -> pd.DataFrame:
    out = q.copy()
    px, se = heston_mc_price(out["option_type"].to_numpy(), out["forward"], out["strike"], out["tau"], out["discount_factor"], p[0], p[1], p[2], p[3], p[4], steps_per_year=steps_per_year, z_s=z_s, z_v=z_v, paths=paths, random_state=random_state, engine=engine)
    out["model_price"] = px
    out["mc_standard_error"] = se
    out["price_residual"] = out["model_price"] - out["mid"]
    out["model_iv"] = np.nan
    out["iv_residual"] = np.nan
    return out


def fit_heston_mc(
    quotes: pd.DataFrame,
    paths_opt: int = 2048,
    paths_final: int = 8192,
    steps_per_year: int = 52,
    random_method: str = "antithetic",
    common_random_numbers: bool = True,
    engine: str = "auto",
    random_state: int = 7,
    weight_col: str = "obs_weight",
    max_nfev: int = 55,
) -> dict:
    t0 = time.perf_counter()
    q = quotes.copy()
    engine_used = _engine_name(engine)
    if q.empty:
        return {"model": "heston", "params": pd.DataFrame(), "fit": pd.DataFrame(), "diag": pd.DataFrame(), "elapsed_sec": 0.0, "engine": engine_used}
    target = q["mid"].to_numpy(dtype=float)
    scale = _scale(q, weight_col)
    max_step = int(max(2, np.ceil(float(q["tau"].max()) * int(steps_per_year)) + 2))
    z_s_opt, z_v_opt = _rng_shocks(paths_opt, max_step, random_state=random_state + 11)
    z_s_final, z_v_final = _rng_shocks(paths_final, max_step, random_state=random_state + 17)
    atm_vol = float(np.nanmedian(q["iv_mid"])) if "iv_mid" in q.columns else 0.7
    v = max(atm_vol**2, 1e-4)
    starts = [np.array([v, 2.0, v, 0.70, -0.45]), np.array([v, 4.0, 0.8 * v, 1.00, -0.70]), np.array([1.2 * v, 1.0, 1.2 * v, 0.40, -0.20])]
    bounds = ([1e-5, 0.05, 1e-5, 0.02, -0.999], [9.0, 12.0, 9.0, 5.0, 0.999])

    def residual(p):
        px, _ = heston_mc_price(q["option_type"].to_numpy(), q["forward"], q["strike"], q["tau"], q["discount_factor"], p[0], p[1], p[2], p[3], p[4], steps_per_year=steps_per_year, z_s=z_s_opt, z_v=z_v_opt, engine=engine_used)
        return (px - target) / scale

    best = None
    for s in starts:
        res = optimize.least_squares(residual, np.clip(s, bounds[0], bounds[1]), bounds=bounds, max_nfev=int(max_nfev), xtol=1e-7, ftol=1e-7, gtol=1e-7)
        val = float(np.nanmean(res.fun**2))
        if best is None or val < best[0]:
            best = (val, res)
    p = best[1].x
    fit = _fit_prices(q, p, steps_per_year, z_s_final, z_v_final, paths_final, random_state, engine=engine_used)
    params = pd.DataFrame([{"v0": p[0], "kappa": p[1], "theta": p[2], "xi": p[3], "rho": p[4], "loss": best[0], "success": bool(best[1].success), "nfev": int(best[1].nfev), "feller_ratio": float(2.0 * p[1] * p[2] / max(p[3] ** 2, 1e-12))}])
    tail = fit[np.abs(fit.get("k", 0.0)) >= 0.14]
    diag = pd.DataFrame([{
        "model": "heston",
        "quotes": len(fit),
        "weighted_price_rmse": float(np.sqrt(np.nanmean(((fit["model_price"] - fit["mid"]) / scale) ** 2))),
        "median_abs_price_error": float(np.nanmedian(np.abs(fit["price_residual"]))),
        "tail_error": float(np.nanmedian(np.abs(tail["price_residual"]))) if not tail.empty else np.nan,
        "median_mc_standard_error": float(np.nanmedian(fit["mc_standard_error"])),
    }])
    conv_rows = []
    for paths in [512, 1024, 2048, 4096, min(int(paths_final), 8192)]:
        if paths > int(paths_final):
            continue
        px, se = heston_mc_price(q["option_type"].to_numpy()[:1], q["forward"].to_numpy()[:1], q["strike"].to_numpy()[:1], q["tau"].to_numpy()[:1], q["discount_factor"].to_numpy()[:1], p[0], p[1], p[2], p[3], p[4], steps_per_year=steps_per_year, paths=paths, random_state=random_state + 29, engine=engine_used)
        conv_rows.append({"paths": paths, "price": float(px[0]), "standard_error": float(se[0])})
    return {
        "model": "heston",
        "params": params,
        "fit": fit,
        "diag": diag,
        "mc_error": fit[["quote_id", "mc_standard_error"]] if "quote_id" in fit.columns else fit[["mc_standard_error"]],
        "mc_convergence": pd.DataFrame(conv_rows).drop_duplicates("paths"),
        "elapsed_sec": time.perf_counter() - t0,
        "engine": engine_used,
        "steps_per_year": int(steps_per_year),
        "paths_final": int(paths_final),
        "random_state": int(random_state),
    }


def heston_prices(quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    params = fit.get("params", pd.DataFrame())
    if quotes.empty or params.empty:
        return quotes.head(0).copy()
    p = params.iloc[0]
    steps = int(fit.get("steps_per_year", 52))
    paths = int(fit.get("paths_final", 4096))
    state = int(fit.get("random_state", 7))
    return _fit_prices(quotes, np.array([p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"]], dtype=float), steps, None, None, paths, state, engine=engine)


__all__ = ["fit_heston_mc", "heston_cf", "heston_mc_price", "heston_prices"]
