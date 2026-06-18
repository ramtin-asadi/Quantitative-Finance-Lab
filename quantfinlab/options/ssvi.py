from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import optimize

from quantfinlab.options.bsm import black76_price


def ssvi_total_var(k, theta, rho, eta, gamma):
    k_arr = np.asarray(k, dtype=float)
    theta_arr = np.maximum(np.asarray(theta, dtype=float), 1e-8)
    phi = float(eta) / np.maximum(theta_arr, 1e-8) ** float(gamma)
    x = phi * k_arr
    return 0.5 * theta_arr * (1.0 + float(rho) * x + np.sqrt((x + float(rho)) ** 2 + 1.0 - float(rho) ** 2))


def _theta_curve(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in q.groupby(["date", "expiry"] if "date" in q.columns else ["expiry"], sort=True):
        near = g.iloc[np.argsort(np.abs(g["k"].to_numpy(dtype=float)))[: min(5, len(g))]]
        theta = float(np.nanmedian(pd.to_numeric(near["iv_mid"], errors="coerce") ** 2 * pd.to_numeric(near["tau"], errors="coerce")))
        values = key if isinstance(key, tuple) else (key,)
        cols = ["date", "expiry"] if "date" in q.columns else ["expiry"]
        row = dict(zip(cols, values, strict=False))
        row.update({"theta": max(theta, 1e-6), "tau": float(np.nanmedian(g["tau"])), "dte_days": float(np.nanmedian(g.get("dte_days", g["tau"] * 365.25)))})
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare(q: pd.DataFrame) -> pd.DataFrame:
    out = q.copy()
    for col in ["date", "expiry"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    if "total_var" not in out.columns:
        out["total_var"] = pd.to_numeric(out["iv_mid"], errors="coerce") ** 2 * pd.to_numeric(out["tau"], errors="coerce")
    return out


def fit_ssvi_surface(quotes: pd.DataFrame, weight_col: str = "obs_weight", engine: str = "auto") -> dict:
    t0 = time.perf_counter()
    q = _prepare(quotes)
    theta = _theta_curve(q)
    keys = ["date", "expiry"] if "date" in q.columns else ["expiry"]
    q_fit = q.drop(columns=[c for c in ["theta"] if c in q.columns])
    z = q_fit.merge(theta[keys + ["theta"]], on=keys, how="inner")
    if z.empty:
        return {"model": "ssvi", "params": pd.DataFrame(), "theta": theta, "fit": pd.DataFrame(), "diag": pd.DataFrame(), "elapsed_sec": time.perf_counter() - t0, "engine": "numpy"}
    target = z["total_var"].to_numpy(dtype=float)
    k = z["k"].to_numpy(dtype=float)
    th = z["theta"].to_numpy(dtype=float)
    if weight_col in z.columns:
        w = pd.to_numeric(z[weight_col], errors="coerce").to_numpy(dtype=float)
    else:
        w = np.ones(len(z), dtype=float)
    w = np.sqrt(np.where(np.isfinite(w) & (w > 0), w, 1.0))

    def residual(p):
        rho, eta, gamma = p
        model = ssvi_total_var(k, th, rho, eta, gamma)
        return (model - target) * w

    starts = [
        np.array([-0.55, 1.0, 0.45]),
        np.array([-0.25, 0.6, 0.35]),
        np.array([-0.75, 1.6, 0.55]),
    ]
    bounds = ([-0.999, 0.02, 0.05], [0.999, 8.0, 0.95])
    best = None
    for s in starts:
        res = optimize.least_squares(residual, s, bounds=bounds, max_nfev=220, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        val = float(np.nanmean(res.fun**2))
        if best is None or val < best[0]:
            best = (val, res)
    p = best[1].x
    params = pd.DataFrame([{"rho": p[0], "eta": p[1], "gamma": p[2], "loss": best[0], "success": bool(best[1].success), "nfev": int(best[1].nfev)}])
    fit = z.copy()
    fit["model_total_var"] = ssvi_total_var(fit["k"], fit["theta"], p[0], p[1], p[2])
    fit["model_iv"] = np.sqrt(np.maximum(fit["model_total_var"], 1e-12) / np.maximum(fit["tau"], 1e-12))
    fit["model_price"] = black76_price(fit["option_type"].to_numpy(), fit["forward"], fit["strike"], fit["tau"], fit["model_iv"], fit["discount_factor"])
    fit["iv_residual"] = fit["model_iv"] - fit["iv_mid"]
    fit["price_residual"] = fit["model_price"] - fit["mid"]
    diag = fit.groupby(keys).agg(
        quotes=("iv_mid", "size"),
        weighted_iv_rmse=("iv_residual", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
        median_abs_iv_error=("iv_residual", lambda x: float(np.nanmedian(np.abs(x)))),
        p90_abs_iv_error=("iv_residual", lambda x: float(np.nanquantile(np.abs(x), 0.90))),
    ).reset_index()
    return {"model": "ssvi", "params": params, "theta": theta, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": "numpy"}


def ssvi_prices(quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    q = _prepare(quotes)
    params = fit.get("params", pd.DataFrame())
    theta = fit.get("theta", pd.DataFrame())
    if q.empty or params.empty or theta.empty:
        return q.head(0).copy()
    keys = ["date", "expiry"] if "date" in theta.columns and "date" in q.columns else ["expiry"]
    p = params.iloc[0]
    q_fit = q.drop(columns=[c for c in ["theta"] if c in q.columns])
    out = q_fit.merge(theta[keys + ["theta"]], on=keys, how="inner")
    out["model_total_var"] = ssvi_total_var(out["k"], out["theta"], p["rho"], p["eta"], p["gamma"])
    out["model_iv"] = np.sqrt(np.maximum(out["model_total_var"], 1e-12) / np.maximum(out["tau"], 1e-12))
    out["model_price"] = black76_price(out["option_type"].to_numpy(), out["forward"], out["strike"], out["tau"], out["model_iv"], out["discount_factor"])
    out["iv_residual"] = out["model_iv"] - out["iv_mid"]
    out["price_residual"] = out["model_price"] - out["mid"]
    return out


__all__ = ["fit_ssvi_surface", "ssvi_total_var", "ssvi_prices"]
