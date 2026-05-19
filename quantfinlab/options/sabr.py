from __future__ import annotations

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
    def _sabr_hagan_iv_numba(f, k, tau, alpha, beta, rho, nu):
        out = np.empty_like(f)
        one_minus_beta = 1.0 - beta
        for i in range(f.size):
            fi = max(f[i], 1e-12)
            ki = max(k[i], 1e-12)
            ti = max(tau[i], 1e-12)
            a = max(alpha, 1e-12)
            n = max(nu, 1e-12)
            log_fk = np.log(fi / ki)
            fk_beta = (fi * ki) ** (0.5 * one_minus_beta)
            corr = 1.0 + (one_minus_beta**2 / 24.0) * log_fk**2 + (one_minus_beta**4 / 1920.0) * log_fk**4
            z = (n / a) * fk_beta * log_fk
            if abs(z) < 1e-8:
                z_over_x = 1.0
            else:
                xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
                z_over_x = z / xz
            term = (
                (one_minus_beta**2 / 24.0) * a * a / ((fi * ki) ** one_minus_beta)
                + 0.25 * rho * beta * n * a / fk_beta
                + (2.0 - 3.0 * rho * rho) * n * n / 24.0
            )
            out[i] = (a / (fk_beta * corr)) * z_over_x * (1.0 + term * ti)
        return out

else:
    _sabr_hagan_iv_numba = None


def sabr_hagan_iv(forward, strike, tau, alpha, beta, rho, nu, engine: str = "auto"):
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(tau, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    f, k, t, alpha_arr, rho_arr, nu_arr = np.broadcast_arrays(f, k, t, alpha_arr, rho_arr, nu_arr)
    if alpha_arr.size == 1 and rho_arr.size == 1 and nu_arr.size == 1 and _engine_name(engine) == "numba" and _sabr_hagan_iv_numba is not None:
        return _sabr_hagan_iv_numba(f.ravel(), k.ravel(), t.ravel(), float(alpha_arr.ravel()[0]), float(beta), float(rho_arr.ravel()[0]), float(nu_arr.ravel()[0])).reshape(f.shape)
    one_minus_beta = 1.0 - float(beta)
    f = np.maximum(f, 1e-12)
    k = np.maximum(k, 1e-12)
    t = np.maximum(t, 1e-12)
    alpha = np.maximum(alpha_arr, 1e-12)
    nu = np.maximum(nu_arr, 1e-12)
    rho = np.clip(rho_arr, -0.999, 0.999)
    log_fk = np.log(f / k)
    fk_beta = (f * k) ** (0.5 * one_minus_beta)
    corr = 1.0 + (one_minus_beta**2 / 24.0) * log_fk**2 + (one_minus_beta**4 / 1920.0) * log_fk**4
    z = (nu / alpha) * fk_beta * log_fk
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    z_over_x = np.where(np.abs(z) < 1e-8, 1.0, z / xz)
    term = (
        (one_minus_beta**2 / 24.0) * alpha**2 / ((f * k) ** one_minus_beta)
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )
    return (alpha / (fk_beta * corr)) * z_over_x * (1.0 + term * t)


def _prepare(q: pd.DataFrame) -> pd.DataFrame:
    out = q.copy()
    for col in ["date", "expiry"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _weight(q: pd.DataFrame, weight_col: str | None) -> np.ndarray:
    if weight_col and weight_col in q.columns:
        w = pd.to_numeric(q[weight_col], errors="coerce").to_numpy(dtype=float)
    elif "surface_weight" in q.columns:
        w = pd.to_numeric(q["surface_weight"], errors="coerce").to_numpy(dtype=float)
    else:
        w = np.ones(len(q), dtype=float)
    return np.sqrt(np.where(np.isfinite(w) & (w > 0), w, 1.0))


def _fit_slice(q: pd.DataFrame, beta: float, weight_col: str, engine: str, start: np.ndarray | None = None):
    f = q["forward"].to_numpy(dtype=float)
    k = q["strike"].to_numpy(dtype=float)
    t = q["tau"].to_numpy(dtype=float)
    y = q["iv_mid"].to_numpy(dtype=float)
    sw = _weight(q, weight_col)
    mask = np.isfinite(f) & np.isfinite(k) & np.isfinite(t) & np.isfinite(y) & (f > 0) & (k > 0) & (t > 0) & (y > 0)
    f, k, t, y, sw = f[mask], k[mask], t[mask], y[mask], sw[mask]
    if len(y) < 5:
        return np.array([np.nan, np.nan, np.nan]), np.nan, False, 0
    atm = float(np.nanmedian(y[np.argsort(np.abs(np.log(k / f)))[: min(4, len(y))]]))
    f_atm = float(np.nanmedian(f))
    alpha0 = max(atm * f_atm ** (1.0 - beta), 1e-4)
    starts = [np.array([alpha0, -0.45, 0.80]), np.array([0.75 * alpha0, -0.75, 1.50]), np.array([1.25 * alpha0, 0.05, 0.45])]
    if start is not None and np.all(np.isfinite(start)):
        starts.insert(0, np.asarray(start, dtype=float))
    bounds = ([1e-8, -0.999, 1e-6], [10.0 * max(alpha0, 1.0), 0.999, 8.0])

    def residual(p):
        model = sabr_hagan_iv(f, k, t, p[0], beta, p[1], p[2], engine=engine)
        return (model - y) * sw

    best = None
    for s in starts:
        s = np.clip(s, bounds[0], bounds[1])
        res = optimize.least_squares(residual, s, bounds=bounds, max_nfev=180, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        val = float(np.nanmean(res.fun**2))
        if best is None or val < best[0]:
            best = (val, res.x, bool(res.success), int(res.nfev))
    return best[1], best[0], best[2], best[3]


def _price_fit(q: pd.DataFrame, params: pd.DataFrame, beta: float, engine: str) -> pd.DataFrame:
    if q.empty or params.empty:
        return q.head(0).copy()
    keys = ["date", "expiry"] if "date" in params.columns and "date" in q.columns else ["expiry"]
    out = q.copy()
    for col in ["alpha", "rho", "nu", "beta"]:
        if col in out.columns and col not in keys:
            out = out.drop(columns=col)
    out = out.merge(params[keys + ["alpha", "rho", "nu"]], on=keys, how="inner")
    out["model_iv"] = sabr_hagan_iv(out["forward"], out["strike"], out["tau"], out["alpha"], beta, out["rho"], out["nu"], engine=engine)
    out["model_price"] = black76_price(out["option_type"].to_numpy(), out["forward"], out["strike"], out["tau"], out["model_iv"], out["discount_factor"])
    out["iv_residual"] = out["model_iv"] - out["iv_mid"]
    out["price_residual"] = out["model_price"] - out["mid"]
    return out


def fit_sabr_surface(
    quotes: pd.DataFrame,
    betas=(1.0, 0.7, 0.5),
    primary_beta: float = 1.0,
    weight_col: str = "obs_weight",
    engine: str = "auto",
) -> dict:
    t0 = time.perf_counter()
    engine_used = _engine_name(engine)
    q = _prepare(quotes)
    beta_rows = []
    stores = {}
    for beta in betas:
        rows = []
        last = None
        group_cols = ["date", "expiry"] if "date" in q.columns else ["expiry"]
        for key, g in q.groupby(group_cols, sort=True):
            if len(g) < 5:
                continue
            p, loss, ok, nfev = _fit_slice(g, float(beta), weight_col, engine_used, start=last)
            if np.all(np.isfinite(p)):
                last = p
            values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(group_cols, values))
            row.update({"beta": float(beta), "alpha": p[0], "rho": p[1], "nu": p[2], "loss": loss, "success": ok, "nfev": nfev, "quotes": len(g), "dte_days": float(np.nanmedian(g.get("dte_days", g["tau"] * 365.25)))})
            rows.append(row)
        params = pd.DataFrame(rows)
        fit = _price_fit(q, params, float(beta), engine_used)
        err = float(np.sqrt(np.nanmean(fit["iv_residual"] ** 2))) if not fit.empty else np.nan
        beta_rows.append({"beta": float(beta), "weighted_iv_rmse": err, "slices": len(params), "success_rate": float(params["success"].mean()) if not params.empty else np.nan})
        stores[float(beta)] = (params, fit)
    primary = float(primary_beta)
    params, fit = stores.get(primary, next(iter(stores.values())) if stores else (pd.DataFrame(), pd.DataFrame()))
    diag = fit.groupby(["date", "expiry"] if "date" in fit.columns else ["expiry"]).agg(
        quotes=("iv_mid", "size"),
        weighted_iv_rmse=("iv_residual", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
        median_abs_iv_error=("iv_residual", lambda x: float(np.nanmedian(np.abs(x)))),
        p90_abs_iv_error=("iv_residual", lambda x: float(np.nanquantile(np.abs(x), 0.90))),
    ).reset_index() if not fit.empty else pd.DataFrame()
    return {"model": "sabr", "beta": primary, "params": params, "fit": fit, "diag": diag, "beta_compare": pd.DataFrame(beta_rows), "elapsed_sec": time.perf_counter() - t0, "engine": engine_used}


def fit_sabr_holdout(
    quotes: pd.DataFrame,
    dates,
    beta: float = 1.0,
    train_mode: str = "anchor_strikes",
    weight_col: str = "obs_weight",
    warm_start: bool = True,
    engine: str = "auto",
) -> dict:
    t0 = time.perf_counter()
    q = _prepare(quotes)
    if "quote_id" not in q.columns:
        q["quote_id"] = np.arange(len(q), dtype=int)
    params_parts = []
    holdout_parts = []
    diag_parts = []
    for d in pd.to_datetime(pd.Series(dates), errors="coerce").dropna().dt.normalize().unique():
        day = q[q["date"].eq(pd.Timestamp(d))].copy()
        anchors = []
        for _, g in day.sort_values(["expiry", "k"]).groupby("expiry"):
            if len(g) < 7:
                continue
            take = g.iloc[np.linspace(0, len(g) - 1, min(7, len(g))).round().astype(int)]
            anchors.append(take)
        train = pd.concat(anchors, ignore_index=False).drop_duplicates("quote_id") if anchors else day.head(0)
        holdout = day[~day["quote_id"].isin(train["quote_id"])].copy()
        if len(train) < 18 or holdout.empty:
            continue
        fit = fit_sabr_surface(train, betas=(beta,), primary_beta=beta, weight_col=weight_col, engine=engine)
        p = fit["params"].copy()
        if p.empty:
            continue
        p["date"] = pd.Timestamp(d)
        params_parts.append(p)
        holdout_parts.append(holdout[["date", "quote_id"]].copy())
        dg = fit["diag"].copy()
        dg["date"] = pd.Timestamp(d)
        diag_parts.append(dg)
    params = pd.concat(params_parts, ignore_index=True) if params_parts else pd.DataFrame()
    holdout_ids = pd.concat(holdout_parts, ignore_index=True) if holdout_parts else pd.DataFrame(columns=["date", "quote_id"])
    diag = pd.concat(diag_parts, ignore_index=True) if diag_parts else pd.DataFrame()
    fit = _price_fit(q.merge(holdout_ids, on=["date", "quote_id"], how="inner"), params, float(beta), _engine_name(engine)) if not params.empty else pd.DataFrame()
    return {"model": "sabr", "beta": float(beta), "params": params, "holdout_ids": holdout_ids, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": _engine_name(engine)}


def sabr_prices(quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    return _price_fit(_prepare(quotes), fit.get("params", pd.DataFrame()), float(fit.get("beta", 1.0)), _engine_name(engine))


__all__ = ["fit_sabr_surface", "fit_sabr_holdout", "sabr_hagan_iv", "sabr_prices"]
