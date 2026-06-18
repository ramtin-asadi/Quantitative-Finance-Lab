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
    if str(engine).lower() == "numba" and not _has_numba:
        return "numpy"
    if str(engine).lower() == "auto":
        return "numba" if _has_numba else "numpy"
    return str(engine).lower()


if _has_numba:

    @njit(cache=True)
    def _svi_total_var_numba(k, a, b, rho, m, sigma):
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))

else:
    _svi_total_var_numba = None


def svi_total_var(k, a, b, rho, m, sigma, engine: str = "auto"):
    k_arr = np.asarray(k, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    scalar_params = all(x.size == 1 for x in [a_arr, b_arr, rho_arr, m_arr, sigma_arr])
    if scalar_params and _engine_name(engine) == "numba" and _svi_total_var_numba is not None:
        return _svi_total_var_numba(
            k_arr,
            float(a_arr.ravel()[0]),
            float(b_arr.ravel()[0]),
            float(rho_arr.ravel()[0]),
            float(m_arr.ravel()[0]),
            float(sigma_arr.ravel()[0]),
        )
    k_arr, a_arr, b_arr, rho_arr, m_arr, sigma_arr = np.broadcast_arrays(k_arr, a_arr, b_arr, rho_arr, m_arr, sigma_arr)
    return a_arr + b_arr * (rho_arr * (k_arr - m_arr) + np.sqrt((k_arr - m_arr) ** 2 + sigma_arr**2))


def svi_iv(k, tau, a, b, rho, m, sigma, engine: str = "auto"):
    w = svi_total_var(k, a, b, rho, m, sigma, engine=engine)
    return np.sqrt(np.maximum(w, 1e-12) / np.maximum(np.asarray(tau, dtype=float), 1e-12))


def _weight(q: pd.DataFrame, weight_col: str | None) -> np.ndarray:
    if weight_col and weight_col in q.columns:
        w = pd.to_numeric(q[weight_col], errors="coerce").to_numpy(dtype=float)
    elif "surface_weight" in q.columns:
        w = pd.to_numeric(q["surface_weight"], errors="coerce").to_numpy(dtype=float)
    else:
        w = np.ones(len(q), dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    return np.sqrt(w / max(float(np.nanmedian(w)), 1e-12))


def _fit_slice(q: pd.DataFrame, weight_col: str | None, engine: str, start: np.ndarray | None = None) -> tuple[np.ndarray, float, bool, int]:
    x = pd.to_numeric(q["k"], errors="coerce").to_numpy(dtype=float)
    tau = float(np.nanmedian(pd.to_numeric(q["tau"], errors="coerce")))
    y = pd.to_numeric(q["iv_mid"], errors="coerce").to_numpy(dtype=float) ** 2 * tau
    sw = _weight(q, weight_col)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y, sw = x[mask], y[mask], sw[mask]
    if len(x) < 5:
        return np.array([np.nan] * 5, dtype=float), np.nan, False, 0
    atm_w = float(np.nanmedian(y[np.argsort(np.abs(x))[: min(4, len(x))]]))
    spread_w = max(float(np.nanpercentile(y, 80) - np.nanpercentile(y, 20)), 1e-4)
    starts = [
        np.array([0.25 * atm_w, 0.50 * spread_w + 1e-4, -0.45, 0.0, 0.12]),
        np.array([0.10 * atm_w, 0.80 * spread_w + 1e-4, -0.70, -0.03, 0.08]),
        np.array([0.40 * atm_w, 0.30 * spread_w + 1e-4, 0.00, 0.02, 0.18]),
    ]
    if start is not None and np.all(np.isfinite(start)):
        starts.insert(0, np.asarray(start, dtype=float))
    bounds = ([-1.0, 1e-7, -0.999, -1.5, 1e-4], [3.0, 10.0, 0.999, 1.5, 2.0])

    def residual(p):
        w = svi_total_var(x, p[0], p[1], p[2], p[3], p[4], engine=engine)
        bad = (~np.isfinite(w)) | (w <= 0)
        r = (w - y) * sw
        if bad.any():
            r = np.where(bad, 1e3 + np.abs(w), r)
        return r

    best = None
    for s in starts:
        s = np.clip(s, bounds[0], bounds[1])
        res = optimize.least_squares(residual, s, bounds=bounds, max_nfev=180, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        val = float(np.nanmean(res.fun**2)) if res.fun.size else np.inf
        if best is None or val < best[0]:
            best = (val, res.x, bool(res.success), int(res.nfev))
    return best[1], best[0], best[2], best[3]


def _prepare(q: pd.DataFrame) -> pd.DataFrame:
    out = q.copy()
    for col in ["date", "expiry"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    if "total_var" not in out.columns:
        out["total_var"] = pd.to_numeric(out["iv_mid"], errors="coerce") ** 2 * pd.to_numeric(out["tau"], errors="coerce")
    return out


def _price_fit(q: pd.DataFrame, params: pd.DataFrame, engine: str) -> pd.DataFrame:
    if q.empty or params.empty:
        return q.head(0).copy()
    out = q.copy()
    keys = ["date", "expiry"] if "date" in params.columns and "date" in out.columns else ["expiry"]
    for col in ["a", "b", "rho", "m", "sigma"]:
        if col in out.columns and col not in keys:
            out = out.drop(columns=col)
    p = params.copy()
    out = out.merge(p[keys + ["a", "b", "rho", "m", "sigma"]], on=keys, how="inner")
    out["model_iv"] = svi_iv(out["k"], out["tau"], out["a"], out["b"], out["rho"], out["m"], out["sigma"], engine=engine)
    out["model_price"] = black76_price(out["option_type"].to_numpy(), out["forward"], out["strike"], out["tau"], out["model_iv"], out["discount_factor"])
    out["iv_residual"] = out["model_iv"] - out["iv_mid"]
    out["price_residual"] = out["model_price"] - out["mid"]
    return out


def fit_svi_surface(quotes: pd.DataFrame, weight_col: str = "obs_weight", engine: str = "auto") -> dict:
    t0 = time.perf_counter()
    engine_used = _engine_name(engine)
    q = _prepare(quotes)
    rows = []
    last = None
    group_cols = ["date", "expiry"] if "date" in q.columns else ["expiry"]
    for key, g in q.groupby(group_cols, sort=True):
        if len(g) < 5:
            continue
        p, loss, ok, nfev = _fit_slice(g, weight_col, engine_used, start=last)
        if np.all(np.isfinite(p)):
            last = p
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, values, strict=False))
        row.update({"a": p[0], "b": p[1], "rho": p[2], "m": p[3], "sigma": p[4], "loss": loss, "success": ok, "nfev": nfev, "quotes": len(g), "dte_days": float(np.nanmedian(g.get("dte_days", g["tau"] * 365.25)))})
        rows.append(row)
    params = pd.DataFrame(rows)
    fit = _price_fit(q, params, engine_used)
    if fit.empty:
        diag = pd.DataFrame()
    else:
        diag = fit.groupby(group_cols).agg(
            quotes=("iv_mid", "size"),
            weighted_iv_rmse=("iv_residual", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            median_abs_iv_error=("iv_residual", lambda x: float(np.nanmedian(np.abs(x)))),
            p90_abs_iv_error=("iv_residual", lambda x: float(np.nanquantile(np.abs(x), 0.90))),
        ).reset_index()
    return {"model": "svi", "params": params, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": engine_used}


def fit_svi_holdout(
    quotes: pd.DataFrame,
    dates,
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
        if day.empty:
            continue
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
        fit = fit_svi_surface(train, weight_col=weight_col, engine=engine)
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
    fit = _price_fit(q.merge(holdout_ids, on=["date", "quote_id"], how="inner"), params, _engine_name(engine)) if not params.empty else pd.DataFrame()
    return {"model": "svi", "params": params, "holdout_ids": holdout_ids, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": _engine_name(engine)}


def svi_prices(quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    return _price_fit(_prepare(quotes), fit.get("params", pd.DataFrame()), _engine_name(engine))


__all__ = ["fit_svi_surface", "fit_svi_holdout", "svi_total_var", "svi_iv", "svi_prices"]
