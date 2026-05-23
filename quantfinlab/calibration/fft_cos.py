from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import optimize

from quantfinlab.options.fourier import cos_prices


def _numeric_column(frame: pd.DataFrame, names, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def calibration_weights(quotes: pd.DataFrame) -> pd.DataFrame:
    out = quotes.copy()
    spread = _numeric_column(out, ("rel_spread", "relative_spread"), 0.10).clip(lower=0.003)
    vega = _numeric_column(out, ("vega",), 1.0).abs().fillna(1.0)
    if "dte_days" not in out.columns:
        out["dte_days"] = pd.to_numeric(out.get("tau"), errors="coerce") * 365.25
    if "calib_scale_px" in out.columns:
        out["calib_scale_px"] = pd.to_numeric(out["calib_scale_px"], errors="coerce").fillna(0.05).clip(lower=1e-5)
    elif "half_spread" in out.columns:
        out["calib_scale_px"] = pd.to_numeric(out["half_spread"], errors="coerce").fillna(0.05).clip(lower=1e-5)
    else:
        out["calib_scale_px"] = (spread * _numeric_column(out, ("mid",), 1.0)).fillna(0.05).clip(lower=1e-5)
    out["obs_weight"] = (1.0 / (spread**2 + 0.03**2)) * np.clip((vega / max(float(np.nanmedian(vega)), 1e-8)) ** 0.35, 0.20, 3.0)
    out["obs_weight"] = out["obs_weight"] / max(float(np.nanmedian(out["obs_weight"])), 1e-12)
    return out


def cos_group_arrays(quotes: pd.DataFrame) -> list[tuple]:
    q = quotes.reset_index(drop=True)
    group_cols = [c for c in ["date", "expiry", "option_type"] if c in q.columns]
    if len(group_cols) < 3:
        pos = np.arange(len(q), dtype=np.int64)
        return [
            (
                pos,
                q["strike"].to_numpy(float),
                q["tau"].to_numpy(float),
                q["spot"].to_numpy(float),
                _numeric_column(q, ("rate",), 0.0).to_numpy(float),
                _numeric_column(q, ("dividend_yield", "implied_dividend_yield"), 0.0).to_numpy(float),
                q["option_type"].to_numpy(),
            )
        ]
    groups = []
    for _, idx in q.groupby(group_cols, sort=False).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        g = q.iloc[pos]
        groups.append(
            (
                pos,
                g["strike"].to_numpy(float),
                g["tau"].to_numpy(float),
                float(pd.to_numeric(g["spot"], errors="coerce").median()),
                float(_numeric_column(g, ("rate",), 0.0).median()),
                float(_numeric_column(g, ("dividend_yield", "implied_dividend_yield"), 0.0).median()),
                str(g["option_type"].iloc[0]),
            )
        )
    return groups


def cos_prices_from_groups(
    model: str,
    params,
    groups: list[tuple],
    n_rows: int,
    *,
    engine: str = "numba",
    n_terms: int = 160,
    truncation_width: float = 12.0,
) -> np.ndarray:
    out = np.empty(int(n_rows), dtype=float)
    for pos, strike, tau, spot, rate, dividend_yield, option_type in groups:
        out[pos] = cos_prices(
            model,
            params,
            strike,
            tau,
            spot,
            rate,
            dividend_yield,
            option_type=option_type,
            n_terms=int(n_terms),
            truncation_width=float(truncation_width),
            engine=engine,
        )
    return out


def cos_prices_grouped(
    model: str,
    params,
    quotes: pd.DataFrame,
    *,
    engine: str = "numba",
    n_terms: int = 160,
    truncation_width: float = 12.0,
) -> np.ndarray:
    groups = cos_group_arrays(quotes)
    return cos_prices_from_groups(model, params, groups, len(quotes), engine=engine, n_terms=n_terms, truncation_width=truncation_width)


def calibration_grid_quotes(
    quotes: pd.DataFrame,
    *,
    min_dte: float = 14.0,
    max_dte: float = 150.0,
    min_vega: float = 0.0,
    max_relative_spread: float = 0.45,
    max_abs_log_moneyness: float = 0.32,
    min_quotes_per_expiry: int = 8,
    min_expiries_per_date: int = 5,
    dte_targets=(14.0, 21.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 150.0),
    k_targets=(-0.30, -0.25, -0.20, -0.16, -0.12, -0.08, -0.04, -0.015, 0.0, 0.015, 0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30),
    min_quotes_per_date: int = 120,
    max_quotes_per_date: int = 200,
    return_steps: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    x = quotes.copy()
    rows = [{"step": "clean quotes", "rows": int(len(x)), "removed": 0}]
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce").dt.normalize()
    if "dte_days" not in x.columns:
        x["dte_days"] = pd.to_numeric(x.get("tau"), errors="coerce") * 365.25
    if "log_moneyness" not in x.columns:
        if "k" in x.columns:
            x["log_moneyness"] = pd.to_numeric(x["k"], errors="coerce")
        else:
            x["log_moneyness"] = np.log(pd.to_numeric(x["strike"], errors="coerce") / pd.to_numeric(x["spot"], errors="coerce"))
    if "moneyness" not in x.columns:
        x["moneyness"] = np.exp(pd.to_numeric(x["log_moneyness"], errors="coerce"))
    if "relative_spread" not in x.columns:
        if "rel_spread" in x.columns:
            x["relative_spread"] = pd.to_numeric(x["rel_spread"], errors="coerce")
        elif {"bid", "ask", "mid"}.issubset(x.columns):
            x["relative_spread"] = (pd.to_numeric(x["ask"], errors="coerce") - pd.to_numeric(x["bid"], errors="coerce")) / pd.to_numeric(x["mid"], errors="coerce").replace(0.0, np.nan)
        else:
            x["relative_spread"] = 0.10
    if "quote_id" not in x.columns:
        x["quote_id"] = np.arange(len(x), dtype=np.int64)
    x["_row_id"] = np.arange(len(x), dtype=np.int64)

    def keep(name, mask):
        nonlocal x
        before = len(x)
        x = x.loc[mask].copy()
        rows.append({"step": name, "rows": int(len(x)), "removed": int(before - len(x))})

    keep("finite market fields", x[["date", "expiry", "spot", "strike", "mid", "tau"]].notna().all(axis=1))
    keep("DTE window", pd.to_numeric(x["dte_days"], errors="coerce").between(float(min_dte), float(max_dte)))
    keep("log-moneyness window", pd.to_numeric(x["log_moneyness"], errors="coerce").abs().le(float(max_abs_log_moneyness)))
    keep("spread window", pd.to_numeric(x["relative_spread"], errors="coerce").between(0.0, float(max_relative_spread)))
    if "vega" in x.columns and float(min_vega) > 0.0:
        keep("vega floor", pd.to_numeric(x["vega"], errors="coerce").abs().ge(float(min_vega)))

    opt = x["option_type"].astype(str).str.lower()
    k = pd.to_numeric(x["log_moneyness"], errors="coerce")
    otm_side = np.where(k < -0.0125, "put", np.where(k > 0.0125, "call", "either"))
    opt_side = np.where(opt.str.startswith("p"), "put", np.where(opt.str.startswith("c"), "call", "other"))
    x["type_penalty"] = np.where(otm_side == "either", 0, np.where(opt_side == otm_side, 0, 1))
    x["quote_score"] = pd.to_numeric(x["relative_spread"], errors="coerce").fillna(1.0) + 0.02 * k.abs().fillna(1.0)
    before = len(x)
    x = x.sort_values(["date", "expiry", "strike", "type_penalty", "quote_score"]).drop_duplicates(["date", "expiry", "strike"], keep="first").copy()
    rows.append({"step": "OTM side by strike", "rows": int(len(x)), "removed": int(before - len(x))})

    x["dte_bucket"] = pd.cut(x["dte_days"], [0, 14, 30, 60, 90, 120, 180], include_lowest=True)
    x["log_moneyness_bucket"] = pd.cut(x["log_moneyness"], [-0.45, -0.30, -0.20, -0.12, -0.06, -0.02, 0.02, 0.06, 0.12, 0.20, 0.30, 0.45], include_lowest=True)
    support_keys = ["date", "expiry", "option_type", "dte_bucket", "log_moneyness_bucket"]
    x["cell_rows"] = x.groupby(support_keys, observed=True)["mid"].transform("size").fillna(1).astype(float)
    counts = x.groupby(["date", "expiry"]).size().rename("expiry_rows")
    x = x.merge(counts.reset_index(), on=["date", "expiry"], how="left")
    before = len(x)
    x = x[x["expiry_rows"].ge(int(min_quotes_per_expiry))].copy()
    rows.append({"step": "expiry support", "rows": int(len(x)), "removed": int(before - len(x))})
    date_expiries = x.groupby("date")["expiry"].nunique()
    good_dates = date_expiries[date_expiries.ge(int(min_expiries_per_date))].index
    before = len(x)
    x = x[x["date"].isin(good_dates)].copy()
    rows.append({"step": "date support", "rows": int(len(x)), "removed": int(before - len(x))})

    dte_targets = np.asarray(dte_targets, dtype=float)
    k_targets = np.asarray(k_targets, dtype=float)
    pieces = []
    for d, day in x.groupby("date", sort=True):
        expiry_table = day.groupby("expiry", as_index=False).agg(dte=("dte_days", "median"), n=("quote_id", "size"))
        expiry_table = expiry_table[expiry_table["n"].ge(int(min_quotes_per_expiry))].copy()
        chosen_expiries = []
        used_expiries = set()
        for target in dte_targets:
            available = expiry_table[~expiry_table["expiry"].isin(used_expiries)].copy()
            if available.empty:
                break
            idx = (available["dte"] - float(target)).abs().idxmin()
            expiry = available.loc[idx, "expiry"]
            chosen_expiries.append(expiry)
            used_expiries.add(expiry)
        ids = []
        selected_targets = {}
        for expiry in chosen_expiries:
            g = day[day["expiry"].eq(expiry)].copy()
            used = set()
            for kt in k_targets:
                r = g[~g["_row_id"].isin(used)].copy()
                if r.empty:
                    break
                score = (pd.to_numeric(r["log_moneyness"], errors="coerce") - float(kt)).abs() + 0.03 * pd.to_numeric(r["relative_spread"], errors="coerce").fillna(1.0)
                idx = score.idxmin()
                rid = int(r.loc[idx, "_row_id"])
                ids.append(rid)
                selected_targets[rid] = (float(kt), float(g["dte_days"].median()))
                used.add(rid)
        selected = day[day["_row_id"].isin(ids)].copy()
        if len(selected) < int(min_quotes_per_date):
            remain = day[~day["_row_id"].isin(selected["_row_id"])].copy()
            if not remain.empty:
                dte_gap = np.min(np.abs(remain["dte_days"].to_numpy(float)[:, None] - dte_targets[None, :]), axis=1)
                k_gap = np.min(np.abs(remain["log_moneyness"].to_numpy(float)[:, None] - k_targets[None, :]), axis=1)
                remain["selection_score"] = 0.003 * dte_gap + k_gap + 0.10 * pd.to_numeric(remain["relative_spread"], errors="coerce").fillna(1.0)
                remain["selection_rank"] = remain["selection_score"].rank(method="first")
                need = max(int(min_quotes_per_date) - len(selected), 0)
                selected = pd.concat([selected, remain[remain["selection_rank"].le(need)]], ignore_index=False)
        if len(selected) > int(max_quotes_per_date):
            dte_gap = np.min(np.abs(selected["dte_days"].to_numpy(float)[:, None] - dte_targets[None, :]), axis=1)
            k_gap = np.min(np.abs(selected["log_moneyness"].to_numpy(float)[:, None] - k_targets[None, :]), axis=1)
            selected["selection_score"] = 0.003 * dte_gap + k_gap + 0.10 * pd.to_numeric(selected["relative_spread"], errors="coerce").fillna(1.0)
            selected["selection_rank"] = selected["selection_score"].rank(method="first")
            selected = selected[selected["selection_rank"].le(int(max_quotes_per_date))].copy()
        for rid, vals in selected_targets.items():
            selected.loc[selected["_row_id"].eq(rid), "k_target"] = vals[0]
            selected.loc[selected["_row_id"].eq(rid), "dte_target"] = vals[1]
        pieces.append(selected)
    out = pd.concat(pieces, ignore_index=True) if pieces else x.iloc[:0].copy()
    rows.append({"step": "daily grid", "rows": int(len(out)), "removed": int(len(x) - len(out))})
    out = calibration_weights(out.drop(columns=[c for c in ["type_penalty", "quote_score", "selection_score", "selection_rank", "expiry_rows", "_row_id"] if c in out.columns], errors="ignore"))
    if "cell_rows" in out.columns:
        out["obs_weight"] = out["obs_weight"] * pd.to_numeric(out["cell_rows"], errors="coerce").fillna(1.0).clip(lower=1.0)
    for name in out.columns:
        dtype_name = str(out[name].dtype).lower()
        if dtype_name == "category" or "interval" in dtype_name:
            out[name] = out[name].astype(str)
    out = out.reset_index(drop=True)
    steps = pd.DataFrame(rows)
    if return_steps:
        return out, steps
    return out


def price_residuals(quotes: pd.DataFrame, model: str, params, *, engine: str = "numba") -> pd.DataFrame:
    out = quotes.copy()
    div = _numeric_column(out, ("dividend_yield", "implied_dividend_yield"), 0.0)
    rate = _numeric_column(out, ("rate",), 0.0)
    out["model_price"] = cos_prices(
        model,
        params,
        out["strike"].to_numpy(float),
        out["tau"].to_numpy(float),
        out["spot"].to_numpy(float),
        rate.to_numpy(float),
        div.to_numpy(float),
        option_type=out["option_type"].to_numpy(),
        engine=engine,
    )
    out["price_residual"] = out["model_price"] - pd.to_numeric(out["mid"], errors="coerce")
    if "iv_mid" in out.columns:
        out["iv_residual"] = out["price_residual"] / pd.to_numeric(out.get("vega", 1.0), errors="coerce").replace(0, np.nan)
    return out


def _bounds_start(model: str, quotes: pd.DataFrame, start=None):
    atm = float(np.nanmedian(quotes.get("iv_mid", pd.Series(0.25, index=quotes.index))))
    key = str(model).lower()
    if key == "bsm":
        base, bounds = np.array([atm]), ([0.03], [3.00])
    elif key == "merton":
        base, bounds = np.array([atm, 0.40, -0.04, 0.20]), ([0.03, 0.0, -1.0, 0.01], [3.00, 8.0, 1.0, 2.0])
    elif key in {"vg", "variance_gamma"}:
        base, bounds = np.array([atm, -0.05, 0.20]), ([0.03, -1.5, 0.01], [3.00, 1.5, 3.0])
    elif key == "heston":
        v = atm * atm
        base, bounds = np.array([v, 2.0, v, 0.60, -0.50]), ([1e-5, 0.05, 1e-5, 0.02, -0.999], [4.0, 12.0, 4.0, 4.0, 0.999])
    elif key == "bates":
        v = atm * atm
        base, bounds = np.array([v, 2.0, v, 0.60, -0.50, 0.30, -0.04, 0.20]), ([1e-5, 0.05, 1e-5, 0.02, -0.999, 0.0, -1.0, 0.01], [4.0, 12.0, 4.0, 4.0, 0.999, 8.0, 1.0, 2.0])
    else:
        raise ValueError(f"unknown Fourier model {model!r}")
    if start is not None:
        arr = np.asarray(start, dtype=float).reshape(-1)
        if arr.size == base.size and np.all(np.isfinite(arr)):
            base = arr
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    return np.clip(base, lo, hi), (lo, hi)


def _valid_params(model: str, p: np.ndarray) -> bool:
    key = str(model).lower()
    if not np.all(np.isfinite(p)):
        return False
    if key in {"vg", "variance_gamma"}:
        sigma, theta, nu = p[:3]
        return sigma > 0 and nu > 0 and (1.0 - theta * nu - 0.5 * sigma * sigma * nu) > 0.0
    return True


def _date_metrics(q: pd.DataFrame, px: np.ndarray, scale: np.ndarray, model: str, success: bool, nfev: int, runtime: float, params: np.ndarray) -> dict:
    mid = pd.to_numeric(q["mid"], errors="coerce").to_numpy(float)
    residual = np.asarray(px, dtype=float) - mid
    half = pd.to_numeric(q.get("half_spread", 0.5 * (q["ask"] - q["bid"]) if {"ask", "bid"}.issubset(q.columns) else 1.0), errors="coerce").fillna(1.0).clip(lower=1e-6).to_numpy(float)
    opt = q["option_type"].astype(str).str.lower()
    dte = pd.to_numeric(q.get("dte_days", q["tau"] * 365.25), errors="coerce")
    m = pd.to_numeric(q.get("moneyness", q["strike"] / q["spot"]), errors="coerce")
    tail = opt.str.startswith("p").to_numpy() & (m.to_numpy(float) <= 0.90)
    short = dte.to_numpy(float) <= 30.0
    inside = ((px >= pd.to_numeric(q.get("bid", q["mid"]), errors="coerce").to_numpy(float)) & (px <= pd.to_numeric(q.get("ask", q["mid"]), errors="coerce").to_numpy(float)))
    row = {
        "model": model,
        "date": pd.Timestamp(q["date"].iloc[0]).normalize() if "date" in q and len(q) else pd.NaT,
        "success": bool(success),
        "nfev": int(nfev),
        "runtime_sec": float(runtime),
        "quotes": int(len(q)),
        "weighted_price_rmse": float(np.sqrt(np.nanmean((residual / scale) ** 2))),
        "median_abs_price_error": float(np.nanmedian(np.abs(residual))),
        "weighted_iv_rmse": float(np.sqrt(np.nanmean((residual / np.maximum(_numeric_column(q, ("vega",), 1.0).abs().to_numpy(float), 1e-6)) ** 2))),
        "bid_ask_hit_rate": float(np.nanmean(inside)),
        "otm_put_rmse": float(np.sqrt(np.nanmean((residual[tail] / scale[tail]) ** 2))) if np.any(tail) else np.nan,
        "short_maturity_rmse": float(np.sqrt(np.nanmean((residual[short] / scale[short]) ** 2))) if np.any(short) else np.nan,
    }
    for i, value in enumerate(params):
        row[f"p{i}"] = float(value)
    if str(model).lower() in {"heston", "bates"} and len(params) >= 5:
        row["feller_ratio"] = float(2.0 * params[1] * params[2] / max(params[3] * params[3], 1e-12))
    return row


def fit_fourier_model(quotes: pd.DataFrame, model: str, *, max_nfev: int = 80, engine: str = "numba") -> dict:
    t0 = time.perf_counter()
    q = calibration_weights(quotes)
    start, bounds = _bounds_start(model, q)
    target = q["mid"].to_numpy(float)
    scale = q["calib_scale_px"].to_numpy(float) / np.sqrt(q["obs_weight"].to_numpy(float))
    groups = cos_group_arrays(q)

    def residual(p):
        if not _valid_params(model, p):
            return np.full_like(target, 1e6, dtype=float)
        px = cos_prices_from_groups(model, p, groups, len(q), n_terms=160, truncation_width=12.0, engine=engine)
        return (px - target) / scale

    res = optimize.least_squares(residual, np.clip(start, bounds[0], bounds[1]), bounds=bounds, max_nfev=int(max_nfev))
    fit = q.copy()
    fit["model_price"] = target + residual(res.x) * scale
    fit["price_residual"] = fit["model_price"] - fit["mid"]
    params = pd.DataFrame([{f"p{i}": v for i, v in enumerate(res.x)} | {"model": model, "loss": float(np.nanmean(res.fun**2)), "success": bool(res.success), "nfev": int(res.nfev)}])
    diag = pd.DataFrame([{"model": model, "quotes": len(fit), "weighted_price_rmse": float(np.sqrt(np.nanmean(res.fun**2))), "median_abs_price_error": float(np.nanmedian(np.abs(fit["price_residual"]))), "runtime": time.perf_counter() - t0}])
    return {"model": model, "params": params, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": engine}


def fit_date_model(
    quotes: pd.DataFrame,
    model: str,
    *,
    start=None,
    max_nfev: int = 80,
    engine: str = "numba",
    n_terms: int = 160,
    truncation_width: float = 12.0,
) -> dict:
    t0 = time.perf_counter()
    q = calibration_weights(quotes)
    if q.empty:
        return {"row": {"model": model, "success": False, "quotes": 0}, "fit": q}
    p0, bounds = _bounds_start(model, q, start=start)
    target = q["mid"].to_numpy(float)
    scale = q["calib_scale_px"].to_numpy(float) / np.sqrt(q["obs_weight"].to_numpy(float))
    groups = cos_group_arrays(q)

    def residual(p):
        if not _valid_params(model, p):
            return np.full_like(target, 1e6, dtype=float)
        px = cos_prices_from_groups(model, p, groups, len(q), n_terms=int(n_terms), truncation_width=float(truncation_width), engine=engine)
        reg = []
        key = str(model).lower()
        if key in {"heston", "bates"}:
            reg = [0.02 * max(p[3] - 2.5, 0.0), 0.01 * max(abs(p[4]) - 0.95, 0.0)]
        if key == "bates":
            reg += [0.02 * max(p[5] - 4.0, 0.0), 0.01 * abs(p[6])]
        if reg:
            return np.r_[(px - target) / scale, np.asarray(reg, dtype=float)]
        return (px - target) / scale

    tries = [p0]
    key = str(model).lower()
    if key in {"heston", "bates"}:
        tries += [np.clip(p0 * np.array([1.1, 0.7, 1.0, 0.8, 1.0] + ([1.0, 1.0, 1.0] if key == "bates" else [])), bounds[0], bounds[1])]
    best = None
    for guess in tries:
        res = optimize.least_squares(residual, np.clip(guess, bounds[0], bounds[1]), bounds=bounds, max_nfev=int(max_nfev), xtol=1e-7, ftol=1e-7, gtol=1e-7)
        loss = float(np.nanmean(res.fun**2))
        if best is None or loss < best[0]:
            best = (loss, res)
    res = best[1]
    px = cos_prices_from_groups(model, res.x, groups, len(q), n_terms=int(n_terms), truncation_width=float(truncation_width), engine=engine)
    fit = q.copy()
    fit["model"] = model
    fit["model_price"] = px
    fit["price_residual"] = px - target
    runtime = time.perf_counter() - t0
    usable = bool(res.success or np.sqrt(max(best[0], 0.0)) < 5.0)
    row = _date_metrics(q, px, scale, model, usable and _valid_params(model, res.x), int(res.nfev), runtime, res.x)
    return {"row": row, "fit": fit, "params": res.x, "result": res}


def fit_daily_models(
    quotes: pd.DataFrame,
    models: list[str] | tuple[str, ...],
    *,
    calibration_dates=None,
    min_quotes: int = 80,
    max_nfev: int = 80,
    engine: str = "numba",
    n_terms: int = 160,
    truncation_width: float = 12.0,
) -> dict:
    q = quotes.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    if calibration_dates is None:
        counts = q.groupby("date").size()
        dates = list(counts[counts >= int(min_quotes)].index)
    else:
        dates = [pd.Timestamp(d).normalize() for d in calibration_dates]
    rows = []
    fits = []
    last: dict[str, np.ndarray] = {}
    for model in models:
        for d in dates:
            day = q[q["date"].eq(d)].copy()
            if len(day) < int(min_quotes):
                continue
            fit = fit_date_model(day, model, start=last.get(str(model)), max_nfev=max_nfev, engine=engine, n_terms=n_terms, truncation_width=truncation_width)
            rows.append(fit["row"])
            if not fit["fit"].empty:
                cols = [c for c in ["date", "expiry", "strike", "option_type", "mid", "bid", "ask", "spot", "tau", "dte_days", "moneyness", "model", "model_price", "price_residual", "calib_scale_px", "obs_weight"] if c in fit["fit"].columns]
                fits.append(fit["fit"][cols].copy())
            if fit["row"].get("success", False):
                last[str(model)] = np.asarray(fit["params"], dtype=float)
    return {"params": pd.DataFrame(rows), "fit": pd.concat(fits, ignore_index=True) if fits else pd.DataFrame()}


def compare_fourier_models(quotes: pd.DataFrame | None = None, fits: dict[str, dict] | None = None, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    if daily is not None:
        q = daily.copy()
        if q.empty:
            return q
        return q.groupby("model", as_index=False).agg(
            dates=("date", "nunique"),
            quotes=("quotes", "sum"),
            success_rate=("success", "mean"),
            weighted_price_rmse=("weighted_price_rmse", "mean"),
            median_abs_price_error=("median_abs_price_error", "median"),
            weighted_iv_rmse=("weighted_iv_rmse", "mean"),
            bid_ask_hit_rate=("bid_ask_hit_rate", "mean"),
            otm_put_rmse=("otm_put_rmse", "mean"),
            short_maturity_rmse=("short_maturity_rmse", "mean"),
            runtime_sec=("runtime_sec", "sum"),
        ).sort_values(["weighted_price_rmse", "runtime_sec"]).reset_index(drop=True)
    rows = []
    fits = fits or {}
    for name, fit in fits.items():
        diag = fit.get("diag", pd.DataFrame())
        if diag.empty:
            rows.append({"model": name, "quotes": 0})
        else:
            row = diag.iloc[0].to_dict()
            row["model"] = name
            row["runtime"] = fit.get("elapsed_sec", row.get("runtime", np.nan))
            rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_price_rmse").reset_index(drop=True)


def warm_start_table(fits: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, fit in fits.items():
        p = fit.get("params", pd.DataFrame())
        if not p.empty:
            row = p.iloc[0].to_dict()
            row["model"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def family_winner(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return ""
    q = comparison.copy()
    if "success_rate" in q.columns:
        q = q[pd.to_numeric(q["success_rate"], errors="coerce").fillna(0.0) > 0.70].copy()
    if q.empty:
        return ""
    sort_cols = [c for c in ["weighted_price_rmse", "runtime_sec", "runtime"] if c in q.columns]
    return str(q.sort_values(sort_cols).iloc[0]["model"])


def calibration_success_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()
    q = daily.copy()
    return q.groupby("model", as_index=False).agg(
        dates=("date", "nunique"),
        success_rate=("success", "mean"),
        failures=("success", lambda x: int((~x.astype(bool)).sum())),
        median_nfev=("nfev", "median"),
        median_runtime_sec=("runtime_sec", "median"),
    )


def residual_by_bucket(fit: pd.DataFrame) -> pd.DataFrame:
    if fit.empty:
        return fit.copy()
    q = fit.copy()
    if "log_moneyness" not in q.columns:
        q["log_moneyness"] = np.log(pd.to_numeric(q["strike"], errors="coerce") / pd.to_numeric(q["spot"], errors="coerce"))
    if "dte_days" not in q.columns:
        q["dte_days"] = pd.to_numeric(q["tau"], errors="coerce") * 365.25
    q["moneyness_bucket"] = pd.cut(q["log_moneyness"], [-0.60, -0.30, -0.15, -0.05, 0.05, 0.15, 0.30, 0.60])
    q["dte_bucket"] = pd.cut(q["dte_days"], [0, 14, 30, 60, 90, 120, 180, 365])
    scale = pd.to_numeric(q.get("calib_scale_px", 1.0), errors="coerce").fillna(1.0).clip(lower=1e-6)
    q["scaled_residual"] = pd.to_numeric(q["price_residual"], errors="coerce") / scale
    return q.groupby(["model", "moneyness_bucket", "dte_bucket"], observed=True).agg(
        median_scaled_residual=("scaled_residual", "median"),
        q25_scaled_residual=("scaled_residual", lambda v: np.nanquantile(v, 0.25)),
        q75_scaled_residual=("scaled_residual", lambda v: np.nanquantile(v, 0.75)),
        rows=("scaled_residual", "size"),
    ).reset_index()


__all__ = [
    "calibration_grid_quotes",
    "calibration_weights",
    "compare_fourier_models",
    "cos_group_arrays",
    "cos_prices_from_groups",
    "cos_prices_grouped",
    "family_winner",
    "fit_daily_models",
    "fit_date_model",
    "fit_fourier_model",
    "price_residuals",
    "calibration_success_table",
    "residual_by_bucket",
    "warm_start_table",
]
