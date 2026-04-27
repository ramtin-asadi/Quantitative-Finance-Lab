from __future__ import annotations

import math
import re
from collections.abc import Iterable
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd

from ..core import CurvePillars, InputError
from .discounting import par_from_df
from .smoothers import fit_curves
from .tenors import _COLUMN_ALIASES, DEFAULT_METHODS, TENOR_PATTERN, tenor_to_years


def normalize_par_yields(
    raw: pd.DataFrame,
    *,
    date_col: str | None = None,
    tenor_cols: list[str] | None = None,
    assume_percent: bool | None = None,
) -> pd.DataFrame:
    """
    Normalize a raw par-yield table:
    - harmonize common column names (e.g., '1 mo' -> '1M')
    - detect/parse date index
    - detect tenor columns and sort by maturity
    - convert yields to float and (optionally) percent -> decimal
    """
    if raw.empty:
        raise InputError("Input DataFrame is empty.")

    data = raw.copy()

    def _normalize_col_name(col: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(col).strip())
        key = cleaned.lower()
        if key in _COLUMN_ALIASES:
            return _COLUMN_ALIASES[key]
        compact = cleaned.replace(" ", "").upper()
        if TENOR_PATTERN.fullmatch(compact):
            return compact
        return cleaned

    data = data.rename(columns={c: _normalize_col_name(c) for c in data.columns})

    normalized_date_col = _normalize_col_name(date_col) if date_col is not None else None

    if normalized_date_col is not None and normalized_date_col in data.columns and normalized_date_col != "date":
        data = data.rename(columns={normalized_date_col: "date"})
    elif normalized_date_col is not None and normalized_date_col.lower() == "date" and "date" in data.columns:
        pass
    elif "date" not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data = data.rename(columns={data.columns[0]: "date"})

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).set_index("date")
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise InputError("Could not detect a date column/index.")

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if tenor_cols is None:
        detected = [c for c in data.columns if TENOR_PATTERN.fullmatch(str(c).strip().upper())]
    else:
        detected = [_normalize_col_name(c) for c in tenor_cols]

    if not detected:
        raise InputError("No tenor columns detected (expected labels like 6M, 2Y, 10Y).")

    detected = sorted(dict.fromkeys(detected), key=tenor_to_years)
    table = data[detected].apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()
    if table.empty:
        raise InputError("No usable tenor data after numeric coercion.")

    if assume_percent is None:
        med = float(np.nanmedian(table.to_numpy(dtype=float)))
        assume_percent = bool(np.isfinite(med) and med > 1.0)

    if assume_percent:
        table = table / 100.0

    return table


def load_par_yields_csv(
    path: str | PathLike[str],
    *,
    date_col: str | None = None,
    tenor_cols: list[str] | None = None,
    assume_percent: bool | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    raw = pd.read_csv(path, **read_csv_kwargs)
    return normalize_par_yields(
        raw,
        date_col=date_col,
        tenor_cols=tenor_cols,
        assume_percent=assume_percent,
    )


def extract_par_curve(
    row: pd.Series | dict,
    tenor_cols: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Extract (labels, T, par) from a row-like object.
    - If tenor_cols is None, detect columns like '1M','6M','1Y','2Y',...
    - Returns par yields as decimals (assumes input already decimals).
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    if tenor_cols is None:
        tenor_cols = [
            c
            for c in row.index.astype(str)
            if TENOR_PATTERN.fullmatch(str(c).strip().replace(" ", "").upper())
        ]

    if not tenor_cols:
        raise InputError("No tenor columns detected. Pass tenor_cols explicitly.")

    y = row[tenor_cols].astype(float)
    mask = np.isfinite(y.values)
    labels = [tenor_cols[i] for i in range(len(tenor_cols)) if mask[i]]
    if not labels:
        raise InputError("All tenor values are NaN/non-finite for this row.")

    par = y.values[mask].astype(float)
    T = np.array([tenor_to_years(label) for label in labels], dtype=float)

    idx = np.argsort(T)
    T = T[idx]
    par = par[idx]
    labels = [labels[i] for i in idx]
    return labels, T, par


def bootstrap_pillars(
    par_curve_row: pd.Series | dict,
    *,
    asof: pd.Timestamp | None = None,
    tenor_cols: list[str] | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
) -> CurvePillars:
    """
    Bootstrap discount factors at observed tenors from a par-yield curve row.

    Convention:
    - For T < 1y: DF(T) = exp(-r*T) if short_end='continuous',
                 or 1/(1+r*T) if short_end='simple'
    - For T >= 1y: solve for DF(T) from par-bond equation with coupon=par yield,
      allowing log-linear interpolation between last known DF and the unknown DF(T).
    """
    labels, T, par = extract_par_curve(par_curve_row, tenor_cols=tenor_cols)
    dfs = bootstrap_from_inputs(
        T=T,
        par=par,
        labels=labels,
        date=asof,
        freq=freq,
        short_end=short_end,
        min_df=min_df,
    )["dfs"]
    return CurvePillars(asof=asof, labels=labels, T=T, par=par, dfs=dfs)


def bootstrap_from_inputs(
    *,
    T: np.ndarray,
    par: np.ndarray,
    labels: list[str],
    date: pd.Timestamp | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
) -> dict:
    d_map: dict[float, float] = {}

    for Ti, ri in zip(T, par, strict=True):
        Ti = float(Ti)
        ri = float(ri)

        if Ti < 1.0:
            d_T = _short_end_df(Ti, ri, short_end=short_end)
            d_map[Ti] = max(float(d_T), min_df)
            continue

        d_T = _solve_df_long_end(Ti, ri, d_map, freq=freq, min_df=min_df)
        if (not np.isfinite(d_T)) or (d_T <= 0):
            d_T = min_df
        d_map[Ti] = max(float(d_T), min_df)

    dfs = np.array([d_map[float(t)] for t in T], dtype=float)
    return {"date": date, "T": T, "par": par, "labels": labels, "dfs": dfs}


def _short_end_df(T: float, r: float, *, short_end: Literal["continuous", "simple"]) -> float:
    if short_end == "continuous":
        return math.exp(-r * T)
    return 1.0 / (1.0 + r * T)


def _price_error_loglinear(
    d_T: float,
    Ti: float,
    t_prev: float,
    d_prev: float,
    times_interp: np.ndarray,
    c: float,
    pv_known: float,
    *,
    freq: int,
    min_df: float,
) -> float:
    d_T = max(float(d_T), min_df)
    pv_interp = 0.0
    if len(times_interp) > 0:
        w = (times_interp - t_prev) / (Ti - t_prev)
        log_d = (1 - w) * np.log(d_prev) + w * np.log(d_T)
        d_interp = np.exp(log_d)
        pv_interp = float(np.sum((c / freq) * d_interp))
    return pv_known + pv_interp + d_T - 1.0


def _solve_df_long_end(
    Ti: float,
    ri: float,
    d_map: dict[float, float],
    *,
    freq: int,
    min_df: float,
) -> float:
    # If there are no prior pillars (e.g., curve starts at 1Y),
    # seed the first long-end DF with a continuous short-end proxy.
    if len(d_map) == 0:
        return float(max(math.exp(-ri * Ti), min_df))

    c = float(ri)
    n = round(Ti * freq)
    times = np.array([k / freq for k in range(1, n + 1)], dtype=float)

    known_T = np.array(sorted(d_map.keys()), dtype=float)
    known_D = np.array([d_map[t] for t in known_T], dtype=float)
    known_D = np.clip(known_D, min_df, None)

    t_prev = float(known_T[-1])
    d_prev = float(known_D[-1])

    times_known = times[times <= t_prev + 1e-12]
    times_interp = times[times > t_prev + 1e-12]

    pv_known = 0.0
    if len(times_known) > 0:
        log_known_D = np.log(known_D)
        log_df_known = np.interp(times_known, known_T, log_known_D)
        d_known = np.exp(log_df_known)
        pv_known = float(np.sum((c / freq) * d_known))

    lo = min_df
    hi = d_prev

    f_lo = _price_error_loglinear(lo, Ti, t_prev, d_prev, times_interp, c, pv_known, freq=freq, min_df=min_df)
    f_hi = _price_error_loglinear(hi, Ti, t_prev, d_prev, times_interp, c, pv_known, freq=freq, min_df=min_df)

    if f_lo * f_hi > 0:
        # fallback: assume coupon DFs can be obtained by log-linear interpolation from known pillars only
        log_known_D = np.log(known_D)
        log_df_cpn = np.interp(times[:-1], known_T, log_known_D, left=log_known_D[0], right=log_known_D[-1])
        d_cpn = np.exp(log_df_cpn)
        pv_coupons = float(np.sum((c / freq) * d_cpn))
        d_T = (1.0 - pv_coupons) / (1.0 + c / freq)
        return float(max(d_T, min_df))

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = _price_error_loglinear(mid, Ti, t_prev, d_prev, times_interp, c, pv_known, freq=freq, min_df=min_df)
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
        if abs(hi - lo) < 1e-12:
            break

    return float(max(0.5 * (lo + hi), min_df))


def rmse_backtest(
    par_yields: pd.DataFrame,
    *,
    methods: Iterable[str] = ("loglinear", "pchip", "nss", "qp"),
    holdouts: list[str] | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
    tenor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Replicates your notebook logic:
    - For each date, optionally hold out a few tenors (if available and interior)
    - Fit curves on training set
    - Compute RMSE on training pillars (IS) and holdouts (OOS)
    """
    methods = [m.lower().strip() for m in methods]
    holdouts = holdouts or ["6M", "2Y", "7Y", "20Y"]
    curve_order = methods

    sse = dict.fromkeys(curve_order, 0.0)
    cnt = dict.fromkeys(curve_order, 0)
    n_dates = dict.fromkeys(curve_order, 0)
    sse_oos = dict.fromkeys(curve_order, 0.0)
    cnt_oos = dict.fromkeys(curve_order, 0)
    n_dates_oos = dict.fromkeys(curve_order, 0)

    failed: list[dict] = []

    for date in par_yields.index:
        row = par_yields.loc[date]
        try:
            pillars_full = bootstrap_pillars(
                row, asof=pd.Timestamp(date), tenor_cols=tenor_cols,
                freq=freq, short_end=short_end, min_df=min_df
            )
        except Exception as e:
            failed.append({"date": date, "method": "bootstrap", "error": str(e)})
            continue

        labels_full = pillars_full.labels
        T_full = pillars_full.T
        par_full = pillars_full.par

        holdout_idx = [labels_full.index(h) for h in holdouts if h in labels_full]
        holdout_idx = sorted({i for i in holdout_idx if 0 < i < len(labels_full) - 1})

        min_train = 4
        if len(T_full) - len(holdout_idx) < min_train:
            holdout_idx = []

        if holdout_idx:
            mask_train = np.ones(len(T_full), dtype=bool)
            mask_train[holdout_idx] = False
            labels_tr = [labels_full[i] for i in range(len(labels_full)) if mask_train[i]]
            T_tr = T_full[mask_train]
            par_tr = par_full[mask_train]

            labels_te = [labels_full[i] for i in range(len(labels_full)) if not mask_train[i]]
            T_te = T_full[~mask_train]
            par_te = par_full[~mask_train]

            boot = bootstrap_from_inputs(
                T=T_tr, par=par_tr, labels=labels_tr, date=pd.Timestamp(date),
                freq=freq, short_end=short_end, min_df=min_df
            )
            pillars = CurvePillars(
                asof=pd.Timestamp(date), labels=labels_tr, T=T_tr, par=par_tr, dfs=boot["dfs"],
                labels_test=labels_te, T_test=T_te, par_test=par_te
            )
        else:
            pillars = pillars_full

        try:
            curves_d = fit_curves(pillars, methods=curve_order, freq=freq, min_df=min_df)
        except Exception as e:
            failed.append({"date": date, "method": "fit_curves", "error": str(e)})
            continue

        for k in curve_order:
            if k not in curves_d:
                continue
            c = curves_d[k]
            try:
                par_fit_tr = par_from_df(c.df, pillars.T, freq=freq)
                err_tr = par_fit_tr - pillars.par
                sse[k] += float(np.sum(err_tr**2))
                cnt[k] += len(err_tr)
                n_dates[k] += 1

                if pillars.T_test is not None and len(pillars.T_test) > 0:
                    par_fit_te = par_from_df(c.df, pillars.T_test, freq=freq)
                    err_te = par_fit_te - pillars.par_test
                    sse_oos[k] += float(np.sum(err_te**2))
                    cnt_oos[k] += len(err_te)
                    n_dates_oos[k] += 1
            except Exception as e:
                failed.append({"date": date, "method": k, "error": str(e)})

    rows = []
    for k in curve_order:
        if cnt[k] == 0:
            continue
        rmse_in = math.sqrt(sse[k] / cnt[k])
        rmse_out = math.sqrt(sse_oos[k] / cnt_oos[k]) if cnt_oos[k] > 0 else float("nan")
        rows.append({
            "method": k,
            "rmse": rmse_in,
            "rmse_oos": rmse_out,
            "n_obs": cnt[k],
            "n_obs_oos": cnt_oos[k],
            "n_dates": n_dates[k],
            "n_dates_oos": n_dates_oos[k],
            "n_failed": len(failed),
        })

    return pd.DataFrame(rows).set_index("method").sort_index()


def normalize_methods(methods: Iterable[str] = DEFAULT_METHODS) -> list[str]:
    vals = [m.lower().strip() for m in methods]
    if not vals:
        raise InputError("At least one curve method is required.")
    return list(dict.fromkeys(vals))


def sort_rmse_table(rmse: pd.DataFrame, *, methods: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Keep a stable method order (if provided) and sort by in-sample RMSE.
    """
    out = rmse.copy()
    if methods is not None:
        ordered = [m for m in normalize_methods(methods) if m in out.index]
        out = out.loc[ordered]
    if "rmse" in out.columns:
        out = out.sort_values("rmse")
    return out


def add_curve_names(
    rmse: pd.DataFrame,
    *,
    curves: dict | None = None,
    method_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Attach the notebook-style display name column used in curve comparison tables.
    """
    out = rmse.copy()
    names = {
        "loglinear": "Log-linear DF",
        "pchip": "PCHIP zero",
        "nss": "NSS",
        "qp": "QP DF",
    }
    if method_names is not None:
        names.update({str(k): str(v) for k, v in method_names.items()})
    if curves is not None:
        for method, curve in curves.items():
            names[str(method)] = getattr(curve, "name", str(method))
    out.insert(0, "name", [names.get(str(m), str(m)) for m in out.index])
    return out


def rank_rmse_table(
    rmse: pd.DataFrame,
    *,
    methods: Iterable[str] | None = None,
    prefer_oos: bool = True,
) -> pd.DataFrame:
    """
    Mirror the notebook's primary-curve ranking:
    sort by OOS RMSE when available, otherwise by in-sample RMSE.
    """
    out = rmse.copy()
    if methods is not None:
        ordered = [m for m in normalize_methods(methods) if m in out.index]
        out = out.loc[ordered]
    if prefer_oos and "rmse_oos" in out.columns and out["rmse_oos"].notna().any():
        sort_cols = ["rmse_oos"]
        if "rmse" in out.columns:
            sort_cols.append("rmse")
        return out.sort_values(sort_cols)
    if "rmse" in out.columns:
        return out.sort_values("rmse")
    return out


def select_primary_curve(
    rmse: pd.DataFrame,
    *,
    methods: Iterable[str] | None = None,
    curves: dict | None = None,
    method_names: dict[str, str] | None = None,
    prefer_oos: bool = True,
) -> tuple[str, str, pd.DataFrame]:
    """
    Select the primary curve method with the same logic used in Notebook 1.

    Returns (method, display_name, ranked_table).
    """
    ranked = rank_rmse_table(rmse, methods=methods, prefer_oos=prefer_oos)
    ranked = add_curve_names(ranked.drop(columns=["name"], errors="ignore"), curves=curves, method_names=method_names)
    method = str(ranked.index[0])
    name = str(ranked.iloc[0]["name"]) if "name" in ranked.columns else method
    return method, name, ranked


def build_zero_curve_panel_from_par_yields(
    par_yields: pd.DataFrame,
    *,
    method: str = "pchip",
    tenors: list[str | float] | None = None,
    as_continuous: bool = True,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
) -> pd.DataFrame:
    """
    Build a date-indexed zero-rate panel from Treasury par-yield curves.

    The helper reuses Project 1 normalization, bootstrapping, and smoothing:
    each curve date is fitted independently, so no future curve information is
    used when Project 4 maps rates onto option quotes.
    """
    if par_yields.empty:
        raise InputError("par_yields is empty.")

    if isinstance(par_yields.index, pd.DatetimeIndex):
        normalized = par_yields.copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized = normalized.sort_index()
        normalized = normalized.apply(pd.to_numeric, errors="coerce")
    else:
        normalized = normalize_par_yields(par_yields)

    tenor_cols = sorted([str(c) for c in normalized.columns], key=tenor_to_years)
    if tenors is None:
        maturity_grid = np.array([tenor_to_years(c) for c in tenor_cols], dtype=float)
    else:
        maturity_grid = []
        for tenor in tenors:
            if isinstance(tenor, (int, float, np.integer, np.floating)):
                maturity_grid.append(float(tenor))
            else:
                maturity_grid.append(float(tenor_to_years(str(tenor))))
        maturity_grid = np.asarray(maturity_grid, dtype=float)

    rows: list[pd.Series] = []
    row_index: list[pd.Timestamp] = []
    curve_method = str(method).lower().strip()

    for date, row in normalized.iterrows():
        try:
            pillars = bootstrap_pillars(
                row,
                asof=pd.Timestamp(date),
                tenor_cols=tenor_cols,
                freq=freq,
                short_end=short_end,
                min_df=min_df,
            )
            curves = fit_curves(pillars, methods=(curve_method,), freq=freq, min_df=min_df)
            curve = curves[curve_method]
            dfs = np.asarray(curve.df(maturity_grid), dtype=float)
        except Exception:
            vals = pd.to_numeric(row[tenor_cols], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(vals)
            if mask.sum() == 0:
                continue
            base_t = np.array([tenor_to_years(c) for c in tenor_cols], dtype=float)[mask]
            base_z = vals[mask]
            zeros = np.interp(maturity_grid, base_t, base_z, left=base_z[0], right=base_z[-1])
            dfs = np.exp(-zeros * maturity_grid)

        dfs = np.clip(dfs, min_df, None)
        if as_continuous:
            rates = -np.log(dfs) / np.clip(maturity_grid, 1e-12, None)
        else:
            rates = np.power(1.0 / dfs, 1.0 / np.clip(maturity_grid, 1e-12, None)) - 1.0
        rows.append(pd.Series(rates, index=maturity_grid))
        row_index.append(pd.Timestamp(date))

    if not rows:
        raise InputError("No zero curves could be built from par_yields.")

    panel = pd.DataFrame(rows, index=pd.DatetimeIndex(row_index))
    panel.columns = panel.columns.astype(float)
    return panel.sort_index()

__all__ = [
    "add_curve_names",
    "bootstrap_from_inputs",
    "bootstrap_pillars",
    "build_zero_curve_panel_from_par_yields",
    "extract_par_curve",
    "load_par_yields_csv",
    "normalize_methods",
    "normalize_par_yields",
    "rank_rmse_table",
    "rmse_backtest",
    "select_primary_curve",
    "sort_rmse_table",
]
