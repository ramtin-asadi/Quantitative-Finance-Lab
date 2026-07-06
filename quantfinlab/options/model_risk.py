from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.bates import bates_prices
from quantfinlab.options.heston import heston_prices
from quantfinlab.options.iv import compute_iv_table
from quantfinlab.options.merton import merton_prices
from quantfinlab.options.sabr import sabr_prices
from quantfinlab.options.ssvi import ssvi_prices
from quantfinlab.options.svi import svi_prices


def choose_model_engine(engine: str = "auto") -> pd.DataFrame:
    """Resolve the numerical engine used for option model-risk calculations.

    Parameters
    ----------
    engine : {'auto', 'numpy', 'numba'}, default='auto'
        Requested backend. ``'auto'`` selects Numba when available and NumPy otherwise.

    Returns
    -------
    pandas.DataFrame
        One-row diagnostic table with requested engine, resolved engine, and Numba
        availability.
    """

    try:
        import numba  # noqa: F401

        has_numba = True
    except Exception:
        has_numba = False
    used = "numba" if str(engine).lower() == "auto" and has_numba else "numpy" if str(engine).lower() == "auto" else str(engine).lower()
    if used == "numba" and not has_numba:
        used = "numpy"
    return pd.DataFrame([{"requested_engine": engine, "engine_used": used, "numba_available": bool(has_numba)}])


def _contract_key(df: pd.DataFrame) -> pd.Series:
    expiry = pd.to_datetime(df["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    strike = pd.to_numeric(df["strike"], errors="coerce").map(lambda x: f"{x:.8g}")
    return df["option_type"].astype(str) + "_" + expiry + "_" + strike


def add_calibration_weights(
    quotes: pd.DataFrame,
    price_uncertainty: str = "half_spread",
    iv_uncertainty: str = "spread_over_vega",
    expiry_balance: bool = True,
) -> pd.DataFrame:
    """Add quote-quality scales and observation weights for option model calibration.

    The function combines bid-ask half-spreads, IV uncertainty, vega information,
    relative spreads, wing distance, and optional expiry balancing to produce price
    scales and bounded observation weights for calibration residuals.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table containing bid, ask, mid, IV, moneyness, expiry, and
        optionally vega columns.
    price_uncertainty : str, default='half_spread'
        Label for the price-uncertainty convention; retained for configuration
        readability.
    iv_uncertainty : str, default='spread_over_vega'
        Label for the IV-uncertainty convention; retained for configuration
        readability.
    expiry_balance : bool, default=True
        If True, downweight densely populated date-expiry groups to reduce slice
        imbalance.

    Returns
    -------
    pandas.DataFrame
        Quote table with ``half_spread``, ``eps_iv``, ``eps_px``, ``calib_scale_px``,
        and ``obs_weight`` columns.
    """

    out = quotes.copy()
    if "contract_key" not in out.columns and {"option_type", "expiry", "strike"}.issubset(out.columns):
        out["contract_key"] = _contract_key(out)
    if "dte_days" not in out.columns:
        out["dte_days"] = pd.to_numeric(out["tau"], errors="coerce") * 365.25
    out["half_spread"] = 0.5 * (pd.to_numeric(out["ask"], errors="coerce") - pd.to_numeric(out["bid"], errors="coerce")).clip(lower=0.0)
    if "vega" in out.columns:
        vega = pd.to_numeric(out["vega"], errors="coerce").abs()
    else:
        vega = pd.Series(np.nanmedian(out["mid"]) * 0.10 if len(out) else 1.0, index=out.index)
    if {"iv_bid", "iv_ask"}.issubset(out.columns):
        direct = 0.5 * (pd.to_numeric(out["iv_ask"], errors="coerce") - pd.to_numeric(out["iv_bid"], errors="coerce")).where(lambda x: np.isfinite(x) & (x > 0))
    else:
        direct = pd.Series(np.nan, index=out.index)
    px_iv = out["half_spread"] / vega.replace(0, np.nan)
    eps_iv = pd.concat([direct, px_iv], axis=1).max(axis=1)
    med_iv = float(np.nanmedian(eps_iv)) if eps_iv.notna().any() else 0.025
    out["eps_iv"] = eps_iv.fillna(med_iv).clip(lower=0.003, upper=0.75)
    out["eps_px"] = np.maximum(out["half_spread"], vega.fillna(0.0) * out["eps_iv"]).clip(lower=1e-6)
    out["calib_scale_px"] = np.maximum(out["half_spread"], 0.25 * vega.fillna(0.0) * out["eps_iv"]).clip(lower=1e-6)
    if "relative_spread" not in out.columns and {"bid", "ask", "mid"}.issubset(out.columns):
        out["relative_spread"] = (out["ask"] - out["bid"]) / out["mid"].replace(0, np.nan)
    spread = pd.to_numeric(out.get("relative_spread"), errors="coerce")
    spread_med = float(np.nanmedian(spread[(spread > 0) & np.isfinite(spread)])) if spread.notna().any() else 0.10
    spread = spread.fillna(spread_med).clip(lower=0.003, upper=2.0)
    w_spread = 1.0 / (spread.to_numpy(dtype=float) ** 2 + 0.03**2)
    w_spread /= max(float(np.nanmedian(w_spread[np.isfinite(w_spread)])), 1e-12)
    vega_med = float(np.nanmedian(vega[(vega > 0) & np.isfinite(vega)])) if np.isfinite(vega).any() else 1.0
    w_vega = np.clip((vega.to_numpy(dtype=float) / max(vega_med, 1e-12)) ** 0.35, 0.20, 3.0)
    w_k = np.exp(-np.maximum(np.abs(pd.to_numeric(out.get("k", 0.0), errors="coerce").to_numpy(dtype=float)) - 0.18, 0.0) / 0.60)
    if expiry_balance and "expiry" in out.columns:
        n_exp = out.groupby(["date", "expiry"])["mid"].transform("count").to_numpy(dtype=float)
        w_tau = 1.0 / np.sqrt(np.maximum(n_exp, 1.0))
        w_tau /= max(float(np.nanmedian(w_tau)), 1e-12)
    else:
        w_tau = 1.0
    w = w_spread * w_vega * w_k * w_tau
    w /= max(float(np.nanmedian(w[np.isfinite(w)])), 1e-12)
    out["obs_weight"] = np.clip(w, 0.05, 20.0)
    return out


def calibration_quotes(
    quotes: pd.DataFrame,
    min_dte: float = 3.0,
    max_dte: float = 120.0,
    min_vega: float = 0.0,
    max_relative_spread: float = 0.85,
    otm_only: bool = True,
) -> pd.DataFrame:
    """Filter an option quote table for model calibration.

    The function keeps positive, finite, sufficiently liquid quotes within a maturity
    range, optional vega floor, and relative-spread cap. When requested, it chooses a
    single out-of-the-money leg per strike to reduce duplicated call/put information.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    min_dte : float, default=3.0
        Minimum days to expiry.
    max_dte : float, default=120.0
        Maximum days to expiry.
    min_vega : float, default=0.0
        Minimum absolute vega.
    max_relative_spread : float, default=0.85
        Maximum relative bid-ask spread.
    otm_only : bool, default=True
        If True, prefer OTM puts below ATM and OTM calls above ATM, while allowing
        either side near ATM.

    Returns
    -------
    pandas.DataFrame
        Filtered, sorted calibration quote table with stable ``quote_id`` and
        ``contract_key`` columns.
    """

    q = quotes.copy()
    if "quote_id" not in q.columns:
        q["quote_id"] = np.arange(len(q), dtype=int)
    if "contract_key" not in q.columns:
        q["contract_key"] = _contract_key(q)
    if "dte_days" not in q.columns:
        q["dte_days"] = q["tau"] * 365.25
    m = (
        (q["mid"] > 0)
        & (q["ask"] >= q["bid"])
        & q["dte_days"].between(float(min_dte), float(max_dte))
        & pd.to_numeric(q["relative_spread"], errors="coerce").between(0.0, float(max_relative_spread))
        & (pd.to_numeric(q.get("vega", 1.0), errors="coerce").abs() >= float(min_vega))
        & np.isfinite(q["iv_mid"])
        & np.isfinite(q["forward"])
    )
    q = q[m].copy()
    if otm_only:
        atm = 0.015
        q["want_type"] = np.where(q["k"] < -atm, "put", np.where(q["k"] > atm, "call", "either"))
        q["type_penalty"] = np.where((q["want_type"].eq("either")) | q["option_type"].eq(q["want_type"]), 0.0, 1.0)
        q["clean_score"] = pd.to_numeric(q["relative_spread"], errors="coerce").fillna(0.30) + 0.02 * q["k"].abs()
        q = q.sort_values(["date", "expiry", "strike", "type_penalty", "clean_score"]).drop_duplicates(["date", "expiry", "strike"], keep="first")
    q = q.sort_values(["date", "expiry", "k"]).reset_index(drop=True)
    q["quote_id"] = np.arange(len(q), dtype=int)
    return q


def choose_surface_date(quotes: pd.DataFrame, min_expiries: int = 4, min_quotes: int = 80, prefer_tail_coverage: bool = True):
    """Select a single quote date with strong volatility-surface coverage.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Calibration-ready quote table containing date, expiry, quote_id, moneyness,
        and spread columns.
    min_expiries : int, default=4
        Minimum number of expiries required for a preferred candidate.
    min_quotes : int, default=80
        Minimum quote count required for a preferred candidate.
    prefer_tail_coverage : bool, default=True
        If True, prefer dates with sufficient left- and right-tail observations.

    Returns
    -------
    pandas.Timestamp
        Selected normalized quote date.
    """

    q = quotes.copy()
    table = q.groupby("date").agg(
        quotes=("quote_id", "size"),
        expiries=("expiry", "nunique"),
        near_atm=("k", lambda x: int((np.abs(x) <= 0.05).sum())),
        left_tail=("k", lambda x: int((x <= -0.18).sum())),
        right_tail=("k", lambda x: int((x >= 0.18).sum())),
        k_width=("k", lambda x: float(np.nanmax(x) - np.nanmin(x))),
        spread=("relative_spread", "median"),
    ).reset_index()
    table["score"] = table["quotes"] + 25 * table["expiries"] + 2 * table["near_atm"] + 4 * table["left_tail"] + 2 * table["right_tail"] + 100 * table["k_width"] - 50 * table["spread"].fillna(0.5)
    good = table[(table["quotes"] >= int(min_quotes)) & (table["expiries"] >= int(min_expiries))].copy()
    if good.empty:
        good = table.copy()
    if prefer_tail_coverage:
        tail = good[(good["left_tail"] + good["right_tail"]) >= 8].copy()
        if not tail.empty:
            good = tail
    return pd.Timestamp(good.sort_values("score", ascending=False).iloc[0]["date"]).normalize()


def balanced_model_quotes(
    quotes: pd.DataFrame,
    target_dtes=(7, 14, 21, 30, 45, 60, 90),
    target_ks=(-0.35, -0.25, -0.17, -0.10, -0.04, 0.00, 0.04, 0.10, 0.17, 0.25, 0.35),
    min_quotes_per_expiry: int = 6,
    prefer_tail_coverage: bool = True,
) -> pd.DataFrame:
    """Select a balanced grid of calibration quotes across target maturities and moneyness nodes.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Calibration-ready quote table.
    target_dtes : iterable, default=(7, 14, 21, 30, 45, 60, 90)
        Target days-to-expiry values.
    target_ks : iterable
        Target log-moneyness values.
    min_quotes_per_expiry : int, default=6
        Minimum selected quotes required for an expiry to be included.
    prefer_tail_coverage : bool, default=True
        Reserved configuration flag for tail-aware selection.

    Returns
    -------
    pandas.DataFrame
        Balanced subset sorted by expiry and log-moneyness.
    """

    q = quotes.copy()
    base = q.groupby("expiry").agg(dte=("dte_days", "median"), n=("quote_id", "size")).reset_index()
    base = base[base["n"] >= int(min_quotes_per_expiry)].copy()
    pieces = []
    used_exp = set()
    for target in target_dtes:
        cand = base[~base["expiry"].isin(used_exp)].copy()
        if cand.empty:
            break
        cand["dist"] = (cand["dte"] - float(target)).abs()
        expiry = cand.sort_values(["dist", "n"], ascending=[True, False]).iloc[0]["expiry"]
        used_exp.add(expiry)
        g = q[q["expiry"].eq(expiry)].sort_values("k").copy()
        rows = []
        used = set()
        for kt in target_ks:
            h = g[~g["quote_id"].isin(used)].copy()
            if h.empty:
                continue
            j = (h["k"] - float(kt)).abs().idxmin()
            rows.append(h.loc[j])
            used.add(h.loc[j, "quote_id"])
        s = pd.DataFrame(rows).drop_duplicates("quote_id") if rows else g.head(0)
        if len(s) >= int(min_quotes_per_expiry):
            pieces.append(s)
    out = pd.concat(pieces, ignore_index=True) if pieces else q.head(0).copy()
    return out.sort_values(["expiry", "k"]).reset_index(drop=True)


def common_model_quotes(quotes: pd.DataFrame, model_quotes: pd.DataFrame | None = None, min_tail_count: int = 6) -> pd.DataFrame:
    """Build or augment a common benchmark quote set for model comparison.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Full calibration quote table.
    model_quotes : pandas.DataFrame, optional
        Preselected benchmark quotes. If absent, a balanced subset is constructed.
    min_tail_count : int, default=6
        Minimum number of tail quotes desired in the benchmark set.

    Returns
    -------
    pandas.DataFrame
        Benchmark quote set sorted by expiry and moneyness, augmented with additional
        tail quotes when necessary.
    """

    q = model_quotes.copy() if model_quotes is not None and not model_quotes.empty else balanced_model_quotes(quotes)
    if int((q["k"].abs() >= 0.14).sum()) < int(min_tail_count):
        tail = quotes[quotes["k"].abs() >= 0.14].sort_values(["expiry", "relative_spread"]).groupby("expiry").head(2)
        q = pd.concat([q, tail], ignore_index=True).drop_duplicates("quote_id")
    return q.sort_values(["expiry", "k"]).reset_index(drop=True)


def _predict(name: str, quotes: pd.DataFrame, fit: dict, engine: str = "auto") -> pd.DataFrame:
    if name == "svi":
        return svi_prices(quotes, fit, engine=engine)
    if name == "ssvi":
        return ssvi_prices(quotes, fit, engine=engine)
    if name == "sabr":
        return sabr_prices(quotes, fit, engine=engine)
    if name == "merton":
        return merton_prices(quotes, fit, engine=engine)
    if name == "heston":
        return heston_prices(quotes, fit, engine=engine)
    if name == "bates":
        return bates_prices(quotes, fit, engine=engine)
    return quotes.head(0).copy()


def _weighted_rmse(x, weight=None) -> float:
    y = np.asarray(x, dtype=float)
    if weight is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(weight, dtype=float)
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.sum(w[mask] * y[mask] * y[mask]) / np.sum(w[mask])))


def _with_price_iv(pred: pd.DataFrame, engine: str = "auto") -> pd.DataFrame:
    out = pred.copy()
    if out.empty or "model_price" not in out.columns:
        return out
    need_iv = "model_iv" not in out.columns or out["model_iv"].isna().any()
    if not need_iv:
        out["iv_residual"] = out["model_iv"] - out["iv_mid"]
        return out
    tmp = out.copy()
    tmp["model_price_for_iv"] = pd.to_numeric(tmp["model_price"], errors="coerce")
    try:
        iv_tmp = compute_iv_table(tmp, price_cols=("model_price_for_iv",), engine=engine)
        model_iv = pd.to_numeric(iv_tmp.get("iv_model_price_for_iv"), errors="coerce")
        success = iv_tmp.get("iv_model_price_for_iv_success", pd.Series(True, index=iv_tmp.index)).fillna(False)
        if "model_iv" in out.columns:
            out["model_iv"] = pd.to_numeric(out["model_iv"], errors="coerce").where(lambda x: x.notna(), model_iv.where(success))
        else:
            out["model_iv"] = model_iv.where(success)
    except Exception:
        if "model_iv" not in out.columns:
            out["model_iv"] = np.nan
    out["iv_residual"] = out["model_iv"] - out["iv_mid"]
    return out


def compare_model_fits(quotes: pd.DataFrame, fits: dict, engine: str = "auto") -> pd.DataFrame:
    """Compare option-pricing model fits on a common quote table.

    For each fitted model, the function reprices the benchmark quotes, computes
    weighted price and IV errors when available, evaluates tail performance, records
    runtime, and applies a pragmatic usability success flag.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Benchmark quote table.
    fits : dict
        Mapping from model name to fit dictionary.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Pricing backend used for model prediction.

    Returns
    -------
    pandas.DataFrame
        Model-comparison table with quote counts, IV quote counts, weighted IV RMSE,
        weighted price RMSE, tail error, runtime, and success indicator.
    """

    rows = []
    for name, fit in fits.items():
        pred = _predict(name, quotes, fit, engine=engine)
        if pred.empty:
            rows.append({"model": name, "benchmark_quotes": 0})
            continue
        pred = _with_price_iv(pred, engine=engine)
        scale = pd.to_numeric(pred["calib_scale_px"] if "calib_scale_px" in pred.columns else pred["half_spread"].clip(lower=1e-6), errors="coerce")
        obs_weight = pd.to_numeric(pred.get("obs_weight", pd.Series(1.0, index=pred.index)), errors="coerce").fillna(1.0)
        price_rmse = _weighted_rmse((pred["model_price"] - pred["mid"]) / scale.replace(0, np.nan))
        iv_rmse = _weighted_rmse(pred["iv_residual"], obs_weight) if "iv_residual" in pred.columns else np.nan
        tail = pred[pred["k"].abs() >= 0.14]
        param_success = bool(fit.get("params", pd.DataFrame()).get("success", pd.Series([True])).fillna(False).mean() > 0.5) if isinstance(fit, dict) else False
        usable_success = bool((np.isfinite(iv_rmse) and iv_rmse < 0.25) or (np.isfinite(price_rmse) and price_rmse < 5.0))
        rows.append({
            "model": name,
            "benchmark_quotes": len(pred),
            "iv_quotes": int(pred["iv_residual"].notna().sum()) if "iv_residual" in pred.columns else 0,
            "weighted_iv_rmse": iv_rmse,
            "weighted_price_rmse": price_rmse,
            "tail_error": _weighted_rmse(tail["iv_residual"], pd.to_numeric(tail.get("obs_weight", pd.Series(1.0, index=tail.index)), errors="coerce")) if not tail.empty and "iv_residual" in tail.columns and tail["iv_residual"].notna().any() else np.nan,
            "runtime": float(fit.get("elapsed_sec", np.nan)),
            "success": bool(param_success or usable_success),
        })
    return pd.DataFrame(rows)


def model_fair_values(
    quotes: pd.DataFrame,
    fits: dict,
    ensemble_method: str = "capped_weighted",
    max_model_weight: float = 0.70,
    min_model_weight: float = 0.30,
    engine: str = "auto",
) -> pd.DataFrame:
    """Build ensemble fair values from multiple fitted option-pricing models.

    The function intersects available holdout IDs when present, reprices quotes with
    each model, assigns inverse-error ensemble weights with optional caps/floors, and
    computes ensemble residuals and model disagreement.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table used for fair-value estimation.
    fits : dict
        Mapping from model name to fit dictionary.
    ensemble_method : str, default='capped_weighted'
        Ensemble method label retained for configuration clarity.
    max_model_weight : float, default=0.70
        Maximum allowed model weight for selected models.
    min_model_weight : float, default=0.30
        Minimum model weight for selected anchor models when present.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Pricing backend.

    Returns
    -------
    pandas.DataFrame
        Quote table with model price columns, ``ensemble_price``,
        ``ensemble_price_residual``, and ``model_disagreement``. Ensemble weights are
        stored in DataFrame attributes.
    """

    base_ids = None
    for fit in fits.values():
        ids = fit.get("holdout_ids", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
        if not ids.empty:
            base_ids = ids if base_ids is None else base_ids.merge(ids, on=["date", "quote_id"], how="inner")
    q = quotes.merge(base_ids, on=["date", "quote_id"], how="inner") if base_ids is not None and not base_ids.empty else quotes.head(0).copy()
    out = q.copy()
    price_cols = []
    errs = {}
    for name, fit in fits.items():
        pred = _predict(name, q, fit, engine=engine)
        if pred.empty:
            continue
        col = f"{name}_price"
        keep = pred[["date", "quote_id", "model_price"]].rename(columns={"model_price": col})
        out = out.merge(keep, on=["date", "quote_id"], how="left")
        price_cols.append(col)
        diag = fit.get("diag", pd.DataFrame())
        errs[name] = float(np.nanmedian(diag.select_dtypes("number").to_numpy())) if not diag.empty else 1.0
    if not price_cols:
        return out.head(0).copy()
    weights = {}
    inv = {name: 1.0 / max(errs.get(name, 1.0), 1e-6) for name in fits if f"{name}_price" in out.columns}
    total = sum(inv.values()) or 1.0
    for name, val in inv.items():
        weights[name] = val / total
    if "svi" in weights and "sabr" in weights:
        weights["svi"] = min(weights["svi"], float(max_model_weight))
        weights["sabr"] = max(weights["sabr"], float(min_model_weight))
    s = sum(weights.values()) or 1.0
    weights = {k: v / s for k, v in weights.items()}
    ensemble = np.zeros(len(out), dtype=float)
    used = np.zeros(len(out), dtype=float)
    for name, w in weights.items():
        col = f"{name}_price"
        if col in out.columns:
            val = out[col].to_numpy(dtype=float)
            mask = np.isfinite(val)
            ensemble[mask] += w * val[mask]
            used[mask] += w
    out["ensemble_price"] = ensemble / np.where(used > 0, used, np.nan)
    out["ensemble_price_residual"] = out["ensemble_price"] - out["mid"]
    out["model_disagreement"] = out[price_cols].std(axis=1).fillna(0.0)
    out.attrs["model_weights"] = weights
    return out[np.isfinite(out["ensemble_price"])].copy()


def residual_scores(
    fair_values: pd.DataFrame,
    residual_col: str = "ensemble_price_residual",
    quote_cost_col: str = "half_spread",
    model_uncertainty_col: str = "model_disagreement",
    score_method: str = "cost_adjusted_z",
) -> pd.DataFrame:
    """Convert ensemble pricing residuals into cost-adjusted z-style signal scores.

    Parameters
    ----------
    fair_values : pandas.DataFrame
        Table containing ensemble residuals, quote costs, and model uncertainty.
    residual_col : str, default='ensemble_price_residual'
        Residual column to score.
    quote_cost_col : str, default='half_spread'
        Entry quote-cost proxy.
    model_uncertainty_col : str, default='model_disagreement'
        Model uncertainty column.
    score_method : str, default='cost_adjusted_z'
        Score-method label retained for readability.

    Returns
    -------
    pandas.DataFrame
        Table with ``total_error``, ``z_residual``, ``watchlist_candidate``, and
        ``strict_candidate`` columns.
    """

    out = fair_values.copy()
    quote_cost = pd.to_numeric(out.get(quote_cost_col, 0.0), errors="coerce").fillna(0.0)
    exit_cost = pd.to_numeric(out.get("expected_exit_half_spread", quote_cost), errors="coerce").fillna(quote_cost)
    model_unc = pd.to_numeric(out.get(model_uncertainty_col, 0.0), errors="coerce").fillna(0.0)
    fit_default = pd.Series(0.0, index=out.index)
    fit_error = pd.to_numeric(out["fit_error"] if "fit_error" in out.columns else fit_default, errors="coerce").fillna(0.0)
    out["total_error"] = quote_cost + 0.5 * exit_cost + 0.5 * model_unc + 0.5 * fit_error
    out["z_residual"] = out[residual_col] / out["total_error"].clip(lower=1e-8)
    out["watchlist_candidate"] = out["z_residual"].abs() > 0.5
    out["strict_candidate"] = out["z_residual"].abs() > 1.0
    return out


def signal_dates(quotes: pd.DataFrame, min_quotes: int = 60, min_expiries: int = 3, min_near_atm_quotes: int = 10):
    """Identify dates with enough option coverage for residual-signal validation.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table with date, quote_id, expiry, and moneyness columns.
    min_quotes : int, default=60
        Minimum quote count per date.
    min_expiries : int, default=3
        Minimum expiry count per date.
    min_near_atm_quotes : int, default=10
        Minimum number of near-ATM quotes per date.

    Returns
    -------
    numpy.ndarray
        Sorted array of eligible normalized dates.
    """

    table = quotes.groupby("date").agg(
        quotes=("quote_id", "size"),
        expiries=("expiry", "nunique"),
        near_atm_quotes=("k", lambda x: int((np.abs(x) <= 0.05).sum())),
    ).reset_index()
    good = table[(table["quotes"] >= int(min_quotes)) & (table["expiries"] >= int(min_expiries)) & (table["near_atm_quotes"] >= int(min_near_atm_quotes))]
    return pd.to_datetime(good["date"]).sort_values().to_numpy()


def next_day_residual_check(scores: pd.DataFrame, option_quotes: pd.DataFrame, hedge_delta_col: str = "delta", cost_model: str = "scheduled", calendar: str = "crypto_24_7", engine: str = "auto") -> pd.DataFrame:
    """Validate residual signals against next-day delta-hedged price changes.

    The function links scored contracts to the next available quote date, computes a
    raw delta-hedged P&L after entry and exit half-spread costs, estimates rolling
    information coefficient direction, and builds a direction-adjusted next-day P&L
    validation table.

    Parameters
    ----------
    scores : pandas.DataFrame
        Residual-scored quote table.
    option_quotes : pandas.DataFrame
        Full option quote table containing next-day observations.
    hedge_delta_col : str, default='delta'
        Delta column used for underlying hedge P&L.
    cost_model : str, default='scheduled'
        Cost-model label retained for compatibility.
    calendar : str, default='crypto_24_7'
        Calendar label retained for configuration metadata.
    engine : str, default='auto'
        Backend label retained for compatibility.

    Returns
    -------
    pandas.DataFrame
        Validation table with next-day prices, hedged P&L, information-coefficient
        diagnostics, and adjusted P&L. Summary and IC tables are stored in attributes.
    """

    if scores.empty:
        return scores.copy()
    q = option_quotes.copy()
    if "contract_key" not in q.columns:
        q["contract_key"] = _contract_key(q)
    rows = []
    dates = np.array(sorted(pd.to_datetime(q["date"]).unique()))
    for d, g in scores.groupby("date", sort=True):
        nxt = dates[dates > np.datetime64(pd.Timestamp(d).normalize())]
        if len(nxt) == 0:
            continue
        dn = pd.Timestamp(nxt[0]).normalize()
        nx = q[q["date"].eq(dn)][["contract_key", "mid", "spot", "half_spread"]].rename(columns={"mid": "next_mid", "spot": "next_spot", "half_spread": "next_half_spread"})
        m = g.merge(nx, on="contract_key", how="inner")
        m["next_date"] = dn
        rows.append(m)
    out = pd.concat(rows, ignore_index=True) if rows else scores.head(0).copy()
    if out.empty:
        out.attrs["summary"] = pd.DataFrame()
        return out
    side = np.sign(out["ensemble_price_residual"]).replace(0, np.nan)
    out["raw_hedged_pnl"] = side * (out["next_mid"] - out["mid"]) + side * (-pd.to_numeric(out.get(hedge_delta_col, 0.0), errors="coerce").fillna(0.0)) * (out["next_spot"] - out["spot"]) - out["half_spread"].fillna(0.0) - out["next_half_spread"].fillna(out["half_spread"])
    ic = out.groupby("date").apply(lambda g: g[["z_residual", "raw_hedged_pnl"]].corr(method="spearman").iloc[0, 1] if len(g) >= 5 else np.nan).rename("ic").reset_index()
    ic["train_ic"] = ic["ic"].expanding(min_periods=2).mean().shift(1)
    ic["signal_direction"] = np.where(ic["train_ic"] > 0.0, 1.0, np.where(ic["train_ic"] < 0.0, -1.0, 0.0))
    out = out.merge(ic[["date", "ic", "train_ic", "signal_direction"]], on="date", how="left")
    out["next_hedged_pnl"] = out["signal_direction"] * out["raw_hedged_pnl"]
    dec = out[out["signal_direction"].ne(0.0)].copy()
    if not dec.empty:
        dec["decile"] = pd.qcut(dec["z_residual"].rank(method="first"), 10, labels=False, duplicates="drop")
        summary = dec.groupby("decile").agg(mean_z=("z_residual", "mean"), mean_next_hedged_pnl=("next_hedged_pnl", "mean"), hit_rate=("next_hedged_pnl", lambda x: float(np.mean(x > 0))), count=("next_hedged_pnl", "size")).reset_index()
    else:
        summary = pd.DataFrame()
    out.attrs["summary"] = summary
    out.attrs["ic_by_date"] = ic
    return out


def residual_entry_schedule(
    validation: pd.DataFrame,
    selector_name: str = "residual_fixed_3d",
    hold_days: int = 3,
    max_entries: int = 80,
    entry_spacing_days: int = 3,
    require_signal_direction: bool = True,
    chronological: bool = True,
) -> pd.DataFrame:
    """Convert residual-validation candidates into a chronological entry schedule.

    Parameters
    ----------
    validation : pandas.DataFrame
        Residual validation table containing watchlist flags, residuals, z-scores, and
        optional signal-direction columns.
    selector_name : str, default='residual_fixed_3d'
        Label assigned to scheduled entries.
    hold_days : int, default=3
        Maximum holding period stored in the schedule.
    max_entries : int, default=80
        Maximum number of entries.
    entry_spacing_days : int, default=3
        Minimum calendar-day spacing between entries.
    require_signal_direction : bool, default=True
        If True, require nonzero signal direction when the column is available.
    chronological : bool, default=True
        Compatibility flag; entries are selected chronologically.

    Returns
    -------
    pandas.DataFrame
        Entry schedule with entry date, contract key, signed quantity, label, score,
        residual diagnostics, and exit-rule flags.
    """

    cols = ["entry_date", "contract_key", "quantity", "label", "entry_score", "entry_residual", "entry_total_error", "entry_z", "max_hold_days", "exit_on_convergence", "exit_on_sign_flip"]
    if validation.empty:
        return pd.DataFrame(columns=cols)
    q = validation[validation["watchlist_candidate"] & np.isfinite(validation["z_residual"])].copy()
    if require_signal_direction and "signal_direction" in q.columns:
        q = q[q["signal_direction"].ne(0.0)].copy()
    if q.empty:
        return pd.DataFrame(columns=cols)
    q["entry_rank"] = q["z_residual"].abs()
    q = q.sort_values(["date", "entry_rank"], ascending=[True, False]).groupby("date", as_index=False).head(1).sort_values("date")
    rows = []
    last = None
    for _, r in q.iterrows():
        d = pd.Timestamp(r["date"]).normalize()
        if last is not None and (d - last).days < int(entry_spacing_days):
            continue
        direction = float(r.get("signal_direction", 1.0))
        qty = direction * np.sign(float(r["ensemble_price_residual"]))
        if qty == 0 or not np.isfinite(qty):
            continue
        rows.append({"entry_date": d, "contract_key": r["contract_key"], "quantity": qty, "label": selector_name, "entry_score": abs(float(r["z_residual"])), "entry_residual": float(r["ensemble_price_residual"]), "entry_total_error": float(r["total_error"]), "entry_z": float(r["z_residual"]), "max_hold_days": int(hold_days), "exit_on_convergence": False, "exit_on_sign_flip": False})
        last = d
        if len(rows) >= int(max_entries):
            break
    return pd.DataFrame(rows, columns=cols)


def market_summary(asset: str, quotes: pd.DataFrame, model_comparison: pd.DataFrame, validation: pd.DataFrame, hedge_comparison: pd.DataFrame, engine: str = "auto") -> pd.DataFrame:
    """Summarize quote, model, validation, and hedge coverage for a market run.

    Parameters
    ----------
    asset : str
        Asset or market label.
    quotes : pandas.DataFrame
        Quote table.
    model_comparison : pandas.DataFrame
        Model-comparison table.
    validation : pandas.DataFrame
        Residual-validation table.
    hedge_comparison : pandas.DataFrame
        Hedge-comparison table.
    engine : str, default='auto'
        Engine label to record.

    Returns
    -------
    pandas.DataFrame
        One-row market summary.
    """

    return pd.DataFrame([{
        "asset": asset,
        "date": pd.Timestamp(quotes["date"].iloc[0]).date() if not quotes.empty and "date" in quotes else pd.NaT,
        "quotes": int(len(quotes)),
        "expiries": int(quotes["expiry"].nunique()) if "expiry" in quotes else 0,
        "models": int(model_comparison["model"].nunique()) if not model_comparison.empty and "model" in model_comparison else 0,
        "validation_rows": int(len(validation)),
        "hedge_runs": int(hedge_comparison["run"].nunique()) if not hedge_comparison.empty and "run" in hedge_comparison else 0,
        "engine": engine,
    }])


__all__ = [
    "add_calibration_weights",
    "balanced_model_quotes",
    "calibration_quotes",
    "choose_model_engine",
    "choose_surface_date",
    "common_model_quotes",
    "compare_model_fits",
    "market_summary",
    "model_fair_values",
    "next_day_residual_check",
    "residual_entry_schedule",
    "residual_scores",
    "signal_dates",
]
