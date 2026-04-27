from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.volatility import realized

from . import bsm, parity


def _option_inputs(frame: pd.DataFrame):
    option_col = "option_type" if "option_type" in frame.columns else "cp"
    forward = frame["forward"] if "forward" in frame.columns else frame.get("f_hat")
    if forward is None:
        forward = frame["spot"] * np.exp(pd.to_numeric(frame.get("rate", 0.0), errors="coerce") * frame["tau"])
    df = frame["discount_factor"] if "discount_factor" in frame.columns else frame.get("df", 1.0)
    return frame[option_col], forward, frame["strike"], frame["tau"], df


def realized_vol_forward_bsm_pricing_comparison(
    quotes: pd.DataFrame,
    realized_vol: pd.DataFrame | pd.Series,
    vol_window: int = 30,
    price_col: str = "mid",
    annualization_days: float = 365.0,
) -> dict[str, pd.DataFrame]:
    """Price quotes with realized volatility in the forward-BSM model."""
    data = realized.align_realized_to_option_expiries(realized_vol, quotes, date_col="date")
    rv_col = f"rv_{int(vol_window)}"
    if rv_col not in data.columns:
        rv_cols = [c for c in data.columns if str(c).startswith("rv_")]
        rv_col = rv_cols[0] if rv_cols else "realized_vol"
    data["realized_vol"] = pd.to_numeric(data.get(rv_col), errors="coerce")
    option_type, forward, strike, tau, df = _option_inputs(data)
    data["realized_vol_forward_bsm_price"] = bsm.forward_bsm_price(
        option_type,
        forward,
        strike,
        tau,
        data["realized_vol"],
        df,
    )
    data["realized_vol_forward_bsm_vega"] = bsm.forward_bsm_vega(
        forward,
        strike,
        tau,
        data["realized_vol"],
        df,
    )
    data["pricing_error"] = data["realized_vol_forward_bsm_price"] - pd.to_numeric(data[price_col], errors="coerce")
    data["abs_pricing_error"] = data["pricing_error"].abs()
    vega_abs = data["realized_vol_forward_bsm_vega"].abs()
    vega_pos = vega_abs[np.isfinite(vega_abs) & (vega_abs > 0)]
    vega_floor = float(max(1e-6, np.nanpercentile(vega_pos, 5) if len(vega_pos) else 1e-6))
    data["vega_scaled_pricing_error"] = data["pricing_error"] / vega_abs.clip(lower=vega_floor)
    if {"bid", "ask"}.issubset(data.columns):
        data["inside_spread_hit"] = (
            (data["realized_vol_forward_bsm_price"] >= pd.to_numeric(data["bid"], errors="coerce"))
            & (data["realized_vol_forward_bsm_price"] <= pd.to_numeric(data["ask"], errors="coerce"))
        ).astype(float)
    data["annualization_days"] = float(annualization_days)
    summary = pricing_error_summary(data, error_col="pricing_error")
    summary["median_vega_scaled_error"] = float(np.nanmedian(data["vega_scaled_pricing_error"]))
    summary["inside_spread_hit_rate"] = (
        float(np.nanmean(data["inside_spread_hit"])) if "inside_spread_hit" in data.columns else np.nan
    )
    return {"table": data, "summary": summary}


def solver_failure_by_log_moneyness(
    iv_table: pd.DataFrame,
    x_col: str = "log_moneyness",
    bins: int | np.ndarray = 16,
) -> pd.DataFrame:
    if iv_table.empty or x_col not in iv_table.columns:
        return pd.DataFrame()
    data = iv_table.copy()
    success = data["iv_success"] if "iv_success" in data.columns else data.get("iv_mid_success", False)
    data["_failed"] = ~pd.Series(success, index=data.index).astype(bool)
    data["_bin"] = pd.cut(pd.to_numeric(data[x_col], errors="coerce"), bins=bins)
    out = data.groupby("_bin", observed=True).agg(
        x_mid=(x_col, "median"),
        failure_rate=("_failed", "mean"),
        n=("_failed", "size"),
    )
    return out.reset_index(drop=True)


def solver_iterations_by_log_moneyness(
    iv_table: pd.DataFrame,
    x_col: str = "log_moneyness",
    bins: int | np.ndarray = 16,
) -> pd.DataFrame:
    if iv_table.empty or x_col not in iv_table.columns:
        return pd.DataFrame()
    data = iv_table.copy()
    iter_col = "iv_iterations" if "iv_iterations" in data.columns else "iv_mid_iters"
    success = data["iv_success"] if "iv_success" in data.columns else data.get("iv_mid_success", False)
    data = data[pd.Series(success, index=data.index).astype(bool)].copy()
    if data.empty or iter_col not in data.columns:
        return pd.DataFrame()
    data["_bin"] = pd.cut(pd.to_numeric(data[x_col], errors="coerce"), bins=bins)
    out = data.groupby("_bin", observed=True).agg(
        x_mid=(x_col, "median"),
        median_iterations=(iter_col, "median"),
        p90_iterations=(iter_col, lambda x: float(np.nanquantile(x, 0.90))),
        n=(iter_col, "size"),
    )
    return out.reset_index(drop=True)


def iv_solver_diagnostics(
    solver_tables: dict[str, pd.DataFrame],
    x_col: str = "log_moneyness",
    bins: int | np.ndarray = 16,
) -> dict[str, pd.DataFrame]:
    """Summarize solver failures and iterations by log-moneyness."""
    failure_frames = []
    iteration_frames = []
    summary_rows: list[dict[str, Any]] = []
    for solver_name, table in solver_tables.items():
        tbl = table.copy()
        if tbl.empty:
            continue
        success = tbl["iv_success"] if "iv_success" in tbl.columns else tbl.get("iv_mid_success", False)
        success = pd.Series(success, index=tbl.index).astype(bool)
        iter_col = "iv_iterations" if "iv_iterations" in tbl.columns else "iv_mid_iters"
        failure = solver_failure_by_log_moneyness(tbl, x_col=x_col, bins=bins)
        if not failure.empty:
            failure["solver"] = solver_name
            failure_frames.append(failure)
        iterations = solver_iterations_by_log_moneyness(tbl, x_col=x_col, bins=bins)
        if not iterations.empty:
            iterations["solver"] = solver_name
            iteration_frames.append(iterations)
        summary_rows.append(
            {
                "solver": solver_name,
                "engine_used": tbl.attrs.get("engine_used", tbl.get("engine_used", pd.Series(["unknown"])).iloc[0]),
                "n": len(tbl),
                "success_rate": float(success.mean()) if len(success) else np.nan,
                "failure_rate": float((~success).mean()) if len(success) else np.nan,
                "median_iterations": float(np.nanmedian(pd.to_numeric(tbl.get(iter_col), errors="coerce")))
                if iter_col in tbl.columns
                else np.nan,
            }
        )
    return {
        "failure_by_log_moneyness": pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(),
        "iterations_by_log_moneyness": pd.concat(iteration_frames, ignore_index=True) if iteration_frames else pd.DataFrame(),
        "summary": pd.DataFrame(summary_rows),
    }


def pricing_error_summary(frame: pd.DataFrame, error_col: str = "pricing_error") -> pd.DataFrame:
    err = pd.to_numeric(frame.get(error_col), errors="coerce")
    finite = err[np.isfinite(err)]
    return pd.DataFrame(
        [
            {
                "n": int(finite.size),
                "mean_error": float(finite.mean()) if finite.size else np.nan,
                "median_error": float(finite.median()) if finite.size else np.nan,
                "median_abs_error": float(finite.abs().median()) if finite.size else np.nan,
                "p90_abs_error": float(finite.abs().quantile(0.90)) if finite.size else np.nan,
                "max_abs_error": float(finite.abs().max()) if finite.size else np.nan,
            }
        ]
    )


def choose_liquid_single_day_for_diagnostics(
    quotes: pd.DataFrame,
    min_pairs: int = 20,
    prefer_dte_range: tuple[int, int] = (21, 60),
) -> pd.Timestamp:
    return parity.choose_liquid_single_day(quotes, min_pairs=min_pairs, prefer_dte_range=prefer_dte_range)


__all__ = [
    "choose_liquid_single_day_for_diagnostics",
    "iv_solver_diagnostics",
    "pricing_error_summary",
    "realized_vol_forward_bsm_pricing_comparison",
    "solver_failure_by_log_moneyness",
    "solver_iterations_by_log_moneyness",
]
