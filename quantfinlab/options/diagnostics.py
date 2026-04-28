from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import bsm, iv, parity, quote_cleaning, rates_dividends


def _prepare_curve_panel(rates: pd.DataFrame) -> pd.DataFrame:
    data = rates.copy()
    lookup = {str(c).strip().lower(): c for c in data.columns}
    date_col = lookup.get("date", data.columns[0])
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data = data.dropna(subset=[date_col]).sort_values(date_col)
    curve = data.set_index(date_col)
    aliases = {
        "1 mo": "1M",
        "2 mo": "2M",
        "3 mo": "3M",
        "4 mo": "4M",
        "6 mo": "6M",
        "1 yr": "1Y",
        "2 yr": "2Y",
        "3 yr": "3Y",
        "5 yr": "5Y",
        "7 yr": "7Y",
        "10 yr": "10Y",
        "20 yr": "20Y",
        "30 yr": "30Y",
    }
    curve = curve.rename(columns={c: aliases.get(str(c).strip().lower(), c) for c in curve.columns})
    for col in curve.columns:
        curve[col] = pd.to_numeric(curve[col], errors="coerce")
    vals = curve.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size and float(np.nanmedian(np.abs(vals))) > 1.0:
        curve = curve / 100.0
    return curve


def _wide_chain_prefilter_report(
    raw: pd.DataFrame,
    *,
    min_dte: int,
    max_dte: int,
    moneyness_range: tuple[float, float],
    max_relative_spread: float,
    closest_atm_pairs: int | None,
    min_pairs_per_expiry: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    report: list[dict[str, Any]] = []

    def add(step: str, frame: pd.DataFrame, previous: int | None) -> int:
        rows = int(len(frame))
        report.append({"step": step, "rows": rows, "removed": np.nan if previous is None else previous - rows})
        return rows

    previous = add("raw wide rows", raw, None)
    required = {"quote_date", "expire_date", "underlying_last", "strike", "c_bid", "c_ask", "p_bid", "p_ask"}
    if raw.empty or not required.issubset(raw.columns):
        return raw.copy(), pd.DataFrame(report)

    out = raw.copy()
    out["_date"] = pd.to_datetime(out["quote_date"], errors="coerce").dt.normalize()
    out["_expiry"] = pd.to_datetime(out["expire_date"], errors="coerce").dt.normalize()
    out["_spot"] = pd.to_numeric(out["underlying_last"], errors="coerce")
    out["_strike"] = pd.to_numeric(out["strike"], errors="coerce")
    for col in ["c_bid", "c_ask", "p_bid", "p_ask"]:
        out[f"_{col}"] = pd.to_numeric(out[col], errors="coerce")
    out["_dte_calendar"] = (
        pd.to_numeric(out["dte"], errors="coerce")
        if "dte" in out.columns
        else (out["_expiry"] - out["_date"]).dt.total_seconds() / 86400.0
    )
    out["_call_mid"] = 0.5 * (out["_c_bid"] + out["_c_ask"])
    out["_put_mid"] = 0.5 * (out["_p_bid"] + out["_p_ask"])
    out["_call_rel_spread"] = (out["_c_ask"] - out["_c_bid"]) / out["_call_mid"].replace(0, np.nan)
    out["_put_rel_spread"] = (out["_p_ask"] - out["_p_bid"]) / out["_put_mid"].replace(0, np.nan)
    out["_moneyness"] = out["_strike"] / out["_spot"].replace(0, np.nan)

    valid_pair = (
        out["_date"].notna()
        & out["_expiry"].notna()
        & (out["_spot"] > 0)
        & (out["_strike"] > 0)
        & (out["_c_bid"] > 0)
        & (out["_c_ask"] >= out["_c_bid"])
        & (out["_p_bid"] > 0)
        & (out["_p_ask"] >= out["_p_bid"])
        & (out["_call_mid"] > 0)
        & (out["_put_mid"] > 0)
    )
    out = out.loc[valid_pair].copy()
    previous = add("valid call/put pair quotes", out, previous)

    spread_ok = (out["_call_rel_spread"] <= max_relative_spread) & (out["_put_rel_spread"] <= max_relative_spread)
    out = out.loc[spread_ok].copy()
    previous = add("relative spread filter", out, previous)

    out = out.loc[(out["_dte_calendar"] >= min_dte) & (out["_dte_calendar"] <= max_dte)].copy()
    previous = add("DTE filter", out, previous)

    lo, hi = moneyness_range
    out = out.loc[(out["_moneyness"] >= lo) & (out["_moneyness"] <= hi)].copy()
    previous = add("moneyness filter", out, previous)

    if closest_atm_pairs and closest_atm_pairs > 0 and not out.empty:
        out["_atm_score"] = np.abs(np.log(out["_moneyness"].clip(lower=1e-12)))
        out["_atm_rank"] = out.groupby(["_date", "_expiry"])["_atm_score"].rank(method="first")
        out = out.loc[out["_atm_rank"] <= int(closest_atm_pairs)].copy()
    previous = add("near-ATM pair preselection", out, previous)

    if min_pairs_per_expiry and min_pairs_per_expiry > 0 and not out.empty:
        counts = out.groupby(["_date", "_expiry"]).size()
        keep = counts[counts >= int(min_pairs_per_expiry)].reset_index()[["_date", "_expiry"]]
        out = out.merge(keep, on=["_date", "_expiry"], how="inner")
    add("liquid expiry slices", out, previous)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore"), pd.DataFrame(report)


def _pair_iv_quotes(iv_table: pd.DataFrame) -> pd.DataFrame:
    data = iv_table.copy()
    data["option_type"] = data["option_type"].map(quote_cleaning.parse_option_type)
    common = ["date", "expiry", "strike"]
    quote_cols = [
        "timestamp",
        "mid",
        "mid_raw",
        "bid",
        "bid_raw",
        "ask",
        "ask_raw",
        "spot",
        "tau",
        "dte",
        "rate",
        "discount_factor",
        "forward",
        "implied_carry",
        "volume",
        "spread",
        "rel_spread",
        "iv_bid",
        "iv_mid",
        "iv_ask",
        "iv_mid_success",
        "source_index",
        "price_unit_detected",
        "valuation_currency",
        "contract_size",
    ]
    keep = [c for c in [*common, *quote_cols] if c in data.columns]
    calls = data[data["option_type"] == "call"][keep].copy()
    puts = data[data["option_type"] == "put"][keep].copy()
    calls = calls.rename(columns={c: f"call_{c}" for c in keep if c not in common})
    puts = puts.rename(columns={c: f"put_{c}" for c in keep if c not in common})
    pairs = calls.merge(puts, on=common, how="inner")
    for col in ["timestamp", "spot", "tau", "dte", "rate", "discount_factor", "forward", "implied_carry"]:
        ccol = f"call_{col}"
        pcol = f"put_{col}"
        if ccol in pairs.columns:
            pairs[col] = pairs[ccol].combine_first(pairs.get(pcol))
    return pairs


def build_atm_iv_panel_from_option_quotes(
    option_quotes: pd.DataFrame,
    *,
    rates: pd.DataFrame | pd.Series | None = None,
    constant_rate: float = 0.0,
    min_dte: int = 7,
    max_dte: int = 120,
    moneyness_range: tuple[float, float] = (0.85, 1.15),
    max_relative_spread: float = 0.20,
    closest_atm_pairs: int | None = 25,
    min_pairs_per_expiry: int = 10,
    annualization_days: float = 365.25,
    solver: str = "lbr_lite",
    engine: str = "auto",
    underlying_default: str | None = "SPX",
) -> pd.DataFrame:
    """Build a clean ATM/near-ATM IV panel using the Project 4 options modules."""
    if option_quotes.empty:
        out = pd.DataFrame()
        out.attrs["cleaning_report"] = pd.DataFrame([{"step": "raw rows", "rows": 0, "removed": np.nan}])
        return out

    prefiltered, wide_report = _wide_chain_prefilter_report(
        option_quotes,
        min_dte=min_dte,
        max_dte=max_dte,
        moneyness_range=moneyness_range,
        max_relative_spread=max_relative_spread,
        closest_atm_pairs=closest_atm_pairs,
        min_pairs_per_expiry=min_pairs_per_expiry,
    )
    long_quotes = quote_cleaning.wide_option_chain_to_long(
        prefiltered,
        underlying_default=underlying_default,
        include_greeks=False,
    )
    long_quotes = quote_cleaning.ensure_option_mid_quotes(long_quotes)
    long_quotes = quote_cleaning.convert_quotes_to_usd_equivalent(long_quotes, unit="auto")
    clean, clean_report = quote_cleaning.clean_option_quotes(
        long_quotes,
        min_dte=min_dte,
        max_dte=max_dte,
        moneyness_range=moneyness_range,
        max_relative_spread=max_relative_spread,
        closest_atm_pairs=closest_atm_pairs,
        min_pairs_per_expiry=min_pairs_per_expiry,
        annualization_days=annualization_days,
    )
    clean_report = clean_report.copy()
    clean_report["step"] = "long-form " + clean_report["step"].astype(str)
    report = pd.concat([wide_report, clean_report], ignore_index=True)
    if "price_unit_detected" in clean.columns:
        unit_counts = clean["price_unit_detected"].astype(str).value_counts(dropna=False)
        report = pd.concat(
            [
                report,
                pd.DataFrame(
                    [
                        {
                            "step": f"price unit detected: {unit}",
                            "rows": int(count),
                            "removed": np.nan,
                        }
                        for unit, count in unit_counts.items()
                    ]
                ),
            ],
            ignore_index=True,
        )

    if clean.empty:
        out = pd.DataFrame()
        out.attrs["cleaning_report"] = report
        return out

    if rates is not None:
        if isinstance(rates, pd.DataFrame):
            clean = rates_dividends.attach_rates_to_options(clean, curve_panel=_prepare_curve_panel(rates))
        else:
            clean = rates_dividends.attach_rates_to_options(clean, rates=rates)
    elif "rate" not in clean.columns:
        clean = rates_dividends.attach_rates_to_options(clean, constant_rate=constant_rate)
    clean = rates_dividends.add_discount_factors(clean)

    forward_table = parity.infer_forwards_from_put_call_parity(clean, price_col="mid")
    report = pd.concat(
        [
            report,
            pd.DataFrame(
                [
                    {
                        "step": "valid forward/parity slices",
                        "rows": int(forward_table["forward"].notna().sum()) if "forward" in forward_table else 0,
                        "removed": np.nan,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    if forward_table.empty:
        out = pd.DataFrame()
        out.attrs["cleaning_report"] = report
        out.attrs["forward_table"] = forward_table
        return out

    fwd_cols = [
        "date",
        "expiry",
        "forward",
        "implied_carry",
        "n_pairs",
        "parity_error_median",
        "parity_error_mad",
        "parity_error_iqr",
    ]
    clean = clean.merge(forward_table[[c for c in fwd_cols if c in forward_table.columns]], on=["date", "expiry"], how="left")
    iv_table = iv.compute_iv_table(clean, price_cols=("bid", "mid", "ask"), solver=solver, engine=engine)
    pairs = _pair_iv_quotes(iv_table)
    if pairs.empty:
        out = pd.DataFrame()
        out.attrs["cleaning_report"] = report
        out.attrs["forward_table"] = forward_table
        return out

    pairs["date"] = pd.to_datetime(pairs["date"], errors="coerce").dt.normalize()
    pairs["expiry"] = pd.to_datetime(pairs["expiry"], errors="coerce").dt.normalize()
    pairs["quote_datetime"] = pd.to_datetime(pairs.get("timestamp"), errors="coerce")
    pairs["dte_calendar"] = pd.to_numeric(pairs["dte"], errors="coerce")
    pairs["dte_trading"] = pairs["dte_calendar"] * 252.0 / float(annualization_days)
    pairs["straddle_bid"] = pd.to_numeric(pairs["call_bid"], errors="coerce") + pd.to_numeric(pairs["put_bid"], errors="coerce")
    pairs["straddle_mid"] = pd.to_numeric(pairs["call_mid"], errors="coerce") + pd.to_numeric(pairs["put_mid"], errors="coerce")
    pairs["straddle_ask"] = pd.to_numeric(pairs["call_ask"], errors="coerce") + pd.to_numeric(pairs["put_ask"], errors="coerce")
    pairs["straddle_spread"] = pairs["straddle_ask"] - pairs["straddle_bid"]
    pairs["straddle_rel_spread"] = pairs["straddle_spread"] / pairs["straddle_mid"].replace(0, np.nan)
    for side in ["bid", "mid", "ask"]:
        pairs[f"atm_iv_{side}"] = np.nanmean(
            np.column_stack(
                [
                    pd.to_numeric(pairs.get(f"call_iv_{side}"), errors="coerce"),
                    pd.to_numeric(pairs.get(f"put_iv_{side}"), errors="coerce"),
                ]
            ),
            axis=1,
        )
    fwd = pd.to_numeric(pairs["forward"], errors="coerce").combine_first(pd.to_numeric(pairs["spot"], errors="coerce"))
    pairs["atm_score"] = np.abs(np.log(pd.to_numeric(pairs["strike"], errors="coerce") / fwd.clip(lower=1e-12)))
    pairs["quote_quality_score"] = pairs["atm_score"] + pairs["straddle_rel_spread"].clip(lower=0).fillna(0)
    pairs = pairs[pairs["atm_iv_mid"].notna() & (pairs["atm_iv_mid"] > 0)].copy()
    selected = pairs.sort_values(["date", "expiry", "quote_quality_score"]).groupby(["date", "expiry"], as_index=False).head(1)

    preferred = [
        "date",
        "quote_datetime",
        "expiry",
        "dte_calendar",
        "dte_trading",
        "tau",
        "spot",
        "forward",
        "strike",
        "rate",
        "discount_factor",
        "call_bid",
        "call_bid_raw",
        "call_mid",
        "call_mid_raw",
        "call_ask",
        "call_ask_raw",
        "put_bid",
        "put_bid_raw",
        "put_mid",
        "put_mid_raw",
        "put_ask",
        "put_ask_raw",
        "straddle_bid",
        "straddle_mid",
        "straddle_ask",
        "atm_iv_bid",
        "atm_iv_mid",
        "atm_iv_ask",
        "straddle_spread",
        "straddle_rel_spread",
        "atm_score",
        "quote_quality_score",
        "call_volume",
        "put_volume",
        "call_price_unit_detected",
        "put_price_unit_detected",
        "call_valuation_currency",
        "put_valuation_currency",
        "call_contract_size",
        "put_contract_size",
        "n_pairs",
        "parity_error_mad",
    ]
    out = selected[[c for c in preferred if c in selected.columns]].sort_values(["date", "expiry"]).reset_index(drop=True)
    report = pd.concat(
        [report, pd.DataFrame([{"step": "final ATM panel rows", "rows": int(len(out)), "removed": np.nan}])],
        ignore_index=True,
    )
    out.attrs["cleaning_report"] = report
    out.attrs["forward_table"] = forward_table
    out.attrs["iv_solver"] = solver
    out.attrs["iv_engine"] = iv_table.attrs.get("engine_used", "unknown")
    return out


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
    from quantfinlab.volatility import realized

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
    "build_atm_iv_panel_from_option_quotes",
    "choose_liquid_single_day_for_diagnostics",
    "iv_solver_diagnostics",
    "pricing_error_summary",
    "realized_vol_forward_bsm_pricing_comparison",
    "solver_failure_by_log_moneyness",
    "solver_iterations_by_log_moneyness",
]
