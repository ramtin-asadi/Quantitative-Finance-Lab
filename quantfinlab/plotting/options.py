from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .curves import LAB_COLORS, set_plot_style


def _ax(ax=None):
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _ordered_strategy_columns(frame: pd.DataFrame) -> list:
    preferred = ["unhedged", "delta", "delta_vega"]
    return [c for c in preferred if c in frame.columns] + [c for c in frame.columns if c not in preferred]


def _format_hedging_axis(ax, n_series: int) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.20)
    ax.tick_params(axis="x", labelrotation=25)
    if n_series > 0:
        ax.legend(loc="best", ncol=min(3, n_series), fontsize=7)


def _relative_spread(frame: pd.DataFrame) -> pd.Series:
    if "rel_spread" in frame.columns:
        return pd.to_numeric(frame["rel_spread"], errors="coerce")
    if {"bid", "ask", "mid"}.issubset(frame.columns):
        return (frame["ask"] - frame["bid"]) / frame["mid"].replace(0, np.nan)
    return pd.Series(np.nan, index=frame.index)


def plot_quote_filter_waterfall(filter_report: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    report = filter_report.copy()
    if "step" not in report.columns or "rows" not in report.columns:
        ax.text(0.5, 0.5, "No filter report", ha="center", va="center")
        return ax
    ax.bar(range(len(report)), report["rows"].astype(float))
    ax.set_xticks(range(len(report)))
    ax.set_xticklabels(report["step"], rotation=35, ha="right")
    ax.set_ylabel("rows")
    ax.set_title(title or "Quote filter waterfall")
    return ax


def plot_clean_vs_dirty_spread(dirty: pd.DataFrame, clean: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    dirty_spread = _relative_spread(dirty).dropna()
    clean_spread = _relative_spread(clean).dropna()
    if dirty_spread.empty and clean_spread.empty:
        ax.text(0.5, 0.5, "No bid/ask spread data", ha="center", va="center")
    else:
        ax.hist(dirty_spread, bins=40, alpha=0.45, label="raw")
        ax.hist(clean_spread, bins=40, alpha=0.65, label="clean")
        ax.legend()
    ax.set_xlabel("relative spread")
    ax.set_ylabel("count")
    ax.set_title(title or "Clean vs dirty spreads")
    return ax


def plot_moneyness_dte_coverage(quotes: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    ax.scatter(quotes.get("moneyness"), quotes.get("dte"), s=8, alpha=0.35)
    ax.set_xlabel("moneyness K/S")
    ax.set_ylabel("days to expiry")
    ax.set_title(title or "Moneyness and DTE coverage")
    return ax


def plot_forward_vs_spot(forward_table: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    if forward_table.empty:
        ax.text(0.5, 0.5, "No forward data", ha="center", va="center")
        return ax
    plot_data = forward_table.sort_values("date")
    ax.plot(plot_data["date"], plot_data["forward"], ".", ms=3, label="forward")
    if "spot" in plot_data.columns:
        ax.plot(plot_data["date"], plot_data["spot"], ".", ms=2, alpha=0.55, label="spot")
    ax.set_ylabel("level")
    ax.set_title(title or "Parity-implied forward vs spot")
    ax.legend(loc="best")
    return ax


def plot_parity_error_by_moneyness(quotes: pd.DataFrame, forward_table: pd.DataFrame | None = None, ax=None, title: str | None = None):
    ax = _ax(ax)
    try:
        from quantfinlab.options import parity

        table = parity.parity_error_table(quotes, forward_table)
    except Exception:
        table = pd.DataFrame()
    if table.empty or "parity_residual" not in table.columns:
        ax.text(0.5, 0.5, "No parity pairs", ha="center", va="center")
    else:
        x = table["moneyness"] if "moneyness" in table.columns else table["strike"]
        ax.scatter(x, table["parity_residual"], s=8, alpha=0.4)
        ax.axhline(0.0, lw=1, color="black", ls="--")
    ax.set_xlabel("moneyness")
    ax.set_ylabel("parity residual")
    ax.set_title(title or "Parity error by moneyness")
    return ax


def _choose_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "date" not in frame.columns:
        return None
    counts = frame.dropna(subset=["date"]).groupby("date").size()
    return pd.Timestamp(counts.idxmax()) if not counts.empty else None


def plot_iv_smile(iv_table: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = iv_table.dropna(subset=["iv_mid"]).copy() if "iv_mid" in iv_table.columns else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "No IV data", ha="center", va="center")
        return ax
    date = _choose_date(data)
    data = data[data["date"] == date].copy()
    xcol = "log_moneyness" if "log_moneyness" in data.columns else "moneyness"
    for expiry, grp in data.groupby("expiry"):
        if len(grp) < 4:
            continue
        grp = grp.sort_values(xcol)
        ax.plot(grp[xcol], grp["iv_mid"], marker="o", ms=2.5, lw=1, label=str(pd.Timestamp(expiry).date()))
        if len(ax.lines) >= 4:
            break
    ax.set_xlabel(xcol)
    ax.set_ylabel("implied vol")
    ax.set_title(title or "IV smile")
    if ax.lines:
        ax.legend(fontsize=7)
    return ax


def plot_iv_term_structure(iv_table: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = iv_table.dropna(subset=["iv_mid"]).copy() if "iv_mid" in iv_table.columns else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "No IV data", ha="center", va="center")
        return ax
    if "log_moneyness" in data.columns:
        data = data[np.abs(data["log_moneyness"]) <= 0.05]
    grouped = data.groupby("dte")["iv_mid"].median().reset_index().sort_values("dte")
    ax.plot(grouped["dte"], grouped["iv_mid"], marker="o", ms=3)
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("ATM implied vol")
    ax.set_title(title or "ATM IV term structure")
    return ax


def plot_iv_bid_ask_band(iv_table: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = iv_table.dropna(subset=["iv_mid"]).copy() if "iv_mid" in iv_table.columns else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "No IV data", ha="center", va="center")
        return ax
    date = _choose_date(data)
    data = data[data["date"] == date].copy()
    expiry = data.groupby("expiry").size().idxmax()
    data = data[data["expiry"] == expiry].sort_values("moneyness")
    x = data["moneyness"]
    y = data["iv_mid"]
    low = data["iv_bid"].combine_first(y) if "iv_bid" in data.columns else y
    high = data["iv_ask"].combine_first(y) if "iv_ask" in data.columns else y
    ax.plot(x, y, lw=1.5, label="mid")
    ax.fill_between(x, np.minimum(low, y), np.maximum(high, y), alpha=0.2, label="bid/ask IV band")
    ax.set_xlabel("moneyness")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "IV bid/mid/ask band")
    ax.legend()
    return ax


def plot_solver_runtime(solver_comparison: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    ax.bar(solver_comparison["solver"], solver_comparison["elapsed_sec"])
    ax.set_ylabel("seconds")
    ax.set_title(title or "Solver runtime")
    return ax


def plot_solver_success(solver_comparison: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    ax.bar(solver_comparison["solver"], solver_comparison["success_rate"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(title or "Solver success")
    return ax


def plot_pricing_error_hist(pricing_errors: pd.DataFrame | pd.Series, ax=None, title: str | None = None):
    ax = _ax(ax)
    values = pricing_errors if isinstance(pricing_errors, pd.Series) else pricing_errors.select_dtypes("number").stack()
    ax.hist(pd.to_numeric(values, errors="coerce").dropna(), bins=30, alpha=0.75)
    ax.set_title(title or "Pricing error summary")
    return ax


def plot_greek_bands(greek_bands: pd.DataFrame, greek: str = "delta", ax=None, title: str | None = None):
    ax = _ax(ax)
    data = greek_bands.dropna(subset=[f"{greek}_mid"]).copy() if f"{greek}_mid" in greek_bands.columns else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "No Greek band data", ha="center", va="center")
        return ax
    date = _choose_date(data)
    data = data[data["date"] == date].sort_values("moneyness")
    ax.plot(data["moneyness"], data[f"{greek}_mid"], label=f"{greek} mid")
    ax.fill_between(data["moneyness"], data[f"{greek}_low"], data[f"{greek}_high"], alpha=0.2)
    ax.set_xlabel("moneyness")
    ax.set_title(title or f"{greek} uncertainty band")
    ax.legend()
    return ax


def plot_realized_vs_implied_vol(rv_iv: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    if rv_iv.empty:
        ax.text(0.5, 0.5, "No RV/IV data", ha="center", va="center")
        return ax
    grouped = rv_iv.groupby("date").agg(realized_vol=("realized_vol", "median"), implied_vol=("implied_vol", "median")).reset_index()
    ax.plot(grouped["date"], grouped["realized_vol"], label="realized")
    ax.plot(grouped["date"], grouped["implied_vol"], label="implied")
    ax.set_ylabel("volatility")
    ax.set_title(title or "Realized vs implied volatility")
    ax.legend()
    return ax


def _result_frame(results: dict, key: str) -> pd.DataFrame:
    if isinstance(results, dict):
        value = results.get(key, pd.DataFrame())
        return value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    return pd.DataFrame()


def _table_from_result(obj, key: str = "table") -> pd.DataFrame:
    if isinstance(obj, dict):
        value = obj.get(key, pd.DataFrame())
        return value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


def _binned_summary(x, y, bins: int = 21) -> pd.DataFrame:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 10:
        return pd.DataFrame(columns=["x_mid", "y_med", "y_q1", "y_q3", "count"])
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    edges = np.unique(np.linspace(np.nanmin(x_arr), np.nanmax(x_arr), int(bins) + 1))
    if len(edges) < 3:
        return pd.DataFrame(columns=["x_mid", "y_med", "y_q1", "y_q3", "count"])
    cut = pd.cut(x_arr, bins=edges, include_lowest=True, duplicates="drop")
    grouped = pd.DataFrame({"x": x_arr, "y": y_arr, "cut": cut}).groupby("cut", observed=False)
    return (
        grouped.agg(
            x_mid=("x", "median"),
            y_med=("y", "median"),
            y_q1=("y", lambda z: np.nanquantile(z, 0.25)),
            y_q3=("y", lambda z: np.nanquantile(z, 0.75)),
            count=("y", "size"),
        )
        .reset_index(drop=True)
    )


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def _strategy_frame(results: dict, value_col: str) -> pd.DataFrame:
    direct = _result_frame(results, value_col)
    if not direct.empty:
        return direct
    components = _result_frame(results, "components")
    if components.empty or value_col not in components.columns:
        return pd.DataFrame()
    date_col = "date" if "date" in components.columns else "trade_date"
    return components.pivot_table(index=date_col, columns="strategy", values=value_col, aggfunc="sum").sort_index()


def plot_single_day_parity_forward_extraction(
    single_day_quotes: pd.DataFrame,
    single_day_forward: pd.DataFrame,
    ax=None,
    title: str | None = None,
):
    ax = _ax(ax)
    quotes = single_day_quotes.copy()
    forward = single_day_forward.copy()
    if quotes.empty:
        ax.text(0.5, 0.5, "No parity data", ha="center", va="center")
        return ax

    from quantfinlab.fixed_income import discounting
    from quantfinlab.options.quote_cleaning import pair_put_call_quotes

    pairs = pair_put_call_quotes(quotes, price_col="mid")
    if pairs.empty:
        ax.text(0.5, 0.5, "No put-call pairs", ha="center", va="center")
        return ax

    if not forward.empty and "expiry" in forward.columns and "n_pairs" in forward.columns:
        exp = forward.sort_values("n_pairs", ascending=False).iloc[0]["expiry"]
    elif "expiry" in forward.columns and not forward.empty:
        exp = forward.iloc[0]["expiry"]
    else:
        exp_col = _first_present(pairs, ("expiry", "call_expiry", "put_expiry"))
        exp = pairs.groupby(exp_col).size().sort_values(ascending=False).index[0] if exp_col else None

    if exp is not None:
        exp_ts = pd.Timestamp(exp)
        exp_col = _first_present(pairs, ("expiry", "call_expiry", "put_expiry"))
        if exp_col:
            pairs = pairs[pd.to_datetime(pairs[exp_col], errors="coerce").eq(exp_ts)].copy()
    pairs = pairs.sort_values("strike")

    rate = pairs.get("call_rate", pairs.get("rate", np.nan))
    tau = pairs.get("call_tau", pairs.get("tau", np.nan))
    df = discounting.discount_factor_from_rate(rate, tau)
    strike = pd.to_numeric(pairs["strike"], errors="coerce")
    f_mid = strike + (pd.to_numeric(pairs["call_mid"], errors="coerce") - pd.to_numeric(pairs["put_mid"], errors="coerce")) / df
    if {"call_bid", "put_ask", "call_ask", "put_bid"}.issubset(pairs.columns):
        f_low = strike + (pd.to_numeric(pairs["call_bid"], errors="coerce") - pd.to_numeric(pairs["put_ask"], errors="coerce")) / df
        f_high = strike + (pd.to_numeric(pairs["call_ask"], errors="coerce") - pd.to_numeric(pairs["put_bid"], errors="coerce")) / df
    else:
        f_low = f_mid
        f_high = f_mid

    if not forward.empty and "forward" in forward.columns:
        if exp is not None and "expiry" in forward.columns:
            frow = forward[pd.to_datetime(forward["expiry"], errors="coerce").eq(pd.Timestamp(exp))]
            f_hat = float(frow["forward"].iloc[0]) if len(frow) else float(np.nanmedian(f_mid))
        else:
            f_hat = float(forward["forward"].iloc[0])
    else:
        f_hat = float(np.nanmedian(f_mid))

    ax.scatter(strike, f_mid, s=18, alpha=0.65, label="f_mid(k)")
    ax.fill_between(strike, np.minimum(f_low, f_high), np.maximum(f_low, f_high), alpha=0.13, label="feasible interval")
    if np.isfinite(f_hat):
        ax.axhline(f_hat, lw=2, label="f_hat")
    ax.set_xlabel("strike")
    ax.set_ylabel("forward level")
    date_col = _first_present(quotes, ("date", "trade_date"))
    date_txt = pd.Timestamp(quotes[date_col].dropna().iloc[0]).date() if date_col and quotes[date_col].notna().any() else ""
    exp_txt = pd.Timestamp(exp).date() if exp is not None else ""
    ax.set_title(title or f"single-day parity forward extraction ({date_txt}, {exp_txt})")
    ax.legend(loc="best")
    return ax


def plot_single_day_forward_iv_skew(iv_table: pd.DataFrame, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = iv_table.dropna(subset=["iv_mid"]).copy() if "iv_mid" in iv_table.columns else pd.DataFrame()
    if data.empty:
        ax.text(0.5, 0.5, "No IV data", ha="center", va="center")
        return ax
    type_col = _first_present(data, ("option_type", "cp"))
    if type_col:
        calls = data[data[type_col].astype(str).str.lower().str.startswith("c")].copy()
        if len(calls) >= 6:
            data = calls
    xcol = "log_moneyness" if "log_moneyness" in data.columns else ("lm_f" if "lm_f" in data.columns else "moneyness")
    exp_col = "expiry" if "expiry" in data.columns else "expiry_datetime" if "expiry_datetime" in data.columns else None
    if exp_col:
        counts = data.groupby(exp_col).size().sort_values(ascending=False)
        chosen_exp = counts.index[0]
        data = data[data[exp_col].eq(chosen_exp)].copy()
    data = data.sort_values(xcol)
    ax.plot(data[xcol], data["iv_mid"], marker="o", lw=1.4, ms=3.5)
    ax.set_xlabel("log-moneyness" if xcol in {"log_moneyness", "lm_f"} else xcol)
    ax.set_ylabel("implied vol")
    ax.set_title(title or "single-day forward-based implied vol skew (mid call)")
    return ax


def plot_market_mid_vs_realized_vol_forward_bsm(rv_pricing, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = _table_from_result(rv_pricing)
    if data.empty or "realized_vol_forward_bsm_price" not in data.columns:
        ax.text(0.5, 0.5, "No realized-vol pricing data", ha="center", va="center")
        return ax
    xcol = "log_moneyness" if "log_moneyness" in data.columns else ("lm_f" if "lm_f" in data.columns else None)
    price_col = "mid" if "mid" in data.columns else _first_present(data, ("price", "market_mid"))
    if xcol is None or price_col is None:
        ax.text(0.5, 0.5, "Need log-moneyness and price data", ha="center", va="center")
        return ax
    sample = data.dropna(subset=[xcol, price_col, "realized_vol_forward_bsm_price"]).copy()
    if sample.empty:
        ax.text(0.5, 0.5, "No finite pricing rows", ha="center", va="center")
        return ax
    bin_mkt = _binned_summary(sample[xcol], sample[price_col])
    bin_rv = _binned_summary(sample[xcol], sample["realized_vol_forward_bsm_price"])
    if bin_mkt.empty or bin_rv.empty:
        ax.scatter(sample[xcol], sample[price_col], s=8, alpha=0.25, label="market mid")
        ax.scatter(sample[xcol], sample["realized_vol_forward_bsm_price"], s=8, alpha=0.25, label="realized-vol bsm")
    else:
        ax.plot(bin_mkt["x_mid"], bin_mkt["y_med"], lw=1.8, label="market mid")
        ax.plot(bin_rv["x_mid"], bin_rv["y_med"], lw=1.8, label="realized-vol bsm")
        ax.fill_between(bin_rv["x_mid"], bin_rv["y_q1"], bin_rv["y_q3"], alpha=0.20, label="realized-vol iqr")
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("option price")
    ax.set_title(title or "market mid vs realized-vol forward-bsm pricing")
    ax.legend(loc="best")
    return ax


def plot_iv_failure_rate_by_log_moneyness(solver_diagnostics, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = _table_from_result(solver_diagnostics, "failure_by_log_moneyness")
    if data.empty:
        ax.text(0.5, 0.5, "No solver failure diagnostics", ha="center", va="center")
        return ax
    for i, (solver, grp) in enumerate(data.groupby("solver")):
        grp = grp.sort_values("x_mid")
        ax.plot(grp["x_mid"], grp["failure_rate"], marker="o", ms=3, lw=1.4, label=str(solver), color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("failure rate")
    ax.set_ylim(bottom=0)
    ax.set_title(title or "IV inversion failure rate by log-moneyness")
    ax.legend(fontsize=7)
    return ax


def plot_iv_iterations_by_log_moneyness(solver_diagnostics, ax=None, title: str | None = None):
    ax = _ax(ax)
    data = _table_from_result(solver_diagnostics, "iterations_by_log_moneyness")
    if data.empty:
        ax.text(0.5, 0.5, "No solver iteration diagnostics", ha="center", va="center")
        return ax
    for i, (solver, grp) in enumerate(data.groupby("solver")):
        grp = grp.sort_values("x_mid")
        ax.plot(
            grp["x_mid"],
            grp["median_iterations"],
            marker="o",
            ms=3,
            lw=1.4,
            label=str(solver),
            color=LAB_COLORS[i % len(LAB_COLORS)],
        )
        if "p90_iterations" in grp.columns:
            ax.fill_between(
                grp["x_mid"],
                grp["median_iterations"],
                grp["p90_iterations"],
                alpha=0.15,
                color=LAB_COLORS[i % len(LAB_COLORS)],
            )
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("median iterations")
    ax.set_title(title or "Successful IV iterations by log-moneyness")
    ax.legend(fontsize=7)
    return ax


def _greek_slice(data: pd.DataFrame, greek: str) -> pd.DataFrame:
    if data.empty:
        return data
    xcol = "log_moneyness" if "log_moneyness" in data.columns else ("lm_f" if "lm_f" in data.columns else None)
    if xcol is None:
        if "moneyness" in data.columns:
            data = data.copy()
            data["log_moneyness"] = np.log(pd.to_numeric(data["moneyness"], errors="coerce"))
            xcol = "log_moneyness"
        else:
            data = data.copy()
            data["_seq"] = np.arange(len(data), dtype=float)
            xcol = "_seq"
    cols = [xcol, f"{greek}_numpy", f"{greek}_jax"]
    date_col = _first_present(data, ("date", "trade_date"))
    exp_col = _first_present(data, ("expiry", "expiry_datetime"))
    if date_col and exp_col:
        score = data.dropna(subset=[xcol]).groupby([date_col, exp_col], dropna=False).size().sort_values(ascending=False)
        if not score.empty:
            date, expiry = score.index[0]
            data = data[(data[date_col].eq(date)) & (data[exp_col].eq(expiry))].copy()
    return data.dropna(subset=[c for c in cols if c in data.columns]).sort_values(xcol)


def plot_numpy_jax_greek_comparison(greek_comparison, greek: str = "delta", ax=None, title: str | None = None):
    ax = _ax(ax)
    data = _table_from_result(greek_comparison, "comparison")
    np_col = f"{greek}_numpy"
    jax_col = f"{greek}_jax"
    if data.empty or np_col not in data.columns or jax_col not in data.columns:
        ax.text(0.5, 0.5, "No NumPy/JAX comparison", ha="center", va="center")
        return ax
    sample = _greek_slice(data, greek)
    xcol = "log_moneyness" if "log_moneyness" in sample.columns else ("lm_f" if "lm_f" in sample.columns else "_seq")
    if sample.empty:
        ax.text(0.5, 0.5, "No finite Greek rows", ha="center", va="center")
        return ax
    ax.plot(sample[xcol], sample[np_col], lw=2.0, label="analytic")
    bands = _table_from_result(greek_comparison, "bands")
    if not bands.empty and f"{greek}_low" in bands.columns and f"{greek}_high" in bands.columns:
        band = bands.reindex(sample.index)
        ax.fill_between(
            sample[xcol],
            pd.to_numeric(band[f"{greek}_low"], errors="coerce"),
            pd.to_numeric(band[f"{greek}_high"], errors="coerce"),
            alpha=0.18,
            label="uncertainty band",
        )
    ax.plot(sample[xcol], sample[jax_col], lw=2.0, ls="--", alpha=0.90, label="jax autodiff")
    ax.set_xlabel("log-moneyness" if xcol in {"log_moneyness", "lm_f"} else "observation")
    ax.set_ylabel(greek)
    ax.set_title(title or greek, fontsize=13)
    ax.legend(loc="best")
    return ax


def plot_greek_error_summary(greek_comparison, ax=None, title: str | None = None):
    ax = _ax(ax)
    summary = _table_from_result(greek_comparison, "summary")
    error_col = "mae" if "mae" in summary.columns else "median_abs_error"
    if summary.empty or "greek" not in summary.columns or error_col not in summary.columns:
        ax.text(0.5, 0.5, "No Greek error summary", ha="center", va="center")
        return ax
    show = summary.set_index("greek")[error_col].sort_values()
    ax.barh(show.index, show.values, color=LAB_COLORS[0])
    ax.set_xlabel("mean absolute error")
    ax.set_title(title or "Greek NumPy/JAX error summary")
    return ax


def plot_greek_uncertainty_bands(greek_bands: pd.DataFrame, greek: str = "delta", ax=None, title: str | None = None):
    ax = _ax(ax)
    if greek_bands.empty:
        ax.text(0.5, 0.5, "No Greek band data", ha="center", va="center")
        return ax
    rows = []
    for name in ["delta", "gamma", "vega", "volga", "vanna", "theta", "rho"]:
        col = f"{name}_band"
        if col in greek_bands.columns:
            vals = pd.to_numeric(greek_bands[col], errors="coerce")
            rows.append(
                {
                    "greek": name,
                    "median_band": float(np.nanmedian(vals)) if vals.notna().any() else np.nan,
                    "p90_band": float(np.nanquantile(vals.dropna(), 0.90)) if vals.notna().any() else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return plot_greek_bands(greek_bands, greek=greek, ax=ax, title=title or "Greek uncertainty from bid/mid/ask IV")
    x = np.arange(len(summary), dtype=float)
    ax.bar(x - 0.18, summary["median_band"], width=0.36, label="median band")
    ax.bar(x + 0.18, summary["p90_band"], width=0.36, alpha=0.70, label="p90 band")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["greek"], rotation=0)
    ax.set_ylabel("band width")
    ax.set_title(title or "Greek uncertainty from bid/mid/ask IV")
    ax.legend(loc="best")
    return ax


def plot_hedging_net_equity(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    nav = _result_frame(hedge_results, "nav")
    if nav.empty:
        ax.text(0.5, 0.5, "No hedging NAV", ha="center", va="center")
        return ax
    columns = _ordered_strategy_columns(nav)
    for i, col in enumerate(columns):
        ax.plot(nav.index, nav[col], lw=1.5, label=str(col), color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_title(title or "net equity: cost-aware banded hedge comparison")
    ax.set_ylabel("equity")
    ax.set_xlabel("date")
    _format_hedging_axis(ax, len(columns))
    return ax


def plot_hedging_rolling_volatility(
    hedge_results: dict,
    window: int = 20,
    annualization_days: float = 365.0,
    annualize: bool = False,
    ax=None,
    title: str | None = None,
):
    ax = _ax(ax)
    pnl = _strategy_frame(hedge_results, "net_pnl")
    if pnl.empty:
        ax.text(0.5, 0.5, "No hedging P&L", ha="center", va="center")
        return ax
    vol = pnl.rolling(int(window)).std()
    if annualize and annualization_days is not None:
        vol = vol * np.sqrt(float(annualization_days))
    columns = _ordered_strategy_columns(vol)
    for i, col in enumerate(columns):
        ax.plot(vol.index, vol[col], lw=1.5, label=str(col), color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_ylabel("std of daily pnl")
    ax.set_xlabel("date")
    ax.set_title(title or f"rolling {int(window)} day pnl volatility")
    _format_hedging_axis(ax, len(columns))
    return ax


def plot_hedging_rolling_turnover(hedge_results: dict, window: int = 21, ax=None, title: str | None = None):
    ax = _ax(ax)
    turnover = _strategy_frame(hedge_results, "turnover")
    if turnover.empty:
        ax.text(0.5, 0.5, "No turnover data", ha="center", va="center")
        return ax
    roll_turn = turnover.rolling(int(window)).mean()
    for col in roll_turn.columns:
        ax.plot(roll_turn.index, roll_turn[col], label=str(col))
    ax.legend(ncol=min(3, len(roll_turn.columns)))
    ax.set_title(title or f"rolling {int(window)} day mean turnover")
    ax.set_ylabel("turnover")
    ax.set_xlabel("date")
    return ax


def plot_hedging_cumulative_pnl(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    pnl = _strategy_frame(hedge_results, "daily_pnl")
    if pnl.empty:
        pnl = _strategy_frame(hedge_results, "net_pnl")
    if pnl.empty:
        ax.text(0.5, 0.5, "No P&L data", ha="center", va="center")
        return ax
    cumulative = pnl.cumsum()
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=str(col))
    ax.legend(ncol=min(3, len(cumulative.columns)))
    ax.set_title(title or "cumulative pnl")
    ax.set_ylabel("cumulative pnl")
    ax.set_xlabel("date")
    return ax


def plot_rolling_residual_delta(hedge_results: dict, window: int = 20, ax=None, title: str | None = None):
    ax = _ax(ax)
    from quantfinlab.backtest import options as opt_backtest

    data = opt_backtest.rolling_residual_delta(hedge_results, window=window)
    if data.empty:
        ax.text(0.5, 0.5, "No residual delta data", ha="center", va="center")
        return ax
    columns = _ordered_strategy_columns(data)
    for i, col in enumerate(columns):
        ax.plot(data.index, data[col], lw=1.5, label=str(col), color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_ylabel("|residual delta|")
    ax.set_xlabel("date")
    ax.set_title(title or "rolling mean absolute residual delta")
    _format_hedging_axis(ax, len(columns))
    return ax


def plot_rolling_residual_vega(hedge_results: dict, window: int = 20, ax=None, title: str | None = None):
    ax = _ax(ax)
    from quantfinlab.backtest import options as opt_backtest

    data = opt_backtest.rolling_residual_vega(hedge_results, window=window)
    if data.empty:
        ax.text(0.5, 0.5, "No residual vega data", ha="center", va="center")
        return ax
    columns = _ordered_strategy_columns(data)
    for i, col in enumerate(columns):
        ax.plot(data.index, data[col], lw=1.5, label=str(col), color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_ylabel("|residual vega|")
    ax.set_xlabel("date")
    ax.set_title(title or "rolling mean absolute residual vega")
    _format_hedging_axis(ax, len(columns))
    return ax


def plot_hedging_nav(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    nav = _result_frame(hedge_results, "nav")
    if nav.empty:
        ax.text(0.5, 0.5, "No hedging NAV", ha="center", va="center")
    else:
        nav.plot(ax=ax)
    ax.set_title(title or "Hedging NAV")
    ax.set_ylabel("NAV")
    return ax


def plot_hedging_drawdown(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    nav = _result_frame(hedge_results, "nav")
    if nav.empty:
        ax.text(0.5, 0.5, "No drawdown data", ha="center", va="center")
    else:
        (nav - nav.cummax()).plot(ax=ax)
    ax.set_title(title or "Hedging drawdown")
    return ax


def plot_hedging_pnl_components(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    components = _result_frame(hedge_results, "components")
    if components.empty:
        ax.text(0.5, 0.5, "No P&L components", ha="center", va="center")
    else:
        comp = components.groupby("strategy")[["option_pnl", "hedge_pnl", "transaction_costs"]].sum()
        comp.plot(kind="bar", ax=ax)
    ax.set_title(title or "Hedging P&L components")
    return ax


def plot_hedge_exposures(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    exposures = _result_frame(hedge_results, "exposures")
    if exposures.empty:
        ax.text(0.5, 0.5, "No exposure data", ha="center", va="center")
    else:
        for strategy, grp in exposures.groupby("strategy"):
            ax.plot(grp["date"], grp["delta_after"], label=f"{strategy} delta")
    ax.set_title(title or "Hedge exposures")
    ax.legend(loc="best")
    return ax


def plot_hedge_trades(hedge_results: dict, ax=None, title: str | None = None):
    ax = _ax(ax)
    trades = _result_frame(hedge_results, "trades")
    if trades.empty:
        ax.text(0.5, 0.5, "No hedge trades", ha="center", va="center")
    else:
        counts = trades.groupby("strategy").size()
        ax.bar(counts.index, counts.values)
    ax.set_ylabel("trade count")
    ax.set_title(title or "Hedge trades")
    return ax


def _finite_limits(x, pad: float = 0.02):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def _grid_xy(grid: dict):
    k = np.asarray(grid["k"], dtype=float)
    tau_days = np.asarray(grid.get("tau_days", np.asarray(grid["tau"], dtype=float) * grid.get("annualization_days", 365.25)), dtype=float)
    return np.meshgrid(k, tau_days)


def _sym_lim(arr, q: float = 0.98, floor: float = 1e-10):
    x = np.abs(np.asarray(arr, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return floor
    lim = float(np.nanquantile(x, q))
    return max(lim, floor)


def quote_coverage_map(ax, quotes: pd.DataFrame, k_col: str = "k", tau_col: str = "tau", iv_col: str = "iv_mid", title: str | None = None):
    ax = _ax(ax)
    data = quotes.dropna(subset=[k_col, tau_col, iv_col]).copy()
    if data.empty:
        ax.text(0.5, 0.5, "No quote data", ha="center", va="center")
        return ax
    tau_days = data[tau_col] * float(data.get("annualization_days", pd.Series(365.25, index=data.index)).iloc[0] if "annualization_days" in data else 365.25)
    hb = ax.hexbin(data[k_col], tau_days, C=data[iv_col], gridsize=(36, 22), mincnt=1, reduce_C_function=np.nanmedian, cmap="viridis")
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("days to expiry")
    ax.set_title(title or "Quote coverage and median IV")
    ax.figure.colorbar(hb, ax=ax, pad=0.01, label="median IV")
    ax.text(0.02, 0.96, f"n={len(data):,}", transform=ax.transAxes, va="top", fontsize=8)
    return ax


def smooth_surface_3d(ax, grid: dict, iv_grid, quotes: pd.DataFrame | None = None, k_col: str = "k", tau_col: str = "tau", iv_col: str = "iv_mid", title: str | None = None, quote_sample: int = 2000, random_state: int | None = None):
    set_plot_style()
    x, y = _grid_xy(grid)
    surf = ax.plot_surface(x, y, np.ma.masked_invalid(iv_grid), cmap="viridis", linewidth=0, antialiased=True, alpha=0.90)
    if quotes is not None and not quotes.empty:
        q = quotes.dropna(subset=[k_col, tau_col, iv_col])
        if len(q) > quote_sample:
            q = q.sample(int(quote_sample), random_state=random_state)
        ax.scatter(q[k_col], q[tau_col] * grid.get("annualization_days", 365.25), q[iv_col], s=3, c="black", alpha=0.16)
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("days")
    ax.set_zlabel("IV")
    ax.set_title(title or "Smooth fitted IV surface")
    ax.view_init(elev=26, azim=-130)
    ax.figure.colorbar(surf, ax=ax, shrink=0.55, pad=0.04, label="IV")
    return ax


def local_vol_surface_3d(ax, lv: dict, x_col: str = "k_spot", z_col: str = "local_vol", title: str | None = None, cap_quantile: float = 0.98):
    set_plot_style()
    x_base = np.asarray(lv[x_col], dtype=float)
    y_base = np.asarray(lv["tau_days"], dtype=float)
    x, y = np.meshgrid(x_base, y_base)
    z = np.asarray(lv[z_col], dtype=float)
    cap = np.nanquantile(z[np.isfinite(z)], cap_quantile) if np.isfinite(z).any() else np.nan
    z = np.where(z <= cap, z, np.nan)
    surf = ax.plot_surface(x, y, np.ma.masked_invalid(z), cmap="magma", linewidth=0, antialiased=True, alpha=0.90)
    ax.set_xlabel("log strike / spot")
    ax.set_ylabel("days")
    ax.set_zlabel("local vol")
    ax.set_title(title or "Dupire local-volatility surface")
    ax.view_init(elev=27, azim=-130)
    ax.figure.colorbar(surf, ax=ax, shrink=0.55, pad=0.04, label="local vol")
    return ax


def smile_slices_comparison(ax, grid: dict, pchip_grid, spline_grid, maturities_days=(30, 60, 120), title: str | None = None):
    ax = _ax(ax)
    k = np.asarray(grid["k"], dtype=float)
    tau_days = np.asarray(grid["tau_days"], dtype=float)
    for i, day in enumerate(maturities_days):
        idx = int(np.nanargmin(np.abs(tau_days - day)))
        color = LAB_COLORS[i % len(LAB_COLORS)]
        ax.plot(k, spline_grid[idx], lw=2.0, color=color, label=f"spline {tau_days[idx]:.0f}d")
        ax.plot(k, pchip_grid[idx], lw=1.2, ls="--", color=color, alpha=0.8)
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "PCHIP vs spline smiles")
    ax.legend(fontsize=7)
    return ax


def residual_histogram(ax, residuals: pd.DataFrame, residual_col: str = "residual_visual", title: str | None = None):
    ax = _ax(ax)
    x = pd.to_numeric(residuals.get(residual_col), errors="coerce").dropna()
    if x.empty:
        ax.text(0.5, 0.5, "No residual data", ha="center", va="center")
        return ax
    ax.hist(x, bins=35, alpha=0.75, color=LAB_COLORS[0])
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_xlabel("IV residual")
    ax.set_ylabel("count")
    ax.set_title(title or "Residual distribution")
    return ax


def local_vol_ratio_map(ax, lv: dict, x_col: str = "k_spot", title: str | None = None, ratio_bounds=(0.50, 1.80)):
    ax = _ax(ax)
    x = np.asarray(lv[x_col], dtype=float)
    ratio = np.asarray(lv["local_vol_to_iv"], dtype=float)
    cf = ax.contourf(x, lv["tau_days"], ratio, levels=np.linspace(ratio_bounds[0], ratio_bounds[1], 15), cmap="viridis", extend="both")
    ax.contour(x, lv["tau_days"], ratio, levels=[1.0], colors="black", linewidths=0.8)
    ax.set_xlabel("log strike / spot")
    ax.set_ylabel("days")
    ax.set_title(title or "Local vol / implied vol")
    ax.figure.colorbar(cf, ax=ax, pad=0.01, label="ratio")
    return ax


def local_vol_slices(ax, lv: dict, x_col: str = "k_spot", maturities_days=(30, 60, 120), title: str | None = None):
    ax = _ax(ax)
    x = np.asarray(lv[x_col], dtype=float)
    tau_days = np.asarray(lv["tau_days"], dtype=float)
    for i, day in enumerate(maturities_days):
        idx = int(np.nanargmin(np.abs(tau_days - day)))
        color = LAB_COLORS[i % len(LAB_COLORS)]
        ax.plot(x, lv["iv"][idx], color=color, lw=1.8, label=f"IV {tau_days[idx]:.0f}d")
        ax.plot(x, lv["local_vol"][idx], color=color, lw=1.5, ls="--", label=f"LV {tau_days[idx]:.0f}d")
    ax.set_xlabel("log strike / spot")
    ax.set_ylabel("volatility")
    ax.set_title(title or "Local vol vs implied vol")
    ax.legend(fontsize=6, ncol=2)
    return ax


def pca_variance_bars(ax, pca: dict, title: str | None = None):
    ax = _ax(ax)
    var = pca.get("explained_variance_table", pd.DataFrame())
    if var.empty:
        ax.text(0.5, 0.5, pca.get("diagnostic", "No PCA"), ha="center", va="center")
        ax.set_title(title or "Surface PCA explained variance")
        return ax
    x = np.arange(len(var)) + 1
    ax.bar(x, var["explained_variance_ratio"], color=LAB_COLORS[0], alpha=0.75)
    ax.plot(x, var["cumulative"], color=LAB_COLORS[3], marker="o", lw=1.8)
    ax.set_xticks(x)
    ax.set_xlabel("component")
    ax.set_ylabel("explained variance")
    ax.set_title(title or "Surface PCA explained variance")
    return ax


def pca_shock_map(ax, pca_shocks: dict, component: int = 1, title: str | None = None):
    ax = _ax(ax)
    arr = np.asarray(pca_shocks.get(f"pc{component}", []), dtype=float)
    grid = pca_shocks.get("grid", {})
    if arr.size == 0 or "k" not in grid:
        ax.text(0.5, 0.5, "No PCA shock", ha="center", va="center")
        ax.set_title(title or f"PC{component} shock")
        return ax
    lim = _sym_lim(arr)
    cf = ax.contourf(grid["k"], grid["tau_days"], arr, levels=17, cmap="coolwarm", vmin=-lim, vmax=lim, extend="both")
    ax.contour(grid["k"], grid["tau_days"], arr, levels=[0.0], colors="black", linewidths=0.7)
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("days")
    ax.set_title(title or f"PC{component} IV shock")
    ax.figure.colorbar(cf, ax=ax, pad=0.01, label="IV shock")
    return ax


def delta_correction_slices(ax, greek_grid: pd.DataFrame, x_col: str = "k_spot", maturities_days=(30, 60, 120), title: str | None = None):
    ax = _ax(ax)
    for i, day in enumerate(maturities_days):
        idx = (greek_grid["tau_days"] - day).abs().idxmin()
        tau = greek_grid.loc[idx, "tau_days"]
        g = greek_grid[np.isclose(greek_grid["tau_days"], tau)].sort_values(x_col)
        ax.plot(g[x_col], g["delta_diff"], lw=1.8, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{tau:.0f}d")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("log strike / spot")
    ax.set_ylabel("delta diff")
    ax.set_title(title or "Delta correction slices")
    ax.legend(fontsize=7)
    return ax


def gamma_correction_slices(ax, greek_grid: pd.DataFrame, x_col: str = "k_spot", maturities_days=(30, 60, 120), title: str | None = None):
    ax = _ax(ax)
    for i, day in enumerate(maturities_days):
        idx = (greek_grid["tau_days"] - day).abs().idxmin()
        tau = greek_grid.loc[idx, "tau_days"]
        g = greek_grid[np.isclose(greek_grid["tau_days"], tau)].sort_values(x_col)
        ax.plot(g[x_col], g["gamma_diff"], lw=1.8, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{tau:.0f}d")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("log strike / spot")
    ax.set_ylabel("gamma diff")
    ax.set_title(title or "Gamma correction slices")
    ax.legend(fontsize=7)
    return ax


def _quiet_axis(ax, message: str, title: str | None = None):
    ax = _ax(ax)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def _days_from_quotes(frame: pd.DataFrame, tau_col: str = "tau") -> pd.Series:
    if "dte_days" in frame.columns:
        return pd.to_numeric(frame["dte_days"], errors="coerce")
    return pd.to_numeric(frame.get(tau_col), errors="coerce") * 365.25


def _smile_expiries(frame: pd.DataFrame, max_slices: int = 4):
    if frame.empty or "expiry" not in frame.columns:
        return []
    counts = frame.groupby("expiry").size().sort_values(ascending=False)
    return list(pd.to_datetime(counts.head(max_slices).index))


def calibration_quote_map(ax, quotes: pd.DataFrame, k_col: str = "k", tau_col: str = "tau", title: str | None = None):
    ax = _ax(ax)
    q = quotes.dropna(subset=[k_col, tau_col]).copy()
    if q.empty:
        return _quiet_axis(ax, "No calibration quotes", title or "Calibration quote map")
    dte = _days_from_quotes(q, tau_col)
    value = pd.to_numeric(q.get("iv_mid", q.get("obs_weight", 1.0)), errors="coerce")
    sc = ax.scatter(q[k_col], dte, c=value, s=14, alpha=0.75, cmap="viridis")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.7)
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("days to expiry")
    ax.set_title(title or "Calibration quote map")
    ax.figure.colorbar(sc, ax=ax, pad=0.01, label="IV" if "iv_mid" in q.columns else "weight")
    return ax


def smile_term_structure(ax, quotes: pd.DataFrame, k_col: str = "k", tau_col: str = "tau", iv_col: str = "iv_mid", title: str | None = None):
    ax = _ax(ax)
    q = quotes.dropna(subset=[k_col, iv_col]).copy()
    if q.empty or "expiry" not in q.columns:
        return _quiet_axis(ax, "No smile data", title or "IV smile term structure")
    for i, expiry in enumerate(_smile_expiries(q, 5)):
        g = q[pd.to_datetime(q["expiry"]).eq(expiry)].sort_values(k_col)
        if len(g) < 4:
            continue
        dte = float(np.nanmedian(_days_from_quotes(g, tau_col)))
        ax.plot(g[k_col], g[iv_col], marker="o", ms=3, lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{dte:.0f}d")
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "IV smile term structure")
    ax.legend(fontsize=7)
    return ax


def model_quote_overlay(ax, quotes: pd.DataFrame, model_quotes: pd.DataFrame, k_col: str = "k", tau_col: str = "tau", title: str | None = None):
    ax = _ax(ax)
    base = quotes.dropna(subset=[k_col, tau_col]).copy()
    chosen = model_quotes.dropna(subset=[k_col, tau_col]).copy()
    if base.empty and chosen.empty:
        return _quiet_axis(ax, "No quote data", title or "Balanced model quotes")
    pieces = []
    for label, frame in [("surface", base), ("model panel", chosen)]:
        if frame.empty or "expiry" not in frame.columns:
            continue
        g = frame.copy()
        g["dte_plot"] = _days_from_quotes(g, tau_col)
        one = g.groupby("expiry").agg(
            dte=("dte_plot", "median"),
            quotes=(k_col, "size"),
            k_min=(k_col, "min"),
            k_max=(k_col, "max"),
        ).reset_index()
        one["panel"] = label
        pieces.append(one)
    if not pieces:
        return _quiet_axis(ax, "No expiry coverage", title or "Balanced model quotes")
    cover = pd.concat(pieces, ignore_index=True)
    expiries = cover.groupby("expiry")["dte"].median().sort_values().index
    x = np.arange(len(expiries), dtype=float)
    width = 0.36
    surface = cover[cover["panel"].eq("surface")].set_index("expiry").reindex(expiries)
    model = cover[cover["panel"].eq("model panel")].set_index("expiry").reindex(expiries)
    ax.bar(x - width / 2, surface["quotes"].fillna(0), width=width, alpha=0.35, label="surface quotes")
    ax.bar(x + width / 2, model["quotes"].fillna(0), width=width, alpha=0.80, label="model quotes")
    for i, expiry in enumerate(expiries):
        row = model.loc[expiry] if expiry in model.index else None
        if row is not None and np.isfinite(row.get("k_min", np.nan)) and np.isfinite(row.get("k_max", np.nan)):
            ax.text(i + width / 2, float(row["quotes"]) + 0.8, f"{row['k_min']:.2f}..{row['k_max']:.2f}", ha="center", va="bottom", rotation=90, fontsize=6)
    labels = [f"{float(cover.loc[cover['expiry'].eq(e), 'dte'].median()):.0f}d" for e in expiries]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("expiry bucket")
    ax.set_ylabel("quote count")
    ax.set_title(title or "Balanced panel coverage by expiry")
    ax.legend(fontsize=7)
    return ax


def svi_smiles(ax, quotes: pd.DataFrame, fit: dict, title: str | None = None):
    ax = _ax(ax)
    f = fit.get("fit", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
    if f.empty:
        return _quiet_axis(ax, "No SVI fit", title or "SVI fitted smiles")
    for i, expiry in enumerate(_smile_expiries(f, 4)):
        g = f[pd.to_datetime(f["expiry"]).eq(expiry)].sort_values("k")
        if len(g) < 4:
            continue
        dte = float(np.nanmedian(_days_from_quotes(g)))
        color = LAB_COLORS[i % len(LAB_COLORS)]
        ax.scatter(g["k"], g["iv_mid"], s=12, alpha=0.50, color=color)
        ax.plot(g["k"], g["model_iv"], lw=1.8, color=color, label=f"{dte:.0f}d")
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "SVI fitted smiles")
    ax.legend(fontsize=7)
    return ax


def svi_ssvi_errors(ax, svi_fit: dict | pd.DataFrame, ssvi_fit: dict | None = None, title: str | None = None):
    ax = _ax(ax)
    if not isinstance(svi_fit, dict) or not isinstance(ssvi_fit, dict):
        return _quiet_axis(ax, "Pass SVI and SSVI fit objects", title or "SVI vs SSVI errors")
    pieces = []
    for name, fit in [("SVI", svi_fit), ("SSVI", ssvi_fit)]:
        f = fit.get("fit", pd.DataFrame()).copy()
        if f.empty or "expiry" not in f.columns or "iv_residual" not in f.columns:
            continue
        f["dte_plot"] = _days_from_quotes(f)
        one = f.groupby("expiry").agg(
            dte=("dte_plot", "median"),
            iv_rmse=("iv_residual", lambda x: float(np.sqrt(np.nanmean(np.asarray(x, dtype=float) ** 2)))),
            quotes=("iv_residual", "size"),
        ).reset_index()
        one["model"] = name
        pieces.append(one)
    if not pieces:
        return _quiet_axis(ax, "No expiry error data", title or "SVI vs SSVI errors")
    err = pd.concat(pieces, ignore_index=True)
    expiries = err.groupby("expiry")["dte"].median().sort_values().index
    x = np.arange(len(expiries), dtype=float)
    width = 0.38
    for offset, name in [(-width / 2, "SVI"), (width / 2, "SSVI")]:
        y = err[err["model"].eq(name)].set_index("expiry").reindex(expiries)["iv_rmse"]
        ax.bar(x + offset, y, width=width, label=name)
    labels = [f"{float(err.loc[err['expiry'].eq(e), 'dte'].median()):.0f}d" for e in expiries]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("expiry")
    ax.set_ylabel("IV RMSE")
    ax.set_title(title or "SVI vs SSVI error by expiry")
    ax.legend(fontsize=7)
    return ax


def ssvi_residuals(ax, quotes: pd.DataFrame, fit: dict, title: str | None = None):
    ax = _ax(ax)
    f = fit.get("fit", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
    if f.empty or "iv_residual" not in f.columns:
        return _quiet_axis(ax, "No SSVI residuals", title or "SSVI residual map")
    q = f.dropna(subset=["k", "iv_residual"]).copy()
    if q.empty:
        return _quiet_axis(ax, "No SSVI residuals", title or "SSVI residual map")
    q["dte_plot"] = _days_from_quotes(q)
    q["expiry_label"] = q.groupby("expiry")["dte_plot"].transform("median").round().astype(int).astype(str) + "d"
    bins = np.array([-np.inf, -0.25, -0.17, -0.10, -0.04, 0.04, 0.10, 0.17, 0.25, np.inf])
    labels = ["<-25", "-25:-17", "-17:-10", "-10:-04", "-04:04", "04:10", "10:17", "17:25", ">25"]
    q["k_bucket"] = pd.cut(q["k"], bins=bins, labels=labels)
    order = q.groupby("expiry")["dte_plot"].median().sort_values().index
    table = q.pivot_table(index="expiry", columns="k_bucket", values="iv_residual", aggfunc="median", observed=False).reindex(order)
    if table.empty:
        return _quiet_axis(ax, "No residual buckets", title or "SSVI residual map")
    lim = _sym_lim(table.to_numpy(dtype=float))
    im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns.astype(str), rotation=35, ha="right")
    y_labels = [f"{float(q.loc[q['expiry'].eq(e), 'dte_plot'].median()):.0f}d" for e in table.index]
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("log-moneyness bucket")
    ax.set_ylabel("expiry")
    ax.set_title(title or "SSVI median IV residual by bucket")
    ax.figure.colorbar(im, ax=ax, pad=0.01, label="model IV - market IV")
    return ax


def sabr_smiles(ax, quotes: pd.DataFrame, fit: dict, title: str | None = None):
    ax = _ax(ax)
    f = fit.get("fit", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
    if f.empty:
        return _quiet_axis(ax, "No SABR fit", title or "SABR fitted smiles")
    for i, expiry in enumerate(_smile_expiries(f, 4)):
        g = f[pd.to_datetime(f["expiry"]).eq(expiry)].sort_values("k")
        if len(g) < 4:
            continue
        dte = float(np.nanmedian(_days_from_quotes(g)))
        color = LAB_COLORS[i % len(LAB_COLORS)]
        ax.scatter(g["k"], g["iv_mid"], s=12, alpha=0.50, color=color)
        ax.plot(g["k"], g["model_iv"], lw=1.8, color=color, label=f"{dte:.0f}d")
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "SABR fitted smiles")
    ax.legend(fontsize=7)
    return ax


def sabr_terms(ax, fit: dict, title: str | None = None):
    ax = _ax(ax)
    p = fit.get("params", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
    if p.empty:
        return _quiet_axis(ax, "No SABR parameters", title or "SABR parameters")
    x = pd.to_numeric(p.get("dte_days", pd.Series(np.arange(len(p)), index=p.index)), errors="coerce")
    order = np.argsort(x.to_numpy(dtype=float))
    x = x.iloc[order]
    p = p.iloc[order]
    ax.plot(x, p["alpha"], marker="o", lw=1.5, label="alpha")
    ax.plot(x, p["nu"], marker="o", lw=1.5, label="nu")
    ax.plot(x, p["rho"], marker="o", lw=1.2, label="rho")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.6)
    ax.set_xlabel("days to expiry")
    ax.set_ylabel("parameter value")
    ax.set_title(title or "SABR parameter term structure")
    ax.legend(fontsize=7)
    return ax


def merton_tail_fit(ax, quotes: pd.DataFrame, fit: dict, title: str | None = None):
    ax = _ax(ax)
    f = fit.get("fit", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
    if f.empty:
        return _quiet_axis(ax, "No Merton fit", title or "Merton tail fit")
    x = pd.to_numeric(f.get("k"), errors="coerce")
    y = pd.to_numeric(f.get("price_residual"), errors="coerce")
    tail_count = int((x.abs() >= 0.14).sum())
    ax.scatter(x, y, s=16, alpha=0.65)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(-0.14, color="black", lw=0.7, ls="--", alpha=0.55)
    ax.axvline(0.14, color="black", lw=0.7, ls="--", alpha=0.55)
    if tail_count < 8:
        ax.text(0.03, 0.95, f"tail diagnostic weak: {tail_count} quotes", transform=ax.transAxes, va="top", fontsize=8)
    else:
        ax.text(0.03, 0.95, f"tail quotes: {tail_count}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("model price - mid")
    ax.set_title(title or "Merton jump-tail fit")
    return ax


def heston_mc_check(ax, fit: dict, title: str | None = None):
    ax = _ax(ax)
    conv = fit.get("mc_convergence", pd.DataFrame()) if isinstance(fit, dict) else pd.DataFrame()
    if conv.empty:
        return _quiet_axis(ax, "No MC convergence data", title or "Heston MC convergence")
    ax.plot(conv["paths"], conv["price"], marker="o", lw=1.5, label="price")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("paths")
    ax.set_ylabel("model price")
    ax2 = ax.twinx()
    ax2.plot(conv["paths"], conv["standard_error"], marker="s", lw=1.2, color=LAB_COLORS[3], label="standard error")
    ax2.set_ylabel("standard error")
    ax.set_title(title or "Heston MC convergence")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7)
    return ax


def heston_bates_fit(ax, quotes: pd.DataFrame, heston_fit: dict, bates_fit: dict, title: str | None = None):
    ax = _ax(ax)
    h = heston_fit.get("fit", pd.DataFrame()) if isinstance(heston_fit, dict) else pd.DataFrame()
    b = bates_fit.get("fit", pd.DataFrame()) if isinstance(bates_fit, dict) else pd.DataFrame()
    if h.empty and b.empty:
        return _quiet_axis(ax, "No simulation fits", title or "Heston vs Bates")
    if not h.empty:
        ax.scatter(h["k"], h["price_residual"], s=16, alpha=0.55, label="heston")
    if not b.empty:
        ax.scatter(b["k"], b["price_residual"], s=16, alpha=0.55, label="bates")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("log strike / forward")
    ax.set_ylabel("model price - mid")
    ax.set_title(title or "Heston vs Bates fit")
    ax.legend(fontsize=7)
    return ax


def model_speed_accuracy(ax, comparison: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if comparison.empty or "model" not in comparison.columns:
        return _quiet_axis(ax, "No comparison data", title or "Model speed and accuracy")
    c = comparison.copy()
    x_col = "runtime_sec" if "runtime_sec" in c.columns else "runtime" if "runtime" in c.columns else "elapsed_sec"
    y_col = "weighted_iv_rmse" if "weighted_iv_rmse" in c.columns else "weighted_price_rmse"
    c[x_col] = pd.to_numeric(c.get(x_col, 0.0), errors="coerce").fillna(0.0)
    c[y_col] = pd.to_numeric(c.get(y_col), errors="coerce")
    c = c.dropna(subset=[y_col])
    if c.empty:
        return _quiet_axis(ax, "No runtime/error data", title or "Model speed and accuracy")
    ax.scatter(c[x_col].clip(lower=1e-6), c[y_col], s=45)
    for _, r in c.iterrows():
        ax.annotate(str(r["model"]), (max(float(r[x_col]), 1e-6), float(r[y_col])), xytext=(4, 2), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("runtime seconds")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(title or "Model error vs runtime")
    return ax


def benchmark_errors(ax, comparison: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if comparison.empty or "model" not in comparison.columns:
        return _quiet_axis(ax, "No benchmark data", title or "Benchmark errors")
    c = comparison.copy()
    value_col = "weighted_iv_rmse" if "weighted_iv_rmse" in c.columns else "weighted_price_rmse"
    c[value_col] = pd.to_numeric(c.get(value_col), errors="coerce")
    c = c.dropna(subset=[value_col]).sort_values(value_col)
    if c.empty:
        return _quiet_axis(ax, "No benchmark errors", title or "Benchmark errors")
    ax.bar(c["model"], c[value_col], color=[LAB_COLORS[i % len(LAB_COLORS)] for i in range(len(c))])
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title or "Common benchmark errors")
    ax.tick_params(axis="x", labelrotation=25)
    return ax


def model_disagreement(ax, fair_values: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if fair_values.empty or "model_disagreement" not in fair_values.columns:
        return _quiet_axis(ax, "No disagreement data", title or "Model disagreement")
    q = fair_values.dropna(subset=["date", "model_disagreement"]).copy()
    if q.empty:
        return _quiet_axis(ax, "No disagreement data", title or "Model disagreement")
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    daily = q.groupby("date").agg(
        median=("model_disagreement", "median"),
        p75=("model_disagreement", lambda x: float(np.nanquantile(x, 0.75))),
        p90=("model_disagreement", lambda x: float(np.nanquantile(x, 0.90))),
        candidates=("watchlist_candidate", lambda x: int(np.nansum(x))) if "watchlist_candidate" in q.columns else ("model_disagreement", "size"),
    ).reset_index().dropna(subset=["date"])
    if daily.empty:
        return _quiet_axis(ax, "No daily disagreement", title or "Model disagreement")
    x = daily["date"]
    ax.plot(x, daily["median"], lw=1.5, label="median")
    ax.plot(x, daily["p90"], lw=1.1, label="p90")
    ax.fill_between(x, daily["p75"], daily["p90"], alpha=0.18, label="p75-p90")
    ax.set_xlabel("date")
    ax.set_ylabel("price disagreement")
    ax.set_title(title or "Daily model disagreement")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=7)
    return ax


def residual_deciles(ax, validation: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    summary = validation.attrs.get("summary", pd.DataFrame()) if isinstance(validation, pd.DataFrame) else pd.DataFrame()
    if summary.empty and isinstance(validation, pd.DataFrame) and {"z_residual", "next_hedged_pnl"}.issubset(validation.columns):
        v = validation.dropna(subset=["z_residual", "next_hedged_pnl"]).copy()
        if not v.empty:
            v["decile"] = pd.qcut(v["z_residual"].rank(method="first"), 10, labels=False, duplicates="drop")
            summary = v.groupby("decile").agg(mean_next_hedged_pnl=("next_hedged_pnl", "mean"), count=("next_hedged_pnl", "size")).reset_index()
    if summary.empty:
        return _quiet_axis(ax, "No residual validation", title or "Residual deciles")
    ax.bar(summary["decile"].astype(int), summary["mean_next_hedged_pnl"], alpha=0.75)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("residual decile")
    ax.set_ylabel("next hedged P&L")
    ax.set_title(title or "Residual deciles vs next hedged P&L")
    return ax


def scheduled_hedge_equity(ax, results: dict, comparison: pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    drawn = False
    for name, result in (results or {}).items():
        nav = result.get("nav", pd.DataFrame()) if isinstance(result, dict) else pd.DataFrame()
        if nav.empty:
            continue
        if "delta" in nav.columns:
            series = nav["delta"]
        else:
            series = nav.iloc[:, 0]
        ax.plot(series.index, series.values, lw=1.6, label=str(name))
        drawn = True
    if not drawn:
        return _quiet_axis(ax, "No scheduled hedge results", title or "Scheduled hedge equity")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("date")
    ax.set_ylabel("cumulative P&L")
    ax.set_title(title or "Scheduled hedge comparison")
    _format_hedging_axis(ax, len(results or {}))
    return ax


__all__ = [
    "benchmark_errors",
    "calibration_quote_map",
    "delta_correction_slices",
    "gamma_correction_slices",
    "heston_bates_fit",
    "heston_mc_check",
    "local_vol_ratio_map",
    "local_vol_slices",
    "local_vol_surface_3d",
    "merton_tail_fit",
    "model_disagreement",
    "model_quote_overlay",
    "model_speed_accuracy",
    "pca_shock_map",
    "pca_variance_bars",
    "plot_clean_vs_dirty_spread",
    "plot_forward_vs_spot",
    "plot_greek_bands",
    "plot_greek_error_summary",
    "plot_greek_uncertainty_bands",
    "plot_hedge_exposures",
    "plot_hedge_trades",
    "plot_hedging_cumulative_pnl",
    "plot_hedging_drawdown",
    "plot_hedging_nav",
    "plot_hedging_net_equity",
    "plot_hedging_pnl_components",
    "plot_hedging_rolling_turnover",
    "plot_hedging_rolling_volatility",
    "plot_iv_bid_ask_band",
    "plot_iv_failure_rate_by_log_moneyness",
    "plot_iv_iterations_by_log_moneyness",
    "plot_iv_smile",
    "plot_iv_term_structure",
    "plot_market_mid_vs_realized_vol_forward_bsm",
    "plot_moneyness_dte_coverage",
    "plot_numpy_jax_greek_comparison",
    "plot_parity_error_by_moneyness",
    "plot_pricing_error_hist",
    "plot_quote_filter_waterfall",
    "plot_realized_vs_implied_vol",
    "plot_rolling_residual_delta",
    "plot_rolling_residual_vega",
    "plot_single_day_forward_iv_skew",
    "plot_single_day_parity_forward_extraction",
    "plot_solver_runtime",
    "plot_solver_success",
    "quote_coverage_map",
    "residual_deciles",
    "residual_histogram",
    "sabr_smiles",
    "sabr_terms",
    "scheduled_hedge_equity",
    "smile_slices_comparison",
    "smooth_surface_3d",
    "smile_term_structure",
    "ssvi_residuals",
    "svi_smiles",
    "svi_ssvi_errors",
]
