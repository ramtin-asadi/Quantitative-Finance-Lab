from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .curves import LAB_COLORS, set_plot_style


def _ax(ax=None):
    if ax is None:
        _, ax = plt.subplots()
    set_plot_style()
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


__all__ = [
    "delta_correction_slices",
    "gamma_correction_slices",
    "local_vol_ratio_map",
    "local_vol_slices",
    "local_vol_surface_3d",
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
    "residual_histogram",
    "smile_slices_comparison",
    "smooth_surface_3d",
]
