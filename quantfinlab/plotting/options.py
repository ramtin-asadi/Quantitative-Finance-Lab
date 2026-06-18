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


def _small_legend(ax, **kwargs):
    defaults = {
        "fontsize": 6,
        "frameon": True,
        "framealpha": 0.88,
        "borderpad": 0.25,
        "labelspacing": 0.25,
        "handlelength": 1.2,
        "handletextpad": 0.35,
        "borderaxespad": 0.25,
    }
    defaults.update(kwargs)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        return ax.legend(handles, labels, **defaults)
    return None


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
        _small_legend(ax, loc="best", ncol=min(3, n_series))


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
        _small_legend(ax)
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
    _small_legend(ax, loc="best")
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
        _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax, loc="best")
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
    _small_legend(ax, loc="best")
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax, loc="best")
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
    _small_legend(ax, loc="best")
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
    _small_legend(ax, ncol=min(3, len(roll_turn.columns)))
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
    _small_legend(ax, ncol=min(3, len(cumulative.columns)))
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
    _small_legend(ax, loc="best")
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
    _small_legend(ax)
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
    _small_legend(ax, ncol=2)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    _small_legend(ax)
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
    ax.legend(lines, [line.get_label() for line in lines], fontsize=6, frameon=True, framealpha=0.88, borderpad=0.25, labelspacing=0.25, handlelength=1.2, handletextpad=0.35)
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
    _small_legend(ax)
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
    _small_legend(ax)
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


def quote_coverage(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    q = quotes.copy()
    if q.empty:
        return _quiet_axis(ax, "No quotes", title or "Quote coverage")
    x = pd.to_numeric(q.get("moneyness", q.get("k_over_s", q.get("strike") / q.get("spot"))), errors="coerce")
    y = pd.to_numeric(q.get("dte_days", q.get("dte", q.get("tau") * 365.25)), errors="coerce")
    ax.scatter(x, y, s=6, alpha=0.25)
    ax.set_xlabel("K / S")
    ax.set_ylabel("DTE")
    ax.set_title(title or "Quote coverage")
    return ax


def dividend_timeline(ax, prices: pd.DataFrame | pd.Series, dividends: pd.DataFrame | pd.Series | None = None, title: str | None = None):
    ax = _ax(ax)
    if isinstance(prices, pd.DataFrame):
        close = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
    else:
        close = pd.Series(prices)
    close.index = pd.to_datetime(close.index, errors="coerce")
    ax.plot(close.index, close.values, lw=1.2, label="close")
    div = dividends
    if div is None and isinstance(prices, pd.DataFrame) and "dividend" in prices.columns:
        div = prices["dividend"]
    if div is not None:
        div = pd.Series(div)
        div.index = pd.to_datetime(div.index, errors="coerce")
        div = div[pd.to_numeric(div, errors="coerce") > 0]
        if not div.empty:
            y = close.reindex(div.index, method="nearest")
            ax.scatter(div.index, y, marker="v", s=35, label="dividend")
    ax.set_title(title or "Underlying and dividends")
    _small_legend(ax)
    return ax


def iv_moneyness(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    q = quotes.dropna(subset=[c for c in ["iv_mid"] if c in quotes.columns]).copy()
    if q.empty:
        return _quiet_axis(ax, "No IV", title or "IV by moneyness")
    x = pd.to_numeric(q.get("moneyness", q.get("k_over_s", q.get("strike") / q.get("spot"))), errors="coerce")
    ax.scatter(x, q["iv_mid"], s=10, alpha=0.45)
    ax.set_xlabel("K / S")
    ax.set_ylabel("IV")
    ax.set_title(title or "IV by moneyness")
    return ax


def sigma_surface(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "sigma_used" not in quotes:
        return _quiet_axis(ax, "No sigma data", title or "Sigma surface")
    q = quotes.copy()
    x = pd.to_numeric(q.get("moneyness", q.get("k_over_s", q.get("strike") / q.get("spot"))), errors="coerce")
    y = pd.to_numeric(q.get("dte_days", q.get("tau") * 365.25), errors="coerce")
    sc = ax.scatter(x, y, c=pd.to_numeric(q["sigma_used"], errors="coerce"), s=6, alpha=0.5, cmap="viridis")
    ax.set_xlabel("K / S")
    ax.set_ylabel("DTE")
    ax.set_title(title or "Sigma used across chain")
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    return ax


def tree_convergence(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No convergence", title or "Tree convergence")
    ax.plot(table["steps"], table["price"], marker="o", lw=1.4)
    if "reference_error" in table:
        ax2 = ax.twinx()
        ax2.plot(table["steps"], table["reference_error"], marker="s", lw=1.0, color=LAB_COLORS[3], label="abs error")
        ax2.set_ylabel("abs error")
    ax.set_xlabel("steps")
    ax.set_ylabel("price")
    ax.set_title(title or "Tree convergence")
    return ax


def tree_boundary(ax, boundary: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if boundary.empty:
        return _quiet_axis(ax, "No boundary", title or "Tree boundary")
    data = boundary.copy()
    x = data["time_to_expiry"] if "time_to_expiry" in data else data["time"] if "time" in data else data.index
    y = data["boundary_over_k"] if "boundary_over_k" in data else data["boundary"] / data["strike"] if {"boundary", "strike"}.issubset(data.columns) else data["boundary"]
    ax.plot(x, y, lw=1.6)
    ax.set_xlabel("time to expiry")
    ax.set_ylabel("boundary S/K")
    ax.set_title(title or "Exercise boundary")
    return ax


def tree_exercise_map(ax, exercise: pd.DataFrame | np.ndarray, title: str | None = None):
    ax = _ax(ax)
    if isinstance(exercise, pd.DataFrame) and {"time_to_expiry", "s_over_k", "exercise"}.issubset(exercise.columns):
        q = exercise.copy()
        q["time_bin"] = pd.cut(pd.to_numeric(q["time_to_expiry"], errors="coerce"), bins=np.linspace(q["time_to_expiry"].min(), q["time_to_expiry"].max(), 41), include_lowest=True)
        q["sk_bin"] = pd.cut(pd.to_numeric(q["s_over_k"], errors="coerce"), bins=np.linspace(q["s_over_k"].min(), q["s_over_k"].max(), 41), include_lowest=True)
        table = q.pivot_table(index="sk_bin", columns="time_bin", values="exercise", aggfunc="max", observed=False)
        im = ax.imshow(table.to_numpy(float), aspect="auto", origin="lower", cmap="magma", extent=[float(q["time_to_expiry"].min()), float(q["time_to_expiry"].max()), float(q["s_over_k"].min()), float(q["s_over_k"].max())])
        ax.set_xlabel("time to expiry")
        ax.set_ylabel("S / K")
        ax.set_title(title or "Tree exercise region")
        ax.figure.colorbar(im, ax=ax, pad=0.01)
        return ax
    values = exercise.to_numpy() if isinstance(exercise, pd.DataFrame) else np.asarray(exercise)
    if values.size == 0:
        return _quiet_axis(ax, "No exercise map", title or "Tree exercise map")
    im = ax.imshow(values.astype(float), aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("time step")
    ax.set_ylabel("S / K grid")
    ax.set_title(title or "Tree exercise region")
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    return ax


def american_premium_heatmap(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "american_premium" not in quotes:
        return _quiet_axis(ax, "No premium data", title or "American premium")
    q = quotes.copy()
    if "option_type" in q.columns and q["option_type"].nunique() > 1:
        q = q[q["option_type"].astype(str).str.lower().str.startswith("p")].copy()
        label = "puts"
    else:
        label = str(q["option_type"].iloc[0]) if "option_type" in q and len(q) else ""
    q["dte_bucket"] = pd.cut(pd.to_numeric(q["dte_days"], errors="coerce"), bins=[0, 14, 30, 60, 90, 120, 180])
    q["m_bucket"] = pd.cut(pd.to_numeric(q["moneyness"], errors="coerce"), bins=[0.65, 0.8, 0.9, 0.97, 1.03, 1.1, 1.25, 1.45])
    table = q.pivot_table(index="dte_bucket", columns="m_bucket", values="american_premium", aggfunc="median", observed=False)
    im = ax.imshow(table.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=7)
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    ax.set_xlabel("K / S bucket")
    ax.set_ylabel("DTE bucket")
    ax.set_title(title or f"American premium heatmap {label}".strip())
    return ax


def pricing_error_distribution(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    col = "pricing_error" if "pricing_error" in quotes.columns else "price_error"
    if quotes.empty or col not in quotes:
        return _quiet_axis(ax, "No pricing errors", title or "Pricing error distribution")
    x = pd.to_numeric(quotes[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(x, bins=80, alpha=0.8)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("model price - mid")
    ax.set_ylabel("contracts")
    ax.set_title(title or "Pricing error distribution")
    return ax


def premium_term_curves(ax, quotes: pd.DataFrame, option_type: str | None = None, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "american_premium" not in quotes:
        return _quiet_axis(ax, "No premium data", title or "Premium Term Curves")
    q = quotes.copy()
    if option_type is not None and "option_type" in q:
        q = q[q["option_type"].astype(str).str.lower().str.startswith(str(option_type).lower()[0])].copy()
    if q.empty:
        return _quiet_axis(ax, "No matching options", title or "Premium Term Curves")
    dte = pd.to_numeric(q.get("dte_days", q.get("tau", np.nan) * 365.25), errors="coerce")
    spot = pd.to_numeric(q.get("spot", 1.0), errors="coerce")
    m = pd.to_numeric(q.get("moneyness", pd.to_numeric(q.get("strike", np.nan), errors="coerce") / spot), errors="coerce")
    q["dte_bucket"] = pd.cut(dte, [0, 14, 30, 60, 90, 120, 180, 365], include_lowest=True)
    q["m_bucket"] = pd.cut(m, [0.65, 0.90, 0.97, 1.03, 1.10, 1.25, 1.50], include_lowest=True)
    q["premium_bps"] = pd.to_numeric(q["american_premium"], errors="coerce") / spot * 10000.0
    rows = []
    for (dte_bucket, m_bucket), part in q.groupby(["dte_bucket", "m_bucket"], observed=True):
        x = 0.5 * (float(dte_bucket.left) + float(dte_bucket.right))
        values = part["premium_bps"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values):
            rows.append({"dte_mid": x, "m_bucket": str(m_bucket), "median": values.median(), "q25": values.quantile(0.25), "q75": values.quantile(0.75), "count": len(values)})
    data = pd.DataFrame(rows)
    if data.empty:
        return _quiet_axis(ax, "No term data", title or "Premium Term Curves")
    for label, part in data.groupby("m_bucket", sort=False):
        part = part.sort_values("dte_mid")
        ax.plot(part["dte_mid"], part["median"], marker="o", lw=1.4, label=label)
        ax.fill_between(part["dte_mid"].to_numpy(float), part["q25"].to_numpy(float), part["q75"].to_numpy(float), alpha=0.14)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("DTE bucket midpoint")
    ax.set_ylabel("median American premium, bps of spot")
    ax.set_title(title or "Premium Term Curves")
    _small_legend(ax, ncol=2)
    return ax


def premium_concentration(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "american_premium" not in quotes:
        return _quiet_axis(ax, "No premium data", title or "Premium Concentration")
    q = quotes.copy()
    if "option_type" not in q:
        q["option_type"] = "option"
    drew = False
    for opt, part in q.groupby(q["option_type"].astype(str).str.lower().str[0]):
        values = pd.to_numeric(part["american_premium"], errors="coerce").clip(lower=0.0).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        if len(values) == 0 or float(values.sum()) <= 0.0:
            continue
        x = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
        y = values.cumsum().to_numpy(float) / float(values.sum())
        ax.plot(x * 100.0, y * 100.0, lw=1.7, label="calls" if opt.startswith("c") else "puts" if opt.startswith("p") else str(opt))
        drew = True
    if not drew:
        return _quiet_axis(ax, "No positive premium", title or "Premium Concentration")
    ax.plot([0, 100], [0, 100], color="black", lw=0.8, ls=":")
    ax.set_xlabel("contracts ranked by premium, cumulative %")
    ax.set_ylabel("total positive American premium, cumulative %")
    ax.set_title(title or "Premium Concentration")
    _small_legend(ax)
    return ax


def pricing_error_spread(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    col = "pricing_error" if "pricing_error" in quotes.columns else "price_error"
    if quotes.empty or col not in quotes:
        return _quiet_axis(ax, "No pricing errors", title or "Spread-Scaled Error")
    q = quotes.copy()
    half = 0.5 * (pd.to_numeric(q.get("ask"), errors="coerce") - pd.to_numeric(q.get("bid"), errors="coerce"))
    if half.isna().all() or (half <= 0).all():
        half = pd.to_numeric(q.get("mid", 1.0), errors="coerce") * pd.to_numeric(q.get("relative_spread", q.get("rel_spread", np.nan)), errors="coerce") * 0.5
    ratio = pd.to_numeric(q[col], errors="coerce") / half.replace(0.0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna().clip(-20.0, 20.0)
    if ratio.empty:
        return _quiet_axis(ax, "No scaled errors", title or "Spread-Scaled Error")
    ax.hist(ratio, bins=80, color=LAB_COLORS[0], alpha=0.82)
    for x, _label in [(-1.0, "bid"), (0.0, "mid"), (1.0, "ask")]:
        ax.axvline(x, color="black" if x == 0.0 else "#6b7280", lw=0.9, ls="-" if x == 0.0 else ":")
    ax.set_xlabel("(model price - mid) / half-spread")
    ax.set_ylabel("contracts")
    ax.set_title(title or "Spread-Scaled Error")
    return ax


def bid_ask_hit_rate(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    price_col = "american_tree_price" if "american_tree_price" in quotes.columns else "model_price" if "model_price" in quotes.columns else None
    if quotes.empty or price_col is None or not {"bid", "ask"}.issubset(quotes.columns):
        return _quiet_axis(ax, "No bid/ask data", title or "Bid/Ask Fit")
    q = quotes.copy()
    dte = pd.to_numeric(q.get("dte_days", q.get("tau", np.nan) * 365.25), errors="coerce")
    q["dte_bucket"] = pd.cut(dte, [0, 14, 30, 60, 90, 120, 180, 365], include_lowest=True)
    q["inside"] = (pd.to_numeric(q[price_col], errors="coerce") >= pd.to_numeric(q["bid"], errors="coerce")) & (pd.to_numeric(q[price_col], errors="coerce") <= pd.to_numeric(q["ask"], errors="coerce"))
    option_side = q["option_type"].astype(str).str.lower().str.startswith("c") if "option_type" in q else pd.Series(False, index=q.index)
    q["option_side"] = np.where(option_side, "calls", "puts")
    table = q.groupby(["dte_bucket", "option_side"], observed=True)["inside"].mean().reset_index()
    if table.empty:
        return _quiet_axis(ax, "No hit-rate data", title or "Bid/Ask Fit")
    mids = {b: 0.5 * (float(b.left) + float(b.right)) for b in table["dte_bucket"].unique()}
    for side, part in table.groupby("option_side"):
        part = part.assign(dte_mid=part["dte_bucket"].map(mids)).sort_values("dte_mid")
        ax.plot(part["dte_mid"], part["inside"] * 100.0, marker="o", lw=1.4, label=side)
    ax.set_ylim(0, 100)
    ax.set_xlabel("DTE bucket midpoint")
    ax.set_ylabel("model inside bid/ask, %")
    ax.set_title(title or "Bid/Ask Fit")
    _small_legend(ax)
    return ax


def boundary_compare(ax, pde_result: dict | pd.DataFrame, tree_result: pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    drew = False
    if isinstance(pde_result, dict):
        boundary = np.asarray(pde_result.get("boundary", []), dtype=float)
        strike = float(pde_result.get("strike", np.nan))
        tau = float(pde_result.get("tau", 1.0))
        if boundary.size and np.isfinite(strike) and strike > 0:
            x = np.linspace(tau, 0.0, len(boundary))
            y = boundary / strike
            mask = np.isfinite(y) & (y > 0)
            if mask.any():
                ax.plot(x[mask], y[mask], lw=1.7, label="PDE")
                drew = True
    elif isinstance(pde_result, pd.DataFrame) and not pde_result.empty:
        x = pde_result["time_to_expiry"] if "time_to_expiry" in pde_result else pde_result.index
        y = pde_result["boundary_over_k"] if "boundary_over_k" in pde_result else pde_result.get("boundary")
        ax.plot(x, y, lw=1.7, label="PDE")
        drew = True
    if tree_result is not None and not tree_result.empty:
        x = tree_result["time_to_expiry"] if "time_to_expiry" in tree_result else tree_result.get("time", tree_result.index)
        y = tree_result["boundary_over_k"] if "boundary_over_k" in tree_result else tree_result.get("boundary")
        ax.plot(x, y, lw=1.2, ls="--", label="tree")
        drew = True
    if not drew:
        return _quiet_axis(ax, "No boundary", title or "Free Boundary")
    ax.set_xlabel("time to expiry")
    ax.set_ylabel("exercise boundary S / K")
    ax.set_title(title or "Free Boundary")
    _small_legend(ax)
    return ax


def pde_tree_gap_curves(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No PDE/tree table", title or "PDE-Tree Gap")
    q = table.copy()
    if "pde_tree_disagreement" not in q:
        if {"pde_price", "american_tree_price"}.issubset(q.columns):
            q["pde_tree_disagreement"] = pd.to_numeric(q["pde_price"], errors="coerce") - pd.to_numeric(q["american_tree_price"], errors="coerce")
        elif {"pde_price", "tree_price"}.issubset(q.columns):
            q["pde_tree_disagreement"] = pd.to_numeric(q["pde_price"], errors="coerce") - pd.to_numeric(q["tree_price"], errors="coerce")
        else:
            return _quiet_axis(ax, "No disagreement column", title or "PDE-Tree Gap")
    q["abs_disagreement"] = pd.to_numeric(q["pde_tree_disagreement"], errors="coerce").abs()
    spot_col = "spot" if "spot" in q.columns else "s" if "s" in q.columns else None
    if spot_col is not None:
        spot = pd.to_numeric(q[spot_col], errors="coerce").replace(0.0, np.nan)
        q["gap_value"] = 10000.0 * q["abs_disagreement"] / spot
        ylabel = "|PDE - tree| (bps of spot)"
    else:
        q["gap_value"] = q["abs_disagreement"]
        ylabel = "|PDE - tree|"
    if "dte_days" not in q.columns:
        return _quiet_axis(ax, "No DTE column", title or "PDE-Tree Gap")
    if "dte_bucket" not in q.columns:
        q["dte_bucket"] = pd.cut(pd.to_numeric(q["dte_days"], errors="coerce"), [0, 14, 30, 60, 90, 120, 180, 365])
    q = q.replace([np.inf, -np.inf], np.nan).dropna(subset=["gap_value", "dte_days", "dte_bucket"])
    if q.empty:
        return _quiet_axis(ax, "No disagreement data", title or "PDE-Tree Gap")
    drew = False
    for i, (opt, group) in enumerate(q.groupby("option_type", observed=True)):
        by = group.groupby("dte_bucket", observed=True).agg(
            x=("dte_days", "median"),
            med=("gap_value", "median"),
            lo=("gap_value", lambda v: np.nanquantile(v, 0.25)),
            hi=("gap_value", lambda v: np.nanquantile(v, 0.75)),
        ).dropna().sort_values("x")
        if by.empty:
            continue
        color = LAB_COLORS[i % len(LAB_COLORS)]
        ax.plot(by["x"], by["med"], marker="o", ms=3, lw=1.5, color=color, label=str(opt))
        ax.fill_between(by["x"].to_numpy(float), by["lo"].to_numpy(float), by["hi"].to_numpy(float), color=color, alpha=0.15, lw=0)
        drew = True
    if not drew:
        return _quiet_axis(ax, "No disagreement data", title or "PDE-Tree Gap")
    ax.set_xlabel("DTE")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "PDE-Tree Gap")
    _small_legend(ax, ncol=2)
    return ax


def pde_disagreement_bars(ax, table: pd.DataFrame, title: str | None = None, n: int = 20):
    return pde_tree_gap_curves(ax, table, title or "PDE-Tree Gap")


def lsm_policy_curve(ax, coefficients: np.ndarray | pd.DataFrame, *, strike: float = 1.0, option_type: str = "put", step: int | None = None, title: str | None = None):
    ax = _ax(ax)
    beta = np.asarray(coefficients, dtype=float)
    if beta.size == 0:
        return _quiet_axis(ax, "No LSM coefficients", title or "LSM Policy")
    if beta.ndim == 1:
        coef = beta
    else:
        idx = int(step) if step is not None else max(0, beta.shape[0] // 2)
        idx = min(max(idx, 0), beta.shape[0] - 1)
        coef = beta[idx]
    x = np.linspace(-0.45, 0.45, 220)
    basis = np.vstack([x ** i for i in range(len(coef))]).T
    cont = basis @ coef
    s_over_k = np.exp(x)
    payoff = np.maximum(s_over_k - 1.0, 0.0) * float(strike) if str(option_type).lower().startswith("c") else np.maximum(1.0 - s_over_k, 0.0) * float(strike)
    exercise = payoff > cont
    ax.plot(x, cont, lw=1.7, label="fitted continuation")
    ax.plot(x, payoff, lw=1.3, color=LAB_COLORS[1], label="immediate exercise")
    if exercise.any():
        idx = np.where(exercise)[0]
        ax.axvspan(x[idx.min()], x[idx.max()], color=LAB_COLORS[1], alpha=0.12, label="exercise region")
        ax.axvline(x[idx[0]], color="black", lw=0.9, ls=":")
    ax.set_xlabel("log(S / K)")
    ax.set_ylabel("value")
    ax.set_title(title or "LSM Policy")
    _small_legend(ax)
    return ax


def overlay_equity_drawdown(ax, results: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    nav = results.get("nav", pd.DataFrame()) if isinstance(results, dict) else results
    dd = results.get("drawdown", pd.DataFrame()) if isinstance(results, dict) else pd.DataFrame()
    if nav.empty:
        return _quiet_axis(ax, "No NAV", title or "Overlay Equity")
    base = nav / nav.iloc[0]
    base.plot(ax=ax, lw=1.2)
    _small_legend(ax, ncol=1)
    ax.set_ylabel("NAV / initial NAV")
    ax.set_title(title or "Overlay Equity")
    if not dd.empty:
        ax2 = ax.twinx()
        dd.min(axis=1).plot(ax=ax2, color="#6b7280", lw=1.0, alpha=0.65, label="worst drawdown")
        ax2.set_ylabel("worst drawdown")
        ax2.grid(False)
    return ax


def strategy_mechanics_bars(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty or "strategy" not in table:
        return _quiet_axis(ax, "No strategy table", title or "Strategy mechanics")
    q = table.copy()
    cols = [c for c in ["total_premium_received", "total_close_cost", "total_spread_cost"] if c in q.columns]
    if not cols:
        return _quiet_axis(ax, "No cashflow data", title or "Strategy mechanics")
    plot = q.set_index("strategy")[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    plot.plot(kind="bar", ax=ax, width=0.82)
    _small_legend(ax, ncol=1)
    ax.set_ylabel("cashflow dollars")
    ax.set_title(title or "Strategy Mechanics")
    ax.tick_params(axis="x", labelrotation=35)
    return ax


def pde_value_map(ax, result: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    values = result.get("values", np.asarray([])) if isinstance(result, dict) else np.asarray(result)
    if np.asarray(values).size == 0:
        return _quiet_axis(ax, "No PDE values", title or "PDE value map")
    arr = np.asarray(values, dtype=float)
    s_grid = np.asarray(result.get("s_grid", np.arange(arr.shape[1])), dtype=float) if isinstance(result, dict) else np.arange(arr.shape[1])
    strike = float(result.get("strike", np.nan)) if isinstance(result, dict) else np.nan
    x = s_grid / strike if np.isfinite(strike) and strike > 0 else s_grid
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", extent=[float(np.nanmin(x)), float(np.nanmax(x)), 0.0, 1.0])
    ax.set_xlabel("S / K")
    ax.set_ylabel("time index fraction")
    ax.set_title(title or "PDE value map")
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    return ax


def pde_value_surface(ax, result: dict | pd.DataFrame, title: str | None = None):
    return pde_value_map(ax, result, title or "PDE value surface")


def pde_exercise_map(ax, result: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if isinstance(result, dict):
        values = np.asarray(result.get("values", []), dtype=float)
        s_grid = np.asarray(result.get("s_grid", np.arange(values.shape[1] if values.ndim == 2 else 0)), dtype=float)
        strike = float(result.get("strike", np.nan))
        option_type = str(result.get("option_type", "put"))
        payoff = np.maximum(s_grid - strike, 0.0) if option_type.lower().startswith("c") else np.maximum(strike - s_grid, 0.0)
        exercise = (values - payoff[None, :]) <= 1e-5 if values.ndim == 2 and np.isfinite(strike) else np.asarray([])
    else:
        exercise = np.asarray(result, dtype=float)
    if exercise.size == 0:
        return _quiet_axis(ax, "No exercise region", title or "PDE exercise region")
    if isinstance(result, dict) and "s_grid" in result and np.isfinite(strike) and strike > 0:
        x = s_grid / strike
        extent = [float(np.nanmin(x)), float(np.nanmax(x)), 0.0, 1.0]
    else:
        extent = None
    im = ax.imshow(exercise.astype(float), aspect="auto", origin="lower", cmap="magma", extent=extent)
    ax.set_xlabel("S / K")
    ax.set_ylabel("time index fraction")
    ax.set_title(title or "PDE exercise region")
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    return ax


def pde_boundary(ax, result: dict | pd.DataFrame, title: str | None = None):
    if isinstance(result, pd.DataFrame):
        data = result
    else:
        boundary = np.asarray(result.get("boundary", []), dtype=float)
        strike = float(result.get("strike", np.nan))
        tau = float(result.get("tau", 1.0))
        data = pd.DataFrame({"time_to_expiry": np.linspace(tau, 0.0, len(boundary)), "boundary": boundary})
        if np.isfinite(strike) and strike > 0:
            data["boundary_over_k"] = data["boundary"] / strike
    return tree_boundary(ax, data, title or "PDE boundary")


def pde_residual(ax, result: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    residuals = result.get("residuals", []) if isinstance(result, dict) else result.get("residual", [])
    if len(residuals) == 0:
        return _quiet_axis(ax, "No residuals", title or "PSOR residual")
    ax.semilogy(np.asarray(residuals, dtype=float), lw=1.2)
    ax.set_xlabel("time step")
    ax.set_ylabel("residual")
    ax.set_title(title or "PSOR residual")
    return ax


def pde_value_slices(ax, result: dict, title: str | None = None):
    ax = _ax(ax)
    values = np.asarray(result.get("values", []), dtype=float)
    s_grid = np.asarray(result.get("s_grid", []), dtype=float)
    strike = float(result.get("strike", np.nan))
    if values.ndim != 2 or s_grid.size == 0 or not np.isfinite(strike) or strike <= 0:
        return _quiet_axis(ax, "No PDE slices", title or "PDE value slices")
    x = s_grid / strike
    option_type = str(result.get("option_type", "put")).lower()
    payoff = np.maximum(x - 1.0, 0.0) if option_type.startswith("c") else np.maximum(1.0 - x, 0.0)
    ax.plot(x, payoff, color="black", lw=1.2, label="payoff")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(round(frac * (values.shape[0] - 1)))
        ax.plot(x, values[idx] / strike, lw=1.0, label=f"t={frac:.2f}")
    ax.set_xlabel("S / K")
    ax.set_ylabel("V / K")
    ax.set_title(title or "PDE value slices")
    _small_legend(ax)
    return ax


def complementarity_gap(ax, result: dict, title: str | None = None):
    ax = _ax(ax)
    values = np.asarray(result.get("values", []), dtype=float)
    s_grid = np.asarray(result.get("s_grid", []), dtype=float)
    strike = float(result.get("strike", np.nan))
    if values.ndim != 2 or s_grid.size == 0 or not np.isfinite(strike):
        return _quiet_axis(ax, "No complementarity gap", title or "Complementarity gap")
    option_type = str(result.get("option_type", "put")).lower()
    payoff = np.maximum(s_grid - strike, 0.0) if option_type.startswith("c") else np.maximum(strike - s_grid, 0.0)
    gap = np.maximum(payoff[None, :] - values, 0.0)
    x = s_grid / strike
    im = ax.imshow(gap, aspect="auto", origin="lower", cmap="magma", extent=[float(np.nanmin(x)), float(np.nanmax(x)), 0.0, 1.0])
    ax.set_xlabel("S / K")
    ax.set_ylabel("time index fraction")
    ax.set_title(title or "Complementarity gap")
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    return ax


def pde_residuals(ax, result: dict | pd.DataFrame, title: str | None = None):
    return pde_residual(ax, result, title or "PSOR residuals")


def method_disagreement_heatmap(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "model_disagreement" not in quotes:
        return _quiet_axis(ax, "No disagreement", title or "Method disagreement")
    q = quotes.copy()
    dte = q["dte_days"] if "dte_days" in q else q["tau"] * 365.25 if "tau" in q else pd.Series(np.nan, index=q.index)
    moneyness = q["moneyness"] if "moneyness" in q else q["strike"] / q["spot"] if {"strike", "spot"}.issubset(q.columns) else pd.Series(np.nan, index=q.index)
    q["dte_bucket"] = pd.cut(pd.to_numeric(dte, errors="coerce"), bins=[0, 14, 30, 60, 90, 120, 180, 365])
    q["m_bucket"] = pd.cut(pd.to_numeric(moneyness, errors="coerce"), bins=np.linspace(0.75, 1.25, 11))
    table = q.pivot_table(index="dte_bucket", columns="m_bucket", values="model_disagreement", aggfunc="median", observed=False)
    im = ax.imshow(table.to_numpy(float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=7)
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    ax.set_title(title or "Method disagreement")
    return ax


def lsm_regression(ax, data: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if data.empty or not {"x", "cashflow"}.issubset(data.columns):
        return _quiet_axis(ax, "No regression data", title or "LSM continuation regression")
    sc = ax.scatter(data["x"], data["cashflow"], c=data.get("exercise", pd.Series(0, index=data.index)), s=10, alpha=0.35, cmap="coolwarm")
    if "continuation" in data.columns:
        ordered = data.sort_values("x")
        ax.plot(ordered["x"], ordered["continuation"], color="black", lw=1.7, label="fitted continuation")
    if "payoff" in data.columns:
        ordered = data.sort_values("x")
        ax.plot(ordered["x"], ordered["payoff"], color="#dc2626", lw=1.2, label="immediate exercise")
    if "boundary_x" in data.columns and data["boundary_x"].notna().any():
        ax.axvline(float(data["boundary_x"].dropna().iloc[0]), color="#111827", lw=1.0, ls=":", label="exercise boundary")
    ax.set_xlabel("log(S/K)")
    ax.set_ylabel("discounted future cashflow")
    ax.set_title(title or "LSM true continuation target")
    _small_legend(ax)
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    return ax


def lsm_boundary(ax, boundary: pd.DataFrame, reference: pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    if boundary.empty:
        return _quiet_axis(ax, "No LSM boundary", title or "LSM boundary")
    x = boundary["step"] if "step" in boundary else boundary.index
    ax.plot(x, boundary["boundary"], lw=1.5, label="LSM")
    if reference is not None and not reference.empty:
        ax.plot(reference.iloc[:, 0], reference.iloc[:, 1], lw=1.0, label="reference")
    ax.set_title(title or "Learned exercise boundary")
    _small_legend(ax)
    return ax


def lsm_policy_gap(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No policy gap", title or "LSM policy gap")
    col = "policy_gap" if "policy_gap" in table else table.select_dtypes("number").columns[-1]
    ax.hist(pd.to_numeric(table[col], errors="coerce").dropna(), bins=25)
    ax.set_title(title or "Policy gap")
    return ax


def lsm_regime_coverage(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No regime coverage", title or "LSM regime coverage")
    q = table.copy()
    x = pd.to_numeric(q.get("moneyness", q.get("median_moneyness")), errors="coerce")
    y = pd.to_numeric(q.get("dte_days", q.get("median_dte")), errors="coerce")
    c = pd.to_numeric(q.get("coverage_rows", q.get("cell_rows", 1)), errors="coerce")
    sc = ax.scatter(x, y, c=c, s=25, cmap="viridis")
    ax.set_xlabel("K / S")
    ax.set_ylabel("DTE")
    ax.set_title(title or "LSM regime coverage")
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    return ax


def lsm_path_convergence(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No LSM convergence", title or "LSM path convergence")
    q = table.copy()
    y = "evaluation_price" if "evaluation_price" in q.columns else "lsm_price"
    degree_col = "basis_degree" if "basis_degree" in q.columns else ("degree" if "degree" in q.columns else None)
    if degree_col is None:
        part = q.sort_values("paths")
        ax.plot(part["paths"], part[y], marker="o", lw=1.2, label="policy")
    else:
        for degree, part in q.groupby(degree_col):
            part = part.sort_values("paths")
            ax.plot(part["paths"], part[y], marker="o", lw=1.2, label=f"degree {degree}")
    if "ci_low" in q.columns and "ci_high" in q.columns:
        part = q.sort_values("paths")
        ax.fill_between(part["paths"], part["ci_low"], part["ci_high"], color="#93c5fd", alpha=0.25)
    ax.set_xscale("log")
    ax.set_xlabel("paths")
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title or "LSM path-count convergence")
    _small_legend(ax)
    return ax


def lsm_exercise_probability(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty or "exercise_probability" not in table:
        return _quiet_axis(ax, "No exercise probability", title or "LSM exercise probability")
    q = table.copy()
    if "dte_bucket" not in q:
        q["dte_bucket"] = pd.cut(pd.to_numeric(q.get("dte_days"), errors="coerce"), [0, 14, 30, 60, 90, 120, 180])
    if "moneyness_bucket" not in q:
        q["moneyness_bucket"] = pd.cut(pd.to_numeric(q.get("moneyness"), errors="coerce"), [0.65, 0.8, 0.9, 0.97, 1.03, 1.1, 1.25, 1.45])
    table2 = q.pivot_table(index="dte_bucket", columns="moneyness_bucket", values="exercise_probability", aggfunc="median", observed=False)
    im = ax.imshow(table2.to_numpy(float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(table2.shape[1]))
    ax.set_xticklabels([str(c) for c in table2.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(table2.shape[0]))
    ax.set_yticklabels([str(i) for i in table2.index], fontsize=7)
    ax.set_xlabel("K / S bucket")
    ax.set_ylabel("DTE bucket")
    ax.set_title(title or "LSM exercise probability")
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    return ax


def lsm_reference_gap(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No reference gap", title or "LSM reference gap")
    q = table.copy()
    gap = "reference_gap" if "reference_gap" in q.columns else "tree_lsm_gap" if "tree_lsm_gap" in q.columns else None
    if gap is None:
        return _quiet_axis(ax, "No gap column", title or "LSM reference gap")
    size = np.sqrt(pd.to_numeric(q.get("coverage_rows", 1), errors="coerce").fillna(1.0)).clip(8, 70)
    x = pd.to_numeric(q["moneyness"], errors="coerce") if "moneyness" in q else pd.Series(1.0, index=q.index)
    y = pd.to_numeric(q["dte_days"], errors="coerce") if "dte_days" in q else pd.Series(0.0, index=q.index)
    sc = ax.scatter(x, y, c=pd.to_numeric(q[gap], errors="coerce"), s=size, cmap="coolwarm", alpha=0.75)
    ax.set_xlabel("K / S")
    ax.set_ylabel("DTE")
    ax.set_title(title or "LSM reference gap by regime")
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    return ax


def method_runtime_error(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No methods", title or "Runtime vs error")
    x = pd.to_numeric(table.get("runtime_sec", table.get("runtime", 0.0)), errors="coerce").clip(lower=1e-8)
    y = pd.to_numeric(table.get("abs_error", table.get("model_disagreement", table.get("price_error", 0.0))), errors="coerce")
    ax.scatter(x, y, s=45)
    for _, row in table.iterrows():
        ax.annotate(str(row.get("method", row.get("model", ""))), (float(row.get("runtime_sec", row.get("runtime", 1e-8))), float(row.get("abs_error", row.get("model_disagreement", row.get("price_error", 0.0))))), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("runtime seconds")
    ax.set_ylabel("error")
    ax.set_title(title or "Runtime vs error")
    return ax


def runtime_accuracy(ax, table: pd.DataFrame, title: str | None = None):
    return method_runtime_error(ax, table, title or "Runtime vs accuracy")


def american_premium_map(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "american_premium" not in quotes:
        return _quiet_axis(ax, "No premium data", title or "American premium")
    x = pd.to_numeric(quotes.get("moneyness", quotes.get("k_over_s", quotes.get("strike") / quotes.get("spot"))), errors="coerce")
    y = pd.to_numeric(quotes.get("dte_days", quotes.get("dte", quotes.get("tau") * 365.25)), errors="coerce")
    sc = ax.scatter(x, y, c=quotes["american_premium"], s=10, cmap="viridis")
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    ax.set_xlabel("K / S")
    ax.set_ylabel("DTE")
    ax.set_title(title or "American premium")
    return ax


def assignment_heatmap(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or "assignment_risk" not in quotes:
        return _quiet_axis(ax, "No assignment risk", title or "Assignment risk")
    q = quotes.copy()
    q["dte_bucket"] = pd.cut(pd.to_numeric(q.get("dte_days", q.get("dte", q.get("tau") * 365.25)), errors="coerce"), bins=[0, 7, 14, 30, 60, 120, 365])
    q["m_bucket"] = pd.cut(pd.to_numeric(q.get("moneyness", q.get("k_over_s", q.get("strike") / q.get("spot"))), errors="coerce"), bins=np.linspace(0.8, 1.2, 9))
    table = q.pivot_table(index="dte_bucket", columns="m_bucket", values="assignment_risk", aggfunc="median", observed=False)
    im = ax.imshow(table.to_numpy(float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([str(c) for c in table.columns], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(i) for i in table.index], fontsize=7)
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    ax.set_title(title or "Assignment risk")
    return ax


def assignment_event_study(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No event study", title or "Assignment risk around ex-dividend")
    q = table.copy()
    if "option_type" in q.columns:
        q = q[q["option_type"].astype(str).str.lower().str.startswith("c")].copy()
    if {"spot", "strike"}.issubset(q.columns):
        q = q[pd.to_numeric(q["spot"], errors="coerce") > pd.to_numeric(q["strike"], errors="coerce")].copy()
    if q.empty:
        return _quiet_axis(ax, "No ITM calls", title or "ITM call assignment risk")
    xcol = "days_to_next_dividend" if "days_to_next_dividend" in table.columns else "event_day"
    ycol = "assignment_risk" if "assignment_risk" in table.columns else table.select_dtypes("number").columns[-1]
    q = q.dropna(subset=[xcol, ycol]).copy()
    if q.empty:
        return _quiet_axis(ax, "No event study", title or "Assignment risk around ex-dividend")
    grouped = q.groupby(xcol)[ycol].agg(["median", "mean", "count"]).reset_index()
    ax.plot(grouped[xcol], grouped["median"], marker="o", lw=1.5, label="median ITM call risk")
    ax.plot(grouped[xcol], grouped["mean"], lw=1.0, alpha=0.7, label="mean")
    ax.fill_between(grouped[xcol].to_numpy(float), grouped["median"].to_numpy(float), 0.0, alpha=0.10)
    ax.invert_xaxis()
    ax.set_xlabel("days to ex-dividend")
    ax.set_ylabel("assignment risk")
    ax.set_title(title or "ITM short-call assignment risk into ex-dividend")
    _small_legend(ax)
    return ax


def dividend_gap_distribution(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty:
        return _quiet_axis(ax, "No dividend gap", title or "Dividend gap")
    q = quotes.copy()
    if "option_type" in q:
        q = q[q["option_type"].astype(str).str.lower().str.startswith("c")].copy()
    if {"spot", "strike"}.issubset(q):
        q = q[pd.to_numeric(q["spot"], errors="coerce") > pd.to_numeric(q["strike"], errors="coerce")].copy()
    gap = pd.to_numeric(q.get("dividend_gap", pd.to_numeric(q.get("next_dividend", 0.0), errors="coerce") - pd.to_numeric(q.get("time_value", 0.0), errors="coerce")), errors="coerce").dropna()
    ax.hist(gap, bins=60, color="#2563eb", alpha=0.8)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("next dividend - time value")
    ax.set_ylabel("ITM calls")
    ax.set_title(title or "Dividend gap distribution")
    return ax


def assignment_component_bars(ax, quotes: pd.DataFrame, title: str | None = None, n: int = 12):
    ax = _ax(ax)
    cols = ["itm_score", "boundary_proximity", "dividend_gap_score", "low_time_value_score", "ex_div_proximity", "model_uncertainty_score"]
    if quotes.empty or not set(cols).issubset(quotes.columns):
        return _quiet_axis(ax, "No components", title or "Assignment components")
    q = quotes.copy()
    if "option_type" in q:
        q = q[q["option_type"].astype(str).str.lower().str.startswith("c")].copy()
    q = q.sort_values("assignment_risk", ascending=False).head(int(n))
    bottom = np.zeros(len(q))
    x = np.arange(len(q))
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#f97316", "#059669", "#6b7280"]
    for col, color in zip(cols, colors, strict=False):
        values = pd.to_numeric(q[col], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, values, bottom=bottom, label=col.replace("_", " "), color=color, alpha=0.85)
        bottom += values
    ax.set_xticks(x)
    expiry = pd.to_datetime(q["expiry"], errors="coerce").dt.strftime("%m-%d") if "expiry" in q else pd.Series("", index=q.index)
    strike = pd.to_numeric(q["strike"], errors="coerce").round(0).astype("Int64").astype(str) if "strike" in q else pd.Series(np.arange(len(q)).astype(str), index=q.index)
    labels = expiry.fillna("") + " " + strike.fillna("")
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("component level")
    ax.set_title(title or "Top short-call assignment risk components")
    _small_legend(ax, ncol=2)
    return ax


def boundary_dividend_scatter(ax, quotes: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty or not {"boundary_distance", "dividend_gap"}.issubset(quotes.columns):
        return _quiet_axis(ax, "No boundary/dividend data", title or "Boundary vs dividend gap")
    q = quotes.copy()
    if "option_type" in q:
        q = q[q["option_type"].astype(str).str.lower().str.startswith("c")].copy()
    if {"spot", "strike"}.issubset(q.columns):
        q = q[pd.to_numeric(q["spot"], errors="coerce") > pd.to_numeric(q["strike"], errors="coerce")].copy()
    sc = ax.scatter(pd.to_numeric(q["boundary_distance"], errors="coerce"), pd.to_numeric(q["dividend_gap"], errors="coerce"), c=pd.to_numeric(q.get("assignment_risk", 0.0), errors="coerce"), s=9, alpha=0.35, cmap="magma")
    ax.axvline(0.0, color="black", lw=0.8)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("boundary distance")
    ax.set_ylabel("dividend - time value")
    ax.set_title(title or "Boundary proximity and dividend gap")
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    return ax


def overlay_equity(ax, results: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    nav = results.get("nav", pd.DataFrame()) if isinstance(results, dict) else results
    if nav.empty:
        return _quiet_axis(ax, "No overlay NAV", title or "Overlay equity")
    base = nav / nav.iloc[0]
    base.plot(ax=ax, lw=1.3)
    ax.set_ylabel("NAV / initial NAV")
    ax.set_title(title or "Overlay equity")
    return ax


def overlay_drawdown(ax, results: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    dd = results.get("drawdown", pd.DataFrame()) if isinstance(results, dict) else results
    if dd.empty:
        return _quiet_axis(ax, "No drawdown", title or "Overlay drawdown")
    dd.plot(ax=ax, lw=1.2)
    ax.set_title(title or "Overlay drawdown")
    return ax


def overlay_return_drawdown(ax, summary: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if summary.empty:
        return _quiet_axis(ax, "No summary", title or "Return vs drawdown")
    x = pd.to_numeric(summary.get("max_drawdown"), errors="coerce")
    y = pd.to_numeric(summary.get("total_return"), errors="coerce")
    ax.scatter(x, y, s=55)
    for _, row in summary.iterrows():
        ax.annotate(str(row.get("strategy", "")), (float(row.get("max_drawdown", np.nan)), float(row.get("total_return", np.nan))), fontsize=7)
    ax.set_xlabel("max drawdown")
    ax.set_ylabel("total return")
    ax.set_title(title or "Strategy return vs drawdown")
    return ax


def premium_close_bars(ax, trades: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if trades.empty or not {"strategy", "event", "cashflow"}.issubset(trades.columns):
        return _quiet_axis(ax, "No trade cashflows", title or "Premium and close costs")
    q = trades.copy()
    q["open_premium"] = np.where(q["event"].eq("open"), pd.to_numeric(q["cashflow"], errors="coerce"), 0.0)
    q["close_cost"] = np.where(q["event"].isin(["roll_close", "assignment_defense_close"]), pd.to_numeric(q["cashflow"], errors="coerce"), 0.0)
    table = q.groupby("strategy")[["open_premium", "close_cost"]].sum()
    table.plot(kind="bar", stacked=True, ax=ax)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("cashflow")
    ax.set_title(title or "Premium received vs close/defense cost")
    return ax


def selected_moneyness(ax, trades: pd.DataFrame, quotes: pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    if trades.empty:
        return _quiet_axis(ax, "No selected contracts", title or "Selected moneyness")
    data = trades.copy()
    if quotes is not None and not quotes.empty and "moneyness" not in data.columns:
        merge_cols = ["contract_key", "moneyness"] + (["option_type"] if "option_type" in quotes.columns else [])
        data = data.merge(quotes[merge_cols].drop_duplicates("contract_key"), on="contract_key", how="left")
    if "moneyness" not in data.columns:
        return _quiet_axis(ax, "No moneyness", title or "Selected moneyness")
    d = pd.to_datetime(data.get("date", data.get("entry_date")), errors="coerce")
    call = data["option_type"].astype(str).str.lower().str.startswith("c") if "option_type" in data else pd.Series(False, index=data.index)
    ax.scatter(d[call], data.loc[call, "moneyness"], s=18, label="calls", alpha=0.75)
    ax.scatter(d[~call], data.loc[~call, "moneyness"], s=18, label="puts", alpha=0.75)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_ylabel("K / S at selection")
    ax.set_title(title or "Selected option moneyness")
    _small_legend(ax)
    return ax


def active_legs(ax, holdings: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if holdings.empty:
        return _quiet_axis(ax, "No holdings", title or "Active legs")
    holdings.plot(ax=ax, lw=1.1)
    ax.set_ylabel("active option legs")
    ax.set_title(title or "Active option legs")
    return ax


def trade_timeline(ax, trades: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if trades.empty or "event" not in trades:
        return _quiet_axis(ax, "No trades", title or "Trade timeline")
    data = trades.copy()
    data["date"] = pd.to_datetime(data.get("date", data.get("entry_date")), errors="coerce")
    events = list(pd.unique(data["event"].astype(str)))
    codes = {name: i for i, name in enumerate(events)}
    y = data["event"].astype(str).map(codes)
    ax.scatter(data["date"], y, s=18, alpha=0.75)
    ax.set_yticks(list(codes.values()))
    ax.set_yticklabels(list(codes.keys()), fontsize=7)
    ax.set_title(title or "Trade event timeline")
    return ax


def monthly_return_bars(ax, nav: pd.DataFrame | dict, title: str | None = None):
    ax = _ax(ax)
    data = nav.get("nav", pd.DataFrame()) if isinstance(nav, dict) else nav
    if data.empty:
        return _quiet_axis(ax, "No NAV", title or "Monthly returns")
    monthly = data.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return _quiet_axis(ax, "Not enough NAV history", title or "Monthly returns")
    monthly.plot(kind="bar", ax=ax, width=0.85)
    ax.set_ylabel("monthly return")
    ax.set_title(title or "Monthly returns")
    return ax


def roll_actions(ax, trades: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if trades.empty:
        return _quiet_axis(ax, "No trades", title or "Roll actions")
    d = pd.to_datetime(trades.get("entry_date", trades.get("date")), errors="coerce")
    y = trades.get("strategy", trades.get("label", pd.Series("trade", index=trades.index))).astype("category").cat.codes
    ax.scatter(d, y, s=18, alpha=0.7)
    ax.set_title(title or "Roll and entry actions")
    return ax


def selected_strikes(ax, trades: pd.DataFrame, quotes: pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    if trades.empty:
        return _quiet_axis(ax, "No selected strikes", title or "Selected strikes")
    data = trades.copy()
    if quotes is not None and not quotes.empty and "strike" not in data.columns:
        keys = ["contract_key", "strike", "option_type"]
        data = data.merge(quotes[keys].drop_duplicates("contract_key"), on="contract_key", how="left")
    if "strike" not in data.columns:
        return _quiet_axis(ax, "No strike column", title or "Selected strikes")
    d = pd.to_datetime(data.get("date", data.get("entry_date")), errors="coerce")
    y = pd.to_numeric(data["strike"], errors="coerce")
    sc = ax.scatter(d, y, c=data.get("quantity", pd.Series(1.0, index=data.index)), s=20, cmap="coolwarm")
    ax.set_ylabel("strike")
    ax.set_title(title or "Selected strikes over time")
    ax.figure.colorbar(sc, ax=ax, pad=0.01)
    return ax


def premium_protection(ax, summary: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if summary.empty:
        return _quiet_axis(ax, "No summary", title or "Premium and protection")
    x = pd.to_numeric(summary.get("net_open_premium_cashflow", summary.get("premium_income", 0.0)), errors="coerce")
    y = pd.to_numeric(summary.get("max_drawdown", 0.0), errors="coerce")
    ax.scatter(x, y, s=45)
    for _, row in summary.iterrows():
        ax.annotate(str(row.get("strategy", "")), (float(row.get("net_open_premium_cashflow", row.get("premium_income", 0.0))), float(row.get("max_drawdown", 0.0))), fontsize=7)
    ax.set_xlabel("net open premium cashflow")
    ax.set_ylabel("max drawdown")
    ax.set_title(title or "Premium/protection tradeoff")
    return ax


def fourier_quote_coverage(ax, quotes: pd.DataFrame, title: str | None = None):
    return quote_coverage(ax, quotes, title or "Fourier quote coverage")


def cf_shape(ax, data: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if data.empty:
        return _quiet_axis(ax, "No CF data", title or "CF shape")
    ax.plot(data["u"], data["real"], label="real")
    ax.plot(data["u"], data["imag"], label="imag")
    _small_legend(ax)
    ax.set_title(title or "Characteristic function")
    return ax


def direct_price_error(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No direct validation", title or "Direct price error")
    x = table.get("strike", table.index)
    y = table.get("price_error", table.select_dtypes("number").iloc[:, -1])
    ax.plot(x, y, marker="o", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title or "Direct price error")
    return ax


def fft_grid(ax, grid: pd.DataFrame, quotes: pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    if grid.empty:
        return _quiet_axis(ax, "No FFT grid", title or "FFT grid")
    ax.plot(grid["strike"], grid["price"], lw=1.2, label="grid")
    if quotes is not None and not quotes.empty:
        ax.scatter(quotes["strike"], quotes["mid"], s=12, alpha=0.5, label="quotes")
    ax.set_xscale("log")
    ax.set_title(title or "FFT strike grid")
    _small_legend(ax)
    return ax


def fft_damping(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No damping data", title or "FFT damping")
    ax.plot(table["alpha"], table["error"], marker="o")
    ax.set_xlabel("alpha")
    ax.set_ylabel("error")
    ax.set_title(title or "Damping sensitivity")
    return ax


def cos_convergence(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No COS convergence", title or "COS convergence")
    q = table.copy()
    y_col = "median_abs_error" if "median_abs_error" in q.columns else "max_abs_error" if "max_abs_error" in q.columns else "error"
    for i, (model, g) in enumerate(q.groupby(q.get("model", pd.Series("COS", index=q.index)), observed=True)):
        g = g.sort_values("n_terms")
        ax.plot(g["n_terms"], pd.to_numeric(g[y_col], errors="coerce"), marker="o", ms=3, lw=1.25, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(model))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("terms")
    ax.set_ylabel("median abs error")
    ax.set_title(title or "COS Convergence")
    _small_legend(ax, ncol=2)
    return ax


def engine_speed(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No speed table", title or "Engine speed")
    ax.bar(table.iloc[:, 0].astype(str), pd.to_numeric(table.get("items_per_sec", table.select_dtypes("number").iloc[:, -1]), errors="coerce"))
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_title(title or "Engine speed")
    return ax


def fft_fit(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No FFT fit", title or "FFT Fit")
    q = table.copy()
    q = q.sort_values("moneyness" if "moneyness" in q.columns else "strike")
    x = pd.to_numeric(q["moneyness"], errors="coerce") if "moneyness" in q.columns else pd.to_numeric(q["strike"], errors="coerce")
    y = pd.to_numeric(q.get("fft_price", q.get("price")), errors="coerce")
    ref = pd.to_numeric(q.get("reference_price", y), errors="coerce")
    err = pd.to_numeric(q.get("abs_error", (y - ref).abs()), errors="coerce").clip(lower=1e-12)
    ax.plot(x, ref, color="black", lw=1.3, label="reference")
    ax.plot(x, y, color=LAB_COLORS[0], lw=1.3, label="FFT")
    ax.fill_between(x.to_numpy(float), np.minimum(y, ref).to_numpy(float), np.maximum(y, ref).to_numpy(float), color=LAB_COLORS[0], alpha=0.12, lw=0)
    ax2 = ax.twinx()
    ax2.plot(x, err, color=LAB_COLORS[6], lw=1.0, alpha=0.85, label="abs error")
    ax2.set_yscale("log")
    ax2.set_ylabel("abs error")
    ax2.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("K / S" if "moneyness" in q.columns else "strike")
    ax.set_ylabel("call price")
    ax.set_title(title or "FFT Fit")
    _small_legend(ax, loc="upper right")
    return ax


def throughput(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No throughput", title or "Throughput")
    q = table.copy()
    if "label" not in q.columns:
        q["label"] = q.get("method", q.get("engine", pd.Series("engine", index=q.index))).astype(str)
    if "prices_per_second" not in q.columns:
        n = pd.to_numeric(q.get("items", q.get("n", 1.0)), errors="coerce")
        runtime = pd.to_numeric(q.get("runtime_sec", q.get("runtime", np.nan)), errors="coerce")
        q["prices_per_second"] = n / runtime.replace(0.0, np.nan)
    by = q.groupby("label", observed=True)["prices_per_second"].median().replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    if by.empty:
        return _quiet_axis(ax, "No throughput", title or "Throughput")
    colors = [LAB_COLORS[i % len(LAB_COLORS)] for i in range(len(by))]
    ax.barh(by.index.astype(str), by.to_numpy(float), color=colors, alpha=0.82)
    ax.set_xscale("log")
    ax.set_xlabel("prices / second")
    ax.set_title(title or "Throughput")
    ax.tick_params(axis="y", labelsize=7)
    return ax


def model_quality(ax, comparison: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if comparison.empty:
        return _quiet_axis(ax, "No model comparison", title or "Model Quality")
    q = comparison.copy().sort_values("weighted_price_rmse")
    y = np.arange(len(q))
    rmse = pd.to_numeric(q["weighted_price_rmse"], errors="coerce")
    ax.barh(y, rmse, color=LAB_COLORS[0], alpha=0.78, label="RMSE")
    if "otm_put_rmse" in q.columns:
        ax.scatter(pd.to_numeric(q["otm_put_rmse"], errors="coerce"), y, color=LAB_COLORS[6], s=36, zorder=3, label="OTM puts")
    ax.set_yticks(y)
    ax.set_yticklabels(q["model"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("weighted error")
    ax.set_title(title or "Model Quality")
    if "bid_ask_hit_rate" in q.columns:
        for yi, hit in zip(y, pd.to_numeric(q["bid_ask_hit_rate"], errors="coerce"), strict=False):
            if np.isfinite(hit):
                ax.text(rmse.max() * 1.03 if np.isfinite(rmse.max()) else 0.0, yi, f"{hit:.0%}", va="center", fontsize=7, color=LAB_COLORS[3])
    _small_legend(ax, loc="lower right")
    return ax


def jump_fit(ax, quotes: pd.DataFrame, fit: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    f = fit.get("fit", pd.DataFrame()) if isinstance(fit, dict) else fit
    if f.empty:
        return _quiet_axis(ax, "No jump fit", title or "Jump fit")
    ax.scatter(f.get("strike"), f.get("mid"), s=12, alpha=0.5, label="mid")
    ax.scatter(f.get("strike"), f.get("model_price"), s=10, alpha=0.6, label="model")
    _small_legend(ax)
    ax.set_title(title or "Jump model fit")
    return ax


def jump_residuals(ax, fit: dict | pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    f = fit.get("fit", pd.DataFrame()) if isinstance(fit, dict) else fit
    if f.empty:
        return _quiet_axis(ax, "No residuals", title or "Jump residuals")
    x = f.get("k", np.log(f.get("strike") / f.get("spot")))
    ax.scatter(x, f.get("price_residual"), s=12, alpha=0.55)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title or "Jump residuals")
    return ax


def sv_fit(ax, quotes: pd.DataFrame, fit: dict | pd.DataFrame, title: str | None = None):
    return jump_fit(ax, quotes, fit, title or "SV fit")


def sv_residuals(ax, fit: dict | pd.DataFrame, title: str | None = None):
    return jump_residuals(ax, fit, title or "SV residuals")


def model_error_runtime(ax, comparison: pd.DataFrame, title: str | None = None):
    return method_runtime_error(ax, comparison.rename(columns={"weighted_price_rmse": "abs_error"}), title or "Model error/runtime")


def density_compare(ax, densities: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if densities.empty:
        return _quiet_axis(ax, "No densities", title or "Density comparison")
    for name, g in densities.groupby("model"):
        ax.plot(g["x"], g["density"], label=str(name))
    _small_legend(ax)
    ax.set_title(title or "Risk-neutral density")
    return ax


def tail_probability(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No tail data", title or "Tail probability")
    ax.bar(table["model"], table["left_tail_probability"])
    ax.set_title(title or "Left-tail probability")
    return ax


def hedge_candidates(ax, candidates: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if candidates.empty:
        return _quiet_axis(ax, "No candidates", title or "Hedge candidates")
    ax.scatter(candidates["strike"], candidates.get("hedge_score", candidates.index), c=candidates.get("dte_days", 0), s=24)
    ax.set_title(title or "Tail hedge candidates")
    return ax


def hedge_equity(ax, results: dict | pd.DataFrame, title: str | None = None):
    return overlay_equity(ax, results, title or "Hedge equity")


def hedge_drawdown(ax, results: dict | pd.DataFrame, title: str | None = None):
    return overlay_drawdown(ax, results, title or "Hedge drawdown")


def fourier_data_overview(ax, quotes: pd.DataFrame, spot: pd.Series | pd.DataFrame | None = None, title: str | None = None):
    ax = _ax(ax)
    if quotes.empty:
        return _quiet_axis(ax, "No quotes", title or "Data Regime")
    q = quotes.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    rows = q.groupby("date").size()
    spread = q.groupby("date")["relative_spread" if "relative_spread" in q else "rel_spread"].median() if ("relative_spread" in q or "rel_spread" in q) else None
    ax.bar(rows.index, rows / max(rows.max(), 1.0), width=1.0, color=LAB_COLORS[7], alpha=0.35, label="rows")
    if spot is not None:
        s = spot["close"] if isinstance(spot, pd.DataFrame) and "close" in spot else pd.Series(spot)
        s.index = pd.to_datetime(s.index, errors="coerce").normalize()
        s = s.sort_index().loc[(s.index >= rows.index.min()) & (s.index <= rows.index.max())]
        if not s.empty:
            ax.plot(s.index, s / s.iloc[0], color=LAB_COLORS[0], lw=1.4, label="spot")
    if spread is not None and not spread.empty:
        ax2 = ax.twinx()
        ax2.plot(spread.index, spread, color=LAB_COLORS[6], lw=1.0, label="spread")
        ax2.set_ylabel("median spread")
        ax2.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("indexed")
    ax.set_xlabel("date")
    ax.set_title(title or "Data Regime")
    _small_legend(ax, loc="upper left")
    return ax


def cf_fingerprint(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No CF data", title or "CF Fingerprint")
    y_col = "magnitude" if "magnitude" in table.columns else "density" if "density" in table.columns else None
    if y_col is None:
        return _quiet_axis(ax, "No CF measure", title or "CF Fingerprint")
    x_col = "u" if "u" in table.columns else "x"
    for i, (name, g) in enumerate(table.groupby("model", observed=True)):
        ax.plot(g[x_col], g[y_col], lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(name))
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or "CF Fingerprint")
    _small_legend(ax, ncol=2)
    return ax


def direct_accuracy(ax, validation: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if validation.empty:
        return _quiet_axis(ax, "No validation", title or "Direct Accuracy")
    q = validation.copy()
    y = "abs_error" if "abs_error" in q.columns else "price_error"
    if "strike" in q.columns:
        for i, (tau, g) in enumerate(q.groupby(q.get("tau", pd.Series("all", index=q.index)), observed=True)):
            ax.plot(g["strike"], np.abs(pd.to_numeric(g[y], errors="coerce")), lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{float(tau) * 365.25:.0f}d" if np.isscalar(tau) and pd.notna(tau) else str(tau))
        ax.set_xlabel("strike")
    else:
        ax.plot(np.abs(pd.to_numeric(q[y], errors="coerce")).to_numpy(), lw=1.0, color=LAB_COLORS[0])
        ax.set_xlabel("case")
    ax.set_ylabel("abs error")
    ax.set_yscale("log")
    ax.set_title(title or "Direct Accuracy")
    _small_legend(ax, ncol=2)
    return ax


def fft_validation(ax, validation: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if validation.empty:
        return _quiet_axis(ax, "No FFT validation", title or "FFT Validation")
    q = validation.copy()
    if "n" in q.columns:
        by = q.groupby(["engine", "n"], observed=True)["median_abs_error"].median().reset_index()
        for i, (engine, g) in enumerate(by.groupby("engine", observed=True)):
            ax.plot(g["n"], g["median_abs_error"], marker="o", ms=3, lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(engine))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
    else:
        by = q.groupby("engine", observed=True)["median_abs_error"].median().sort_values()
        ax.bar(by.index.astype(str), by.to_numpy(float), color=LAB_COLORS[: len(by)])
        ax.set_xlabel("engine")
    ax.set_ylabel("median abs error")
    ax.set_yscale("log")
    ax.set_title(title or "FFT Validation")
    _small_legend(ax)
    return ax


def cos_validation(ax, validation: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if validation.empty:
        return _quiet_axis(ax, "No COS validation", title or "COS Validation")
    q = validation.copy()
    for i, (model, g) in enumerate(q.groupby("model", observed=True)):
        x = g["n_terms"] if "n_terms" in g.columns else np.arange(len(g))
        y = g["max_abs_error"] if "max_abs_error" in g.columns else g.get("median_abs_error", pd.Series(np.nan, index=g.index))
        ax.plot(x, y, marker="o", ms=3, lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(model))
    if "n_terms" in q.columns:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("terms")
    ax.set_ylabel("error")
    ax.set_yscale("log")
    ax.set_title(title or "COS Validation")
    _small_legend(ax, ncol=2)
    return ax


def engine_frontier(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No engine table", title or "Engine Frontier")
    q = table.copy()
    x = pd.to_numeric(q.get("runtime_sec", q.get("runtime", 1.0)), errors="coerce")
    y = pd.to_numeric(q.get("median_abs_error", q.get("max_abs_error", q.get("error", 1.0))), errors="coerce")
    labels = q.get("engine", q.get("method", q.get("model", pd.Series("", index=q.index)))).astype(str)
    ax.scatter(x, y, s=55, color=[LAB_COLORS[i % len(LAB_COLORS)] for i in range(len(q))], alpha=0.85)
    for xi, yi, lab in zip(x, y, labels, strict=False):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(lab), (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("seconds")
    ax.set_ylabel("error")
    ax.set_title(title or "Engine Frontier")
    return ax


def daily_calibration_error(ax, daily: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if daily.empty:
        return _quiet_axis(ax, "No daily fits", title or "Daily RMSE")
    q = daily.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    for i, (model, g) in enumerate(q.groupby("model", observed=True)):
        g = g.sort_values("date")
        ax.plot(g["date"], g["weighted_price_rmse"], lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(model))
    ax.set_xlabel("date")
    ax.set_ylabel("weighted RMSE")
    ax.set_title(title or "Daily RMSE")
    _small_legend(ax, ncol=2)
    return ax


def calibration_parameters(ax, daily: pd.DataFrame, title: str | None = None, param: str = "p0"):
    ax = _ax(ax)
    if daily.empty or param not in daily.columns:
        return _quiet_axis(ax, "No parameters", title or "Parameters")
    q = daily.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    for i, (model, g) in enumerate(q.groupby("model", observed=True)):
        ax.plot(g.sort_values("date")["date"], g.sort_values("date")[param], lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(model))
    ax.set_xlabel("date")
    ax.set_ylabel(param)
    ax.set_title(title or "Parameters")
    _small_legend(ax, ncol=2)
    return ax


def model_tournament(ax, comparison: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if comparison.empty:
        return _quiet_axis(ax, "No comparison", title or "Model Tournament")
    q = comparison.sort_values("weighted_price_rmse").copy()
    ax.bar(q["model"].astype(str), q["weighted_price_rmse"], color=LAB_COLORS[0], alpha=0.78, label="RMSE")
    ax.set_ylabel("weighted RMSE")
    if "bid_ask_hit_rate" in q.columns:
        ax2 = ax.twinx()
        ax2.plot(q["model"].astype(str), q["bid_ask_hit_rate"], marker="o", ms=3, color=LAB_COLORS[3], label="hit rate")
        ax2.set_ylabel("hit rate")
        ax2.tick_params(axis="y", labelsize=7)
    ax.set_title(title or "Model Tournament")
    ax.tick_params(axis="x", rotation=15)
    return ax


def residual_smiles(ax, residuals: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if residuals.empty:
        return _quiet_axis(ax, "No residuals", title or "Residual Smiles")
    q = residuals.copy()
    x_col = "x" if "x" in q.columns else "log_moneyness_mid" if "log_moneyness_mid" in q.columns else None
    if x_col is None and "moneyness_bucket" in q.columns:
        q["x"] = np.arange(len(q))
        x_col = "x"
    y_col = "median_scaled_residual" if "median_scaled_residual" in q.columns else "price_residual"
    for i, (model, g) in enumerate(q.groupby("model", observed=True)):
        g = g.sort_values(x_col)
        ax.plot(g[x_col], g[y_col], marker="o", ms=3, lw=1.1, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(model))
        if {"q25_scaled_residual", "q75_scaled_residual"}.issubset(g.columns):
            ax.fill_between(g[x_col].to_numpy(float), g["q25_scaled_residual"].to_numpy(float), g["q75_scaled_residual"].to_numpy(float), color=LAB_COLORS[i % len(LAB_COLORS)], alpha=0.12, lw=0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("scaled residual")
    ax.set_title(title or "Residual Smiles")
    _small_legend(ax, ncol=2)
    return ax


def tail_density(ax, densities: pd.DataFrame, title: str | None = None):
    return density_compare(ax, densities, title or "Tail Density")


def tail_probability_series(ax, table: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if table.empty:
        return _quiet_axis(ax, "No tail series", title or "Tail Probability")
    q = table.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    y_col = "tail_prob_80" if "tail_prob_80" in q.columns else "left_tail_probability"
    for i, (model, g) in enumerate(q.groupby("model", observed=True)):
        ax.plot(g.sort_values("date")["date"], g.sort_values("date")[y_col], lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(model))
    ax.set_xlabel("date")
    ax.set_ylabel("probability")
    ax.set_title(title or "Tail Probability")
    _small_legend(ax, ncol=2)
    return ax


def hedge_nav(ax, results: dict | pd.DataFrame, title: str | None = None):
    return overlay_equity(ax, results, title or "Hedge NAV")


def hedge_selection(ax, schedule: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if schedule.empty:
        return _quiet_axis(ax, "No schedule", title or "Hedge Selection")
    q = schedule.copy()
    q["date"] = pd.to_datetime(q.get("date", q.get("entry_date")), errors="coerce")
    for i, (name, g) in enumerate(q.groupby(q.get("selector", q.get("strategy", pd.Series("hedge", index=q.index))), observed=True)):
        y = pd.to_numeric(g.get("moneyness", g.get("entry_moneyness", np.nan)), errors="coerce")
        ax.plot(g["date"], y, marker="o", ms=2.5, lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(name))
    ax.set_xlabel("date")
    ax.set_ylabel("K / S")
    ax.set_title(title or "Hedge Selection")
    _small_legend(ax, ncol=2)
    return ax


def hedge_cost_payoff(ax, trades: pd.DataFrame, title: str | None = None):
    ax = _ax(ax)
    if trades.empty:
        return _quiet_axis(ax, "No trades", title or "Hedge Cashflow")
    q = trades.copy()
    if "strategy" not in q.columns:
        q["strategy"] = "hedge"
    q["premium_spent"] = np.where(q.get("event", "").astype(str).eq("open"), -pd.to_numeric(q.get("cashflow", 0.0), errors="coerce"), 0.0)
    q["payoff"] = np.where(q.get("event", "").astype(str).isin(["expiry_settlement", "roll_close"]), pd.to_numeric(q.get("cashflow", 0.0), errors="coerce"), 0.0)
    by = q.groupby("strategy")[["premium_spent", "payoff"]].sum()
    by.plot(kind="bar", stacked=False, ax=ax, color=[LAB_COLORS[6], LAB_COLORS[3]], width=0.75)
    ax.set_ylabel("cashflow")
    ax.set_title(title or "Hedge Cashflow")
    ax.tick_params(axis="x", rotation=15)
    _small_legend(ax)
    return ax


__all__ = [
    "benchmark_errors",
    "american_premium_map",
    "american_premium_heatmap",
    "assignment_event_study",
    "assignment_heatmap",
    "bid_ask_hit_rate",
    "boundary_compare",
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
    "model_error_runtime",
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
    "quote_coverage",
    "dividend_timeline",
    "iv_moneyness",
    "sigma_surface",
    "tree_convergence",
    "tree_exercise_map",
    "tree_boundary",
    "pde_value_map",
    "pde_value_surface",
    "pde_value_slices",
    "pde_exercise_map",
    "pde_boundary",
    "pde_residual",
    "pde_residuals",
    "pde_tree_gap_curves",
    "pde_disagreement_bars",
    "complementarity_gap",
    "pricing_error_distribution",
    "pricing_error_spread",
    "method_disagreement_heatmap",
    "lsm_regression",
    "lsm_boundary",
    "lsm_policy_curve",
    "lsm_policy_gap",
    "lsm_regime_coverage",
    "lsm_path_convergence",
    "lsm_exercise_probability",
    "lsm_reference_gap",
    "method_runtime_error",
    "runtime_accuracy",
    "dividend_gap_distribution",
    "assignment_component_bars",
    "boundary_dividend_scatter",
    "overlay_equity",
    "overlay_drawdown",
    "overlay_equity_drawdown",
    "overlay_return_drawdown",
    "premium_close_bars",
    "premium_concentration",
    "premium_term_curves",
    "selected_moneyness",
    "active_legs",
    "strategy_mechanics_bars",
    "trade_timeline",
    "monthly_return_bars",
    "roll_actions",
    "selected_strikes",
    "premium_protection",
    "fourier_quote_coverage",
    "fourier_data_overview",
    "cf_fingerprint",
    "direct_accuracy",
    "fft_validation",
    "fft_fit",
    "cos_validation",
    "engine_frontier",
    "throughput",
    "model_quality",
    "daily_calibration_error",
    "calibration_parameters",
    "model_tournament",
    "residual_smiles",
    "tail_density",
    "tail_probability_series",
    "hedge_nav",
    "hedge_selection",
    "hedge_cost_payoff",
    "cf_shape",
    "direct_price_error",
    "fft_grid",
    "fft_damping",
    "cos_convergence",
    "engine_speed",
    "jump_fit",
    "jump_residuals",
    "sv_fit",
    "sv_residuals",
    "density_compare",
    "tail_probability",
    "hedge_candidates",
    "hedge_equity",
    "hedge_drawdown",
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
