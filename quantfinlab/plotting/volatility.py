from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .curves import LAB_COLORS, set_plot_style


def _clean_series(values, *, index=None) -> pd.Series:
    s = pd.Series(values, index=index)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not isinstance(s.index, pd.DatetimeIndex):
        idx = pd.to_datetime(s.index, errors="coerce")
        if idx.notna().all():
            s.index = pd.DatetimeIndex(idx)
    return s.dropna().sort_index()


def _date_values(frame: pd.DataFrame, date_col: str = "date") -> pd.Series:
    if date_col not in frame.columns:
        raise KeyError(f"Missing required date column: {date_col}")
    return pd.to_datetime(frame[date_col], errors="coerce")


def format_date_axis(
    ax: plt.Axes,
    *,
    max_ticks: int = 4,
    labelsize: float = 8.0,
    rotation: float = 0.0,
) -> plt.Axes:
    """Use compact date ticks that fit inside small subplot panels."""
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max(3, int(max_ticks)), interval_multiples=True)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=labelsize, rotation=rotation, pad=2)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right" if rotation else "center")
    ax.margins(x=0.01)
    return ax


def plot_spot_and_realized_vol(
    ax: plt.Axes,
    spot: pd.Series,
    returns: pd.Series,
    *,
    annualization: float = 252.0,
    vol_window: int = 21,
    title: str = "Spot and realized volatility",
) -> plt.Axes:
    """Plot underlying spot with a secondary-axis rolling realized-vol overlay."""
    set_plot_style()
    spot_s = _clean_series(spot)
    ret_s = _clean_series(returns)
    ax.plot(spot_s.index, spot_s.values, lw=1.0, color=LAB_COLORS[0], label="spot")
    ax.set_ylabel("Spot")
    ax_vol = ax.twinx()
    rv = ret_s.rolling(int(vol_window)).std(ddof=1) * np.sqrt(float(annualization))
    ax_vol.plot(rv.index, rv.values, lw=0.85, alpha=0.7, color=LAB_COLORS[3], label=f"{vol_window}d RV")
    ax_vol.set_ylabel("Ann. vol")
    ax.set_title(title)
    format_date_axis(ax)
    return ax


def plot_iv_forecast_vol(
    ax: plt.Axes,
    vrp_panel: pd.DataFrame,
    *,
    date_col: str = "date",
    iv_col: str = "atm_iv_mid",
    forecast_col: str = "forecast_vol_ann",
    realized_col: str | None = None,
    title: str = "IV vs forecast volatility",
) -> plt.Axes:
    """Plot ATM implied volatility against matched model forecast volatility."""
    set_plot_style()
    if vrp_panel is None or vrp_panel.empty:
        ax.text(0.5, 0.5, "No volatility panel", ha="center", va="center")
        ax.axis("off")
        return ax
    dates = _date_values(vrp_panel, date_col)
    if iv_col in vrp_panel.columns:
        ax.plot(dates, pd.to_numeric(vrp_panel[iv_col], errors="coerce"), lw=1.0, label="ATM IV")
    if forecast_col in vrp_panel.columns:
        ax.plot(dates, pd.to_numeric(vrp_panel[forecast_col], errors="coerce"), lw=1.0, label="forecast vol")
    if realized_col and realized_col in vrp_panel.columns:
        ax.plot(
            dates,
            pd.to_numeric(vrp_panel[realized_col], errors="coerce"),
            lw=0.9,
            alpha=0.75,
            label="future realized vol",
        )
    ax.set_title(title)
    ax.set_ylabel("Ann. vol")
    ax.legend(loc="best")
    format_date_axis(ax)
    return ax


def plot_vrp_variance_spread(
    ax: plt.Axes,
    vrp_panel: pd.DataFrame,
    *,
    date_col: str = "date",
    vrp_col: str = "vrp_var",
    title: str = "VRP variance spread",
) -> plt.Axes:
    """Plot variance risk premium, defined as IV variance minus forecast variance."""
    set_plot_style()
    if vrp_panel is None or vrp_panel.empty or vrp_col not in vrp_panel.columns:
        ax.text(0.5, 0.5, "No VRP data", ha="center", va="center")
        ax.axis("off")
        return ax
    dates = _date_values(vrp_panel, date_col)
    ax.plot(dates, pd.to_numeric(vrp_panel[vrp_col], errors="coerce"), lw=1.0, color=LAB_COLORS[2])
    ax.axhline(0.0, color="#222222", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Variance spread")
    format_date_axis(ax)
    return ax


def plot_vrp_rank_zscore(
    ax: plt.Axes,
    vrp_panel: pd.DataFrame,
    *,
    date_col: str = "date",
    rank_col: str = "vrp_rank",
    z_col: str = "vrp_z",
    title: str = "VRP rank and z-score",
) -> plt.Axes:
    """Plot rolling VRP rank and z-score on one panel."""
    set_plot_style()
    if vrp_panel is None or vrp_panel.empty:
        ax.text(0.5, 0.5, "No VRP data", ha="center", va="center")
        ax.axis("off")
        return ax
    dates = _date_values(vrp_panel, date_col)
    if rank_col in vrp_panel.columns:
        ax.plot(dates, pd.to_numeric(vrp_panel[rank_col], errors="coerce"), lw=1.0, label="rank")
    if z_col in vrp_panel.columns:
        ax.plot(dates, pd.to_numeric(vrp_panel[z_col], errors="coerce"), lw=0.9, alpha=0.8, label="z")
    ax.set_title(title)
    ax.legend(loc="best")
    format_date_axis(ax)
    return ax


def plot_overlay_nav(
    ax: plt.Axes,
    equity: pd.DataFrame,
    *,
    date_col: str = "date",
    strategy_col: str = "strategy",
    nav_col: str = "nav",
    title: str = "Strategy NAV",
) -> plt.Axes:
    """Plot NAV paths from an overlay equity table."""
    set_plot_style()
    if equity is None or equity.empty:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center")
        ax.axis("off")
        return ax
    for i, (name, group) in enumerate(equity.groupby(strategy_col, sort=True)):
        dates = _date_values(group, date_col)
        ax.plot(dates, pd.to_numeric(group[nav_col], errors="coerce"), lw=1.0, label=str(name), color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.legend(loc="best")
    format_date_axis(ax)
    return ax


def plot_overlay_drawdowns(
    ax: plt.Axes,
    equity: pd.DataFrame,
    *,
    date_col: str = "date",
    strategy_col: str = "strategy",
    drawdown_col: str = "drawdown",
    title: str = "Drawdowns",
) -> plt.Axes:
    """Plot drawdowns from an overlay equity table."""
    set_plot_style()
    if equity is None or equity.empty:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center")
        ax.axis("off")
        return ax
    for i, (name, group) in enumerate(equity.groupby(strategy_col, sort=True)):
        dates = _date_values(group, date_col)
        ax.plot(
            dates,
            pd.to_numeric(group[drawdown_col], errors="coerce"),
            lw=1.0,
            label=str(name),
            color=LAB_COLORS[i % len(LAB_COLORS)],
        )
    ax.axhline(0.0, color="#222222", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    format_date_axis(ax)
    return ax


def plot_selected_model_counts_by_horizon(
    ax: plt.Axes,
    selected_model_counts: pd.DataFrame,
    *,
    model_col: str = "selected_model",
    horizon_col: str = "horizon",
    count_col: str = "n",
    title: str = "Selected model counts by horizon",
) -> plt.Axes:
    """Stacked bar chart of selected forecast model counts by horizon."""
    set_plot_style()
    if selected_model_counts is None or selected_model_counts.empty:
        ax.text(0.5, 0.5, "No selection counts", ha="center", va="center")
        ax.axis("off")
        return ax
    counts = (
        selected_model_counts.pivot(index=model_col, columns=horizon_col, values=count_col)
        .fillna(0.0)
        .sort_index()
    )
    bottom = np.zeros(len(counts.columns), dtype=float)
    x = [str(h) for h in counts.columns]
    for i, (model_name, row) in enumerate(counts.iterrows()):
        values = row.to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=str(model_name), color=LAB_COLORS[i % len(LAB_COLORS)])
        bottom += values
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return ax


def plot_qlike_heatmap(
    ax: plt.Axes,
    score_pivot: pd.DataFrame,
    *,
    model_order: Sequence[str] | None = None,
    title: str = "QLIKE heatmap",
) -> plt.Axes:
    """Heatmap of forecast QLIKE scores by model and horizon."""
    set_plot_style()
    if score_pivot is None or score_pivot.empty:
        ax.text(0.5, 0.5, "No QLIKE scores", ha="center", va="center")
        ax.axis("off")
        return ax
    heat = score_pivot.copy()
    if model_order is not None:
        heat = heat.reindex([m for m in model_order if m in heat.index])
    heat = heat.dropna(how="all")
    if heat.empty:
        ax.text(0.5, 0.5, "No QLIKE scores", ha="center", va="center")
        ax.axis("off")
        return ax
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([str(c) for c in heat.columns])
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([str(i) for i in heat.index], fontsize=7)
    ax.set_xlabel("Horizon")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_summary_pnl_drawdown_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    strategy_col: str = "strategy",
    pnl_col: str = "total_net_pnl",
    drawdown_col: str = "max_drawdown",
    title: str = "Total P&L and max drawdown",
) -> plt.Axes:
    """Side-by-side bar chart for strategy total P&L and max drawdown."""
    set_plot_style()
    if summary is None or summary.empty:
        ax.text(0.5, 0.5, "No summary data", ha="center", va="center")
        ax.axis("off")
        return ax
    tbl = summary.reset_index(drop=True)
    x = np.arange(len(tbl))
    ax.bar(x - 0.18, pd.to_numeric(tbl[pnl_col], errors="coerce"), width=0.36, label="total P&L", color=LAB_COLORS[0])
    ax.bar(
        x + 0.18,
        pd.to_numeric(tbl[drawdown_col], errors="coerce"),
        width=0.36,
        label="max drawdown",
        color=LAB_COLORS[3],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tbl[strategy_col].astype(str), rotation=45, ha="right")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def _rough_quiet(ax: plt.Axes, text: str, title: str | None = None) -> plt.Axes:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title or text)
    return ax


def _rough_legend(ax: plt.Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, frameon=True, framealpha=0.88, **kwargs)


def price_and_variance(ax: plt.Axes, spot, variance, title: str | None = None) -> plt.Axes:
    s = spot["close"] if isinstance(spot, pd.DataFrame) and "close" in spot.columns else pd.Series(spot)
    v = pd.Series(variance)
    if s.empty or v.empty:
        return _rough_quiet(ax, "No price/variance data", title)
    ax.plot(s.index, s, lw=1.2, color=LAB_COLORS[0], label="price")
    ax2 = ax.twinx()
    ax2.plot(v.index, np.sqrt(pd.to_numeric(v, errors="coerce").clip(lower=0.0)), lw=0.9, color=LAB_COLORS[1], alpha=0.75, label="sqrt variance")
    ax.set_ylabel("price")
    ax2.set_ylabel("volatility")
    ax.set_title(title or "Price and realized variance")
    _rough_legend(ax, loc="upper left")
    _rough_legend(ax2, loc="upper right")
    return ax


def fbm_roughness(ax: plt.Axes, fbm: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if fbm.empty:
        return _rough_quiet(ax, "No fBM paths", title)
    for i, (h, g) in enumerate(fbm.groupby("h", sort=True)):
        one = g[g["path"].eq(g["path"].min())]
        ax.plot(one["t"], one["x"], lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"H={float(h):.2f}")
    ax.set_xlabel("time")
    ax.set_ylabel("path")
    ax.set_title(title or "Fractional Brownian roughness")
    _rough_legend(ax, ncol=2)
    return ax


def moment_ladder(ax: plt.Axes, scaling: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if scaling.empty:
        return _rough_quiet(ax, "No scaling data", title)
    q = scaling.copy()
    q["log_lag"] = np.log(pd.to_numeric(q.get("lag"), errors="coerce"))
    q["log_moment"] = np.log(pd.to_numeric(q.get("moment"), errors="coerce"))
    for i, (order, g) in enumerate(q.dropna().groupby("q", sort=True)):
        ax.plot(g["log_lag"], g["log_moment"], marker="o", ms=3, lw=1.1, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"q={float(order):g}")
    ax.set_xlabel("log lag")
    ax.set_ylabel("log moment")
    ax.set_title(title or "Moment scaling")
    _rough_legend(ax, ncol=2)
    return ax


def hurst_linearity(ax: plt.Axes, hurst: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if hurst.empty:
        return _rough_quiet(ax, "No H estimates", title)
    q = hurst.copy()
    ax.scatter(q["q"], q["slope"], s=35, color=LAB_COLORS[0], label="estimated slopes")
    h = float(pd.to_numeric(q["h"], errors="coerce").median())
    x = np.linspace(0.0, float(q["q"].max()) * 1.05, 100)
    ax.plot(x, h * x, color=LAB_COLORS[1], lw=1.5, label=f"median H={h:.3f}")
    ax.set_xlabel("q")
    ax.set_ylabel("slope")
    ax.set_title(title or "qH linearity")
    _rough_legend(ax)
    return ax


def roughness_clock(ax: plt.Axes, history: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if history.empty:
        return _rough_quiet(ax, "No roughness history", title)
    date_col = "date" if "date" in history.columns else history.index.name
    x = history[date_col] if date_col in history.columns else history.index
    y = history["h"] if "h" in history.columns else history.select_dtypes(include=[np.number]).iloc[:, 0]
    ax.plot(pd.to_datetime(x), y, lw=1.2, color=LAB_COLORS[2])
    ax.axhline(pd.to_numeric(y, errors="coerce").median(), color=LAB_COLORS[1], lw=1.0, ls="--")
    ax.set_ylabel("H")
    ax.set_title(title or "Roughness through time")
    return ax


def forecast_race(ax: plt.Axes, scores: pd.DataFrame, metric: str = "qlike_var", title: str | None = None) -> plt.Axes:
    if scores.empty or metric not in scores.columns:
        return _rough_quiet(ax, "No forecast scores", title)
    q = scores.copy()
    piv = q.pivot_table(index="model", columns="horizon", values=metric, aggfunc="mean")
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.astype(str))
    ax.set_xlabel("horizon")
    ax.set_title(title or "Forecast race")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def forecast_path(ax: plt.Axes, forecast_data: pd.DataFrame, model: str = "rough_kernel", horizon: int = 21, title: str | None = None) -> plt.Axes:
    q = forecast_data.copy()
    q = q[(q.get("model").astype(str).eq(model)) & (pd.to_numeric(q.get("horizon"), errors="coerce").eq(int(horizon)))]
    if q.empty:
        return _rough_quiet(ax, "No forecast path", title)
    ax.plot(q["date"], q["forecast_vol_ann"], lw=1.2, label="forecast")
    if "realized_vol_ann" in q.columns:
        ax.plot(q["date"], q["realized_vol_ann"], lw=1.0, alpha=0.8, label="realized")
    ax.set_ylabel("annualized vol")
    ax.set_title(title or f"{horizon}-day forecast")
    _rough_legend(ax)
    return ax


def smile_slices(ax: plt.Axes, quotes: pd.DataFrame, fit: dict | None = None, maturities_days=(7, 14, 30, 60, 90), title: str | None = None) -> plt.Axes:
    if quotes.empty:
        return _rough_quiet(ax, "No smile data", title)
    q = quotes.copy()
    if "k" not in q.columns:
        q["k"] = np.log(q["strike"] / q.get("forward", q["spot"]))
    for i, d in enumerate(maturities_days):
        tau = float(d) / 365.25
        if fit is not None:
            from quantfinlab.options.surface import surface_iv

            k = np.linspace(float(q["k"].quantile(0.05)), float(q["k"].quantile(0.95)), 80)
            sigma = surface_iv(fit, k, np.full_like(k, tau))
            ax.plot(k, sigma, lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{d}d")
        else:
            g = q[np.isclose(pd.to_numeric(q["tau"], errors="coerce"), tau, atol=max(3 / 365.25, 0.04 * tau))].sort_values("k")
            if len(g) >= 3:
                ax.plot(g["k"], g["iv_mid"], marker="o", ms=2.5, lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{d}d")
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "Smile slices")
    _rough_legend(ax, ncol=2)
    return ax


def smooth_surface(ax: plt.Axes, grid: dict, sigma_grid, quotes: pd.DataFrame | None = None, title: str | None = None) -> plt.Axes:
    z = np.asarray(sigma_grid, dtype=float)
    if z.size == 0:
        return _rough_quiet(ax, "No surface grid", title)
    im = ax.imshow(z, origin="lower", aspect="auto", extent=[min(grid["k"]), max(grid["k"]), min(grid["tau"]) * 365.25, max(grid["tau"]) * 365.25], cmap="magma")
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("days")
    ax.set_title(title or "Smooth IV surface")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def skew_power(ax: plt.Axes, psi: pd.DataFrame, skew_fit: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if psi.empty:
        return _rough_quiet(ax, "No skew data", title)
    x = pd.to_numeric(psi["tau"], errors="coerce")
    y = pd.to_numeric(psi["atm_skew"], errors="coerce").abs()
    ax.scatter(x, y, s=28, color=LAB_COLORS[0], label="ATM skew")
    if skew_fit is not None and not skew_fit.empty and np.isfinite(skew_fit.iloc[0].get("c", np.nan)):
        c = float(skew_fit.iloc[0]["c"])
        alpha = float(skew_fit.iloc[0]["alpha"])
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(xs, c * xs ** (-alpha), color=LAB_COLORS[1], lw=1.4, label=f"alpha={alpha:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tau")
    ax.set_ylabel("|ATM skew|")
    ax.set_title(title or "ATM skew power law")
    _rough_legend(ax)
    return ax


def skew_history(ax: plt.Axes, table: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if table.empty:
        return _rough_quiet(ax, "No skew history", title)
    ax.plot(pd.to_datetime(table["date"]), table["h"], lw=1.2, color=LAB_COLORS[0])
    ax.axhline(pd.to_numeric(table["h"], errors="coerce").median(), color=LAB_COLORS[1], ls="--", lw=1.0)
    ax.set_ylabel("H from skew")
    ax.set_title(title or "Skew-implied roughness")
    return ax


def fit_comparison(ax: plt.Axes, heston_fit: pd.DataFrame, rough_fit: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if (heston_fit is None or heston_fit.empty) and (rough_fit is None or rough_fit.empty):
        return _rough_quiet(ax, "No model fits", title)
    frames = []
    if heston_fit is not None and not heston_fit.empty:
        frames.append(heston_fit.assign(model_label="heston"))
    if rough_fit is not None and not rough_fit.empty:
        frames.append(rough_fit.assign(model_label="rough_heston"))
    q = pd.concat(frames, ignore_index=True)
    if "k" not in q.columns:
        q["k"] = np.log(q["strike"] / q["spot"])
    use_iv = {"iv_mid", "vega", "price_residual"}.issubset(q.columns)
    market = q.drop_duplicates([c for c in ["date", "expiry", "strike", "option_type"] if c in q.columns])
    if use_iv:
        q["model_iv"] = pd.to_numeric(q["iv_mid"], errors="coerce") + pd.to_numeric(q["price_residual"], errors="coerce") / pd.to_numeric(q["vega"], errors="coerce").replace(0.0, np.nan)
        ax.scatter(market["k"], market["iv_mid"], s=10, alpha=0.35, color="black", label="market")
        y_col = "model_iv"
        y_label = "implied vol"
    else:
        ax.scatter(market["k"], market["mid"], s=10, alpha=0.35, color="black", label="market")
        y_col = "model_price"
        y_label = "price"
    for i, (name, g) in enumerate(q.groupby("model_label", sort=True)):
        by = g.sort_values("k")
        ax.plot(by["k"], by[y_col], lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=str(name))
    ax.set_xlabel("log moneyness")
    ax.set_ylabel(y_label)
    ax.set_title(title or "Model fit comparison")
    _rough_legend(ax)
    return ax


def maturity_error(ax: plt.Axes, heston_buckets: pd.DataFrame, rough_buckets: pd.DataFrame, title: str | None = None) -> plt.Axes:
    frames = []
    for name, frame in [("heston", heston_buckets), ("rough_heston", rough_buckets)]:
        if frame is not None and not frame.empty:
            frames.append(frame.assign(model=name))
    if not frames:
        return _rough_quiet(ax, "No residual buckets", title)
    q = pd.concat(frames, ignore_index=True)
    by = q.groupby("model")["median_scaled_residual"].apply(lambda x: float(np.nanmean(np.abs(x)))).sort_values()
    ax.bar(by.index.astype(str), by.values, color=LAB_COLORS[: len(by)])
    ax.set_ylabel("mean |scaled residual|")
    ax.set_title(title or "Error by maturity and moneyness")
    ax.tick_params(axis="x", rotation=15)
    return ax


def rbergomi_paths(ax: plt.Axes, sim: dict | pd.DataFrame, title: str | None = None) -> plt.Axes:
    if not isinstance(sim, dict) or "variance" not in sim:
        return _rough_quiet(ax, "No rBergomi paths", title)
    t = np.asarray(sim.get("time"))
    v = np.asarray(sim["variance"])
    n = min(20, v.shape[0])
    for i in range(n):
        ax.plot(t, v[i], lw=0.6, alpha=0.45, color=LAB_COLORS[i % len(LAB_COLORS)])
    ax.set_xlabel("time")
    ax.set_ylabel("variance")
    ax.set_title(title or "rBergomi variance paths")
    return ax


def rough_smile_map(ax: plt.Axes, smile: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if smile.empty:
        return _rough_quiet(ax, "No rough smile", title)
    piv = smile.pivot_table(index="tau_days", columns="k", values="iv", aggfunc="mean")
    im = ax.imshow(piv.to_numpy(dtype=float), origin="lower", aspect="auto", extent=[float(piv.columns.min()), float(piv.columns.max()), float(piv.index.min()), float(piv.index.max())], cmap="viridis")
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("days")
    ax.set_title(title or "Rough smile map")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def rbergomi_fit(ax: plt.Axes, rbergomi_iv: pd.DataFrame, quotes: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if rbergomi_iv.empty:
        return _rough_quiet(ax, "No rBergomi fit", title)
    for i, (d, g) in enumerate(rbergomi_iv.groupby("tau_days", sort=True)):
        ax.plot(g["k"], g["iv"], lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{float(d):.0f}d")
    if quotes is not None and not quotes.empty and "iv_mid" in quotes.columns:
        q = quotes.copy()
        if "k" not in q.columns:
            q["k"] = np.log(q["strike"] / q.get("forward", q["spot"]))
        ax.scatter(q["k"], q["iv_mid"], s=8, color="black", alpha=0.25, label="market")
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("implied vol")
    ax.set_title(title or "rBergomi smile fit")
    _rough_legend(ax, ncol=2)
    return ax


def riccati_stability(ax: plt.Axes, table: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if table.empty:
        return _rough_quiet(ax, "No Riccati data", title)
    ax.plot(table["riccati_steps"], table["price"], marker="o", lw=1.2, color=LAB_COLORS[0])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("steps")
    ax.set_ylabel("price")
    ax.set_title(title or "Fractional Riccati stability")
    return ax


def parameter_paths(ax: plt.Axes, params: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if params.empty:
        return _rough_quiet(ax, "No parameter paths", title)
    q = params.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce")
    cols = [c for c in ["h", "p0", "p1", "p2", "p3", "p4", "rho"] if c in q.columns]
    for i, col in enumerate(cols[:5]):
        ax.plot(q["date"], q[col], lw=1.0, color=LAB_COLORS[i % len(LAB_COLORS)], label=col)
    ax.set_title(title or "Parameter paths")
    _rough_legend(ax, ncol=2)
    return ax


def delta_slices(ax: plt.Axes, delta_map: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if delta_map.empty:
        return _rough_quiet(ax, "No delta map", title)
    for i, (d, g) in enumerate(delta_map.groupby("tau_days", sort=True)):
        g = g.sort_values("k")
        ax.plot(g["k"], g["rough_heston_delta"], lw=1.2, color=LAB_COLORS[i % len(LAB_COLORS)], label=f"{float(d):.0f}d")
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("delta")
    ax.set_title(title or "Rough-Heston delta slices")
    _rough_legend(ax, ncol=2)
    return ax


def delta_gap(ax: plt.Axes, delta_map: pd.DataFrame, base_col: str = "bsm_delta", title: str | None = None) -> plt.Axes:
    if delta_map.empty:
        return _rough_quiet(ax, "No delta gaps", title)
    q = delta_map.copy()
    gap = q["rough_heston_delta"] - q[base_col]
    piv = q.assign(gap=gap).pivot_table(index="tau_days", columns="k", values="gap", aggfunc="mean")
    im = ax.imshow(piv.to_numpy(dtype=float), origin="lower", aspect="auto", extent=[float(piv.columns.min()), float(piv.columns.max()), float(piv.index.min()), float(piv.index.max())], cmap="coolwarm")
    ax.set_xlabel("log moneyness")
    ax.set_ylabel("days")
    ax.set_title(title or "Rough-model delta gap")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def rough_dashboard_summary(ax: plt.Axes, summary: pd.DataFrame, title: str | None = None) -> plt.Axes:
    if summary.empty:
        return _rough_quiet(ax, "No summary", title)
    ax.axis("off")
    text = summary.head(8).round(4).to_string(index=False)
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=8, transform=ax.transAxes)
    ax.set_title(title or "Rough-volatility summary")
    return ax


__all__ = [
    "delta_gap",
    "delta_slices",
    "fbm_roughness",
    "fit_comparison",
    "format_date_axis",
    "forecast_path",
    "forecast_race",
    "hurst_linearity",
    "maturity_error",
    "moment_ladder",
    "parameter_paths",
    "plot_iv_forecast_vol",
    "plot_overlay_drawdowns",
    "plot_overlay_nav",
    "plot_qlike_heatmap",
    "plot_selected_model_counts_by_horizon",
    "plot_spot_and_realized_vol",
    "plot_summary_pnl_drawdown_bars",
    "plot_vrp_rank_zscore",
    "plot_vrp_variance_spread",
    "price_and_variance",
    "rbergomi_fit",
    "rbergomi_paths",
    "riccati_stability",
    "rough_dashboard_summary",
    "rough_smile_map",
    "roughness_clock",
    "skew_history",
    "skew_power",
    "smile_slices",
    "smooth_surface",
]
