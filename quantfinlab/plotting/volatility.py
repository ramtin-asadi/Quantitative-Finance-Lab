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


def plot_volatility_transfer_grid(
    *,
    spot: pd.Series,
    returns: pd.Series,
    vrp_panel: pd.DataFrame,
    equity: pd.DataFrame,
    selected_model_counts: pd.DataFrame,
    score_pivot: pd.DataFrame,
    summary: pd.DataFrame,
    model_order: Sequence[str] | None = None,
    annualization: float = 252.0,
    figsize: tuple[float, float] = (15.0, 11.0),
) -> tuple[plt.Figure, np.ndarray]:
    """Create the 3x3 volatility forecasting/VRP/overlay transfer plot grid."""
    set_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.ravel()
    plot_spot_and_realized_vol(
        axes[0],
        spot,
        returns,
        annualization=annualization,
        title="Spot and 21d realized vol",
    )
    plot_selected_model_counts_by_horizon(axes[1], selected_model_counts, title="Selected model counts by horizon")
    plot_qlike_heatmap(axes[2], score_pivot, model_order=model_order, title="QLIKE heatmap")
    plot_iv_forecast_vol(axes[3], vrp_panel, title="IV vs forecast vol")
    plot_vrp_variance_spread(axes[4], vrp_panel, title="VRP variance spread")
    plot_vrp_rank_zscore(axes[5], vrp_panel, title="VRP rank/z")
    plot_overlay_nav(axes[6], equity, title="Strategy NAV")
    plot_overlay_drawdowns(axes[7], equity, title="Drawdowns")
    plot_summary_pnl_drawdown_bars(axes[8], summary, title="Total P&L and max drawdown")
    fig.tight_layout(pad=1.2, w_pad=1.2, h_pad=1.4)
    return fig, axes


__all__ = [
    "format_date_axis",
    "plot_iv_forecast_vol",
    "plot_overlay_drawdowns",
    "plot_overlay_nav",
    "plot_qlike_heatmap",
    "plot_selected_model_counts_by_horizon",
    "plot_spot_and_realized_vol",
    "plot_summary_pnl_drawdown_bars",
    "plot_volatility_transfer_grid",
    "plot_vrp_rank_zscore",
    "plot_vrp_variance_spread",
]
