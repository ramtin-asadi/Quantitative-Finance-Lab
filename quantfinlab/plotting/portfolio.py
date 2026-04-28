from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from quantfinlab.plotting.curves import choose_heatmap_cmap, set_plot_style
from quantfinlab.portfolio import selection


def _get_ax(ax=None):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def format_portfolio_time_axis(ax, *, rotation: float = 25.0, date_format: str = "%Y-%m"):
    ax.xaxis.set_major_formatter(DateFormatter(date_format))
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, labelrotation=rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    return ax


def show_portfolio_xaxis_like_risk_module(ax):
    """Keep x tick labels visible on multi-row portfolio panels."""
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax.xaxis.set_ticks_position("bottom")
    return ax


def apply_portfolio_subplot_layout(
    fig,
    axes=None,
    *,
    hspace: float = 0.42,
    wspace: float = 0.28,
    bottom: float = 0.08,
    top: float = 0.94,
):
    """
    Apply the same practical subplot spacing style used by the risk notebook.

    The individual date-axis formatter only rotates and shows labels; this
    figure-level helper owns the vertical spacing so lower-row x-axis labels
    are not clipped or hidden.
    """
    if axes is not None:
        for ax in np.asarray(axes, dtype=object).reshape(-1):
            if hasattr(ax, "tick_params"):
                show_portfolio_xaxis_like_risk_module(ax)
    fig.subplots_adjust(hspace=hspace, wspace=wspace, bottom=bottom, top=top)
    return fig


def _format_date_axis(ax):
    return format_portfolio_time_axis(ax)


def _labels_for_strategies(strategies: Sequence[str], summary: pd.DataFrame | None = None) -> list[str]:
    labels = []
    for name in strategies:
        row = summary.loc[name].to_dict() if summary is not None and name in summary.index else None
        labels.append(selection.strategy_display_label(name, row))
    return labels


def plot_strategy_nav(
    nav: pd.DataFrame,
    strategies: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    summary: pd.DataFrame | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if nav.empty:
        ax.text(0.5, 0.5, "No NAV data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    strategies = list(strategies or nav.columns)
    strategies = [s for s in strategies if s in nav.columns]
    labels_use = list(labels) if labels is not None else _labels_for_strategies(strategies, summary)
    for name, label in zip(strategies, labels_use, strict=False):
        s = nav[name].dropna()
        ax.plot(s.index, s.values, label=label)
    ax.set_title(title or "Strategy NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    if strategies:
        ax.legend(loc="best", fontsize=8)
    return _format_date_axis(ax)


def plot_strategy_drawdowns(
    nav: pd.DataFrame,
    strategies: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    summary: pd.DataFrame | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if nav.empty:
        ax.text(0.5, 0.5, "No NAV data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    strategies = list(strategies or nav.columns)
    strategies = [s for s in strategies if s in nav.columns]
    labels_use = list(labels) if labels is not None else _labels_for_strategies(strategies, summary)
    for name, label in zip(strategies, labels_use, strict=False):
        s = nav[name].dropna()
        dd = selection.calc_drawdown(s)
        ax.plot(dd.index, dd.values, label=label)
    ax.set_title(title or "Strategy Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    if strategies:
        ax.legend(loc="best", fontsize=8)
    return _format_date_axis(ax)


def heatmap_matrix(
    grid_results: pd.DataFrame,
    *,
    optimizer: str,
    metric: str,
    mu_order: Sequence[str] = selection.MU_ORDER,
    cov_order: Sequence[str] = selection.COV_ORDER,
) -> pd.DataFrame:
    df = grid_results[grid_results["Optimizer"].eq(optimizer)].copy()
    df = df[df["Mu model"].isin(mu_order) & df["Covariance model"].isin(cov_order)]
    if df.empty:
        return pd.DataFrame(index=mu_order, columns=cov_order, dtype=float)
    mat = df.pivot_table(index="Mu model", columns="Covariance model", values=metric, aggfunc="mean")
    return mat.reindex(index=mu_order, columns=cov_order)


def _annotate_heatmap(ax, values: np.ndarray, fmt: str):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=10, color="white")


def plot_grid_heatmap(
    grid_results: pd.DataFrame,
    optimizer: str,
    metric: str,
    *,
    ax=None,
    title: str | None = None,
    mu_order: Sequence[str] = selection.MU_ORDER,
    cov_order: Sequence[str] = selection.COV_ORDER,
    annotate: bool = True,
    fmt: str | None = None,
    cmap: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    mat = heatmap_matrix(
        grid_results,
        optimizer=optimizer,
        metric=metric,
        mu_order=mu_order,
        cov_order=cov_order,
    )
    values = mat.to_numpy(dtype=float)
    if values.size == 0 or np.all(~np.isfinite(values)):
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    cmap_use = cmap or choose_heatmap_cmap(metric_name=metric)
    im = ax.imshow(values, aspect="auto", cmap=cmap_use)
    if annotate:
        if fmt is None:
            fmt = ".1f" if metric.lower().endswith("n") or metric == "Effective N" else ".2f"
        _annotate_heatmap(ax, values, fmt)
    ax.set_xticks(range(len(cov_order)))
    ax.set_xticklabels(cov_order, rotation=35, ha="right")
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, pad=2)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Covariance model")
    ax.set_yticks(range(len(mu_order)))
    ax.set_yticklabels(mu_order)
    ax.set_ylabel("Mu model")
    ax.set_title(title or f"{optimizer}: {metric}")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_metric_bar(
    grid_results: pd.DataFrame,
    strategies: Sequence[str],
    *,
    metric: str,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    present = [s for s in strategies if s in grid_results.index]
    if not present:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = grid_results.loc[present, metric].astype(float)
    labels_use = list(labels) if labels is not None else _labels_for_strategies(present, grid_results)
    ax.barh(labels_use, vals.values)
    ax.set_title(title or metric)
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_turnover_bar(grid_results: pd.DataFrame, strategies: Sequence[str], *, ax=None, title: str | None = None):
    return plot_metric_bar(grid_results, strategies, metric="Turnover", ax=ax, title=title or "Turnover")


def plot_effective_n_bar(grid_results: pd.DataFrame, strategies: Sequence[str], *, ax=None, title: str | None = None):
    return plot_metric_bar(grid_results, strategies, metric="Effective N", ax=ax, title=title or "Effective N")


def plot_finalist_metric_bar(
    grid_results: pd.DataFrame,
    strategies: Sequence[str],
    *,
    metric: str = "Sharpe",
    ax=None,
    title: str | None = None,
    baseline: str = "EW",
    include_baseline: bool = True,
):
    strategies_use = (
        selection.with_baseline(strategies, available=grid_results.index, baseline=baseline)
        if include_baseline
        else list(strategies)
    )
    return plot_metric_bar(
        grid_results,
        strategies_use,
        metric=metric,
        ax=ax,
        title=title or f"Finalist {metric}",
    )


def plot_risk_return_scatter(
    grid_results: pd.DataFrame,
    strategies: Sequence[str],
    *,
    ax=None,
    title: str | None = None,
    risk_col: str = "Vol",
    return_col: str = "CAGR",
    color_col: str = "Sharpe",
    annotate: bool = True,
    summary: pd.DataFrame | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    present = [s for s in strategies if s in grid_results.index]
    if not present:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    df = grid_results.loc[present].copy()
    x = df[risk_col].astype(float)
    y = df[return_col].astype(float)
    c = df[color_col].astype(float) if color_col in df.columns else None
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap=choose_heatmap_cmap(metric_name=color_col),
        s=70,
        edgecolor="white",
        linewidth=0.8,
    )
    if c is not None and np.isfinite(c).any():
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_col)
    if annotate:
        label_source = summary if summary is not None else grid_results
        labels = _labels_for_strategies(present, label_source)
        for xi, yi, label in zip(x, y, labels, strict=False):
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_title(title or "Finalist return vs risk")
    ax.set_xlabel(risk_col)
    ax.set_ylabel(return_col)
    ax.grid(True, alpha=0.3)
    return ax


def plot_turnover(turnover: pd.DataFrame | pd.Series, strategies: Sequence[str] | None = None, *, ax=None, title: str | None = None):
    set_plot_style()
    ax = _get_ax(ax)
    if isinstance(turnover, pd.Series):
        turnover.plot(ax=ax)
    else:
        strategies = list(strategies or turnover.columns)
        turnover[[s for s in strategies if s in turnover.columns]].plot(ax=ax)
    ax.set_title(title or "Turnover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    return _format_date_axis(ax)


def plot_weights(weights: pd.DataFrame, *, ax=None, title: str | None = None, top_n: int = 10):
    set_plot_style()
    ax = _get_ax(ax)
    if weights.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    last = weights.iloc[-1].astype(float)
    last = last[last > 0].sort_values(ascending=False).head(int(top_n)).sort_values()
    ax.barh(last.index, last.values)
    ax.set_title(title or "Top Weights")
    ax.set_xlabel("Weight")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_fixed_mu_covariance_comparison(nav: pd.DataFrame, strategies: Sequence[str], *, ax=None, title: str | None = None):
    return plot_strategy_nav(
        nav,
        strategies=strategies,
        ax=ax,
        title=title or "MV covariance comparison: fixed mu = BayesSteinMomentum",
    )


def plot_fixed_cov_mu_comparison(nav: pd.DataFrame, strategies: Sequence[str], *, ax=None, title: str | None = None):
    return plot_strategy_nav(
        nav,
        strategies=strategies,
        ax=ax,
        title=title or "Mu comparison: fixed covariance = EWMA",
    )


plot_nav = plot_strategy_nav
plot_drawdowns = plot_strategy_drawdowns
def plot_finalist_nav(
    nav: pd.DataFrame,
    strategies: Sequence[str],
    *,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    summary: pd.DataFrame | None = None,
    baseline: str = "EW",
    include_baseline: bool = True,
):
    strategies_use = (
        selection.with_baseline(strategies, available=nav.columns, baseline=baseline)
        if include_baseline
        else list(strategies)
    )
    return plot_strategy_nav(nav, strategies_use, ax=ax, title=title, labels=labels, summary=summary)


def plot_finalist_drawdowns(
    nav: pd.DataFrame,
    strategies: Sequence[str],
    *,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    summary: pd.DataFrame | None = None,
    baseline: str = "EW",
    include_baseline: bool = True,
):
    strategies_use = (
        selection.with_baseline(strategies, available=nav.columns, baseline=baseline)
        if include_baseline
        else list(strategies)
    )
    return plot_strategy_drawdowns(nav, strategies_use, ax=ax, title=title, labels=labels, summary=summary)


__all__ = [
    "apply_portfolio_subplot_layout",
    "format_portfolio_time_axis",
    "heatmap_matrix",
    "plot_drawdowns",
    "plot_effective_n_bar",
    "plot_finalist_drawdowns",
    "plot_finalist_metric_bar",
    "plot_finalist_nav",
    "plot_fixed_cov_mu_comparison",
    "plot_fixed_mu_covariance_comparison",
    "plot_grid_heatmap",
    "plot_metric_bar",
    "plot_nav",
    "plot_risk_return_scatter",
    "plot_strategy_drawdowns",
    "plot_strategy_nav",
    "plot_turnover",
    "plot_turnover_bar",
    "plot_weights",
    "show_portfolio_xaxis_like_risk_module",
]
