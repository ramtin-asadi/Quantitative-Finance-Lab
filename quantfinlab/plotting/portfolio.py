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
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=10, color="black")


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


def plot_active_nav(
    returns: pd.DataFrame,
    strategy: str,
    benchmark: str,
    *,
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if returns.empty or strategy not in returns.columns or benchmark not in returns.columns:
        ax.text(0.5, 0.5, "Missing active returns", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    active = returns[strategy].reindex(returns.index).fillna(0.0) - returns[benchmark].reindex(returns.index).fillna(0.0)
    nav = (1.0 + active).cumprod()
    ax.plot(nav.index, nav.values, color="#0072B2", lw=1.4)
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title or f"Active NAV: {strategy} vs {benchmark}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Active growth")
    ax.grid(True, alpha=0.3)
    return _format_date_axis(ax)


def plot_rolling_active_metrics(
    returns: pd.DataFrame,
    strategy: str,
    benchmark: str,
    *,
    window: int = 126,
    metric: str = "active_return",
    annualization: float = 252.0,
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if returns.empty or strategy not in returns.columns or benchmark not in returns.columns:
        ax.text(0.5, 0.5, "Missing active returns", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    active = returns[strategy].reindex(returns.index).fillna(0.0) - returns[benchmark].reindex(returns.index).fillna(0.0)
    if metric.lower() in {"ir", "information_ratio", "rolling_ir"}:
        roll = active.rolling(int(window)).mean() * float(annualization)
        te = active.rolling(int(window)).std() * np.sqrt(float(annualization))
        series = (roll / te.replace(0.0, np.nan)).rename("Rolling IR")
        ylabel = "Information ratio"
    else:
        series = (active.rolling(int(window)).mean() * float(annualization)).rename("Rolling active return")
        ylabel = "Annualized active return"
    ax.plot(series.index, series.values, color="#CC79A7", lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title or series.name)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _format_date_axis(ax)


def plot_active_weights_heatmap(
    weights: pd.DataFrame,
    benchmark_weights: pd.Series | dict[str, float] | None = None,
    *,
    last_n: int = 48,
    ax=None,
    title: str | None = None,
    cmap: str = "coolwarm",
):
    set_plot_style()
    ax = _get_ax(ax)
    if weights.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    W = weights.tail(int(last_n)).copy().astype(float)
    if benchmark_weights is not None:
        bench = pd.Series(benchmark_weights, dtype=float).reindex(W.columns).fillna(0.0)
        W = W.subtract(bench, axis=1)
    W = W.loc[:, W.abs().sum(axis=0).sort_values(ascending=False).index]
    vals = W.T.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if vals.size else 0.0
    vmax = max(float(vmax), 0.01)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title or "Active weights")
    ax.set_yticks(range(len(W.columns)))
    ax.set_yticklabels(W.columns)
    step = max(1, len(W.index) // 8)
    locs = list(range(0, len(W.index), step))
    ax.set_xticks(locs)
    ax.set_xticklabels([pd.Timestamp(W.index[i]).strftime("%Y-%m") for i in locs], rotation=45, ha="right", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_latest_weights(
    weights: pd.DataFrame,
    benchmark_weights: pd.Series | dict[str, float],
    *,
    ax=None,
    title: str | None = None,
    mode: str = "side_by_side",
):
    set_plot_style()
    ax = _get_ax(ax)
    if weights.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    latest = weights.iloc[-1].astype(float)
    bench = pd.Series(benchmark_weights, dtype=float).reindex(latest.index).fillna(0.0)
    order = (latest - bench).abs().sort_values(ascending=True).index
    if mode == "active":
        vals = (latest - bench).reindex(order)
        ax.barh(vals.index, vals.values, color=np.where(vals.values >= 0, "#0072B2", "#D55E00"))
        ax.axvline(0.0, color="black", lw=0.8)
        ax.set_xlabel("Active weight")
    else:
        y = np.arange(len(order))
        ax.barh(y - 0.18, bench.reindex(order).values, height=0.35, label="Benchmark", color="#999999")
        ax.barh(y + 0.18, latest.reindex(order).values, height=0.35, label="BL", color="#0072B2")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("Weight")
    ax.set_title(title or "Latest weights")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_view_selection_counts(selection_log: pd.DataFrame, *, ax=None, title: str | None = None):
    set_plot_style()
    ax = _get_ax(ax)
    if selection_log is None or selection_log.empty:
        ax.text(0.5, 0.5, "No selection log", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    counts = selection_log[selection_log["kept"].astype(bool)].groupby("view_family").size().sort_values()
    ax.barh(counts.index, counts.values, color="#009E73")
    ax.set_title(title or "Selected views by family")
    ax.set_xlabel("Selections")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_view_q_and_confidence(
    view_log: pd.DataFrame,
    confidence_log: pd.DataFrame | None = None,
    *,
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if view_log is None or view_log.empty:
        ax.text(0.5, 0.5, "No view log", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    q_col = "q_tilt_final" if "q_tilt_final" in view_log.columns else "q_tilt"
    q = view_log.groupby("view_family")[q_col].agg(lambda x: np.mean(np.abs(pd.to_numeric(x, errors="coerce")))).sort_values()
    ax.barh(q.index, q.values, color="#0072B2", alpha=0.85, label="Avg abs q")
    ax.set_xlabel("Average absolute q")
    ax.set_title(title or "View q tilt")
    ax.grid(True, axis="x", alpha=0.3)
    if confidence_log is not None and not confidence_log.empty and "confidence" in confidence_log.columns:
        ax2 = ax.twiny()
        conf = confidence_log.groupby("view_family")["confidence"].mean().reindex(q.index)
        ax2.plot(conf.values, range(len(conf.index)), marker="o", color="#D55E00", lw=1.0, label="Confidence")
        ax2.set_xlabel("Avg confidence")
        ax2.set_xlim(0, 1)
    return ax


def plot_stress_summary(
    stress_summary: pd.DataFrame,
    *,
    value_col: str = "return",
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if stress_summary is None or stress_summary.empty:
        ax.text(0.5, 0.5, "No stress summary", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    mat = stress_summary.pivot_table(index="window", columns="strategy", values=value_col, aggfunc="mean")
    mat.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title or "Stress-window performance")
    ax.set_xlabel("")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_posterior_shift_heatmap(
    posterior_mu: pd.DataFrame,
    prior_mu: pd.DataFrame,
    *,
    last_n: int = 36,
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if posterior_mu.empty or prior_mu.empty:
        ax.text(0.5, 0.5, "No posterior shifts", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    shift = posterior_mu.subtract(prior_mu, fill_value=0.0).tail(int(last_n))
    shift = shift.loc[:, shift.abs().sum(axis=0).sort_values(ascending=False).index]
    vals = shift.T.to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(vals))) if vals.size else 0.0, 0.01)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title or "Posterior return shifts")
    ax.set_yticks(range(len(shift.columns)))
    ax.set_yticklabels(shift.columns)
    step = max(1, len(shift.index) // 8)
    locs = list(range(0, len(shift.index), step))
    ax.set_xticks(locs)
    ax.set_xticklabels([pd.Timestamp(shift.index[i]).strftime("%Y-%m") for i in locs], rotation=45, ha="right", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    "plot_active_nav",
    "plot_active_weights_heatmap",
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
    "plot_latest_weights",
    "plot_posterior_shift_heatmap",
    "plot_risk_return_scatter",
    "plot_rolling_active_metrics",
    "plot_stress_summary",
    "plot_strategy_drawdowns",
    "plot_strategy_nav",
    "plot_turnover",
    "plot_turnover_bar",
    "plot_view_q_and_confidence",
    "plot_view_selection_counts",
    "plot_weights",
    "show_portfolio_xaxis_like_risk_module",
]
