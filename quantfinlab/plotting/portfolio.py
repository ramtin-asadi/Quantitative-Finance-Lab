from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from quantfinlab.plotting.curves import LAB_COLORS, choose_heatmap_cmap, set_plot_style
from quantfinlab.portfolio import selection
from quantfinlab.risk.utils import _excess_returns


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


def _add_inset_colorbar(im, ax, *, label: str | None = None):
    cax = ax.inset_axes([0.965, 0.12, 0.025, 0.76])
    cbar = ax.figure.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    return cbar


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
    apply_style: bool = True,
):
    if apply_style:
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
    apply_style: bool = True,
):
    if apply_style:
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
    _add_inset_colorbar(im, ax)
    return ax


def plot_q_tilt_heatmap(
    view_log: pd.DataFrame,
    *,
    last_n: int = 48,
    ax=None,
    title: str | None = None,
    q_col: str | None = None,
    date_col: str = "date",
    group_col: str | None = None,
    cmap: str = "viridis",
):
    set_plot_style()
    ax = _get_ax(ax)
    if view_log is None or view_log.empty:
        ax.text(0.5, 0.5, "No view log", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    data = view_log.copy()
    q_candidates = ["q_tilt_final", "q_tilt", "q", "final_q"]
    q_col = q_col or next((col for col in q_candidates if col in data.columns), None)
    if q_col is None or date_col not in data.columns:
        ax.text(0.5, 0.5, "Missing q tilt data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    group_candidates = ["family_display_name", "view_family", "view_id", "view"]
    group_col = group_col or next((col for col in group_candidates if col in data.columns), None)
    if group_col is None:
        ax.text(0.5, 0.5, "Missing view labels", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[q_col] = pd.to_numeric(data[q_col], errors="coerce")
    data = data.dropna(subset=[date_col, group_col, q_col])
    if data.empty:
        ax.text(0.5, 0.5, "No q tilt data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    q_map = data.pivot_table(index=group_col, columns=date_col, values=q_col, aggfunc="mean")
    q_map = q_map.reindex(q_map.abs().sum(axis=1).sort_values(ascending=False).index)
    q_map = q_map.iloc[:, -int(last_n):] if int(last_n) > 0 else q_map
    vals = q_map.to_numpy(dtype=float)

    im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_title(title or "View q tilt over time")
    ax.set_yticks(range(len(q_map.index)))
    ax.set_yticklabels(q_map.index)

    step = max(1, len(q_map.columns) // 8)
    locs = list(range(0, len(q_map.columns), step))
    ax.set_xticks(locs)
    ax.set_xticklabels(
        [pd.Timestamp(q_map.columns[i]).strftime("%Y-%m") for i in locs],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    _add_inset_colorbar(im, ax, label="q tilt")
    return ax


def plot_view_count_timeline(
    candidate_log: pd.DataFrame,
    selected_log: pd.DataFrame | None = None,
    *,
    ax=None,
    title: str | None = None,
    date_col: str = "date",
):
    set_plot_style()
    ax = _get_ax(ax)
    if candidate_log is None or candidate_log.empty or date_col not in candidate_log.columns:
        ax.text(0.5, 0.5, "No view counts", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    candidates = candidate_log.copy()
    candidates[date_col] = pd.to_datetime(candidates[date_col], errors="coerce")
    candidate_counts = candidates.dropna(subset=[date_col]).groupby(date_col).size()

    if selected_log is not None and not selected_log.empty and date_col in selected_log.columns:
        selected = selected_log.copy()
        selected[date_col] = pd.to_datetime(selected[date_col], errors="coerce")
        selected_counts = selected.dropna(subset=[date_col]).groupby(date_col).size()
    elif "kept" in candidates.columns:
        kept = candidates["kept"].astype(bool)
        selected_counts = candidates.loc[kept].dropna(subset=[date_col]).groupby(date_col).size()
    else:
        selected_counts = pd.Series(dtype=float)

    counts = pd.concat(
        {
            "Candidate views": candidate_counts,
            "Selected views": selected_counts,
        },
        axis=1,
    ).fillna(0.0)

    ax.plot(counts.index, counts["Candidate views"], color=LAB_COLORS[0], lw=1.4, label="Candidate")
    ax.plot(counts.index, counts["Selected views"], color=LAB_COLORS[1], lw=1.4, label="Selected")
    ax.set_title(title or "Candidate and selected views")
    ax.set_xlabel("Date")
    ax.set_ylabel("View count")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _format_date_axis(ax)


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
    ax.barh(counts.index, counts.values, color=LAB_COLORS[2])
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
    group_col = "family_display_name" if "family_display_name" in view_log.columns else "view_family"
    q = (
        view_log.groupby(group_col)[q_col]
        .agg(lambda x: np.mean(np.abs(pd.to_numeric(x, errors="coerce"))))
        .sort_values()
    )
    ax.barh(q.index, q.values, color=LAB_COLORS[0], alpha=0.85, label="Avg abs q")
    ax.set_xlabel("Average absolute q")
    ax.set_title(title or "View q tilt")
    ax.grid(True, axis="x", alpha=0.3)
    if (
        confidence_log is not None
        and not confidence_log.empty
        and "confidence" in confidence_log.columns
    ):
        conf_data = confidence_log.copy()
        if group_col in conf_data.columns:
            conf_group_col = group_col
        elif (
            group_col == "family_display_name"
            and "view_family" in conf_data.columns
            and "view_family" in view_log.columns
        ):
            label_map = (
                view_log[["view_family", "family_display_name"]]
                .dropna()
                .drop_duplicates("view_family")
                .set_index("view_family")["family_display_name"]
            )
            conf_data["_display_group"] = conf_data["view_family"].map(label_map).fillna(conf_data["view_family"])
            conf_group_col = "_display_group"
        elif "view_family" in conf_data.columns:
            conf_group_col = "view_family"
        else:
            conf_group_col = None
        if conf_group_col is None:
            return ax
        ax2 = ax.twiny()
        conf = conf_data.groupby(conf_group_col)["confidence"].mean().reindex(q.index)
        ax2.plot(conf.values, range(len(conf.index)), marker="o", color=LAB_COLORS[1], lw=1.0, label="Confidence")
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", top=False, labeltop=False, length=0)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, loc="lower right", fontsize=7, frameon=False)
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


def _short_names(names: Sequence[str], short_labels: Mapping[str, str] | None = None, labels: Sequence[str] | None = None):
    if labels is not None:
        return list(labels)
    if short_labels is None:
        return list(names)
    return [short_labels.get(name, name) for name in names]


def _small_legend(ax, *, ncol: int = 2):
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=7, ncol=ncol, frameon=True, framealpha=0.85)
    return ax


def _average_weight_matrix(weights_by_name: Mapping[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, weights in weights_by_name.items():
        if isinstance(weights, pd.DataFrame):
            rows[name] = weights.astype(float).mean(axis=0)
        else:
            rows[name] = pd.Series(weights, dtype=float)
    return pd.DataFrame(rows).fillna(0.0)


def plot_finalist_nav(
    nav: pd.DataFrame,
    strategies: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    summary: pd.DataFrame | None = None,
    baseline: str = "EW",
    include_baseline: bool = True,
    short_labels: Mapping[str, str] | None = None,
):
    strategies_use = (
        selection.with_baseline(strategies, available=nav.columns, baseline=baseline)
        if strategies is not None and include_baseline
        else list(strategies or nav.columns)
    )
    labels_use = _short_names(strategies_use, short_labels, labels)
    ax = plot_strategy_nav(nav, strategies_use, ax=ax, title=title, labels=labels_use, summary=summary)
    return _small_legend(ax)


def plot_finalist_drawdowns(
    nav: pd.DataFrame,
    strategies: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    summary: pd.DataFrame | None = None,
    baseline: str = "EW",
    include_baseline: bool = True,
    short_labels: Mapping[str, str] | None = None,
):
    strategies_use = (
        selection.with_baseline(strategies, available=nav.columns, baseline=baseline)
        if strategies is not None and include_baseline
        else list(strategies or nav.columns)
    )
    labels_use = _short_names(strategies_use, short_labels, labels)
    ax = plot_strategy_drawdowns(nav, strategies_use, ax=ax, title=title, labels=labels_use, summary=summary)
    return _small_legend(ax)


def plot_finalist_metric_bar(
    grid_results: pd.DataFrame,
    strategies: Sequence[str] | None = None,
    *,
    metric: str = "Sharpe",
    ax=None,
    title: str | None = None,
    baseline: str = "EW",
    include_baseline: bool = True,
    short_labels: Mapping[str, str] | None = None,
    labels: Sequence[str] | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    strategies_use = list(strategies or grid_results.index)
    if strategies is not None and include_baseline:
        strategies_use = selection.with_baseline(strategies_use, available=grid_results.index, baseline=baseline)
    present = [s for s in strategies_use if s in grid_results.index and metric in grid_results.columns]
    if not present:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = grid_results.loc[present, metric].astype(float)
    names = _short_names(present, short_labels, labels)
    ax.barh(names, vals.values)
    ax.set_title(title or metric)
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_risk_return_scatter(
    grid_results: pd.DataFrame,
    strategies: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
    risk_col: str = "Vol",
    return_col: str = "CAGR",
    color_col: str | None = "Sharpe",
    size_col: str | None = None,
    short_labels: Mapping[str, str] | None = None,
    apply_style: bool = True,
):
    if apply_style:
        set_plot_style()
    ax = _get_ax(ax)
    present = [s for s in list(strategies or grid_results.index) if s in grid_results.index]
    if not present or risk_col not in grid_results.columns or return_col not in grid_results.columns:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    df = grid_results.loc[present].copy()
    colors = df[color_col].astype(float) if color_col in df.columns else "#0072B2"
    sizes = 90.0
    if size_col in df.columns:
        raw = df[size_col].abs().astype(float)
        sizes = 50.0 + 260.0 * raw / max(float(raw.max()), 1e-12)
    scatter = ax.scatter(df[risk_col], df[return_col], c=colors, s=sizes, alpha=0.82, edgecolor="white", linewidth=0.6)
    for name, label in zip(present, _short_names(present, short_labels), strict=False):
        ax.annotate(label, (df.loc[name, risk_col], df.loc[name, return_col]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_title(title or "Risk-return")
    ax.set_xlabel(risk_col)
    ax.set_ylabel(return_col)
    ax.grid(True, alpha=0.3)
    if color_col in df.columns:
        ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=color_col)
    return ax


def plot_average_weight_heatmap(
    weights_by_name: Mapping[str, pd.DataFrame | pd.Series],
    *,
    ax=None,
    title: str | None = None,
    short_labels: Mapping[str, str] | None = None,
    cmap: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    avg = _average_weight_matrix(weights_by_name)
    if avg.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    avg = avg.loc[avg.sum(axis=1).sort_values(ascending=False).index]
    vals = avg.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap or "Blues")
    ax.set_title(title or "Average weights")
    ax.set_yticks(range(len(avg.index)))
    ax.set_yticklabels(avg.index)
    ax.set_xticks(range(len(avg.columns)))
    ax.set_xticklabels(_short_names(list(avg.columns), short_labels), rotation=35, ha="right")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_sleeve_exposure_bar(
    weights_by_name: Mapping[str, pd.DataFrame | pd.Series],
    sleeve_map: Mapping[str, str],
    *,
    ax=None,
    title: str | None = None,
    short_labels: Mapping[str, str] | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    avg = _average_weight_matrix(weights_by_name)
    if avg.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    sleeves = pd.Series(sleeve_map, dtype=str)
    rows = {}
    for sleeve in sleeves.dropna().unique():
        members = sleeves[sleeves.eq(sleeve)].index.intersection(avg.index)
        rows[sleeve] = avg.loc[members].sum(axis=0) if len(members) else pd.Series(0.0, index=avg.columns)
    exposure = pd.DataFrame(rows).fillna(0.0)
    exposure.index = _short_names(list(exposure.index), short_labels)
    exposure.plot(kind="bar", stacked=True, ax=ax, width=0.82)
    ax.set_title(title or "Sleeve exposure")
    ax.set_xlabel("")
    ax.set_ylabel("Average weight")
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    return _small_legend(ax, ncol=1)


def plot_cvar_budget_path(path: pd.DataFrame, *, ax=None, title: str | None = None):
    set_plot_style()
    ax = _get_ax(ax)
    if path is None or path.empty:
        ax.text(0.5, 0.5, "No path", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    x = path["budget_scale"] if "budget_scale" in path.columns else np.arange(len(path))
    if "expected_return" in path.columns:
        ax.plot(x, path["expected_return"], marker="o", label="Expected return")
    if "cvar_loss" in path.columns:
        ax.plot(x, path["cvar_loss"], marker="o", label="CVaR loss")
    if "cvar_budget" in path.columns:
        ax.plot(x, path["cvar_budget"], ls="--", label="Budget")
    ax.set_title(title or "CVaR budget path")
    ax.set_xlabel("Budget scale")
    ax.grid(True, alpha=0.3)
    return _small_legend(ax)


def plot_risk_contribution_bar(table: pd.DataFrame, *, ax=None, title: str | None = None):
    set_plot_style()
    ax = _get_ax(ax)
    if table is None or table.empty:
        ax.text(0.5, 0.5, "No risk contributions", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    t = table.copy()
    if "asset" in t.columns:
        t = t.set_index("asset")
    col = "percent_risk_contribution" if "percent_risk_contribution" in t.columns else "risk_contribution"
    vals = t[col].astype(float).sort_values()
    ax.barh(vals.index, vals.values)
    ax.set_title(title or "Risk contribution")
    ax.set_xlabel("Percent of risk" if col.startswith("percent") else "Risk contribution")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_hrp_dendrogram(
    cov_ann,
    *,
    labels: Sequence[str] | None = None,
    linkage_method: str = "average",
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    cov = pd.DataFrame(cov_ann, index=labels, columns=labels) if labels is not None else pd.DataFrame(cov_ann)
    labels_use = [str(x) for x in cov.index]
    std = np.sqrt(np.diag(cov.to_numpy(dtype=float))).clip(min=1e-12)
    corr = np.clip(cov.to_numpy(dtype=float) / np.outer(std, std), -1.0, 1.0)
    dist = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    np.fill_diagonal(dist, 0.0)
    dendrogram(linkage(squareform(dist, checks=False), method=linkage_method), labels=labels_use, ax=ax, leaf_rotation=60, leaf_font_size=7)
    ax.set_title(title or "HRP clustering")
    ax.grid(False)
    return ax


def plot_nco_cluster_weights(
    cluster_table: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    *,
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if isinstance(weights, pd.DataFrame):
        w = weights.iloc[-1].astype(float)
    else:
        w = pd.Series(weights, dtype=float)
    if cluster_table is None or cluster_table.empty:
        ax.text(0.5, 0.5, "No clusters", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    t = cluster_table.copy()
    if "asset" not in t.columns:
        t = t.reset_index().rename(columns={t.index.name or "index": "asset"})
    sleeve = t.set_index("asset")["cluster"].astype(str)
    vals = w.groupby(sleeve.reindex(w.index).fillna("NA")).sum().sort_values()
    ax.barh(vals.index, vals.values)
    ax.set_title(title or "NCO-MV clusters")
    ax.set_xlabel("Latest weight")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_wasserstein_radius_path(path: pd.DataFrame, *, ax=None, title: str | None = None):
    set_plot_style()
    ax = _get_ax(ax)
    if path is None or path.empty or "radius" not in path.columns:
        ax.text(0.5, 0.5, "No path", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    x = path["radius"].astype(float)
    for col, label in [("robust_return", "Robust return"), ("volatility", "Volatility"), ("effective_n", "Effective N")]:
        if col in path.columns:
            ax.plot(x, path[col].astype(float), marker="o", label=label)
    ax.set_title(title or "Wasserstein radius path")
    ax.set_xlabel("Radius")
    ax.grid(True, alpha=0.3)
    return _small_legend(ax)


def _frame_for_heatmap(data, *, fill_diag: float | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        out = data.copy().astype(float)
    else:
        arr = np.asarray(data, dtype=float)
        out = pd.DataFrame(arr)
    out = out.replace([np.inf, -np.inf], np.nan)
    if fill_diag is not None and out.shape[0] == out.shape[1]:
        vals = out.to_numpy(dtype=float).copy()
        np.fill_diagonal(vals, float(fill_diag))
        out = pd.DataFrame(vals, index=out.index, columns=out.columns)
    return out


def _heatmap_ticks(ax, labels: Sequence[str], *, max_labels: int = 24):
    labels = [str(x) for x in labels]
    n = len(labels)
    if n <= max_labels:
        locs = np.arange(n)
        shown = labels
    else:
        step = max(int(np.ceil(n / max_labels)), 1)
        locs = np.arange(0, n, step)
        shown = [labels[i] for i in locs]
    ax.set_xticks(locs)
    ax.set_xticklabels(shown, rotation=90, fontsize=6)
    ax.set_yticks(locs)
    ax.set_yticklabels(shown, fontsize=6)


def corr_heatmap(ax, corr, *, title: str | None = None, cmap: str = "coolwarm"):
    mat = _frame_for_heatmap(corr, fill_diag=1.0)
    vals = mat.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_title(title or "Correlation")
    _heatmap_ticks(ax, list(mat.index))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def tail_heatmap(ax, tail, *, title: str | None = None, cmap: str = "magma"):
    mat = _frame_for_heatmap(tail, fill_diag=1.0)
    vals = mat.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(1e-12, float(np.nanmax(vals))))
    ax.set_title(title or "Tail dependence")
    _heatmap_ticks(ax, list(mat.index))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def distance_heatmap(ax, distance, *, title: str | None = None, cmap: str = "viridis_r"):
    mat = _frame_for_heatmap(distance, fill_diag=0.0)
    vals = mat.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_title(title or "Distance")
    _heatmap_ticks(ax, list(mat.index))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def network_graph(
    ax,
    graph,
    *,
    title: str | None = None,
    seed: int = 17,
):
    try:
        import networkx as nx
    except Exception:
        ax.text(0.5, 0.5, "networkx unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    if graph is None or graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    draw_graph = graph.copy()
    pos = nx.spring_layout(draw_graph, weight="weight", seed=int(seed), iterations=80)
    strength = pd.Series(dict(draw_graph.degree(weight="weight")), dtype=float)
    if strength.empty or float(strength.max()) <= 1e-12:
        sizes = [16.0 for _ in draw_graph.nodes()]
    else:
        sizes = [18.0 + 85.0 * float(strength.get(n, 0.0)) / float(strength.max()) for n in draw_graph.nodes()]
    weights = np.asarray([float(d.get("weight", 0.0)) for _, _, d in draw_graph.edges(data=True)], dtype=float)
    high = max(float(weights.max()) if len(weights) else 1.0, 1e-12)
    widths = 0.14 + (0.65 if draw_graph.number_of_edges() > 1000 else 2.0) * weights / high
    edge_alpha = 0.22 if draw_graph.number_of_edges() > 1000 else 0.68
    nx.draw_networkx_edges(
        draw_graph,
        pos,
        ax=ax,
        width=widths,
        alpha=edge_alpha,
        edge_color=weights if len(weights) else "#355c7d",
        edge_cmap=__import__("matplotlib").colormaps["plasma"],
        edge_vmin=0.0,
        edge_vmax=high,
    )
    nx.draw_networkx_nodes(
        draw_graph,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=list(strength.reindex(draw_graph.nodes()).fillna(0.0)),
        cmap="viridis",
        alpha=0.88,
        linewidths=0.25,
        edgecolors="white",
    )
    ax.set_title(title or "Network")
    ax.set_axis_off()
    return ax


def centrality_heatmap(ax, centrality: pd.DataFrame, *, title: str | None = None, cmap: str = "viridis"):
    df = pd.DataFrame(centrality).copy()
    cols = [c for c in ["degree", "eigenvector", "betweenness", "closeness", "combined"] if c in df.columns]
    df = df[cols] if cols else df.select_dtypes(include=[np.number])
    if df.empty:
        ax.text(0.5, 0.5, "No centrality", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = df.astype(float).to_numpy()
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title or "Centrality scores")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(x) for x in df.index], fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def centrality_score_bars(
    ax,
    scores: pd.Series | pd.DataFrame,
    *,
    column: str = "combined",
    top_n: int = 20,
    title: str | None = None,
):
    if isinstance(scores, pd.DataFrame):
        if column in scores.columns:
            vals = scores[column]
        else:
            vals = scores.select_dtypes(include=[np.number]).iloc[:, 0]
    else:
        vals = pd.Series(scores, dtype=float)
    vals = vals.dropna().sort_values(ascending=True).tail(int(top_n))
    if vals.empty:
        ax.text(0.5, 0.5, "No scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.barh(vals.index.astype(str), vals.values)
    ax.set_title(title or "Centrality scores")
    ax.set_xlabel(column)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def score_heatmap(ax, scores: pd.DataFrame, *, title: str | None = None, cmap: str = "viridis"):
    df = pd.DataFrame(scores).select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if df.empty:
        ax.text(0.5, 0.5, "No scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = df.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_title(title or "Scores")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(x) for x in df.index], fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def weight_heatmap(
    ax,
    weights: pd.DataFrame | pd.Series,
    *,
    title: str | None = None,
    top_n: int = 40,
    cmap: str = "Blues",
):
    if isinstance(weights, pd.Series):
        df = weights.to_frame("weight")
    else:
        df = pd.DataFrame(weights).copy()
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    order = df.abs().max(axis=1).sort_values(ascending=False).head(int(top_n)).index
    df = df.loc[order]
    if df.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    im = ax.imshow(df.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_title(title or "Weights")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([str(x) for x in df.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(x) for x in df.index], fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def _comparison_matrix(data: pd.DataFrame, *, value: str) -> pd.DataFrame:
    df = pd.DataFrame(data).copy()
    if df.empty:
        return pd.DataFrame()
    lookup = {str(c).lower(): c for c in df.columns}
    val_col = lookup.get(str(value).lower(), value)
    if val_col not in df.columns:
        return pd.DataFrame()
    needed = {"dependence", "network", "centrality"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()
    dfx = df.copy()
    dfx["setup"] = dfx["dependence"].astype(str) + " / " + dfx["network"].astype(str)
    if "direction" in dfx.columns:
        dfx["setup"] = dfx["setup"] + " / " + dfx["direction"].astype(str)
    mat = dfx.pivot_table(index="centrality", columns="setup", values=val_col, aggfunc="mean")
    return mat.sort_index(axis=0).sort_index(axis=1)


def performance_heatmap(
    ax,
    comparison: pd.DataFrame,
    *,
    value: str,
    title: str | None = None,
    cmap: str | None = None,
    annotate: bool = False,
):
    mat = _comparison_matrix(comparison, value=value)
    if mat.empty:
        ax.text(0.5, 0.5, "No comparison", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = mat.to_numpy(dtype=float)
    cmap_use = cmap or choose_heatmap_cmap(metric_name=value)
    im = ax.imshow(vals, aspect="auto", cmap=cmap_use)
    if annotate:
        _annotate_heatmap(ax, vals, ".2f")
    ax.set_title(title or value)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=50, ha="right", fontsize=6)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def network_comparison_grid(ax, comparison: pd.DataFrame, *, value: str, title: str | None = None):
    return performance_heatmap(ax, comparison, value=value, title=title)


def target_distribution(ax, data: pd.DataFrame, *, raw_col: str, scaled_col: str, title: str | None = None):
    set_plot_style()
    df = data[[raw_col, scaled_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        ax.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.hist(df[raw_col], bins=60, alpha=0.45, density=True, color=LAB_COLORS[0], label=raw_col)
    ax.hist(df[scaled_col], bins=60, alpha=0.45, density=True, color=LAB_COLORS[1], label=scaled_col)
    ax.axvline(0.0, color=LAB_COLORS[2], lw=1.0)
    ax.set_title(title or "Target distributions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def target_by_asset(ax, data: pd.DataFrame, *, target_col: str, asset_col: str = "asset", title: str | None = None):
    set_plot_style()
    df = data[[asset_col, target_col]].dropna()
    if df.empty:
        ax.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    order = df.groupby(asset_col)[target_col].median().sort_values().index
    vals = [df.loc[df[asset_col].eq(asset), target_col].to_numpy(dtype=float) for asset in order]
    ax.boxplot(vals, labels=order, vert=False, showfliers=False, patch_artist=True)
    for patch in ax.artists:
        patch.set_facecolor(LAB_COLORS[0])
        patch.set_alpha(0.35)
    ax.axvline(0.0, color=LAB_COLORS[2], lw=1.0)
    ax.set_title(title or f"{target_col} by asset")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def target_stability(ax, stats: pd.DataFrame, *, title: str | None = None):
    set_plot_style()
    if stats is None or stats.empty:
        ax.text(0.5, 0.5, "No stability data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    if "mean" in stats.columns:
        ax.plot(stats.index, stats["mean"], color=LAB_COLORS[0], label="rolling mean")
    if "vol" in stats.columns:
        ax.plot(stats.index, stats["vol"], color=LAB_COLORS[1], label="rolling volatility")
    ax.axhline(0.0, color=LAB_COLORS[2], lw=0.8, alpha=0.5)
    ax.set_title(title or "Target stability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return _format_date_axis(ax)


def feature_importance_bars(ax, importance: pd.Series | pd.DataFrame, *, top_n: int = 20, title: str | None = None):
    set_plot_style()
    if isinstance(importance, pd.DataFrame):
        vals = importance.iloc[:, 0]
    else:
        vals = pd.Series(importance, dtype=float)
    vals = vals.dropna().sort_values().tail(int(top_n))
    if vals.empty:
        ax.text(0.5, 0.5, "No importance", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.barh(vals.index, vals.values, color=LAB_COLORS[0], alpha=0.82)
    ax.set_title(title or "Feature importance")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def feature_correlation_map(ax, corr: pd.DataFrame, *, title: str | None = None, annotate: bool = False):
    set_plot_style()
    c = pd.DataFrame(corr).astype(float)
    if c.empty:
        ax.text(0.5, 0.5, "No correlation", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = c.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(c.columns)))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(c.index)))
    ax.set_yticklabels(c.index, fontsize=6)
    if annotate and len(c) <= 12:
        _annotate_heatmap(ax, vals, ".1f")
    ax.set_title(title or "Feature correlation")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def pca_explained_variance(ax, explained: pd.DataFrame, *, title: str | None = None):
    set_plot_style()
    if explained is None or explained.empty:
        ax.text(0.5, 0.5, "No PCA data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    x = np.arange(1, len(explained) + 1)
    ratio = explained["explained_variance_ratio"].to_numpy(dtype=float)
    cumulative = explained["cumulative"].to_numpy(dtype=float)
    ax.bar(x, ratio, color=LAB_COLORS[0], alpha=0.65, label="component")
    ax.plot(x, cumulative, color=LAB_COLORS[1], marker="o", lw=1.8, label="cumulative")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Component")
    ax.set_title(title or "PCA explained variance")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def forecast_scatter(ax, data: pd.DataFrame, *, y_col: str, pred_col: str, title: str | None = None):
    set_plot_style()
    d = data[[y_col, pred_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No forecast data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.scatter(d[pred_col], d[y_col], s=14, alpha=0.35, color=LAB_COLORS[0], edgecolor="none")
    lo = float(np.nanmin([d[pred_col].min(), d[y_col].min()]))
    hi = float(np.nanmax([d[pred_col].max(), d[y_col].max()]))
    ax.plot([lo, hi], [lo, hi], color=LAB_COLORS[1], lw=1.0)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Realized")
    ax.set_title(title or "Prediction vs realized")
    ax.grid(True, alpha=0.25)
    return ax


def forecast_buckets_chart(ax, bucket_table: pd.DataFrame, *, title: str | None = None):
    set_plot_style()
    if bucket_table is None or bucket_table.empty:
        ax.text(0.5, 0.5, "No buckets", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = bucket_table["mean"] if "mean" in bucket_table.columns else bucket_table.iloc[:, 0]
    ax.bar(vals.index.astype(str), vals.values, color=LAB_COLORS[0], alpha=0.82)
    ax.axhline(0.0, color=LAB_COLORS[2], lw=0.8)
    ax.set_title(title or "Realized outcome by forecast bucket")
    ax.set_xlabel("Forecast bucket")
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def rolling_ic_chart(ax, data: pd.DataFrame, *, date_col: str, asset_col: str, y_col: str, pred_col: str, title: str | None = None):
    set_plot_style()
    from quantfinlab.ml.evaluation import rolling_rank_ic

    s = rolling_rank_ic(data, date_col=date_col, asset_col=asset_col, y_col=y_col, pred_col=pred_col)
    if s.empty:
        ax.text(0.5, 0.5, "No IC data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.plot(s.index, s.values, color=LAB_COLORS[0], lw=1.5)
    ax.axhline(0.0, color=LAB_COLORS[2], lw=0.8)
    ax.set_title(title or "Rolling rank IC")
    ax.grid(True, alpha=0.25)
    return _format_date_axis(ax)


def coverage_reliability(ax, data: pd.DataFrame, *, y_col: str, low_col: str, high_col: str, title: str | None = None):
    set_plot_style()
    d = data[[y_col, low_col, high_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No interval data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    d = d.assign(width=d[high_col] - d[low_col], hit=d[y_col].between(d[low_col], d[high_col]))
    ranks = d["width"].rank(method="first")
    d["bin"] = pd.qcut(ranks, 5, labels=False) + 1
    rel = d.groupby("bin").agg(coverage=("hit", "mean"), width=("width", "mean"))
    ax.bar(rel.index.astype(str), rel["coverage"], color=LAB_COLORS[0], alpha=0.78)
    ax.axhline(0.80, color=LAB_COLORS[1], ls="--", lw=1.2, label="target 80%")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title or "Interval coverage reliability")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def uncertainty_error_map(ax, data: pd.DataFrame, *, width_col: str, y_col: str, pred_col: str, title: str | None = None):
    set_plot_style()
    d = data[[width_col, y_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No uncertainty data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    err = (d[y_col] - d[pred_col]).abs()
    ax.scatter(d[width_col], err, s=14, alpha=0.35, color=LAB_COLORS[0], edgecolor="none")
    ax.set_xlabel("Forecast interval width")
    ax.set_ylabel("Absolute error")
    ax.set_title(title or "Uncertainty vs error")
    ax.grid(True, alpha=0.25)
    return ax


def disagreement_error_map(ax, data: pd.DataFrame, *, disagreement_col: str, y_col: str, pred_col: str, title: str | None = None):
    set_plot_style()
    d = data[[disagreement_col, y_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No disagreement data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    err = (d[y_col] - d[pred_col]).abs()
    ax.scatter(d[disagreement_col], err, s=14, alpha=0.35, color=LAB_COLORS[1], edgecolor="none")
    ax.set_xlabel("Model disagreement")
    ax.set_ylabel("Absolute error")
    ax.set_title(title or "Disagreement vs error")
    ax.grid(True, alpha=0.25)
    return ax


def conformal_shift_chart(ax, data: pd.DataFrame, *, raw_low: str, raw_high: str, conf_low: str, conf_high: str, title: str | None = None):
    set_plot_style()
    cols = [raw_low, raw_high, conf_low, conf_high]
    d = data[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No conformal data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    raw_width = d[raw_high] - d[raw_low]
    conf_width = d[conf_high] - d[conf_low]
    ax.hist(raw_width, bins=40, alpha=0.50, color=LAB_COLORS[0], label="raw")
    ax.hist(conf_width, bins=40, alpha=0.45, color=LAB_COLORS[1], label="conformal")
    ax.set_title(title or "Raw vs conformal interval width")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def kelly_weight_path(ax, weights: pd.DataFrame, *, title: str | None = None):
    set_plot_style()
    W = pd.DataFrame(weights).astype(float)
    if W.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    top = W.mean().sort_values(ascending=False).head(8).index
    W[top].plot.area(ax=ax, alpha=0.75, linewidth=0)
    ax.set_title(title or "Kelly weight path")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.20)
    return _format_date_axis(ax)


def confidence_exposure_map(ax, data: pd.DataFrame, *, weight_frame: pd.DataFrame, title: str | None = None):
    set_plot_style()
    if data.empty or weight_frame.empty or "c_width" not in data.columns:
        ax.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    expo = weight_frame.sum(axis=1).rename("exposure")
    conf = data.groupby("date")["c_width"].mean()
    d = pd.concat([conf, expo], axis=1).dropna()
    ax.scatter(d["c_width"], d["exposure"], s=28, color=LAB_COLORS[0], alpha=0.60)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Gross exposure")
    ax.set_title(title or "Confidence vs exposure")
    ax.grid(True, alpha=0.25)
    return ax


def kelly_shrinkage_curve(ax, data: pd.DataFrame, *, mu_col: str, mu_adj_col: str, title: str | None = None):
    set_plot_style()
    d = data[[mu_col, mu_adj_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.empty:
        ax.text(0.5, 0.5, "No shrinkage data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.scatter(d[mu_col], d[mu_adj_col], s=12, color=LAB_COLORS[0], alpha=0.35)
    lo = float(np.nanmin([d[mu_col].min(), d[mu_adj_col].min()]))
    hi = float(np.nanmax([d[mu_col].max(), d[mu_adj_col].max()]))
    ax.plot([lo, hi], [lo, hi], color=LAB_COLORS[1], lw=1.0)
    ax.axhline(0.0, color=LAB_COLORS[2], lw=0.7)
    ax.axvline(0.0, color=LAB_COLORS[2], lw=0.7)
    ax.set_xlabel("Raw mu")
    ax.set_ylabel("Adjusted mu")
    ax.set_title(title or "Kelly forecast shrinkage")
    ax.grid(True, alpha=0.25)
    return ax


def forecast_to_weight_map(ax, data: pd.DataFrame, *, weight_frame: pd.DataFrame, title: str | None = None):
    set_plot_style()
    if data.empty or weight_frame.empty or not {"date", "asset", "mu_adj"}.issubset(data.columns):
        ax.text(0.5, 0.5, "No forecast/weight data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    w_long = weight_frame.stack().rename("weight").reset_index()
    w_long.columns = ["date", "asset", "weight"]
    d = data[["date", "asset", "mu_adj"]].merge(w_long, on=["date", "asset"], how="inner").dropna()
    ax.scatter(d["mu_adj"], d["weight"], s=16, alpha=0.45, color=LAB_COLORS[0], edgecolor="none")
    ax.axvline(0.0, color=LAB_COLORS[2], lw=0.8)
    ax.set_xlabel("Adjusted forecast")
    ax.set_ylabel("Weight")
    ax.set_title(title or "Forecast to weight")
    ax.grid(True, alpha=0.25)
    return ax


def plot_rolling_sharpe(
    ax,
    returns: pd.DataFrame,
    *,
    window: int = 252,
    rf_daily: float | pd.Series = 0.0,
    title: str | None = None,
):
    set_plot_style()
    R = pd.DataFrame(returns).astype(float)
    if R.empty:
        ax.text(0.5, 0.5, "No returns", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    excess = _excess_returns(R, rf_daily)
    volatility = R if np.isscalar(rf_daily) else excess
    roll = excess.rolling(int(window)).mean() / volatility.rolling(int(window)).std(ddof=1)
    roll = roll * np.sqrt(252.0)
    roll.plot(ax=ax, lw=1.1)
    ax.axhline(0.0, color=LAB_COLORS[2], lw=0.8)
    ax.set_title(title or "Rolling Sharpe")
    ax.grid(True, alpha=0.25)
    return _format_date_axis(ax)


def policy_exposure_path(ax, weights: pd.DataFrame, *, cash_ticker: str = "SHY", title: str | None = None):
    set_plot_style()
    W = pd.DataFrame(weights).copy()
    if W.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    W.index = pd.to_datetime(W.index)
    risky = W.drop(columns=[cash_ticker], errors="ignore").sum(axis=1)
    cash = W[cash_ticker] if cash_ticker in W.columns else 1.0 - risky
    ax.plot(risky.index, risky, label="risky exposure", color=LAB_COLORS[0], lw=1.8)
    ax.plot(cash.index, cash, label="cash", color=LAB_COLORS[1], lw=1.2, ls="--")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title or "Policy exposure")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    return _format_date_axis(ax)


def active_return_curve(
    ax,
    strategy_returns: pd.DataFrame,
    *,
    strategy: str,
    benchmark: str,
    title: str | None = None,
):
    set_plot_style()
    R = pd.DataFrame(strategy_returns).copy()
    if R.empty or strategy not in R.columns or benchmark not in R.columns:
        ax.text(0.5, 0.5, "No active return data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    active = (R[strategy] - R[benchmark]).dropna()
    nav = (1.0 + active).cumprod() - 1.0
    ax.plot(nav.index, nav, color=LAB_COLORS[0], lw=1.8)
    ax.axhline(0.0, color=LAB_COLORS[2], lw=0.8)
    ax.set_title(title or f"{strategy} active return")
    ax.grid(True, alpha=0.25)
    return _format_date_axis(ax)


def policy_weight_area(ax, weights: pd.DataFrame, *, title: str | None = None, top_n: int = 8):
    set_plot_style()
    W = pd.DataFrame(weights).copy()
    if W.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    W.index = pd.to_datetime(W.index)
    cols = W.abs().mean().sort_values(ascending=False).head(int(top_n)).index.tolist()
    other = W.drop(columns=cols, errors="ignore").sum(axis=1)
    plot_data = W[cols].copy()
    if W.shape[1] > len(cols):
        plot_data["Other"] = other
    plot_data.plot.area(ax=ax, linewidth=0.0, alpha=0.86)
    ax.set_ylim(0.0, max(1.0, float(plot_data.sum(axis=1).max())))
    ax.set_title(title or "Policy weights")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    ax.grid(True, alpha=0.20)
    return _format_date_axis(ax)


def regime_exposure_boxplot(
    ax,
    weights: pd.DataFrame,
    regime_features: pd.DataFrame,
    *,
    title: str | None = None,
):
    set_plot_style()
    W = pd.DataFrame(weights).copy()
    R = pd.DataFrame(regime_features).copy()
    if W.empty or R.empty:
        ax.text(0.5, 0.5, "No regime exposure data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    W.index = pd.to_datetime(W.index)
    R.index = pd.to_datetime(R.index)
    risky = W.drop(columns=[c for c in W.columns if str(c).upper() in {"SHY", "CASH"}], errors="ignore").sum(axis=1)
    probs = R.reindex(R.index.union(W.index)).sort_index().ffill().reindex(W.index)
    prob_cols = [c for c in probs.columns if str(c).startswith("p_")]
    if prob_cols:
        labels = probs[prob_cols].idxmax(axis=1).str.replace("p_", "", regex=False)
    else:
        labels = pd.Series("all", index=W.index)
    data = [risky.loc[labels.eq(label)].dropna().to_numpy() for label in sorted(labels.dropna().unique())]
    names = sorted(labels.dropna().unique())
    if not data:
        ax.text(0.5, 0.5, "No labeled regimes", ha="center", va="center", transform=ax.transAxes)
        return ax
    ax.boxplot(data, labels=names, patch_artist=True, boxprops={"facecolor": LAB_COLORS[8], "alpha": 0.55})
    ax.set_ylabel("Risky exposure")
    ax.set_title(title or "Exposure by regime")
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def forecast_policy_sensitivity(
    ax,
    weights: pd.DataFrame,
    forecast_features: pd.DataFrame,
    *,
    title: str | None = None,
):
    set_plot_style()
    W = pd.DataFrame(weights).copy()
    F = pd.DataFrame(forecast_features).copy()
    if W.empty or F.empty or not {"date", "asset"}.issubset(F.columns):
        ax.text(0.5, 0.5, "No forecast sensitivity data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    W.index = pd.to_datetime(W.index)
    F["date"] = pd.to_datetime(F["date"])
    score_col = "tcn_alpha" if "tcn_alpha" in F.columns else ("tcn_rank" if "tcn_rank" in F.columns else None)
    if score_col is None:
        ax.text(0.5, 0.5, "No TCN score column", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    w_long = W.stack().rename("weight").reset_index()
    w_long.columns = ["date", "asset", "weight"]
    d = F[["date", "asset", score_col]].merge(w_long, on=["date", "asset"], how="inner").dropna()
    ax.scatter(d[score_col], d["weight"], s=14, alpha=0.35, color=LAB_COLORS[0], edgecolor="none")
    if len(d) > 5 and d[score_col].std(ddof=0) > 0:
        coef = np.polyfit(d[score_col], d["weight"], deg=1)
        x = np.linspace(float(d[score_col].min()), float(d[score_col].max()), 100)
        ax.plot(x, coef[0] * x + coef[1], color=LAB_COLORS[1], lw=1.5)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Policy weight")
    ax.set_title(title or "Forecast sensitivity")
    ax.grid(True, alpha=0.25)
    return ax


def turnover_cost_curve(
    ax,
    weights_by_strategy: Mapping[str, pd.DataFrame],
    *,
    cost_bps: float = 10.0,
    title: str | None = None,
):
    set_plot_style()
    rows = []
    for name, weights in weights_by_strategy.items():
        W = pd.DataFrame(weights).copy().sort_index()
        if W.empty:
            continue
        turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
        rows.append(
            {
                "Strategy": str(name),
                "Annual Turnover": float(turnover.mean() * 52.0),
                "Cost Drag": float(turnover.mean() * 52.0 * float(cost_bps) / 10000.0),
            }
        )
    tbl = pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()
    if tbl.empty:
        ax.text(0.5, 0.5, "No turnover data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    tbl["Annual Turnover"].plot(kind="bar", ax=ax, color=LAB_COLORS[0])
    ax.set_title(title or "Turnover and cost")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def concentration_path(ax, weights: pd.DataFrame, *, title: str | None = None):
    set_plot_style()
    W = pd.DataFrame(weights).copy()
    if W.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    W.index = pd.to_datetime(W.index)
    risky = W.drop(columns=[c for c in W.columns if str(c).upper() in {"SHY", "CASH"}], errors="ignore")
    hhi = (risky.clip(lower=0.0) ** 2).sum(axis=1)
    max_w = risky.max(axis=1)
    ax.plot(hhi.index, hhi, label="HHI", color=LAB_COLORS[0], lw=1.6)
    ax.plot(max_w.index, max_w, label="max weight", color=LAB_COLORS[1], lw=1.3, ls="--")
    ax.set_title(title or "Concentration")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    return _format_date_axis(ax)


__all__ = [
    "active_return_curve",
    "apply_portfolio_subplot_layout",
    "centrality_heatmap",
    "centrality_score_bars",
    "concentration_path",
    "corr_heatmap",
    "distance_heatmap",
    "format_portfolio_time_axis",
    "heatmap_matrix",
    "network_comparison_grid",
    "network_graph",
    "performance_heatmap",
    "plot_active_nav",
    "plot_active_weights_heatmap",
    "plot_average_weight_heatmap",
    "confidence_exposure_map",
    "conformal_shift_chart",
    "plot_cvar_budget_path",
    "plot_drawdowns",
    "plot_effective_n_bar",
    "disagreement_error_map",
    "feature_correlation_map",
    "feature_importance_bars",
    "forecast_buckets_chart",
    "forecast_scatter",
    "forecast_to_weight_map",
    "forecast_policy_sensitivity",
    "kelly_shrinkage_curve",
    "kelly_weight_path",
    "pca_explained_variance",
    "plot_finalist_drawdowns",
    "plot_finalist_metric_bar",
    "plot_finalist_nav",
    "plot_fixed_cov_mu_comparison",
    "plot_fixed_mu_covariance_comparison",
    "plot_grid_heatmap",
    "plot_hrp_dendrogram",
    "plot_latest_weights",
    "plot_metric_bar",
    "plot_nav",
    "plot_nco_cluster_weights",
    "plot_posterior_shift_heatmap",
    "plot_q_tilt_heatmap",
    "plot_risk_contribution_bar",
    "plot_risk_return_scatter",
    "plot_rolling_sharpe",
    "plot_rolling_active_metrics",
    "plot_sleeve_exposure_bar",
    "policy_exposure_path",
    "policy_weight_area",
    "plot_strategy_drawdowns",
    "plot_strategy_nav",
    "plot_stress_summary",
    "plot_turnover",
    "plot_turnover_bar",
    "plot_view_count_timeline",
    "plot_view_q_and_confidence",
    "plot_view_selection_counts",
    "plot_wasserstein_radius_path",
    "plot_weights",
    "regime_exposure_boxplot",
    "rolling_ic_chart",
    "score_heatmap",
    "show_portfolio_xaxis_like_risk_module",
    "tail_heatmap",
    "target_by_asset",
    "target_distribution",
    "target_stability",
    "turnover_cost_curve",
    "uncertainty_error_map",
    "coverage_reliability",
    "weight_heatmap",
]
