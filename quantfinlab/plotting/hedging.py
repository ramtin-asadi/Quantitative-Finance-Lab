from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantfinlab.plotting.curves import choose_heatmap_cmap, set_plot_style
from quantfinlab.risk import drawdown_series

HEDGE_COLORS = [
    "#069AF3",
    "#FE420F",
    "#00008B",
    "#008080",
    "#CC79A7",
    "#9614fa",
    "#DC143C",
    "#7BC8F6",
    "#0072B2",
    "#04D8B2",
    "#800080",
    "#FF8072",
]


def _get_ax(ax=None):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _bar_colors(n: int, *, offset: int = 0, colors: Sequence[str] | None = None) -> list[str]:
    palette = list(colors or HEDGE_COLORS)
    return [palette[(i + offset) % len(palette)] for i in range(max(int(n), 0))]


def _model_color_map(models: Sequence[str], colors: Sequence[str] | None = None) -> dict[str, str]:
    palette = list(colors or HEDGE_COLORS)
    return {str(model): palette[i % len(palette)] for i, model in enumerate(pd.Index(models).dropna().unique())}


def _inset_colorbar(im, ax, *, label: str | None = None):
    cax = ax.inset_axes([0.965, 0.13, 0.025, 0.74])
    cbar = ax.figure.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    return cbar


def _heatmap(mat: pd.DataFrame, *, ax=None, title: str | None = None, cmap=None, fmt: str = ".2f"):
    set_plot_style()
    ax = _get_ax(ax)
    if mat.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    values = mat.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
        vmax = float(np.nanpercentile(np.abs(finite), 95))
        vmin = -vmax if np.nanmin(finite) < 0 else float(np.nanpercentile(finite, 5))
    else:
        vmin, vmax = 0.0, 1.0
    use_cmap = cmap or ("coolwarm" if vmin < 0 else "viridis")
    im = ax.imshow(values, aspect="auto", cmap=use_cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
    ax.set_title(title or "")
    if mat.size <= 40:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7)
    _inset_colorbar(im, ax)
    return ax


def plot_diag_heatmap(tab: pd.DataFrame, *, ax=None, title: str = "relationship diagnostics"):
    cols = [c for c in ["target_vol", "hedge_vol", "corr", "r2", "beta_iqr"] if c in tab.columns]
    mat = tab.set_index("relationship")[cols] if not tab.empty and cols else pd.DataFrame()
    return _heatmap(mat, ax=ax, title=title)


def plot_metric_heatmap(
    tab: pd.DataFrame,
    metric: str,
    *,
    ax=None,
    title: str | None = None,
    cmap=None,
):
    mat = tab.pivot(index="relationship", columns="model", values=metric) if not tab.empty else pd.DataFrame()
    return _heatmap(mat, ax=ax, title=title or metric, cmap=cmap)


def plot_score_heatmap(tab: pd.DataFrame, *, ax=None, title: str = "hedge score"):
    return plot_metric_heatmap(tab, "score", ax=ax, title=title)


def plot_best_score_bar(
    best: pd.DataFrame,
    *,
    ax=None,
    title: str = "best hedge score",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    need = {"relationship", "best_model", "score"}
    if best.empty or not need.issubset(best.columns):
        ax.text(0.5, 0.5, "No score table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    data = best[list(need)].copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["score"]).sort_values("score")
    cmap = _model_color_map(data["best_model"], colors=colors)
    ax.barh(data["relationship"], data["score"], color=[cmap[str(m)] for m in data["best_model"]], alpha=0.86)
    ax.set_title(title)
    ax.set_xlabel("score")
    ax.grid(True, axis="x", alpha=0.25)
    handles = [ax.barh([], [], color=color, label=model) for model, color in cmap.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    return ax


def plot_risk_reduction_scatter(
    tab: pd.DataFrame,
    *,
    ax=None,
    title: str = "vol vs ES reduction",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    need = {"model", "vol_red", "es_red"}
    if tab.empty or not need.issubset(tab.columns):
        ax.text(0.5, 0.5, "No model table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    data = tab.copy()
    cmap = _model_color_map(data["model"], colors=colors)
    for model, g in data.groupby("model"):
        size = 40 + 120 * pd.to_numeric(g.get("score", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
        ax.scatter(g["vol_red"], g["es_red"], s=size, alpha=0.68, label=model, color=cmap[str(model)], edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("vol reduction")
    ax.set_ylabel("ES reduction")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    return ax


def plot_score_cost_scatter(
    tab: pd.DataFrame,
    *,
    ax=None,
    title: str = "score vs cost drag",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    need = {"model", "score", "cost_drag_ann"}
    if tab.empty or not need.issubset(tab.columns):
        ax.text(0.5, 0.5, "No score table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    cmap = _model_color_map(tab["model"], colors=colors)
    for model, g in tab.groupby("model"):
        ax.scatter(g["cost_drag_ann"], g["score"], s=58, alpha=0.72, label=model, color=cmap[str(model)], edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("annual cost drag")
    ax.set_ylabel("score")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    return ax


def plot_turnover_cost_scatter(
    tab: pd.DataFrame,
    *,
    ax=None,
    title: str = "turnover and cost",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    need = {"model", "turnover_ann", "cost_drag_ann"}
    if tab.empty or not need.issubset(tab.columns):
        ax.text(0.5, 0.5, "No cost table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    cmap = _model_color_map(tab["model"], colors=colors)
    for model, g in tab.groupby("model"):
        ax.scatter(g["turnover_ann"], g["cost_drag_ann"], s=58, alpha=0.72, label=model, color=cmap[str(model)], edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("annual turnover")
    ax.set_ylabel("annual cost drag")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    return ax


def plot_model_counts(
    best: pd.DataFrame,
    *,
    ax=None,
    title: str = "best model counts",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    if best.empty or "best_model" not in best.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    counts = best["best_model"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color=_bar_colors(len(counts), colors=colors), alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("relationships")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def plot_score_gap(
    tab: pd.DataFrame,
    *,
    ax=None,
    title: str = "best minus second",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    if tab.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    gaps = {}
    for name, g in tab.groupby("relationship"):
        vals = g["score"].dropna().sort_values(ascending=False)
        gaps[name] = float(vals.iloc[0] - vals.iloc[1]) if len(vals) >= 2 else np.nan
    s = pd.Series(gaps).sort_values()
    ax.barh(s.index, s.values, color=_bar_colors(len(s), offset=2, colors=colors), alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("score gap")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def plot_beta_paths(beta: pd.DataFrame, *, ax=None, title: str | None = None):
    set_plot_style()
    ax = _get_ax(ax)
    if beta is None or beta.empty:
        ax.text(0.5, 0.5, "No beta", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    beta.dropna(how="all").plot(ax=ax, lw=1.2)
    ax.axhline(0.0, lw=0.8, color="black", alpha=0.45)
    ax.set_title(title or "beta path")
    ax.set_xlabel("date")
    ax.grid(True, alpha=0.25)
    return ax


def plot_beta_grid(beta_by_name: Mapping[str, pd.DataFrame], keys: Sequence[str] | None = None, *, ncols: int = 2):
    import matplotlib.pyplot as plt

    keys = list(keys or beta_by_name.keys())
    keys = [k for k in keys if k in beta_by_name]
    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No beta paths", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, np.asarray([ax])
    rows = int(np.ceil(len(keys) / int(ncols)))
    fig, axes = plt.subplots(rows, int(ncols), figsize=(6.8 * int(ncols), 3.0 * rows), squeeze=False)
    arr = axes.ravel()
    for i, key in enumerate(keys):
        plot_beta_paths(beta_by_name[key], ax=arr[i], title=key)
    for j in range(len(keys), len(arr)):
        arr[j].axis("off")
    fig.tight_layout()
    return fig, arr


def plot_traded_beta(
    desired: pd.DataFrame,
    traded: pd.DataFrame,
    *,
    hedge: str | None = None,
    ax=None,
    title: str | None = None,
):
    set_plot_style()
    ax = _get_ax(ax)
    if desired.empty or traded.empty:
        ax.text(0.5, 0.5, "No beta", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    h = hedge or str(desired.columns[0])
    desired[h].dropna().plot(ax=ax, lw=1.1, label="desired", alpha=0.85)
    traded[h].dropna().plot(ax=ax, lw=1.4, label="traded", alpha=0.85)
    ax.set_title(title or f"{h}: desired vs traded beta")
    ax.axhline(0.0, lw=0.8, color="black", alpha=0.45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def plot_price_grid(px: pd.DataFrame, pairs: Sequence[tuple[str, str, str]], *, ncols: int = 2):
    import matplotlib.pyplot as plt

    cols = [str(c).strip().lower() for c in px.columns]
    data = px.copy()
    data.columns = cols
    pairs = [p for p in pairs if p[1] in data.columns and p[2] in data.columns]
    if not pairs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No prices", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, np.asarray([ax])
    rows = int(np.ceil(len(pairs) / int(ncols)))
    fig, axes = plt.subplots(rows, int(ncols), figsize=(6.8 * int(ncols), 3.0 * rows), squeeze=False)
    arr = axes.ravel()
    for i, (name, a, b) in enumerate(pairs):
        sub = data[[a, b]].dropna()
        if not sub.empty:
            norm = sub / sub.iloc[0]
            arr[i].plot(norm.index, norm[a], label=a)
            arr[i].plot(norm.index, norm[b], label=b)
        arr[i].set_title(name)
        arr[i].legend(fontsize=8)
        arr[i].grid(True, alpha=0.25)
    for j in range(len(pairs), len(arr)):
        arr[j].axis("off")
    fig.tight_layout()
    return fig, arr


def plot_nav_dd(bt: Mapping[str, object], strategies: Sequence[str], *, ax_nav=None, ax_dd=None, title: str = ""):
    import matplotlib.pyplot as plt

    set_plot_style()
    if ax_nav is None or ax_dd is None:
        _, (ax_nav, ax_dd) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True)
    for s in strategies:
        if s not in bt:
            continue
        nav = pd.Series(bt[s].net_values, dtype=float)
        ax_nav.plot(nav.index, nav / nav.iloc[0], label=s)
        dd = drawdown_series(nav, input_kind="nav")
        ax_dd.plot(dd.index, dd, label=s)
    ax_nav.set_title(title or "net value")
    ax_nav.legend(fontsize=8)
    ax_nav.grid(True, alpha=0.25)
    ax_dd.set_title("drawdown")
    ax_dd.grid(True, alpha=0.25)
    return ax_nav, ax_dd


def plot_z_grid(z_by_name: Mapping[str, pd.DataFrame], keys: Sequence[str] | None = None, *, ncols: int = 2):
    import matplotlib.pyplot as plt

    keys = [k for k in list(keys or z_by_name.keys()) if k in z_by_name]
    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No z-scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, np.asarray([ax])
    rows = int(np.ceil(len(keys) / int(ncols)))
    fig, axes = plt.subplots(rows, int(ncols), figsize=(6.8 * int(ncols), 2.7 * rows), squeeze=False)
    arr = axes.ravel()
    for i, key in enumerate(keys):
        z = z_by_name[key]["z"].dropna()
        arr[i].plot(z.index, z.values, lw=1.0)
        arr[i].axhline(2.0, ls="--", lw=0.8, color="tab:red")
        arr[i].axhline(-2.0, ls="--", lw=0.8, color="tab:green")
        arr[i].axhline(0.0, lw=0.8, color="black", alpha=0.4)
        arr[i].set_title(key)
        arr[i].grid(True, alpha=0.25)
    for j in range(len(keys), len(arr)):
        arr[j].axis("off")
    fig.tight_layout()
    return fig, arr


def plot_resid_nav(bt: Mapping[str, object], strategies: Sequence[str] | None = None, *, ax=None, title: str = "residual strategy nav"):
    set_plot_style()
    ax = _get_ax(ax)
    strategies = list(strategies or bt.keys())
    used = False
    for s in strategies:
        if s not in bt:
            continue
        nav = pd.Series(bt[s].net_values, dtype=float)
        if len(nav):
            ax.plot(nav.index, nav / nav.iloc[0], label=s)
            used = True
    if not used:
        ax.text(0.5, 0.5, "No residual backtests", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    return ax


def plot_quality_bar(
    tab: pd.DataFrame,
    *,
    metric: str = "best_vol_red",
    top: int = 8,
    ax=None,
    title: str = "best vol reduction",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    if tab.empty or "relationship" not in tab.columns or metric not in tab.columns:
        ax.text(0.5, 0.5, "No quality table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    data = tab[["relationship", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna().sort_values(metric).tail(int(top))
    if data.empty:
        ax.text(0.5, 0.5, "No quality table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.barh(data["relationship"], data[metric], color=_bar_colors(len(data), colors=colors), alpha=0.85)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def plot_resid_gate_counts(
    gate: pd.DataFrame,
    *,
    ax=None,
    title: str = "eligible residual gates",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    if gate.empty or "beta_source" not in gate.columns or "eligible" not in gate.columns:
        ax.text(0.5, 0.5, "No gate table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    counts = gate.assign(eligible=gate["eligible"].astype(bool)).groupby("beta_source")["eligible"].sum().sort_values()
    if counts.empty:
        ax.text(0.5, 0.5, "No gate table", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.barh(counts.index, counts.values, color=_bar_colors(len(counts), offset=3, colors=colors), alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("eligible candidates")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def plot_resid_return_bar(
    tab: pd.DataFrame,
    *,
    metric: str = "net_return",
    top: int = 8,
    ax=None,
    title: str = "residual trading net return",
    colors: Sequence[str] | None = None,
):
    set_plot_style(colors=list(colors or HEDGE_COLORS))
    ax = _get_ax(ax)
    need = {"pair", "beta_source", metric}
    if tab.empty or not need.issubset(tab.columns):
        ax.text(0.5, 0.5, "No residual trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    data = tab[["pair", "beta_source", metric]].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna().sort_values(metric).tail(int(top))
    if data.empty:
        ax.text(0.5, 0.5, "No residual trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    labels = data["pair"].astype(str) + " | " + data["beta_source"].astype(str).str.replace("_", " ")
    ax.barh(labels, data[metric], color=_bar_colors(len(data), offset=6, colors=colors), alpha=0.85)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    return ax


__all__ = [
    "HEDGE_COLORS",
    "plot_beta_grid",
    "plot_beta_paths",
    "plot_best_score_bar",
    "plot_diag_heatmap",
    "plot_metric_heatmap",
    "plot_model_counts",
    "plot_nav_dd",
    "plot_price_grid",
    "plot_quality_bar",
    "plot_resid_gate_counts",
    "plot_resid_nav",
    "plot_resid_return_bar",
    "plot_risk_reduction_scatter",
    "plot_score_cost_scatter",
    "plot_score_gap",
    "plot_score_heatmap",
    "plot_traded_beta",
    "plot_turnover_cost_scatter",
    "plot_z_grid",
]
