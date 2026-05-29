from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from quantfinlab.plotting.curves import choose_heatmap_cmap, set_plot_style
from quantfinlab.plotting.risk import shorten_label


def _empty(ax: plt.Axes, text: str = "No data") -> plt.Axes:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return ax


def _date_ticks(ax: plt.Axes, index: pd.Index, n: int = 8) -> None:
    if len(index) == 0:
        return
    locs = np.linspace(0, len(index) - 1, min(int(n), len(index))).astype(int)
    labels = [pd.Timestamp(index[i]).strftime("%Y-%m") for i in locs]
    ax.set_xticks(locs)
    ax.set_xticklabels(labels, rotation=35, ha="right")


def feature_corr(
    ax: plt.Axes,
    x: pd.DataFrame,
    *,
    title: str = "Feature correlation",
    annotate: bool = False,
) -> plt.Axes:
    set_plot_style()
    if x is None or x.empty:
        return _empty(ax)
    corr = x.corr()
    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap=choose_heatmap_cmap("correlation"))
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([shorten_label(c, max_len=16) for c in corr.columns], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([shorten_label(c, max_len=16) for c in corr.index], fontsize=7)
    if annotate and len(corr) <= 16:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def pca_explained(
    ax: plt.Axes,
    explained: pd.DataFrame,
    *,
    title: str = "PCA explained variance",
) -> plt.Axes:
    set_plot_style()
    if explained is None or explained.empty:
        return _empty(ax)
    ratios = explained["explained_variance_ratio"].astype(float)
    ax.bar(range(len(ratios)), ratios.values, alpha=0.75, label="component")
    ax.plot(range(len(ratios)), explained["cumulative"].astype(float).values, marker="o", lw=1.8, label="cumulative")
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(ratios.index.astype(str), rotation=0)
    ax.set_ylim(0.0, min(1.05, max(0.25, float(explained["cumulative"].max()) * 1.08)))
    ax.set_title(title)
    ax.set_ylabel("Share of variance")
    ax.legend(loc="best")
    return ax


def pca_loadings(
    ax: plt.Axes,
    loadings: pd.DataFrame,
    *,
    title: str = "PCA loadings",
    top_n: int | None = 18,
) -> plt.Axes:
    set_plot_style()
    if loadings is None or loadings.empty:
        return _empty(ax)
    L = loadings.copy()
    if top_n is not None and L.shape[0] > int(top_n):
        order = L.abs().max(axis=1).sort_values(ascending=False).head(int(top_n)).index
        L = L.loc[order]
    vmax = max(float(np.nanmax(np.abs(L.to_numpy(dtype=float)))), 1e-6)
    im = ax.imshow(L.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(L.index)))
    ax.set_yticklabels([shorten_label(i, max_len=22) for i in L.index], fontsize=7)
    ax.set_xticks(range(len(L.columns)))
    ax.set_xticklabels(L.columns.astype(str))
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def model_quality(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    metric: str = "economic_separation",
    title: str | None = None,
) -> plt.Axes:
    set_plot_style()
    if table is None or table.empty:
        return _empty(ax)
    t = table.copy()
    if "model" in t.columns:
        t = t.set_index("model")
    if metric not in t.columns:
        for candidate in ["balanced_accuracy", "macro_f1", "economic_separation", "posterior_confidence", "silhouette"]:
            if candidate in t.columns:
                metric = candidate
                break
    if metric not in t.columns:
        return _empty(ax, "Metric missing")
    s = t[metric].astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    if s.empty:
        return _empty(ax, "Metric missing")
    ax.barh([shorten_label(i, max_len=28) for i in s.index], s.values)
    ax.set_title(title or metric.replace("_", " ").title())
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def regime_probabilities(
    ax: plt.Axes,
    proba: pd.DataFrame,
    *,
    title: str = "Regime probabilities",
) -> plt.Axes:
    set_plot_style()
    if proba is None or proba.empty:
        return _empty(ax)
    p = proba.astype(float).clip(lower=0.0)
    denom = p.sum(axis=1).replace(0.0, np.nan)
    p = p.div(denom, axis=0).fillna(0.0)
    ax.stackplot(p.index, p.T.values, labels=[shorten_label(c) for c in p.columns], alpha=0.82)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", ncol=min(4, len(p.columns)), fontsize=7)
    return ax


def regime_profiles(
    ax: plt.Axes,
    profile: pd.DataFrame,
    *,
    cols: list[str] | None = None,
    title: str = "Regime profile",
) -> plt.Axes:
    set_plot_style()
    if profile is None or profile.empty:
        return _empty(ax)
    keep = cols if cols is not None else [c for c in profile.columns if c not in {"observations", "share"}]
    keep = [c for c in keep if c in profile.columns]
    if not keep:
        return _empty(ax, "No profile columns")
    z = profile[keep].astype(float)
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = z.fillna(0.0)
    vmax = max(float(np.nanmax(np.abs(z.to_numpy(dtype=float)))), 1e-6)
    im = ax.imshow(z.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels([f"state {i}" for i in z.index])
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels([shorten_label(c, max_len=18) for c in z.columns], rotation=45, ha="right", fontsize=7)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def transition_heatmap(
    ax: plt.Axes,
    labels_or_matrix,
    *,
    title: str = "Transition matrix",
) -> plt.Axes:
    set_plot_style()
    if isinstance(labels_or_matrix, pd.DataFrame):
        mat = labels_or_matrix.astype(float)
    else:
        lab = pd.Series(labels_or_matrix).dropna().astype(int)
        if len(lab) < 2:
            return _empty(ax)
        states = sorted(lab.unique())
        counts = pd.DataFrame(0.0, index=states, columns=states)
        for a, b in zip(lab.iloc[:-1], lab.iloc[1:], strict=False):
            counts.loc[int(a), int(b)] += 1.0
        mat = counts.div(counts.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    im = ax.imshow(mat.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns.astype(str))
    ax.set_yticklabels(mat.index.astype(str))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def class_balance(
    ax: plt.Axes,
    labels,
    *,
    title: str = "Class balance",
) -> plt.Axes:
    set_plot_style()
    lab = pd.Series(labels).dropna().astype(int)
    if lab.empty:
        return _empty(ax)
    share = lab.value_counts(normalize=True).sort_index()
    ax.bar([f"state {i}" for i in share.index], share.values)
    ax.set_ylim(0.0, max(0.05, float(share.max()) * 1.25))
    ax.set_title(title)
    ax.set_ylabel("Share")
    return ax


def confusion(
    ax: plt.Axes,
    matrix,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Confusion matrix",
) -> plt.Axes:
    set_plot_style()
    if isinstance(matrix, pd.DataFrame):
        mat = matrix.astype(float)
        xlabels = mat.columns.astype(str).tolist()
        ylabels = mat.index.astype(str).tolist()
    else:
        arr = np.asarray(matrix, dtype=float)
        mat = pd.DataFrame(arr)
        xlabels = labels if labels is not None else [str(i) for i in mat.columns]
        ylabels = labels if labels is not None else [str(i) for i in mat.index]
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="Blues")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Realized")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def feature_importance(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    top_n: int = 15,
    title: str = "Feature importance",
) -> plt.Axes:
    set_plot_style()
    if table is None or table.empty:
        return _empty(ax)
    t = table.copy()
    if "feature" in t.columns:
        t = t.set_index("feature")
    col = "permutation_importance" if "permutation_importance" in t.columns and t["permutation_importance"].notna().any() else "importance"
    s = t[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(int(top_n)).sort_values()
    if s.empty:
        return _empty(ax)
    ax.barh([shorten_label(i, max_len=28) for i in s.index], s.values)
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " "))
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def probability_confidence(
    ax: plt.Axes,
    proba: pd.DataFrame,
    *,
    window: int = 63,
    title: str = "Probability confidence",
) -> plt.Axes:
    set_plot_style()
    if proba is None or proba.empty:
        return _empty(ax)
    conf = proba.astype(float).max(axis=1)
    ax.plot(conf.index, conf.values, lw=1.0, alpha=0.55, label="max probability")
    ax.plot(conf.index, conf.rolling(int(window)).mean().values, lw=1.8, label=f"{window}d mean")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.set_ylabel("Confidence")
    ax.legend(loc="best")
    return ax


def weight_heatmap(
    ax: plt.Axes,
    weights: pd.DataFrame,
    *,
    last_n: int | None = 60,
    title: str = "Weights",
) -> plt.Axes:
    set_plot_style()
    if weights is None or weights.empty:
        return _empty(ax)
    W = weights.copy().astype(float).fillna(0.0)
    if last_n is not None:
        W = W.tail(int(last_n))
    W = W.loc[:, W.abs().sum(axis=0).sort_values(ascending=False).index]
    im = ax.imshow(W.T.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.35, float(W.max().max())))
    ax.set_yticks(range(len(W.columns)))
    ax.set_yticklabels(W.columns, fontsize=8)
    _date_ticks(ax, W.index)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def risky_allocation(
    ax: plt.Axes,
    series: pd.Series,
    *,
    title: str = "Risky allocation",
) -> plt.Axes:
    set_plot_style()
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return _empty(ax)
    ax.plot(s.index, s.values, lw=1.6)
    ax.fill_between(s.index, 0.0, s.values, alpha=0.15)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    return ax


__all__ = [
    "class_balance",
    "confusion",
    "feature_corr",
    "feature_importance",
    "model_quality",
    "pca_explained",
    "pca_loadings",
    "probability_confidence",
    "regime_probabilities",
    "regime_profiles",
    "risky_allocation",
    "transition_heatmap",
    "weight_heatmap",
]
