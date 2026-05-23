from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.macro.indicators import BLOCK_COLUMNS
from quantfinlab.plotting.curves import set_plot_style
from quantfinlab.portfolio.selection import calc_drawdown


def _get_ax(ax=None):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _heatmap(ax, data: pd.DataFrame, *, title: str, cmap: str = "coolwarm"):
    if data is None or data.empty:
        ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = data.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if vals.size else 0.0
    vmax = max(float(vmax), 1e-8)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=7)
    for i, row in enumerate(data.index):
        for j, col in enumerate(data.columns):
            value = data.loc[row, col]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_macro_blocks(data: pd.DataFrame, *, ax=None, title: str = "Latest condition blocks"):
    set_plot_style()
    ax = _get_ax(ax)
    cols = [c for c in BLOCK_COLUMNS if c in data.columns]
    if not cols:
        ax.text(0.5, 0.5, "No blocks", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    latest = data[cols].dropna(how="all").iloc[-1].sort_values()
    colors = ["#069AF3" if value <= 0 else "#FE420F" for value in latest]
    ax.barh(latest.index, latest.values, color=colors)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Latest z-score")
    ax.grid(True, alpha=0.3)
    return ax


def plot_block_correlation(data: pd.DataFrame, *, ax=None, title: str = "Block correlation"):
    set_plot_style()
    ax = _get_ax(ax)
    cols = [c for c in BLOCK_COLUMNS if c in data.columns]
    corr = data[cols].corr() if cols else pd.DataFrame()
    return _heatmap(ax, corr, title=title)


def plot_fci_models(fci_models: pd.DataFrame, *, ax=None, title: str = "FCI models"):
    set_plot_style()
    ax = _get_ax(ax)
    if fci_models is None or fci_models.empty:
        ax.text(0.5, 0.5, "No FCI data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    fci_models.plot(ax=ax, lw=1.1)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    return ax


def plot_fci_vs_nfci(
    fci: pd.Series | pd.DataFrame,
    nfci: pd.Series | None = None,
    *,
    ax=None,
    title: str = "Selected FCI",
):
    set_plot_style()
    ax = _get_ax(ax)
    fci_s = fci.iloc[:, 0] if isinstance(fci, pd.DataFrame) else pd.Series(fci)
    fci_s.dropna().plot(ax=ax, lw=1.3, label=fci_s.name or "FCI")
    if nfci is not None:
        pd.Series(nfci).reindex(fci_s.index).dropna().plot(ax=ax, lw=1.0, label="NFCI")
    else:
        pct = fci_s.expanding(24).rank(pct=True)
        high = pct[pct >= 0.80]
        if len(high):
            ax.scatter(high.index, fci_s.reindex(high.index), s=8, color="#D55E00", label="High stress")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    return ax


def plot_fci_model_scores(scoreboard: pd.DataFrame, *, ax=None, title: str = "FCI score"):
    set_plot_style()
    ax = _get_ax(ax)
    if scoreboard is None or scoreboard.empty or "final_score" not in scoreboard.columns:
        ax.text(0.5, 0.5, "No scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = scoreboard["final_score"].sort_values()
    ax.barh(vals.index, vals.values)
    ax.set_title(title)
    ax.set_xlabel("Final score")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_fci_quintile_stress(report: pd.DataFrame, *, ax=None, title: str = "Stress by FCI quintile"):
    set_plot_style()
    ax = _get_ax(ax)
    if report is None or report.empty or "future_stress" not in report.columns:
        ax.text(0.5, 0.5, "No quintile report", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    report["future_stress"].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Future stress")
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_sector_regime_heatmap(table: pd.DataFrame, *, ax=None, title: str = "ETF return by FCI regime"):
    set_plot_style()
    ax = _get_ax(ax)
    return _heatmap(ax, table, title=title, cmap="RdYlGn")


def plot_defensive_cyclical_spread(spread: pd.Series, *, ax=None, title: str = "Spread"):
    set_plot_style()
    ax = _get_ax(ax)
    s = pd.Series(spread).dropna()
    if s.empty:
        ax.text(0.5, 0.5, "No spread", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    s.plot(ax=ax, lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    return ax


def plot_strategy_growth(nav: pd.DataFrame, *, ax=None, title: str = "Strategy growth"):
    set_plot_style()
    ax = _get_ax(ax)
    if nav is None or nav.empty:
        ax.text(0.5, 0.5, "No strategy NAV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    nav.plot(ax=ax, lw=1.1)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Growth of 1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    return ax


def plot_strategy_drawdowns(nav: pd.DataFrame, *, ax=None, title: str = "Strategy drawdowns"):
    set_plot_style()
    ax = _get_ax(ax)
    if nav is None or nav.empty:
        ax.text(0.5, 0.5, "No strategy NAV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    dd = nav.apply(calc_drawdown)
    dd.plot(ax=ax, lw=1.0)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    return ax


def plot_strategy_weights(weights: pd.DataFrame, *, ax=None, title: str = "Strategy weights"):
    set_plot_style()
    ax = _get_ax(ax)
    if weights is None or weights.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    w = weights.tail(36).loc[:, weights.tail(36).sum(axis=0).sort_values(ascending=False).index]
    im = ax.imshow(w.T.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0.0)
    ax.set_title(title)
    ax.set_yticks(range(len(w.columns)))
    ax.set_yticklabels(w.columns, fontsize=7)
    step = max(1, len(w.index) // 6)
    locs = list(range(0, len(w.index), step))
    ax.set_xticks(locs)
    ax.set_xticklabels([pd.Timestamp(w.index[i]).strftime("%Y-%m") for i in locs], rotation=45, ha="right", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_latest_scores(
    table: pd.DataFrame,
    *,
    ax=None,
    title: str = "Latest scores and weights",
    score_col: str = "final_score",
    weight_col: str = "portfolio_weight",
):
    set_plot_style()
    ax = _get_ax(ax)
    if table is None or table.empty or score_col not in table.columns:
        ax.text(0.5, 0.5, "No latest scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    t = table.copy()
    if "sector" in t.columns:
        t = t.set_index("sector")
    t = t.sort_values(score_col).tail(12)
    ax.barh(t.index, t[score_col].astype(float), label="Score")
    if weight_col in t.columns:
        ax2 = ax.twiny()
        ax2.plot(t[weight_col].astype(float), range(len(t)), marker="o", color="#D55E00", lw=1.0, label="Weight")
        ax2.set_xlabel("Weight")
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.grid(True, axis="x", alpha=0.3)
    return ax


__all__ = [
    "plot_block_correlation",
    "plot_defensive_cyclical_spread",
    "plot_fci_model_scores",
    "plot_fci_models",
    "plot_fci_quintile_stress",
    "plot_fci_vs_nfci",
    "plot_latest_scores",
    "plot_macro_blocks",
    "plot_sector_regime_heatmap",
    "plot_strategy_drawdowns",
    "plot_strategy_growth",
    "plot_strategy_weights",
]
