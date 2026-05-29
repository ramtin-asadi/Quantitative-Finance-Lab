from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quantfinlab.plotting.curves import set_plot_style
from quantfinlab.portfolio.selection import calc_drawdown


def _result_series(results, attr):
    if hasattr(results, attr):
        return getattr(results, attr)
    if isinstance(results, Mapping):
        rows = {}
        for name, res in results.items():
            if hasattr(res, attr):
                rows[str(name)] = getattr(res, attr)
            elif isinstance(res, Mapping) and attr in res:
                rows[str(name)] = pd.Series(res[attr])
        return pd.DataFrame(rows)
    return pd.DataFrame(results)


def _heatmap(ax, data, title=None, cmap="coolwarm", center=False):
    set_plot_style()
    df = pd.DataFrame(data).dropna(how="all")
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    vals = df.to_numpy(dtype=float)
    if center:
        vmax = max(float(np.nanmax(np.abs(vals))), 1e-12)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(vals, aspect="auto", cmap=cmap)
    ax.set_title(title or "")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(x) for x in df.index])
    step = max(1, len(df.columns) // 8)
    locs = list(range(0, len(df.columns), step))
    ax.set_xticks(locs)
    labels = []
    for i in locs:
        value = df.columns[i]
        labels.append(pd.Timestamp(value).strftime("%Y-%m") if isinstance(value, pd.Timestamp) else str(value))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def factor_growth(ax, factors, title=None):
    set_plot_style()
    f = pd.DataFrame(factors).astype(float)
    growth = (1.0 + f).cumprod()
    growth.plot(ax=ax, lw=1.2)
    ax.set_title(title or "Factor Growth")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


def factor_drawdowns(ax, factors, title=None):
    set_plot_style()
    f = pd.DataFrame(factors).astype(float)
    dd = (1.0 + f).cumprod().apply(calc_drawdown)
    dd.plot(ax=ax, lw=1.1)
    ax.set_title(title or "Factor Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


def factor_corr(ax, factors, title=None):
    set_plot_style()
    corr = pd.DataFrame(factors).astype(float).corr()
    vals = corr.to_numpy(dtype=float)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title or "Factor Correlation")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def factor_state_history(ax, z_factor, title=None):
    z = pd.DataFrame(z_factor).astype(float).tail(84).T
    return _heatmap(ax, z, title or "Factor State History", cmap="coolwarm", center=True)


def latest_factor_state(ax, z_factor, title=None):
    set_plot_style()
    z = pd.DataFrame(z_factor).dropna(how="all")
    if z.empty:
        ax.text(0.5, 0.5, "No factor state", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    last = z.iloc[-1].sort_values()
    ax.barh(last.index, last.values)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_title(title or "Latest Factor State")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def beta_heatmap(ax, beta, date=None, title=None):
    set_plot_style()
    if isinstance(beta.columns, pd.MultiIndex):
        dt = pd.Timestamp(date) if date is not None else beta.dropna(how="all").index.max()
        row = beta.loc[:dt].dropna(how="all").iloc[-1].unstack("factor")
    else:
        row = pd.DataFrame(beta)
    vals = row.to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(vals))), 1e-12)
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title or "Factor Betas")
    ax.set_xticks(range(len(row.columns)))
    ax.set_xticklabels(row.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(row.index)))
    ax.set_yticklabels(row.index)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def r2_history(ax, r2, title=None):
    set_plot_style()
    df = pd.DataFrame(r2).astype(float)
    df.mean(axis=1).dropna().plot(ax=ax, lw=1.4, color="#0072B2", label="Average R2")
    ax.set_title(title or "Rolling Factor Model R2")
    ax.set_ylabel("R2")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    return ax


def score_history(ax, scores, title=None):
    s = pd.DataFrame(scores).astype(float).tail(84).T
    return _heatmap(ax, s, title or "Score History", cmap="coolwarm", center=True)


def latest_score_bar(ax, scores, title=None):
    set_plot_style()
    s = pd.Series(scores.iloc[-1] if isinstance(scores, pd.DataFrame) else scores, dtype=float).dropna().sort_values()
    if s.empty:
        ax.text(0.5, 0.5, "No scores", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    ax.barh(s.index, s.values)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_title(title or "Latest Scores")
    ax.grid(True, axis="x", alpha=0.25)
    return ax


def signal_ic(ax, ic, title=None):
    set_plot_style()
    df = pd.DataFrame(ic)
    if "rank_ic" in df.columns:
        vals = df["rank_ic"].astype(float).sort_values()
        ax.barh(vals.index, vals.values)
        ax.axvline(0.0, color="black", lw=0.8)
    else:
        df.astype(float).plot(ax=ax, lw=1.1)
        ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title or "Signal Rank IC")
    ax.grid(True, alpha=0.25)
    return ax


def top_bottom_payoff(ax, spread, title=None):
    set_plot_style()
    df = pd.DataFrame(spread)
    if "top_minus_bottom" in df.columns:
        vals = df["top_minus_bottom"].astype(float).sort_values()
        ax.barh(vals.index, vals.values)
        ax.axvline(0.0, color="black", lw=0.8)
    else:
        df.astype(float).plot(ax=ax, lw=1.1)
        ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title or "Top-minus-Bottom")
    ax.grid(True, alpha=0.25)
    return ax


def strategy_nav(ax, results, title=None):
    set_plot_style()
    nav = _result_series(results, "net_values")
    if nav.empty:
        ax.text(0.5, 0.5, "No NAV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    nav = nav.divide(nav.dropna().iloc[0])
    nav.plot(ax=ax, lw=1.2)
    ax.set_title(title or "Strategy NAV")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


def strategy_drawdowns(ax, results, title=None):
    set_plot_style()
    nav = _result_series(results, "net_values")
    if nav.empty:
        ax.text(0.5, 0.5, "No NAV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    dd = nav.apply(calc_drawdown)
    dd.plot(ax=ax, lw=1.1)
    ax.set_title(title or "Strategy Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


def rolling_active_return(ax, results, benchmark, window=12, title=None):
    set_plot_style()
    ret = _result_series(results, "net_returns")
    if ret.empty or benchmark not in ret.columns:
        ax.text(0.5, 0.5, "Missing benchmark", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    bench = ret[benchmark]
    for col in ret.columns:
        if col == benchmark:
            continue
        active = ((1.0 + ret[col]).rolling(int(window)).apply(np.prod, raw=True) - 1.0) - (
            (1.0 + bench).rolling(int(window)).apply(np.prod, raw=True) - 1.0
        )
        ax.plot(active.index, active.values, lw=1.0, label=col)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title or f"Rolling Active Return vs {benchmark}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


def weight_history(ax, weights, title=None):
    w = pd.DataFrame(weights).astype(float).tail(60)
    w = w.loc[:, w.mean().sort_values(ascending=False).index]
    return _heatmap(ax, w.T, title or "Weight History", cmap="Blues", center=False)


def turnover_history(ax, results, title=None):
    set_plot_style()
    turnover = _result_series(results, "turnover")
    if turnover.empty:
        ax.text(0.5, 0.5, "No turnover", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    turnover.plot(ax=ax, lw=1.1)
    ax.set_title(title or "Turnover")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


def factor_exposure(ax, exposure, title=None):
    set_plot_style()
    e = pd.DataFrame(exposure).astype(float).dropna(how="all")
    if e.empty:
        ax.text(0.5, 0.5, "No exposure", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    e.plot(ax=ax, lw=1.1)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title or "Portfolio Factor Exposure")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return ax


__all__ = [
    "beta_heatmap",
    "factor_corr",
    "factor_drawdowns",
    "factor_exposure",
    "factor_growth",
    "factor_state_history",
    "latest_factor_state",
    "latest_score_bar",
    "r2_history",
    "rolling_active_return",
    "score_history",
    "signal_ic",
    "strategy_drawdowns",
    "strategy_nav",
    "top_bottom_payoff",
    "turnover_history",
    "weight_history",
]
