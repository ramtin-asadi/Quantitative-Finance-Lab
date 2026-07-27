from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk import (
    capm as capm_mod,
    drawdown as drawdown_mod,
    performance as performance_mod,
    var_backtesting,
)
from quantfinlab.risk.utils import _excess_returns

from .curves import LAB_COLORS, choose_heatmap_cmap, set_plot_style

_LABEL_REPLACEMENTS = (
    ("MaxSharpe (FrontierGrid)", "MS-FG"),
    ("MaxSharpe", "MS"),
    ("Ridge MV", "Ridge"),
    ("RidgeMV", "Ridge"),
    ("SampleCov", "Samp"),
    ("Sample", "Samp"),
    ("LedoitWolf", "LW"),
    ("BayesSteinMomentum", "BSM"),
    ("BayesStein", "BS"),
    ("Momentum", "Mom"))


def shorten_label(label: object, *, max_len: int = 24) -> str:
    text = str(label)
    for old, new in _LABEL_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace("(", " ").replace(")", " ").replace(",", " ")
    text = " ".join(text.replace("_", " ").split())
    if len(text) <= int(max_len):
        return text
    tokens = text.split()
    if len(tokens) > 1:
        compact = " ".join(tokens[: max(1, len(tokens) - 1)])
        if len(compact) <= int(max_len):
            return compact
    return text[: max(int(max_len) - 3, 1)].rstrip() + "..."


def short_label_map(names, *, max_len: int = 24, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    overrides = {str(k): str(v) for k, v in (overrides or {}).items()}
    out: dict[str, str] = {}
    used: dict[str, int] = {}
    for name in [str(n) for n in names]:
        label = overrides.get(name, shorten_label(name, max_len=max_len))
        base = label
        used[base] = used.get(base, 0) + 1
        if used[base] > 1:
            suffix = f" {used[base]}"
            label = f"{base[: max(int(max_len) - len(suffix), 1)].rstrip()}{suffix}"
        out[name] = label
    return out


def make_color_map(names, palette=LAB_COLORS) -> dict[str, str]:
    names_list = [str(n) for n in names]
    return {n: palette[i % len(palette)] for i, n in enumerate(names_list)}


def auto_grid(
    n_panels: int,
    *,
    ncols: int = 2,
    figsize: tuple[float, float] = (11, 7),
    sharex: bool = False,
    sharey: bool = False,
):
    n = int(n_panels)
    if n <= 0:
        raise InputError("n_panels must be positive.")
    c = max(int(ncols), 1)
    rows = int(np.ceil(n / c))
    fig, axes = plt.subplots(rows, c, figsize=figsize, sharex=sharex, sharey=sharey)
    axes_arr = np.asarray([axes]) if isinstance(axes, plt.Axes) else np.asarray(axes).reshape(-1)
    return fig, axes_arr


def turn_off_unused_axes(axes, *, used: int) -> None:
    arr = np.asarray(axes).reshape(-1)
    for i in range(max(int(used), 0), len(arr)):
        arr[i].axis("off")


def _coerce_object_returns(objects) -> dict[str, pd.Series]:
    if isinstance(objects, pd.DataFrame):
        data = {str(c): objects[c] for c in objects.columns}
    elif isinstance(objects, Mapping):
        data = {str(k): v for k, v in objects.items()}
    else:
        raise InputError("objects must be a dict[name -> return series] or DataFrame.")
    out: dict[str, pd.Series] = {}
    for name, val in data.items():
        s = pd.Series(val, copy=True)
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if s.empty:
            continue
        if not isinstance(s.index, pd.DatetimeIndex):
            idx = pd.to_datetime(s.index, errors="coerce")
            if idx.notna().all():
                s.index = pd.DatetimeIndex(idx)
                s = s.sort_index()
        out[str(name)] = s
    if not out:
        raise InputError("No non-empty return series remain after cleaning.")
    return out


def _as_series(x) -> pd.Series:
    return pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def plot_nav_compare(
    ax: plt.Axes,
    objects,
    *,
    colors: dict[str, str] | None = None,
    title: str = "Cumulative NAV",
) -> plt.Axes:
    set_plot_style()
    obj = _coerce_object_returns(objects)
    for name, r in obj.items():
        nav = performance_mod.nav_series(r)
        ax.plot(nav.index, nav.values, lw=1.8, label=shorten_label(name), color=(colors or {}).get(name))
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.legend()
    return ax


def plot_drawdown_compare_objects(
    ax: plt.Axes,
    objects,
    *,
    colors: dict[str, str] | None = None,
    title: str = "Drawdown",
) -> plt.Axes:
    set_plot_style()
    obj = _coerce_object_returns(objects)
    for name, r in obj.items():
        dd = drawdown_mod.drawdown_series(r, input_kind="returns")
        ax.plot(dd.index, dd.values, lw=1.4, label=shorten_label(name), color=(colors or {}).get(name))
    ax.axhline(0.0, color="#444", lw=1.0)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend()
    return ax


def plot_rolling_vol(
    ax: plt.Axes,
    returns,
    *,
    windows: Sequence[int] = (20, 60, 252),
    annualization: float = 252.0,
    name: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    set_plot_style()
    r = _as_series(returns)
    wlist = [int(w) for w in windows if int(w) > 1]
    if len(wlist) == 0:
        raise InputError("windows must contain at least one integer > 1.")
    for w in wlist:
        rv = r.rolling(w).std(ddof=1) * np.sqrt(float(annualization))
        ax.plot(rv.index, rv.values, lw=1.4, label=f"{w}d")
    ax.set_title(title or (f"Rolling Vol - {shorten_label(name)}" if name else "Rolling Vol"))
    ax.set_ylabel("Ann. Vol")
    ax.legend()
    return ax


def plot_rolling_volatility(
    ax: plt.Axes,
    objects,
    *,
    windows: Sequence[int] = (20, 60, 252),
    annualization: float = 252.0,
    title: str = "Rolling Volatility",
) -> plt.Axes:
    set_plot_style()
    if isinstance(objects, (Mapping, pd.DataFrame)):
        obj = _coerce_object_returns(objects)
        w = max(int(x) for x in windows if int(x) > 1)
        for name, r in obj.items():
            rv = r.rolling(w).std(ddof=1) * np.sqrt(float(annualization))
            ax.plot(rv.index, rv.values, lw=1.2, label=shorten_label(name))
        ax.set_title(title)
        ax.set_ylabel(f"Ann. Vol ({w}d)")
        ax.legend(ncol=2)
        return ax
    return plot_rolling_vol(
        ax,
        objects,
        windows=windows,
        annualization=annualization,
        title=title,
    )


def plot_tail_shape_bars(
    ax: plt.Axes,
    tail_shape_table: pd.DataFrame,
    *,
    metric: str = "tail_ratio_95_05",
    title: str | None = None,
) -> plt.Axes:
    set_plot_style()
    tbl = tail_shape_table
    if tbl is None or tbl.empty or metric not in tbl.columns:
        ax.text(0.5, 0.5, "No tail data", ha="center", va="center")
        ax.axis("off")
        return ax
    s = tbl[metric].astype(float).sort_values()
    ax.barh(s.index.astype(str), s.values)
    ax.set_title(title or metric)
    return ax


def plot_var_method_bars(
    ax: plt.Axes,
    var_es_table: pd.DataFrame,
    *,
    metric_contains: str = "var",
    title: str = "VaR/ES by Method",
) -> plt.Axes:
    set_plot_style()
    tbl = var_es_table
    if tbl is None or tbl.empty:
        ax.text(0.5, 0.5, "No VaR/ES data", ha="center", va="center")
        ax.axis("off")
        return ax
    cols = [c for c in tbl.columns if metric_contains.lower() in str(c).lower()]
    if not cols:
        ax.text(0.5, 0.5, "Metric not found", ha="center", va="center")
        ax.axis("off")
        return ax
    tbl[cols].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Loss")
    return ax


def plot_var_backtest(
    ax: plt.Axes,
    returns,
    *,
    alpha: float = 0.05,
    lookback: int = 252,
    method: str = "hist",
    methods: Sequence[str] | None = None,
    name: str | None = None,
    backtest: Mapping[str, Any] | None = None,
) -> plt.Axes:
    set_plot_style()
    method_norm = str(method).strip().lower()
    chosen_method = method_norm
    if method_norm == "best":
        table = var_backtesting.var_backtest_table(
            {"_object": pd.Series(returns)},
            alpha=alpha,
            methods=(list(methods) if methods is not None else ["hist", "cf", "fhs"]),
            lookback=lookback,
        )
        best_map = var_backtesting.best_var_methods(table)
        chosen_method = str(best_map.get("_object", "hist"))

    st = (
        backtest
        if backtest is not None
        else var_backtesting.breach_stats(
            returns,
            alpha=alpha,
            lookback=lookback,
            method=chosen_method,
        )
    )
    z = st["series"]
    if z.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.axis("off")
        return ax
    br = st["breach"]
    ax.plot(z.index, z["ret"].values, lw=0.9, alpha=0.9, label="return")
    ax.plot(
        z.index,
        z["var_q"].values,
        lw=1.8,
        ls="--",
        label=f"rolling VaR q({int(alpha * 100)}%) [{chosen_method}]",
    )
    ax.scatter(z.index[br], z.loc[br, "ret"].values, s=14, marker="x", label="breach")
    ax.set_title(f"VaR - {shorten_label(name)}" if name else "VaR Backtest")
    ax.set_ylabel("Daily Return")
    ax.legend()
    return ax


plot_var_exceptions = plot_var_backtest


def plot_var_backtest_summary(
    ax: plt.Axes,
    var_backtest_table: pd.DataFrame,
    *,
    title: str = "VaR Backtest Accuracy",
) -> plt.Axes:
    set_plot_style()
    tbl = var_backtest_table
    if tbl is None or tbl.empty:
        ax.text(0.5, 0.5, "No backtest data", ha="center", va="center")
        ax.axis("off")
        return ax
    if isinstance(tbl.index, pd.MultiIndex) and "is_best" in tbl.columns:
        show = tbl[tbl["is_best"]].copy()
        labels = [shorten_label(i[0]) for i in show.index]
    else:
        show = tbl.copy()
        labels = [shorten_label(i) for i in show.index]
    metric = "accuracy_score" if "accuracy_score" in show.columns else "abs_coverage_error"
    s = pd.Series(show[metric].to_numpy(dtype=float), index=labels).sort_values()
    ax.barh(s.index, s.values)
    ax.set_title(title)
    ax.set_xlabel(metric)
    return ax


def plot_stress_bar(
    ax: plt.Axes,
    stress_tbl: pd.DataFrame,
    *,
    window: str,
    metric: str = "cum_return",
    ascending: bool = True,
) -> plt.Axes:
    set_plot_style()
    if stress_tbl is None or stress_tbl.empty:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center")
        ax.axis("off")
        return ax
    if metric not in stress_tbl.columns:
        raise InputError(f"metric {metric!r} not in stress table.")
    if window not in stress_tbl.index:
        ax.text(0.5, 0.5, "Window not found", ha="center", va="center")
        ax.axis("off")
        return ax
    sub = stress_tbl.loc[window]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T
    if "object" not in sub.columns:
        raise InputError("stress_tbl must include 'object' column.")
    s = pd.Series(sub[metric].to_numpy(dtype=float), index=[shorten_label(v) for v in sub["object"].astype(str)], dtype=float)
    s = s.sort_values(ascending=bool(ascending))
    ax.barh(s.index, s.values)
    ax.set_title(shorten_label(f"{window} {metric}", max_len=30))
    ax.set_xlabel(metric)
    return ax


def plot_stress_heatmap(
    ax: plt.Axes,
    stress_table: pd.DataFrame,
    *,
    metric: str = "cum_return",
    title: str = "Historical Stress Windows",
    cmap: str | None = None,
) -> plt.Axes:
    set_plot_style()
    tbl = stress_table
    if tbl is None or tbl.empty or metric not in tbl.columns:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center")
        ax.axis("off")
        return ax
    data = tbl.copy()
    if "object" not in data.columns:
        if "object" in data.index.names:
            data = data.reset_index()
        else:
            ax.text(0.5, 0.5, "No object column", ha="center", va="center")
            ax.axis("off")
            return ax
    if "window" not in data.columns:
        data = data.reset_index().rename(columns={data.index.name or "index": "window"})
    piv = data.pivot_table(index="window", columns="object", values=metric, aggfunc="first")
    if piv.empty:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center")
        ax.axis("off")
        return ax
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap or choose_heatmap_cmap(metric_name=metric, kind="stress"))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([shorten_label(c) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.astype(str))
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_capm_scatter(
    ax: plt.Axes,
    returns,
    market_ret,
    *,
    rf_daily: float | pd.Series = 0.0,
    name: str | None = None,
    color: str | None = None,
) -> plt.Axes:
    set_plot_style()
    r = pd.to_numeric(pd.Series(returns), errors="coerce")
    m = pd.to_numeric(pd.Series(market_ret), errors="coerce")
    z = pd.concat([m.rename("x"), r.rename("y")], axis=1).dropna()
    if z.empty:
        ax.text(0.5, 0.5, "No CAPM data", ha="center", va="center")
        ax.axis("off")
        return ax
    x = _excess_returns(z["x"], rf_daily)
    y = _excess_returns(z["y"], rf_daily)
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    x, y = z["x"], z["y"]
    alpha, beta, r2 = capm_mod.capm_ols(y, x)
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    dot_color = LAB_COLORS[7]
    line_color = color if color is not None else LAB_COLORS[1]
    ax.scatter(xv, yv, s=10, alpha=0.10, color=dot_color)
    if np.isfinite(alpha) and np.isfinite(beta):
        xs = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 200)
        ax.plot(xs, alpha + beta * xs, lw=2.0, color=line_color)
    ax.axhline(0.0, color="#444", lw=1.0)
    ax.axvline(0.0, color="#444", lw=1.0)
    ax.set_title(f"CAPM - {shorten_label(name)}" if name else "CAPM Fit")
    ax.set_xlabel("Market Excess Return")
    ax.set_ylabel("Object Excess Return")
    if np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(r2):
        ax.text(
            0.02,
            0.98,
            f"alpha(d): {alpha:.4f}\nbeta: {beta:.3f}\nr2: {r2:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )
    return ax


def plot_rolling_beta_compare(
    ax: plt.Axes,
    capm_roll: dict[str, pd.DataFrame],
    *,
    window: int = 252,
    metric: str = "beta",
    title: str | None = None,
) -> plt.Axes:
    set_plot_style()
    col = f"{metric}_{int(window)}"
    found = 0
    for name, df in capm_roll.items():
        if df is None or df.empty or col not in df.columns:
            continue
        ax.plot(df.index, df[col].values, lw=1.4, label=shorten_label(name))
        found += 1
    if found == 0:
        ax.text(0.5, 0.5, "No rolling data", ha="center", va="center")
        ax.axis("off")
        return ax
    if metric == "beta":
        ax.axhline(1.0, color="#444", lw=1.0, ls="--")
        ax.set_title(title or f"Rolling Beta ({window}d)")
        ax.set_ylabel("Beta")
    else:
        ax.axhline(0.0, color="#444", lw=1.0)
        ax.set_title(title or f"Rolling Correlation ({window}d)")
        ax.set_ylabel("Correlation")
    ax.legend(ncol=2)
    return ax


def plot_rolling_beta(
    ax: plt.Axes,
    rolling_beta: dict[str, pd.DataFrame],
    *,
    window: int | None = None,
    title: str = "Rolling CAPM Beta",
) -> plt.Axes:
    if window is None:
        windows: list[int] = []
        for df in rolling_beta.values():
            windows.extend(int(str(c).split("_", 1)[1]) for c in df.columns if str(c).startswith("beta_"))
        window = max(windows) if windows else 252
    return plot_rolling_beta_compare(ax, rolling_beta, window=int(window), metric="beta", title=title)


def plot_corr_heatmap(
    ax: plt.Axes,
    corr: pd.DataFrame,
    *,
    annotate: bool = True,
    cmap: str | None = None,
    title: str = "Correlation Matrix",
) -> plt.Axes:
    set_plot_style()
    if corr is None or corr.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        ax.axis("off")
        return ax
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap=cmap or choose_heatmap_cmap(metric_name="correlation"))
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([shorten_label(c) for c in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([shorten_label(i) for i in corr.index])
    ax.set_title(title)
    if annotate:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_top_contrib(
    ax: plt.Axes,
    contrib: pd.Series | pd.DataFrame | dict[str, float],
    *,
    title: str = "Top Contributions",
    k: int = 10,
) -> plt.Axes:
    set_plot_style()
    if isinstance(contrib, pd.Series):
        s = contrib.copy()
    elif isinstance(contrib, pd.DataFrame):
        if contrib.shape[0] == 1:
            s = contrib.iloc[0]
        elif contrib.shape[1] == 1:
            s = contrib.iloc[:, 0]
        else:
            raise InputError("Contribution DataFrame must have one row or one column.")
    elif isinstance(contrib, dict):
        s = pd.Series(contrib, dtype=float)
    else:
        s = pd.Series(contrib, dtype=float)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if s.empty:
        ax.text(0.5, 0.5, "No contribution data", ha="center", va="center")
        ax.axis("off")
        return ax
    s.index = [str(i) for i in s.index]
    top_idx = s.abs().sort_values(ascending=False).head(int(max(k, 1))).index
    show = s.loc[top_idx].sort_values()
    ax.barh(show.index, show.values)
    ax.set_title(title)
    ax.set_xlabel("Contribution")
    return ax


def plot_contribution_bars(
    ax: plt.Axes,
    contribution_table: pd.Series | pd.DataFrame | dict[str, float],
    *,
    top_k: int = 10,
    title: str = "Top Contributions",
) -> plt.Axes:
    if isinstance(contribution_table, pd.DataFrame) and contribution_table.shape[0] > 1 and contribution_table.shape[1] > 1:
        contribution_table = contribution_table.apply(pd.to_numeric, errors="coerce").abs().mean(axis=0)
    return plot_top_contrib(ax, contribution_table, title=title, k=top_k)


__all__ = [
    "LAB_COLORS",
    "auto_grid",
    "make_color_map",
    "plot_capm_scatter",
    "plot_contribution_bars",
    "plot_corr_heatmap",
    "plot_drawdown_compare_objects",
    "plot_nav_compare",
    "plot_rolling_beta",
    "plot_rolling_beta_compare",
    "plot_rolling_vol",
    "plot_rolling_volatility",
    "plot_stress_bar",
    "plot_stress_heatmap",
    "plot_tail_shape_bars",
    "plot_top_contrib",
    "plot_var_backtest",
    "plot_var_backtest_summary",
    "plot_var_exceptions",
    "plot_var_method_bars",
    "set_plot_style",
    "short_label_map",
    "shorten_label",
    "turn_off_unused_axes",
]
