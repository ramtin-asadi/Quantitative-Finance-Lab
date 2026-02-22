from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from . import fixed_income as fi
from . import risk as rk
from .core import PortfolioState

LAB_COLORS = ["#069AF3","#FE420F", "#00008B", "#008080", "#800080",
          "#7BC8F6", "#0072B2","#04D8B2", "#CC79A7", "#FF8072", "#9614fa", "#DC143C"]

# Single source of truth for style
_LAB_STYLE = {
    "figure.figsize": (6, 3),
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 6,
}

_STYLE_APPLIED = False


def set_plot_style(colors: list[str] | None = None, *, force: bool = False) -> None:
    """
    Apply Quantitative-Finance-Lab plotting defaults globally.

    - Call inside your plotting functions (recommended) so notebooks stay clean.
    - Set force=True to re-apply after a user changes rcParams.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return

    palette = colors or LAB_COLORS
    mpl.rcParams.update(_LAB_STYLE)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)

    _STYLE_APPLIED = True


def draw_market_par_points(
    ax: plt.Axes,
    maturities: np.ndarray,
    par_yields: np.ndarray,
    *,
    label: str = "Market par",
) -> None:
    ax.plot(
        np.asarray(maturities, dtype=float),
        np.asarray(par_yields, dtype=float) * 100.0,
        "o",
        markersize=5,
        markeredgecolor="black",
        markerfacecolor="white",
        label=label,
        zorder=5,
    )


def draw_curve_lines(
    ax: plt.Axes,
    curve_table: pd.DataFrame,
    *,
    scale: float = 1.0,
    label_map: dict[str, str] | None = None,
) -> None:
    if curve_table.empty:
        return
    for method in curve_table.columns:
        label = label_map.get(str(method), str(method)) if label_map is not None else str(method)
        ax.plot(curve_table.index, curve_table[method] * scale, label=label)


def style_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: bool = True,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def plot_par_yields_history(
    ax: plt.Axes,
    par_yields: pd.DataFrame,
    *,
    title: str = "Par Yields Over Time",
) -> None:
    if par_yields is None or par_yields.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for col in par_yields.columns:
        ax.plot(par_yields.index, par_yields[col] * 100.0, label=str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.legend(ncol=4)


def plot_yield_curve_snapshots(
    ax: plt.Axes,
    par_yields: pd.DataFrame,
    *,
    tenor_cols: list[str] | None = None,
    sample_dates: list[pd.Timestamp] | None = None,
    title: str = "Yield Curve Snapshots (Par Yields)",
) -> None:
    if par_yields is None or par_yields.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    cols = tenor_cols if tenor_cols is not None else [str(c) for c in par_yields.columns]
    cols = [c for c in cols if c in par_yields.columns]
    if not cols:
        ax.text(0.5, 0.5, "No tenor columns", ha="center", va="center")
        ax.axis("off")
        return

    idx = par_yields.index
    if sample_dates is None:
        sample_dates = [idx[0], idx[len(idx) // 2], idx[max(0, len(idx) - 252)], idx[-1]]

    x = np.arange(len(cols))
    for d in sample_dates:
        dts = pd.Timestamp(d)
        if dts not in idx:
            pos = idx.searchsorted(dts, side="right") - 1
            if pos < 0:
                continue
            dts = pd.Timestamp(idx[pos])
        y = par_yields.loc[dts, cols].astype(float)
        mask = np.isfinite(y.values)
        ax.plot(x[mask], y.values[mask] * 100.0, marker="o", label=dts.strftime("%Y-%m-%d"))

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Yield (%)")
    ax.legend()


def plot_par_fit(
    ax: plt.Axes,
    market_row: pd.Series | dict,
    curves: dict[str, fi.Curve],
    *,
    rmse: pd.DataFrame | None = None,
    tenor_cols: list[str] | None = None,
    freq: int = 2,
    grid_points: int = 200,
    title: str = "Par Yield Curve Fit",
) -> None:
    _, T, par = fi.extract_par_curve(market_row, tenor_cols=tenor_cols)
    grid = np.linspace(max(1 / 12, float(np.min(T))), float(np.max(T)), grid_points)
    par_table = fi.par_curve_table(curves, grid=grid, freq=freq)

    labels: dict[str, str] = {}
    for method, curve in curves.items():
        label = curve.name
        if rmse is not None and method in rmse.index:
            label = f"{label} (IS {rmse.loc[method, 'rmse']:.6f}, OOS {rmse.loc[method, 'rmse_oos']:.6f})"
        labels[method] = label

    draw_market_par_points(ax, T, par)
    draw_curve_lines(ax, par_table, scale=100.0, label_map=labels)
    style_axis(ax, title=title, xlabel="Maturity (Years)", ylabel="Par Yield (%)")


def _plot_curve_table(
    ax: plt.Axes,
    curve_table: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    if curve_table.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    draw_curve_lines(ax, curve_table, scale=scale)
    style_axis(ax, title=title, xlabel="Maturity (Years)", ylabel=ylabel)


def plot_zero_curves(ax: plt.Axes, zero_table: pd.DataFrame, *, title: str = "Zero Curves") -> None:
    _plot_curve_table(ax, zero_table, title=title, ylabel="Zero Rate (%)", scale=100.0)


def plot_discount_curves(ax: plt.Axes, df_table: pd.DataFrame, *, title: str = "Discount Curves") -> None:
    _plot_curve_table(ax, df_table, title=title, ylabel="Discount Factor", scale=1.0)


def plot_forward_curves(
    ax: plt.Axes,
    forward_table: pd.DataFrame,
    *,
    title: str = "Forward Curves",
) -> None:
    _plot_curve_table(ax, forward_table, title=title, ylabel="Forward Rate (%)", scale=100.0)


def plot_total_pv(ax: plt.Axes, total_pv: pd.DataFrame, *, title: str = "Synthetic Book Total PV") -> None:
    if total_pv.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for method in total_pv.columns:
        ax.plot(total_pv.index, total_pv[method], label=str(method))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("PV")
    ax.legend()


def plot_bucket_pv(
    ax: plt.Axes,
    bucket_pv: pd.DataFrame,
    *,
    last_date: pd.Timestamp | None = None,
    title: str = "Bucket PV (Last Date)",
) -> None:
    if bucket_pv.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    use_date = pd.Timestamp(last_date) if last_date is not None else pd.Timestamp(bucket_pv.index.max())
    last = bucket_pv.loc[use_date]
    data = last.unstack(level=0)
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Maturity (Years)")
    ax.set_ylabel("PV")
    ax.legend()


def plot_risk_metric(
    ax: plt.Axes,
    risk: pd.DataFrame,
    *,
    metric: str,
    title: str | None = None,
) -> None:
    if risk.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    if metric not in {"pv01", "convexity"}:
        raise fi.InputError("metric must be 'pv01' or 'convexity'.")
    sub = risk.xs(metric, axis=1, level=1)
    for method in sub.columns:
        ax.plot(sub.index, sub[method], label=str(method))
    ax.set_title(title or metric.upper())
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.legend()


def plot_rmse_bars(
    ax: plt.Axes,
    rmse_df: pd.DataFrame,
    *,
    title: str = "RMSE (IS vs OOS)",
) -> None:
    if rmse_df is None or rmse_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    cols = [c for c in ["rmse", "rmse_oos"] if c in rmse_df.columns]
    if not cols:
        ax.text(0.5, 0.5, "No RMSE columns", ha="center", va="center")
        ax.axis("off")
        return
    show = rmse_df[cols].copy()
    show.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("RMSE")
    ax.legend()


def plot_bond_metric_bar(
    ax: plt.Axes,
    bond_table: pd.DataFrame,
    *,
    metric: str = "pv01",
    title: str = "Bond PV01 by Method",
) -> None:
    if bond_table is None or bond_table.empty or metric not in bond_table.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    bond_table[metric].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric)


def plot_krd_heatmap(
    ax: plt.Axes,
    krd_df: pd.DataFrame,
    *,
    method: str,
    keys: list[int] | tuple[int, ...] | None = None,
    title: str | None = None,
):
    if krd_df.empty or method not in krd_df.columns.get_level_values(0):
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return None
    if keys is None:
        keys = sorted(set(int(k) for k in krd_df.columns.get_level_values(1)))
    sub = krd_df[method].reindex(columns=list(keys))
    im = ax.imshow(sub.values.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title(title or f"KRD - {method}")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k}Y" for k in keys])
    n = len(sub.index)
    if n > 1:
        tick_idx = np.linspace(0, n - 1, 6).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([pd.Timestamp(sub.index[i]).strftime("%Y") for i in tick_idx])
    else:
        ax.set_xticks([0])
        ax.set_xticklabels([pd.Timestamp(sub.index[0]).strftime("%Y")])
    return im


def draw_table(
    ax: plt.Axes,
    table_df: pd.DataFrame,
    *,
    title: str | None = None,
    float_fmt: str = "{:.6g}",
    scale: tuple[float, float] = (1.0, 1.3),
) -> None:
    ax.axis("off")
    if table_df is None or table_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    show = table_df.copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: float_fmt.format(float(x)) if np.isfinite(x) else "nan")

    tbl = ax.table(
        cellText=show.values,
        rowLabels=show.index.tolist(),
        colLabels=show.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.scale(*scale)
    if title:
        ax.set_title(title)


# ----------------------------
# Portfolio plotting helpers
# ----------------------------

def make_color_map(names, palette=LAB_COLORS) -> dict[str, str]:
    vals = [str(n) for n in names]
    return {name: palette[i % len(palette)] for i, name in enumerate(vals)}


def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _result_field(result, key: str):
    if isinstance(result, dict):
        if key not in result:
            raise KeyError(f"Result is missing key {key!r}.")
        return result[key]
    return result[key]


def _drawdown(values: pd.Series) -> pd.Series:
    v = _as_series(values).astype(float)
    if v.empty:
        return v
    return v / v.cummax() - 1.0


def _format_date_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="x", labelrotation=25)


def plot_net_equity(
    ax: plt.Axes,
    result,
    *,
    name: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> None:
    s = _as_series(_result_field(result, "net_values")).dropna()
    if s.empty:
        ax.text(0.5, 0.5, "No net equity data", ha="center", va="center")
        ax.axis("off")
        return
    ax.plot(s.index, s.values, color=color)
    ax.set_title(title or f"{name or 'Strategy'} - Net Equity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    _format_date_axis(ax)


def plot_drawdown(
    ax: plt.Axes,
    result,
    *,
    name: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> None:
    s = _as_series(_result_field(result, "net_values")).dropna()
    if s.empty:
        ax.text(0.5, 0.5, "No net equity data", ha="center", va="center")
        ax.axis("off")
        return
    dd = _drawdown(s)
    ax.plot(dd.index, dd.values, color=color)
    ax.set_title(title or f"{name or 'Strategy'} - Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    _format_date_axis(ax)


def plot_top_weights(
    ax: plt.Axes,
    result,
    *,
    name: str | None = None,
    color: str | None = None,
    k: int = 10,
    title: str | None = None,
) -> None:
    wdf = pd.DataFrame(_result_field(result, "weights"))
    if wdf.empty:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center")
        ax.axis("off")
        return
    last_dt = pd.Timestamp(wdf.index[-1])
    w_last = wdf.loc[last_dt].astype(float)
    w_last = w_last[w_last > 0].sort_values(ascending=False)
    if w_last.empty:
        ax.text(0.5, 0.5, "No positive weights", ha="center", va="center")
        ax.axis("off")
        return
    top = w_last.head(int(k)).sort_values()
    ax.barh(top.index, top.values, color=color)
    ax.set_title(title or f"{name or 'Strategy'} - Top {min(int(k), len(top))} Weights ({last_dt.date()})")
    ax.set_xlabel("Weight")


def _cache_state(cache: dict, dt: pd.Timestamp):
    if dt in cache:
        st = cache[dt]
    elif pd.Timestamp(dt) in cache:
        st = cache[pd.Timestamp(dt)]
    else:
        return None
    if isinstance(st, PortfolioState):
        return st.as_dict()
    return st


def plot_top_risk_contrib(
    ax: plt.Axes,
    result,
    cache: dict,
    *,
    cov_key: str = "oas",
    name: str | None = None,
    color: str | None = None,
    k: int = 10,
    title: str | None = None,
) -> None:
    try:
        vol_rc, _ = rk.portfolio_contribution_snapshot(
            {"backtest": result, "state_cache": cache, "cov_key": cov_key}
        )
    except Exception:
        ax.text(0.5, 0.5, "No risk contribution data", ha="center", va="center")
        ax.axis("off")
        return

    wdf = pd.DataFrame(_result_field(result, "weights"))
    last_dt = pd.Timestamp(wdf.index[-1]) if not wdf.empty else None
    plot_top_contrib(
        ax,
        vol_rc,
        title=title or f"{name or 'Strategy'} - Top {min(int(k), len(vol_rc))} Risk Contributions ({last_dt.date() if last_dt is not None else 'n/a'})",
        k=int(k),
    )
    if color is not None:
        for patch in ax.patches:
            patch.set_color(color)
    ax.set_xlabel("Contribution to volatility")


def plot_net_equity_compare(
    ax: plt.Axes,
    results: dict,
    *,
    colors: dict[str, str] | None = None,
    title: str = "Net Equity (Comparison)",
) -> None:
    if not results:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.axis("off")
        return
    for name, res in results.items():
        s = _as_series(_result_field(res, "net_values")).dropna()
        if s.empty:
            continue
        ax.plot(s.index, s.values, label=str(name), color=(colors or {}).get(str(name)))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    _format_date_axis(ax)


def plot_drawdown_compare(
    ax: plt.Axes,
    results: dict,
    *,
    colors: dict[str, str] | None = None,
    title: str = "Drawdown (Comparison)",
) -> None:
    if not results:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.axis("off")
        return
    for name, res in results.items():
        s = _as_series(_result_field(res, "net_values")).dropna()
        if s.empty:
            continue
        dd = _drawdown(s)
        ax.plot(dd.index, dd.values, label=str(name), color=(colors or {}).get(str(name)))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    _format_date_axis(ax)


def _strategy_metrics(
    results: dict,
    *,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name, res in results.items():
        net_returns = _as_series(_result_field(res, "net_returns")).dropna().astype(float)
        net_values = _as_series(_result_field(res, "net_values")).dropna().astype(float)
        if net_returns.empty or net_values.empty:
            rows.append({"strategy": str(name), "CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan})
            continue
        years = len(net_returns) / float(annualization)
        cagr = float(net_values.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
        ann_vol = (
            float(net_returns.std(ddof=1) * np.sqrt(float(annualization)))
            if len(net_returns) > 1
            else np.nan
        )
        sd = float(net_returns.std(ddof=1)) if len(net_returns) > 1 else np.nan
        sharpe = (
            float((net_returns - float(rf_daily)).mean() / sd * np.sqrt(float(annualization)))
            if np.isfinite(sd) and sd > 0
            else np.nan
        )
        rows.append({"strategy": str(name), "CAGR": cagr, "AnnVol": ann_vol, "Sharpe": sharpe})
    return pd.DataFrame(rows).set_index("strategy")


def plot_sharpe_compare(
    ax: plt.Axes,
    results: dict,
    *,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
    colors: dict[str, str] | None = None,
    title: str = "Realized Sharpe (Net)",
) -> None:
    if not results:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.axis("off")
        return

    metrics = _strategy_metrics(results, rf_daily=rf_daily, annualization=annualization)
    if metrics.empty or "Sharpe" not in metrics.columns:
        ax.text(0.5, 0.5, "No Sharpe data", ha="center", va="center")
        ax.axis("off")
        return

    s = metrics["Sharpe"].sort_values()
    bar_colors = [(colors or {}).get(str(n)) for n in s.index]
    ax.barh(s.index.tolist(), s.values, color=bar_colors)
    ax.set_title(title)
    ax.set_xlabel("Sharpe")


def plot_risk_return_scatter(
    ax: plt.Axes,
    results: dict,
    *,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
    colors: dict[str, str] | None = None,
    title: str = "Realized Risk-Return (Net)",
) -> None:
    if not results:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.axis("off")
        return

    metrics = _strategy_metrics(results, rf_daily=rf_daily, annualization=annualization)
    if metrics.empty:
        ax.text(0.5, 0.5, "No metric data", ha="center", va="center")
        ax.axis("off")
        return

    for name in metrics.index:
        x = float(metrics.loc[name, "AnnVol"])
        y = float(metrics.loc[name, "CAGR"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter([x], [y], color=(colors or {}).get(str(name)))
        ax.annotate(str(name), (x, y), fontsize=8, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("CAGR")


# ----------------------------
# Risk notebook plotting wrappers
# ----------------------------

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
        raise fi.InputError("n_panels must be positive.")
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
    elif isinstance(objects, dict):
        data = {str(k): v for k, v in objects.items()}
    else:
        raise fi.InputError("objects must be a dict[name -> return series] or DataFrame.")
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
        raise fi.InputError("No non-empty return series remain after cleaning.")
    return out


def plot_nav_compare(ax: plt.Axes, objects, *, colors: dict[str, str] | None = None, title: str = "Cumulative NAV") -> None:
    obj = _coerce_object_returns(objects)
    for name, r in obj.items():
        nav = (1.0 + r).cumprod()
        ax.plot(nav.index, nav.values, lw=1.8, label=name, color=(colors or {}).get(name))
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.legend()


def plot_drawdown_compare_objects(
    ax: plt.Axes,
    objects,
    *,
    colors: dict[str, str] | None = None,
    title: str = "Drawdown",
) -> None:
    obj = _coerce_object_returns(objects)
    for name, r in obj.items():
        nav = (1.0 + r).cumprod()
        dd = nav / nav.cummax() - 1.0
        ax.plot(dd.index, dd.values, lw=1.4, label=name, color=(colors or {}).get(name))
    ax.axhline(0.0, color="#444", lw=1.0)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend()


def plot_rolling_vol(
    ax: plt.Axes,
    returns,
    *,
    windows: list[int] | tuple[int, ...] = (20, 60, 252),
    annualization: float = 252.0,
    name: str | None = None,
) -> None:
    r = pd.to_numeric(pd.Series(returns), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    wlist = [int(w) for w in windows if int(w) > 1]
    if len(wlist) == 0:
        raise fi.InputError("windows must contain at least one integer > 1.")
    for w in wlist:
        rv = r.rolling(w).std(ddof=1) * np.sqrt(float(annualization))
        ax.plot(rv.index, rv.values, lw=1.4, label=f"{w}d")
    ax.set_title(f"Rolling Volatility - {name}" if name else "Rolling Volatility")
    ax.set_ylabel("Ann. Vol")
    ax.legend()


def plot_var_backtest(
    ax: plt.Axes,
    returns,
    *,
    alpha: float = 0.05,
    lookback: int = 252,
    method: str = "hist",
    methods: list[str] | tuple[str, ...] | None = None,
    name: str | None = None,
) -> None:
    method_norm = str(method).strip().lower()
    chosen_method = method_norm
    if method_norm == "best":
        table = rk.var_backtest_table(
            {"_object": pd.Series(returns)},
            alpha=alpha,
            methods=(list(methods) if methods is not None else list(rk.VAR_BACKTEST_METHODS)),
            lookback=lookback,
        )
        best_map = rk.best_var_methods(table)
        chosen_method = str(best_map.get("_object", "hist"))

    st = rk.breach_stats(returns, alpha=alpha, lookback=lookback, method=chosen_method)
    z = st["series"]
    if z.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        ax.axis("off")
        return
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
    ax.set_title(f"VaR Backtest - {name}" if name else "VaR Backtest")
    ax.set_ylabel("Daily Return")
    ax.legend()


def plot_stress_bar(
    ax: plt.Axes,
    stress_tbl: pd.DataFrame,
    *,
    window: str,
    metric: str = "cum_return",
    ascending: bool = True,
) -> None:
    if stress_tbl is None or stress_tbl.empty:
        ax.text(0.5, 0.5, "No stress data", ha="center", va="center")
        ax.axis("off")
        return
    if metric not in stress_tbl.columns:
        raise fi.InputError(f"metric {metric!r} not in stress table.")
    if window not in stress_tbl.index:
        ax.text(0.5, 0.5, "Window not found", ha="center", va="center")
        ax.axis("off")
        return
    sub = stress_tbl.loc[window]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T
    if "object" not in sub.columns:
        raise fi.InputError("stress_tbl must include 'object' column.")
    s = pd.Series(sub[metric].to_numpy(dtype=float), index=sub["object"].astype(str).tolist(), dtype=float)
    s = s.sort_values(ascending=bool(ascending))
    ax.barh(s.index, s.values)
    ax.set_title(f"{window} - {metric}")
    ax.set_xlabel(metric)


def plot_capm_scatter(
    ax: plt.Axes,
    returns,
    market_ret,
    *,
    rf_daily: float = 0.0,
    name: str | None = None,
    color: str | None = None,
) -> None:
    r = pd.to_numeric(pd.Series(returns), errors="coerce")
    m = pd.to_numeric(pd.Series(market_ret), errors="coerce")
    z = pd.concat([m.rename("x"), r.rename("y")], axis=1).dropna()
    if z.empty:
        ax.text(0.5, 0.5, "No CAPM data", ha="center", va="center")
        ax.axis("off")
        return
    x = z["x"] - float(rf_daily)
    y = z["y"] - float(rf_daily)
    alpha, beta, r2 = rk.capm_ols(y, x)
    xv = x.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    dot_color = LAB_COLORS[0]
    line_color = color if color is not None else LAB_COLORS[1]
    ax.scatter(xv, yv, s=10, alpha=0.10, color=dot_color)
    if np.isfinite(alpha) and np.isfinite(beta):
        xs = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 200)
        ax.plot(xs, alpha + beta * xs, lw=2.0, color=line_color)
    ax.axhline(0.0, color="#444", lw=1.0)
    ax.axvline(0.0, color="#444", lw=1.0)
    ax.set_title(f"CAPM Fit - {name}" if name else "CAPM Fit")
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


def plot_rolling_beta_compare(
    ax: plt.Axes,
    capm_roll: dict[str, pd.DataFrame],
    *,
    window: int = 252,
    metric: str = "beta",
) -> None:
    col = f"{metric}_{int(window)}"
    found = 0
    for name, df in capm_roll.items():
        if df is None or df.empty or col not in df.columns:
            continue
        ax.plot(df.index, df[col].values, lw=1.4, label=str(name))
        found += 1
    if found == 0:
        ax.text(0.5, 0.5, "No rolling data", ha="center", va="center")
        ax.axis("off")
        return
    if metric == "beta":
        ax.axhline(1.0, color="#444", lw=1.0, ls="--")
        ax.set_title(f"Rolling Beta ({window}d)")
        ax.set_ylabel("Beta")
    else:
        ax.axhline(0.0, color="#444", lw=1.0)
        ax.set_title(f"Rolling Correlation ({window}d)")
        ax.set_ylabel("Correlation")
    ax.legend(ncol=2)


def plot_corr_heatmap(
    ax: plt.Axes,
    corr: pd.DataFrame,
    *,
    annotate: bool = True,
    cmap: str = "Spectral",
) -> None:
    if corr is None or corr.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center")
        ax.axis("off")
        return
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Matrix")
    if annotate:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_top_contrib(
    ax: plt.Axes,
    contrib: pd.Series | pd.DataFrame | dict[str, float],
    *,
    title: str = "Top Contributions",
    k: int = 10,
) -> None:
    if isinstance(contrib, pd.Series):
        s = contrib.copy()
    elif isinstance(contrib, pd.DataFrame):
        if contrib.shape[0] == 1:
            s = contrib.iloc[0]
        elif contrib.shape[1] == 1:
            s = contrib.iloc[:, 0]
        else:
            raise fi.InputError("Contribution DataFrame must have one row or one column.")
    elif isinstance(contrib, dict):
        s = pd.Series(contrib, dtype=float)
    else:
        s = pd.Series(contrib, dtype=float)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if s.empty:
        ax.text(0.5, 0.5, "No contribution data", ha="center", va="center")
        ax.axis("off")
        return
    s.index = [str(i) for i in s.index]
    top_idx = s.abs().sort_values(ascending=False).head(int(max(k, 1))).index
    show = s.loc[top_idx].sort_values()
    ax.barh(show.index, show.values)
    ax.set_title(title)
    ax.set_xlabel("Contribution")


set_plot_style()
