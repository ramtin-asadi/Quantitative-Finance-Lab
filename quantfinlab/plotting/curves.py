from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from ..common.contracts import Curve
from ..fixed_income import bootstrap, discounting

LAB_COLORS = ["#069AF3","#FE420F", "#00008B", "#008080" , "#CC79A7",
          "#9614fa", "#DC143C", "#7BC8F6", "#0072B2","#04D8B2", "#800080", "#FF8072"]


def set_plot_style(colors: list[str] | None = None) -> None:
    palette = list(colors) if colors is not None else LAB_COLORS
    plt.rcParams["axes.prop_cycle"] = cycler(color=palette)
    plt.rcParams.update(
        {
            "figure.figsize": (6, 3),
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 7,
        }
    )


def choose_heatmap_cmap(metric_name: str | None = None, kind: str | None = None) -> str:
    return "coolwarm"


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
    curves: dict[str, Curve],
    *,
    rmse: pd.DataFrame | None = None,
    tenor_cols: list[str] | None = None,
    freq: int = 2,
    grid_points: int = 200,
    title: str = "Par Yield Curve Fit",
) -> None:
    _, T, par = bootstrap.extract_par_curve(market_row, tenor_cols=tenor_cols)
    grid = np.linspace(max(1 / 12, float(np.min(T))), float(np.max(T)), grid_points)
    par_table = discounting.par_curve_table(curves, grid=grid, freq=freq)

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

__all__ = [
    "LAB_COLORS",
    "draw_curve_lines",
    "draw_market_par_points",
    "plot_discount_curves",
    "plot_forward_curves",
    "plot_par_fit",
    "plot_par_yields_history",
    "plot_rmse_bars",
    "plot_yield_curve_snapshots",
    "plot_zero_curves",
    "set_plot_style",
    "style_axis",
]
