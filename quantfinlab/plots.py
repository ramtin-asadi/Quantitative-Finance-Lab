from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from . import fixed_income as fi

LAB_COLORS = [
    "#069AF3", "#FE420F", "#00008B", "#008080", "#800080",
    "#7BC8F6", "#0072B2", "#04D8B2", "#CC79A7", "#DC143C",
    "#9614fa", "#76FF7B", "#FF8072",
]

# Single source of truth for style
_LAB_STYLE = {
    "figure.figsize": (6, 3),
    "figure.dpi": 300,
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
    im = ax.imshow(sub.values.T, aspect="auto", origin="lower")
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


set_plot_style()