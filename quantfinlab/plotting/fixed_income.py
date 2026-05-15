from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.errors import InputError
from .curves import choose_heatmap_cmap


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
        raise InputError("metric must be 'pv01' or 'convexity'.")
    sub = risk.xs(metric, axis=1, level=1)
    for method in sub.columns:
        ax.plot(sub.index, sub[method], label=str(method))
    ax.set_title(title or metric.upper())
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
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
    method: str | None = None,
    keys: list[int] | tuple[int, ...] | None = None,
    title: str | None = None,
):
    if krd_df is None or krd_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return None

    if isinstance(krd_df.columns, pd.MultiIndex):
        if method is None:
            method = str(krd_df.columns.get_level_values(0)[0])
        if method not in krd_df.columns.get_level_values(0):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            return None
        if keys is None:
            keys = sorted({int(k) for k in krd_df.columns.get_level_values(1)})
        sub = krd_df[method].reindex(columns=list(keys))
        default_title = f"KRD - {method}"
    elif {"date", "key", "krd"}.issubset(krd_df.columns):
        use = krd_df.copy()
        if method is not None and "strategy" in use.columns:
            use = use[use["strategy"] == method]
        if use.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            return None
        if keys is None:
            keys = sorted({int(k) for k in use["key"]})
        sub = use.pivot_table(index="date", columns="key", values="krd").sort_index().reindex(columns=list(keys))
        default_title = "KRD"
    else:
        sub = krd_df.copy()
        if keys is None:
            keys = [int(k) for k in sub.columns]
        sub = sub.reindex(columns=list(keys))
        default_title = "KRD"

    im = ax.imshow(
        sub.values.T,
        aspect="auto",
        origin="lower",
        cmap=choose_heatmap_cmap(metric_name="krd", kind="risk"),
    )
    ax.set_title(title or default_title)
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


def _drawdown(nav: pd.Series) -> pd.Series:
    nav = nav.dropna().astype(float)
    if nav.empty:
        return nav
    return nav / nav.cummax() - 1.0


def plot_ladder_nav(
    ax: plt.Axes,
    navs: pd.Series | pd.DataFrame | dict[str, pd.Series],
    *,
    title: str = "Ladder NAV",
) -> None:
    if isinstance(navs, dict):
        data = pd.DataFrame(navs)
    elif isinstance(navs, pd.Series):
        data = navs.to_frame(navs.name or "NAV")
    else:
        data = navs
    if data is None or data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for col in data.columns:
        ax.plot(data.index, data[col], label=str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend()


def plot_ladder_drawdown(
    ax: plt.Axes,
    navs: pd.Series | pd.DataFrame | dict[str, pd.Series],
    *,
    title: str = "Ladder Drawdown",
) -> None:
    if isinstance(navs, dict):
        data = pd.DataFrame(navs)
    elif isinstance(navs, pd.Series):
        data = navs.to_frame(navs.name or "NAV")
    else:
        data = navs
    if data is None or data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for col in data.columns:
        ax.plot(data.index, _drawdown(data[col]), label=str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()


def plot_ladder_cumulative_return(
    ax: plt.Axes,
    returns: pd.Series | pd.DataFrame | dict[str, pd.Series],
    *,
    title: str = "Cumulative Return",
) -> None:
    if isinstance(returns, dict):
        data = pd.DataFrame(returns)
    elif isinstance(returns, pd.Series):
        data = returns.to_frame(returns.name or "Return")
    else:
        data = returns
    if data is None or data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for col in data.columns:
        cum = (1.0 + data[col].fillna(0.0)).cumprod() - 1.0
        ax.plot(cum.index, cum, label=str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend()


def plot_ladder_weights(
    ax: plt.Axes,
    weights: pd.DataFrame,
    *,
    title: str = "Ladder Weights",
) -> None:
    if weights is None or weights.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for col in weights.columns:
        ax.plot(weights.index, weights[col], label=f"{int(col)}Y" if isinstance(col, (int, float, np.integer)) else str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend()


def plot_carry_return_contributions(
    ax: plt.Axes,
    carry: pd.DataFrame,
    *,
    cumulative: bool = False,
    title: str | None = None,
) -> None:
    if carry is None or carry.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    cols = [c for c in ["coupon_carry_ret", "roll_ret", "curve_move_ret"] if c in carry.columns]
    if not cols:
        ax.text(0.5, 0.5, "No carry columns", ha="center", va="center")
        ax.axis("off")
        return
    labels = {
        "coupon_carry_ret": "Coupon + cash carry",
        "roll_ret": "Roll-down",
        "curve_move_ret": "Curve move",
    }
    data = carry[cols].fillna(0.0)
    if cumulative:
        data = data.cumsum()
    for col in cols:
        ax.plot(data.index, data[col], label=labels.get(col, col))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title or ("Cumulative Carry / Roll / Curve-Move" if cumulative else "Carry / Roll / Curve-Move"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Return contribution")
    ax.legend()


def plot_duration_tracking(
    ax: plt.Axes,
    duration: pd.Series | pd.DataFrame,
    *,
    target: float | None = None,
    band: float | None = None,
    title: str = "Duration Tracking",
) -> None:
    if duration is None or len(duration) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    data = duration.to_frame("duration") if isinstance(duration, pd.Series) else duration
    for col in data.columns:
        ax.plot(data.index, data[col], label=str(col))
    if target is not None:
        ax.axhline(float(target), color="black", linestyle="--", linewidth=0.8, label="Target")
        if band is not None:
            ax.axhline(float(target) + float(band), color="black", linestyle=":", linewidth=0.8)
            ax.axhline(float(target) - float(band), color="black", linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Duration")
    ax.legend()


def plot_risk_timeseries(
    ax: plt.Axes,
    risk_frames: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    metric: str,
    title: str | None = None,
) -> None:
    if isinstance(risk_frames, dict):
        plotted = False
        for label, frame in risk_frames.items():
            if frame is not None and not frame.empty and metric in frame.columns:
                ax.plot(frame.index, frame[metric], label=str(label))
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            return
    else:
        frame = risk_frames
        if frame is None or frame.empty or metric not in frame.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            return
        ax.plot(frame.index, frame[metric], label=metric)
    ax.set_title(title or metric)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.legend()


def plot_krd_lines(
    ax: plt.Axes,
    krd_df: pd.DataFrame,
    *,
    value: str = "krd",
    keys: list[int] | tuple[int, ...] | None = None,
    title: str | None = None,
) -> None:
    if krd_df is None or krd_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    if {"date", "key", value}.issubset(krd_df.columns):
        sub = krd_df.pivot_table(index="date", columns="key", values=value).sort_index()
    else:
        sub = krd_df.copy()
    if keys is not None:
        sub = sub.reindex(columns=list(keys))
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    for key in sub.columns:
        ax.plot(sub.index, sub[key], label=f"{int(key)}Y")
    ax.set_title(title or ("Key-Rate PV01" if value == "key_rate_pv01" else "Key-Rate Duration"))
    ax.set_xlabel("Date")
    ax.set_ylabel(value)
    ax.legend()


def plot_latest_krd_bar(
    ax: plt.Axes,
    krd_df: pd.DataFrame,
    *,
    value: str = "krd",
    title: str | None = None,
) -> None:
    if krd_df is None or krd_df.empty or value not in krd_df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    latest_date = pd.Timestamp(krd_df["date"].max())
    latest = krd_df[krd_df["date"] == latest_date].sort_values("key").set_index("key")
    ax.bar([f"{int(k)}Y" for k in latest.index], latest[value].values)
    ax.set_title(title or f"Latest {value} ({latest_date.date()})")
    ax.set_xlabel("Key maturity")
    ax.set_ylabel(value)


def plot_trade_summary(
    ax: plt.Axes,
    trades: pd.DataFrame,
    *,
    title: str = "Trade Summary",
) -> None:
    if trades is None or trades.empty or "notional" not in trades.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    show = trades.groupby("reason")["notional"].sum().sort_values(ascending=False)
    show.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Reason")
    ax.set_ylabel("Notional")


def plot_curve_snapshots(
    ax: plt.Axes,
    curves: pd.DataFrame,
    maturities,
    dates,
    *,
    tenor_cols: list[str] | tuple[str, ...] | None = None,
    title: str = "Curve snapshots",
    ylabel: str = "Yield (%)",
) -> None:
    if curves is None or curves.empty:
        ax.text(0.5, 0.5, "No curve data", ha="center", va="center")
        ax.axis("off")
        return
    maturities = np.asarray(maturities, dtype=float)
    use_cols = list(tenor_cols) if tenor_cols is not None else list(curves.columns)
    for date in pd.DatetimeIndex(dates):
        if date in curves.index:
            ax.plot(maturities, curves.loc[date, use_cols].to_numpy(float) * 100, marker="o", label=date.strftime("%Y-%m"))
    ax.set_title(title)
    ax.set_xlabel("Maturity in years")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_pca_diagnostics(
    ax: plt.Axes,
    maturities,
    pca_fit: dict,
    *,
    title: str = "PCA loadings",
) -> None:
    labels = ["pc1 level", "pc2 slope", "pc3 curvature"]
    maturities = np.asarray(maturities, dtype=float)
    loadings = np.asarray(pca_fit["loadings"], dtype=float)
    explained = np.asarray(pca_fit.get("explained", np.repeat(np.nan, loadings.shape[1])), dtype=float)
    for j in range(min(3, loadings.shape[1])):
        suffix = f" ({explained[j]:.0%})" if np.isfinite(explained[j]) else ""
        ax.plot(maturities, loadings[:, j], marker="o", label=f"{labels[j]}{suffix}")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Maturity in years")
    ax.set_ylabel("Loading")
    ax.legend(fontsize=8)


def plot_count_bar(
    ax: plt.Axes,
    counts: pd.Series | pd.DataFrame,
    *,
    title: str = "Counts",
    ylabel: str = "Months",
) -> None:
    if counts is None or len(counts) == 0:
        ax.text(0.5, 0.5, "No count data", ha="center", va="center")
        ax.axis("off")
        return
    data = counts.iloc[:, 0] if isinstance(counts, pd.DataFrame) else counts
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)


def plot_target_duration_path(
    ax: plt.Axes,
    target_data: pd.Series | pd.DataFrame,
    *,
    column: str = "target duration",
    neutral: float = 5.0,
    title: str = "Target duration",
) -> None:
    if target_data is None or len(target_data) == 0:
        ax.text(0.5, 0.5, "No target data", ha="center", va="center")
        ax.axis("off")
        return
    series = target_data[column] if isinstance(target_data, pd.DataFrame) and column in target_data else pd.Series(target_data)
    ax.plot(series.index, series.to_numpy(float), label=column)
    ax.axhline(float(neutral), color="black", lw=0.8, label="neutral")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Years")
    ax.legend(fontsize=8)


def plot_signed_notional(
    ax: plt.Axes,
    swap_log: pd.DataFrame,
    *,
    title: str = "Synthetic swap signed notional",
) -> None:
    if swap_log is None or swap_log.empty or "signed notional" not in swap_log.columns:
        ax.text(0.5, 0.5, "No swap data", ha="center", va="center")
        ax.axis("off")
        return
    ax.plot(swap_log.index, swap_log["signed notional"], label="signed notional")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Notional")
    ax.legend(fontsize=8)


def plot_short_rate_fan_comparison(
    ax: plt.Axes,
    vasicek_paths: pd.DataFrame,
    cir_paths: pd.DataFrame,
    *,
    title: str = "Short-rate fan comparison",
) -> None:
    for paths, color, label in [(vasicek_paths, "#069AF3", "vasicek"), (cir_paths, "#FE420F", "cir")]:
        q = paths.quantile([0.05, 0.50, 0.95], axis=1).T
        ax.fill_between(q.index, q[0.05] * 100, q[0.95] * 100, alpha=0.15, color=color)
        ax.plot(q.index, q[0.50] * 100, color=color, lw=1.8, label=f"{label} median")
    ax.set_title(title)
    ax.set_xlabel("Years")
    ax.set_ylabel("Short rate (%)")
    ax.legend()


def plot_hw_g2_loadings(
    ax: plt.Axes,
    maturities,
    hw_fit: dict,
    g2_fit: dict,
    *,
    title: str = "HW1F and G2++ loading structure",
) -> None:
    maturities = np.asarray(maturities, dtype=float)
    ax.plot(maturities, hw_fit["loading"], marker="o", label="hw1f loading")
    ax.plot(maturities, g2_fit["loadings"][:, 0], marker="o", label="g2 first loading")
    ax.plot(maturities, g2_fit["loadings"][:, 1], marker="o", label="g2 second loading")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Maturity in years")
    ax.set_ylabel("Loading")
    ax.legend()


def plot_hw_g2_scenario_fan(
    ax: plt.Axes,
    maturities,
    base_curve,
    hw_curves,
    g2_curves,
    *,
    title: str = "One-month yield-change fan",
) -> None:
    maturities = np.asarray(maturities, dtype=float)
    base = np.asarray(base_curve, dtype=float)
    for curves, color, label in [(hw_curves, "#069AF3", "hw1f"), (g2_curves, "#008080", "g2++")]:
        changes_bp = (np.asarray(curves, dtype=float) - base[None, :]) * 10000.0
        q = np.quantile(changes_bp, [0.10, 0.50, 0.90], axis=0)
        ax.fill_between(maturities, q[0], q[2], alpha=0.16, color=color)
        ax.plot(maturities, q[1], color=color, lw=1.8, label=f"{label} median")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_title(title)
    ax.set_xlabel("Maturity in years")
    ax.set_ylabel("Yield change (bp)")
    ax.legend()


def plot_scenario_pnl_bars(
    ax: plt.Axes,
    scenario_summary: pd.DataFrame,
    *,
    title: str = "Scenario P&L by portfolio",
) -> None:
    if scenario_summary is None or scenario_summary.empty:
        ax.text(0.5, 0.5, "No scenario data", ha="center", va="center")
        ax.axis("off")
        return
    scenario_order = [s for s in ["parallel -100 bp", "parallel -50 bp", "parallel +50 bp", "parallel +100 bp"] if s in scenario_summary.index]
    plot_data = scenario_summary.reindex(scenario_order) if scenario_order else scenario_summary
    plot_data.mul(100).plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("KRD-implied PnL (% NAV)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)


def plot_effective_duration_comparison(
    ax: plt.Axes,
    durations: pd.DataFrame,
    *,
    title: str = "Effective duration comparison",
) -> None:
    if durations is None or durations.empty:
        ax.text(0.5, 0.5, "No duration data", ha="center", va="center")
        ax.axis("off")
        return
    for col in durations.columns:
        ax.plot(durations.index, durations[col], label=str(col))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Effective duration")
    ax.legend(fontsize=8)


def plot_rolling_active_return(
    ax: plt.Axes,
    active_returns: pd.DataFrame,
    *,
    window: int = 12,
    title: str = "Rolling active return",
) -> None:
    if active_returns is None or active_returns.empty:
        ax.text(0.5, 0.5, "No active return data", ha="center", va="center")
        ax.axis("off")
        return
    data = (1.0 + active_returns.fillna(0.0)).rolling(int(window)).apply(np.prod, raw=True) - 1.0
    for col in data.columns:
        ax.plot(data.index, data[col] * 100, label=str(col))
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{window}-month active return (%)")
    ax.legend(fontsize=8)


__all__ = [
    "plot_bond_metric_bar",
    "plot_bucket_pv",
    "plot_carry_return_contributions",
    "plot_count_bar",
    "plot_curve_snapshots",
    "plot_duration_tracking",
    "plot_effective_duration_comparison",
    "plot_krd_heatmap",
    "plot_krd_lines",
    "plot_ladder_cumulative_return",
    "plot_ladder_drawdown",
    "plot_ladder_nav",
    "plot_ladder_weights",
    "plot_latest_krd_bar",
    "plot_hw_g2_loadings",
    "plot_hw_g2_scenario_fan",
    "plot_risk_metric",
    "plot_risk_timeseries",
    "plot_scenario_pnl_bars",
    "plot_short_rate_fan_comparison",
    "plot_signed_notional",
    "plot_target_duration_path",
    "plot_rolling_active_return",
    "plot_total_pv",
    "plot_trade_summary",
]
