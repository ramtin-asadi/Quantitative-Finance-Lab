"""Atomic plots for fundamental research and issuer reports.

Every public function draws on one caller-supplied axis and returns that same
axis. The module intentionally does not apply a global plotting style so notebook
``rcParams`` remain under the caller's control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


def _plot_colors(count: int) -> list[str]:
    colors = rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return [colors[number % len(colors)] for number in range(count)]


def _clean_label(value: object) -> str:
    text = str(value).replace("_", " ").strip()
    for prefix in ("corporate ", "financial "):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
    if text.lower().endswith(" score"):
        text = text[:-6]
    replacements = {
        "cfo": "CFO",
        "fcf": "FCF",
        "roa": "ROA",
        "roe": "ROE",
        "roic proxy": "ROIC",
        "fin roa": "ROA",
        "fin roe": "ROE",
        "fin tangible bvps": "Tangible book value per share",
        "ttm": "TTM",
    }
    return replacements.get(text.lower(), text.title())


def _empty_axis(ax: Axes, message: str, *, title: str | None = None) -> Axes:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _history_frame(
    history: pd.DataFrame,
    *,
    date_columns: Sequence[str] = (
        "latest_quarter_end",
        "period_end",
        "decision_date",
        "date",
    ),
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame.")
    frame = history.copy()
    date_column = next((column for column in date_columns if column in frame), None)
    if date_column is not None:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
    else:
        dates = pd.Series(pd.to_datetime(frame.index, errors="coerce"), index=frame.index)
    order = np.argsort(dates.to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    dates = dates.iloc[order].reset_index(drop=True)
    return frame, dates


def _issuer_ticks(ax: Axes, dates: pd.Series, size: int) -> None:
    if size <= 0:
        ax.set_xticks([])
        return
    ticks = np.unique(np.linspace(0, size - 1, min(6, size)).round().astype(int))
    parsed = pd.to_datetime(dates, errors="coerce")
    labels = [
        (
            parsed.iloc[position].strftime("%Y-%m")
            if pd.notna(parsed.iloc[position])
            else str(position + 1)
        )
        for position in ticks
    ]
    ax.set_xticks(ticks, labels, rotation=45, ha="right")


def _money_scale(series: Sequence[pd.Series]) -> tuple[float, str]:
    finite_parts = []
    for values in series:
        numeric = _numeric(values).dropna()
        if not numeric.empty:
            finite_parts.append(numeric.abs())
    magnitude = pd.concat(finite_parts).max() if finite_parts else 0.0
    if magnitude >= 1e12:
        return 1e12, "$ trillions"
    if magnitude >= 1e9:
        return 1e9, "$ billions"
    if magnitude >= 1e6:
        return 1e6, "$ millions"
    if magnitude >= 1e3:
        return 1e3, "$ thousands"
    return 1.0, "$"


def _filter_ticker(history: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
    if ticker is None or "ticker" not in history:
        return history
    return history[history["ticker"].astype(str).eq(str(ticker))]


def _is_financial(history: pd.DataFrame) -> bool:
    for column in ("company_type", "score_family"):
        if column in history:
            values = history[column].dropna().astype(str).str.lower()
            if values.str.startswith("financial").any():
                return True
            if values.str.startswith("corporate").any():
                return False
    for column in ("fin_roa", "fin_roe", "fin_tangible_bvps"):
        if column in history and _numeric(history[column]).notna().any():
            return True
    return False


def _earnings_source(
    history: pd.DataFrame,
    *,
    minimum_observations: int,
) -> tuple[pd.Series, str]:
    candidates: list[tuple[pd.Series, str]] = []
    if "revenue_q" in history:
        candidates.append((_numeric(history["revenue_q"]), "Quarterly revenue"))
    for column in ("revenue_ttm", "revenue"):
        if column in history:
            candidates.append((_numeric(history[column]), "TTM revenue"))
    if "pretax_income_q" in history:
        pretax = (_numeric(history["pretax_income_q"]), "Quarterly pretax income")
    else:
        pretax = (pd.Series(np.nan, index=history.index, dtype=float), "Quarterly pretax income")

    for values, label in candidates:
        if values.notna().sum() >= int(minimum_observations):
            return values, label
    if pretax[0].notna().any():
        return pretax
    if candidates:
        return max(candidates, key=lambda item: int(item[0].notna().sum()))
    return pretax


def plot_statement_coverage(
    coverage: pd.DataFrame,
    *,
    ax: Axes,
    title: str | None = None,
    annotate: bool = True,
) -> Axes:
    """Plot annual statement-field coverage as a bounded heatmap."""

    if not isinstance(coverage, pd.DataFrame) or coverage.empty:
        return _empty_axis(ax, "No statement coverage", title=title or "Statement coverage")
    values = coverage.apply(pd.to_numeric, errors="coerce").T
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).any():
        return _empty_axis(ax, "No statement coverage", title=title or "Statement coverage")

    matrix = values.to_numpy(dtype=float)
    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(values.columns)), [str(value) for value in values.columns])
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_yticks(range(len(values.index)), [_clean_label(value) for value in values.index])
    if annotate and matrix.size <= 120:
        for row, column in np.argwhere(np.isfinite(matrix)):
            value = matrix[row, column]
            color = "white" if value >= 0.62 else "#222222"
            ax.text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title or "Point-in-time statement coverage")
    ax.set_xlabel("Year")
    return ax


def plot_reconstruction_sources(
    sources: pd.DataFrame,
    *,
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot TTM reconstruction-method shares by statement field."""

    if not isinstance(sources, pd.DataFrame) or sources.empty:
        return _empty_axis(
            ax,
            "No reconstruction-source data",
            title=title or "TTM reconstruction sources",
        )
    values = sources.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = values.loc[values.sum(axis=1).gt(0), values.sum(axis=0).gt(0)]
    if values.empty:
        return _empty_axis(
            ax,
            "No reconstruction-source data",
            title=title or "TTM reconstruction sources",
        )

    positions = np.arange(len(values))
    left = np.zeros(len(values), dtype=float)
    for number, column in enumerate(values.columns):
        width = values[column].to_numpy(dtype=float)
        ax.barh(
            positions,
            width,
            left=left,
            height=0.72,
            color=_plot_colors(len(values.columns))[number],
            label=_clean_label(column),
        )
        left += width
    ax.set_yticks(positions, [_clean_label(value) for value in values.index])
    ax.set_xlim(0.0, max(1.0, float(np.nanmax(left))))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of issuers")
    ax.set_title(title or "TTM reconstruction sources")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


def plot_score_counts(
    scores: pd.DataFrame,
    *,
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot the number of scoreable issuers through time."""

    if not isinstance(scores, pd.DataFrame) or scores.empty or "decision_date" not in scores:
        return _empty_axis(ax, "No score history", title=title or "Scoreable issuers")
    score_column = next(
        (column for column in ("selection_score", "final_score", "base_score") if column in scores),
        None,
    )
    if score_column is None:
        return _empty_axis(ax, "No score column", title=title or "Scoreable issuers")
    family_column = next(
        (column for column in ("company_type", "score_family") if column in scores),
        None,
    )
    frame = scores.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce")
    frame[score_column] = _numeric(frame[score_column])
    frame = frame.dropna(subset=["decision_date", score_column])
    if frame.empty:
        return _empty_axis(ax, "No score history", title=title or "Scoreable issuers")
    if family_column is None:
        counts = frame.groupby("decision_date").size().rename("All issuers").to_frame()
    else:
        counts = (
            frame.groupby(["decision_date", family_column])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
    positions = np.arange(len(counts))
    for number, column in enumerate(counts.columns):
        ax.plot(
            positions,
            counts[column].to_numpy(dtype=float),
            color=_plot_colors(len(counts.columns))[number],
            linewidth=1.8,
            label=_clean_label(column),
        )
    _issuer_ticks(ax, pd.Series(counts.index), len(counts))
    ax.set_title(title or "Scoreable issuers through time")
    ax.set_ylabel("Issuers")
    ax.set_xlabel("")
    if len(counts.columns) > 1:
        ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_score_weights(
    score_weights: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    company_type: str,
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot walk-forward score-block weights for one company type."""

    if isinstance(score_weights, Mapping):
        key = next(
            (value for value in score_weights if str(value).lower() == str(company_type).lower()),
            None,
        )
        weights = score_weights.get(key, pd.DataFrame()) if key is not None else pd.DataFrame()
    else:
        weights = score_weights
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return _empty_axis(
            ax,
            "No score-weight history",
            title=title or f"{company_type.title()} score weights",
        )

    frame, dates = _history_frame(weights)
    excluded = {"decision_date", "date", "period_end", "latest_quarter_end"}
    columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        values = _numeric(frame[column])
        if values.notna().any():
            frame[column] = values
            columns.append(column)
    if not columns:
        return _empty_axis(
            ax,
            "No score-weight history",
            title=title or f"{company_type.title()} score weights",
        )

    positions = np.arange(len(frame))
    for number, column in enumerate(columns):
        ax.plot(
            positions,
            frame[column],
            linewidth=1.7,
            color=_plot_colors(len(columns))[number],
            label=_clean_label(column),
        )
    _issuer_ticks(ax, dates, len(frame))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.set_title(title or f"{company_type.title()} walk-forward score weights")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_rank_ic(
    validation: pd.DataFrame,
    *,
    score: str = "selection_score",
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot mean rank information coefficient by forward horizon."""

    if not isinstance(validation, pd.DataFrame) or validation.empty:
        return _empty_axis(ax, "No rank-IC data", title=title or "Rank IC")
    frame = validation.copy()
    if "model" in frame:
        frame = frame[frame["model"].astype(str).eq(str(score))]
    elif score in frame.index:
        frame = frame.loc[[score]].reset_index(drop=True)
    horizon_column = next(
        (column for column in ("horizon_months", "horizon") if column in frame),
        None,
    )
    value_column = next(
        (column for column in ("mean_rank_ic", "rank_ic", "mean_ic") if column in frame),
        None,
    )
    if horizon_column is None or value_column is None or frame.empty:
        return _empty_axis(ax, "No rank-IC data", title=title or "Rank IC")
    frame[value_column] = _numeric(frame[value_column])
    frame = frame.dropna(subset=[value_column]).sort_values(horizon_column)
    if frame.empty:
        return _empty_axis(ax, "No rank-IC data", title=title or "Rank IC")

    positions = np.arange(len(frame))
    values = frame[value_column].to_numpy(dtype=float)
    colors = np.where(values >= 0.0, _plot_colors(1)[0], "#9AA0A6")
    ax.bar(positions, values, width=0.68, color=colors)
    labels = [
        f"{value:g}m" if np.issubdtype(type(value), np.number) else str(value)
        for value in frame[horizon_column]
    ]
    ax.set_xticks(positions, labels)
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Forward horizon")
    ax.set_ylabel("Mean rank IC")
    ax.set_title(title or f"{_clean_label(score)} rank IC")
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_bucket_returns(
    bucket_returns: pd.DataFrame | pd.Series,
    *,
    horizon: int = 12,
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot average forward return by score bucket for one horizon."""

    if isinstance(bucket_returns, pd.Series):
        frame = bucket_returns.rename("mean").rename_axis("bucket").reset_index()
    elif isinstance(bucket_returns, pd.DataFrame):
        frame = bucket_returns.copy()
    else:
        raise TypeError("bucket_returns must be a pandas DataFrame or Series.")
    if frame.empty:
        return _empty_axis(ax, "No bucket returns", title=title or "Score-bucket returns")

    horizon_column = next(
        (column for column in ("horizon_months", "horizon") if column in frame),
        None,
    )
    if horizon_column is not None:
        frame = frame[pd.to_numeric(frame[horizon_column], errors="coerce").eq(float(horizon))]
    bucket_column = next(
        (column for column in ("bucket", "quintile", "score_bucket") if column in frame),
        None,
    )
    value_column = next(
        (
            column
            for column in ("mean", "mean_return", "forward_return", "return")
            if column in frame
        ),
        None,
    )
    if value_column is None and horizon in frame.columns:
        value_column = horizon
    if bucket_column is None and frame.index.name is not None:
        frame = frame.reset_index()
        bucket_column = frame.columns[0]
    if bucket_column is None or value_column is None:
        return _empty_axis(ax, "No bucket returns", title=title or "Score-bucket returns")
    frame[value_column] = _numeric(frame[value_column])
    frame = frame.dropna(subset=[value_column]).sort_values(bucket_column)
    if frame.empty:
        return _empty_axis(ax, "No bucket returns", title=title or "Score-bucket returns")

    positions = np.arange(len(frame))
    values = frame[value_column].to_numpy(dtype=float)
    ax.bar(positions, values, width=0.7, color=_plot_colors(1)[0])
    ax.set_xticks(positions, [f"Q{value}" for value in frame[bucket_column]])
    ax.axhline(0.0, color="#555555", linewidth=0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Score bucket")
    ax.set_ylabel("Forward return")
    ax.set_title(title or f"Average {horizon}-month return by score bucket")
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_selection_mix(
    selections: pd.DataFrame,
    *,
    top_n: int = 15,
    group: str = "industry_group",
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot the latest selected portfolio composition as sorted horizontal bars."""

    if not isinstance(selections, pd.DataFrame) or selections.empty:
        return _empty_axis(ax, "No stock selections", title=title or "Selection mix")
    frame = selections.copy()
    if "top_n" in frame:
        frame = frame[pd.to_numeric(frame["top_n"], errors="coerce").eq(int(top_n))]
    latest_date = None
    if "decision_date" in frame:
        frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce")
        latest_date = frame["decision_date"].max()
        frame = frame[frame["decision_date"].eq(latest_date)]
    group_column = (
        group
        if group in frame
        else next(
            (column for column in ("industry", "company_type", "score_family") if column in frame),
            None,
        )
    )
    if group_column is None or frame.empty:
        return _empty_axis(ax, "No selection groups", title=title or "Selection mix")
    groups = frame[group_column].fillna("Unclassified").astype(str)
    if "ticker" in frame:
        counts = frame.assign(_group=groups).groupby("_group")["ticker"].nunique()
    else:
        counts = groups.value_counts()
    counts = counts[counts.gt(0)].sort_values(ascending=True, kind="stable")
    if counts.empty:
        return _empty_axis(ax, "No selection groups", title=title or "Selection mix")

    positions = np.arange(len(counts))
    ax.barh(
        positions,
        counts.to_numpy(dtype=float),
        height=0.7,
        color=_plot_colors(len(counts)),
    )
    ax.set_yticks(positions, [_clean_label(value) for value in counts.index])
    total = float(counts.sum())
    for position, value in enumerate(counts):
        ax.text(
            float(value) + max(total * 0.01, 0.05),
            position,
            f"{int(value)} · {value / total:.0%}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0.0, max(float(counts.max()) * 1.3, 1.0))
    date_text = f" · {latest_date:%Y-%m}" if pd.notna(latest_date) else ""
    ax.set_title(title or f"Top {top_n} composition{date_text}")
    ax.set_xlabel("Selected companies")
    ax.set_ylabel("")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


def plot_revenue_margin(
    history: pd.DataFrame,
    *,
    ax: Axes,
    ticker: str | None = None,
    title: str | None = None,
    minimum_observations: int = 3,
) -> Axes:
    """Plot issuer earnings with an appropriate operating or income comparator."""

    filtered = _filter_ticker(history, ticker)
    frame, dates = _history_frame(filtered)
    if frame.empty:
        return _empty_axis(ax, "No earnings history", title=title or "Issuer earnings")
    earnings, earnings_label = _earnings_source(
        frame,
        minimum_observations=minimum_observations,
    )
    if not earnings.notna().any():
        return _empty_axis(ax, "No earnings history", title=title or "Issuer earnings")

    financial = _is_financial(frame)
    if financial:
        comparison_column = next(
            (
                column
                for column in ("net_income_q", "net_income")
                if column in frame and _numeric(frame[column]).notna().any()
            ),
            None,
        )
        comparison_label = (
            "Quarterly net income" if comparison_column == "net_income_q" else "TTM net income"
        )
    else:
        comparison_column = next(
            (
                column
                for column in ("operating_margin", "gross_margin")
                if column in frame and _numeric(frame[column]).notna().any()
            ),
            None,
        )
        comparison_label = (
            "Operating margin" if comparison_column == "operating_margin" else "Gross margin"
        )

    money_series = [earnings]
    if financial and comparison_column is not None:
        money_series.append(_numeric(frame[comparison_column]))
    divisor, unit = _money_scale(money_series)
    positions = np.arange(len(frame))
    ax.bar(
        positions,
        earnings / divisor,
        width=0.68,
        color=_plot_colors(1)[0],
        label=earnings_label,
    )
    _issuer_ticks(ax, dates, len(frame))
    ax.set_ylabel(f"{earnings_label} · {unit}")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.2)

    if comparison_column is not None:
        comparison = _numeric(frame[comparison_column])
        right = ax.twinx()
        if financial:
            right.plot(
                positions,
                comparison / divisor,
                color=_plot_colors(2)[1],
                marker="o",
                markersize=3,
                linewidth=1.8,
            )
            right.set_ylabel(f"{comparison_label} · {unit}")
        else:
            right.plot(
                positions,
                comparison,
                color=_plot_colors(2)[1],
                marker="o",
                markersize=3,
                linewidth=1.8,
            )
            right.set_ylabel(comparison_label)
            right.yaxis.set_major_formatter(PercentFormatter(1.0))
        right.grid(False)
    label = f"{ticker} · " if ticker else ""
    default_title = (
        f"{label}{earnings_label} and {comparison_label.lower()}"
        if comparison_column is not None
        else f"{label}{earnings_label}"
    )
    ax.set_title(title or default_title)
    return ax


def plot_cash_flow(
    history: pd.DataFrame,
    *,
    ax: Axes,
    ticker: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot quarterly net income, operating cash flow, and free cash flow."""

    filtered = _filter_ticker(history, ticker)
    frame, dates = _history_frame(filtered)
    series: list[tuple[str, pd.Series]] = []
    if "net_income_q" in frame:
        series.append(("Net income", _numeric(frame["net_income_q"])))
    if "cfo_q" in frame:
        cfo = _numeric(frame["cfo_q"])
        series.append(("Operating cash flow", cfo))
        if "capex_q" in frame:
            series.append(("Free cash flow", cfo - _numeric(frame["capex_q"])))
        elif "free_cash_flow_q" in frame:
            series.append(("Free cash flow", _numeric(frame["free_cash_flow_q"])))
    if not series or not any(values.notna().any() for _, values in series):
        return _empty_axis(ax, "No quarterly cash-flow history", title=title or "Cash flow")

    divisor, unit = _money_scale([values for _, values in series])
    positions = np.arange(len(frame))
    width = min(0.72 / len(series), 0.24)
    midpoint = (len(series) - 1) / 2.0
    for number, (label, values) in enumerate(series):
        ax.bar(
            positions + (number - midpoint) * width,
            values / divisor,
            width=width,
            color=_plot_colors(len(series))[number],
            label=label,
        )
    _issuer_ticks(ax, dates, len(frame))
    label = f"{ticker} · " if ticker else ""
    ax.set_title(title or f"{label}Quarterly cash flow")
    ax.set_ylabel(unit)
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_profitability(
    history: pd.DataFrame,
    *,
    ax: Axes,
    ticker: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot the issuer's available corporate or financial profitability measures."""

    filtered = _filter_ticker(history, ticker)
    frame, dates = _history_frame(filtered)
    if _is_financial(frame):
        candidates = (("fin_roa", "ROA"), ("fin_roe", "ROE"))
    else:
        candidates = (
            ("operating_margin", "Operating margin"),
            ("roa", "ROA"),
            ("roe", "ROE"),
        )
    series = [
        (column, label)
        for column, label in candidates
        if column in frame and _numeric(frame[column]).notna().any()
    ]
    if not series:
        return _empty_axis(
            ax,
            "No profitability history",
            title=title or "Profitability",
        )

    positions = np.arange(len(frame))
    for number, (column, label) in enumerate(series):
        ax.plot(
            positions,
            _numeric(frame[column]),
            color=_plot_colors(len(series))[number],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
        )
    _issuer_ticks(ax, dates, len(frame))
    label = f"{ticker} · " if ticker else ""
    ax.set_title(title or f"{label}Profitability")
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_financial_position(
    history: pd.DataFrame,
    *,
    ax: Axes,
    ticker: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot per-share book values for financials or balance-sheet amounts otherwise."""

    filtered = _filter_ticker(history, ticker)
    frame, dates = _history_frame(filtered)
    financial = _is_financial(frame)
    per_share_candidates = (
        ("book_value_per_share", "Book value per share"),
        ("fin_tangible_bvps", "Tangible book value per share"),
        ("tangible_book_value_per_share", "Tangible book value per share"),
    )
    amount_candidates = (
        ("total_debt", "Total debt"),
        ("total_assets", "Total assets"),
        ("common_equity", "Common equity"),
    )
    if financial:
        selected = []
        used_labels: set[str] = set()
        for column, label in per_share_candidates:
            if (
                column in frame
                and label not in used_labels
                and _numeric(frame[column]).notna().any()
            ):
                selected.append((column, label))
                used_labels.add(label)
        if selected:
            divisor, unit = 1.0, "$ per share"
            default_title = "Book value per share"
        else:
            selected = [
                (column, label)
                for column, label in amount_candidates
                if column in frame and _numeric(frame[column]).notna().any()
            ]
            divisor, unit = _money_scale([_numeric(frame[column]) for column, _ in selected])
            default_title = "Financial position"
    else:
        selected = [
            (column, label)
            for column, label in amount_candidates
            if column in frame and _numeric(frame[column]).notna().any()
        ]
        divisor, unit = _money_scale([_numeric(frame[column]) for column, _ in selected])
        default_title = "Financial position"
    if not selected:
        return _empty_axis(
            ax,
            "No financial-position history",
            title=title or "Financial position",
        )

    positions = np.arange(len(frame))
    for number, (column, label) in enumerate(selected):
        ax.plot(
            positions,
            _numeric(frame[column]) / divisor,
            color=_plot_colors(len(selected))[number],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=label,
        )
    _issuer_ticks(ax, dates, len(frame))
    label = f"{ticker} · " if ticker else ""
    ax.set_title(title or f"{label}{default_title}")
    ax.set_ylabel(unit)
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


def plot_peer_percentiles(
    comparison: pd.DataFrame | pd.Series,
    *,
    ax: Axes,
    title: str | None = None,
) -> Axes:
    """Plot favorable peer percentiles on an honest zero-to-one scale."""

    if isinstance(comparison, pd.Series):
        values = comparison.copy()
    elif isinstance(comparison, pd.DataFrame):
        frame = comparison.copy()
        if "metric" in frame and isinstance(frame.index, pd.RangeIndex):
            frame = frame.set_index("metric")
        column = next(
            (
                value
                for value in (
                    "favorable percentile",
                    "favorable_percentile",
                    "percentile",
                )
                if value in frame
            ),
            None,
        )
        if column is None:
            return _empty_axis(
                ax,
                "No peer percentiles",
                title=title or "Peer percentiles",
            )
        values = frame[column]
    else:
        raise TypeError("comparison must be a pandas DataFrame or Series.")
    values = _numeric(values).dropna().clip(0.0, 1.0).sort_values()
    if values.empty:
        return _empty_axis(ax, "No peer percentiles", title=title or "Peer percentiles")

    positions = np.arange(len(values))
    ax.hlines(positions, 0.5, values.to_numpy(dtype=float), color="#C8CDD2", linewidth=4)
    colors = np.where(
        values.to_numpy(dtype=float) >= 0.5,
        _plot_colors(1)[0],
        "#9AA0A6",
    )
    ax.scatter(values, positions, s=45, color=colors, zorder=3)
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_yticks(positions, [_clean_label(value) for value in values.index])
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Favorable percentile")
    ax.set_title(title or "Peer-group percentiles")
    ax.grid(True, axis="x", alpha=0.2)
    return ax


def plot_score_history(
    scores: pd.DataFrame,
    *,
    ax: Axes,
    ticker: str | None = None,
    title: str | None = None,
) -> Axes:
    """Plot available selection, fundamental, and momentum scores through time."""

    filtered = _filter_ticker(scores, ticker)
    frame, dates = _history_frame(
        filtered,
        date_columns=("decision_date", "date", "latest_quarter_end", "period_end"),
    )
    candidates = (
        ("selection_score", "Selection score"),
        ("final_score", "Fundamental score"),
        ("momentum_score", "Momentum score"),
    )
    selected = [
        (column, label)
        for column, label in candidates
        if column in frame and _numeric(frame[column]).notna().any()
    ]
    if not selected:
        return _empty_axis(ax, "No score history", title=title or "Score history")

    positions = np.arange(len(frame))
    all_values = []
    for number, (column, label) in enumerate(selected):
        values = _numeric(frame[column])
        all_values.append(values)
        ax.plot(
            positions,
            values,
            color=_plot_colors(len(selected))[number],
            linewidth=1.8,
            label=label,
        )
    _issuer_ticks(ax, dates, len(frame))
    finite = pd.concat(all_values).dropna()
    if not finite.empty and finite.between(0.0, 1.0).all():
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ylabel = "Percentile score"
    else:
        if not finite.empty and finite.between(0.0, 100.0).all():
            ax.set_ylim(0.0, 100.0)
        ylabel = "Score"
    label = f"{ticker} · " if ticker else ""
    ax.set_title(title or f"{label}Score history")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, axis="y", alpha=0.2)
    return ax


__all__ = [
    "plot_bucket_returns",
    "plot_cash_flow",
    "plot_financial_position",
    "plot_peer_percentiles",
    "plot_profitability",
    "plot_rank_ic",
    "plot_reconstruction_sources",
    "plot_revenue_margin",
    "plot_score_counts",
    "plot_score_history",
    "plot_score_weights",
    "plot_selection_mix",
    "plot_statement_coverage",
]
