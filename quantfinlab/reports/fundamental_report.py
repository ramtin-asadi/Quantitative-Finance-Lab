"""Issuer-level fundamental report orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.contracts import FundamentalReportArtifacts
from quantfinlab.fundamentals.analysis import warning_penalty_weights

try:  # optional notebook display
    from IPython.display import display as ipy_display
except ModuleNotFoundError:  # pragma: no cover - IPython is a notebook dependency
    ipy_display = None


corporate_sections = {
    "profitability": (
        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "gross_profitability_assets",
        "roa",
        "roe",
        "roic_proxy",
        "roic",
    ),
    "cash_quality": (
        "cfo_assets",
        "fcf_assets",
        "cfo_net_income",
        "fcf_conversion",
        "total_accruals",
        "positive_cfo_frequency",
        "positive_fcf_frequency",
    ),
    "growth": (
        "revenue_growth",
        "operating_income_growth",
        "net_income_growth",
        "cfo_growth",
        "fcf_growth",
        "revenue_per_share_growth",
        "eps_growth",
    ),
    "financial_strength": (
        "current_ratio",
        "cash_ratio",
        "debt_equity",
        "debt_assets",
        "net_debt_assets",
        "liabilities_assets",
        "interest_coverage",
        "cfo_debt",
        "fcf_debt",
        "cash_assets",
    ),
    "efficiency": (
        "asset_turnover",
        "receivable_turnover",
        "receivables_turnover",
        "inventory_turnover",
        "cash_conversion_cycle",
        "working_capital_revenue",
    ),
    "capital_allocation": (
        "net_shareholder_yield",
        "shareholder_yield",
        "dividend_yield",
        "repurchase_yield",
        "issuance_yield",
        "share_count_dilution",
        "per_share_growth_spread",
        "reinvestment_proxy",
        "reinvestment_rate",
        "reinvestment_quality",
    ),
    "valuation": (
        "earnings_yield",
        "fcf_yield",
        "sales_yield",
        "book_to_market",
        "price_earnings",
        "price_book",
        "ev_ebit",
        "ev_fcf",
        "enterprise_value_ebit",
        "enterprise_value_fcf",
        "ebit_ev",
        "sales_ev",
    ),
    "dupont": (
        "net_margin",
        "asset_turnover",
        "equity_multiplier",
        "dupont_roe",
        "dupont_gap",
    ),
}

financial_sections = {
    "profitability": (
        "fin_roa",
        "fin_roe",
        "fin_pretax_assets",
        "fin_net_margin",
        "pretax_return_on_assets",
        "fin_positive_earnings_frequency",
    ),
    "cash_quality": (
        "fin_net_income_variability",
        "fin_roa_variability",
        "fin_roe_variability",
        "positive_earnings_frequency",
        "positive_cfo_frequency",
    ),
    "growth": (
        "fin_net_income_growth",
        "fin_revenue_growth",
        "fin_equity_growth",
        "revenue_growth",
        "fin_bvps_growth",
        "fin_tbvps_growth",
    ),
    "financial_strength": (
        "fin_equity_assets",
        "fin_tangible_equity_assets",
        "fin_liabilities_assets",
        "fin_assets_equity",
        "tangible_equity_assets",
        "fin_equity_assets_change",
    ),
    "efficiency": (
        "revenue_assets",
        "operating_expense_ratio",
        "fin_revenue_assets",
        "fin_operating_expense_ratio",
        "fin_pretax_margin",
    ),
    "capital_allocation": (
        "fin_net_payout_yield",
        "fin_dividend_yield",
        "fin_repurchase_yield",
        "fin_issuance_yield",
        "dividend_yield",
        "repurchase_yield",
        "issuance_yield",
        "fin_share_dilution",
    ),
    "valuation": (
        "fin_earnings_yield",
        "fin_book_to_market",
        "fin_tangible_book_to_market",
        "fin_revenue_market_cap",
        "price_earnings",
        "price_book",
    ),
    "dupont": (
        "fin_roe",
        "net_margin",
        "fin_assets_equity",
    ),
}

corporate_peer_metrics = (
    "operating_margin",
    "roa",
    "cfo_assets",
    "revenue_growth",
    "net_debt_assets",
    "earnings_yield",
    "fcf_yield",
    "net_shareholder_yield",
)

financial_peer_metrics = (
    "fin_roa",
    "fin_roe",
    "fin_equity_assets",
    "fin_assets_equity",
    "fin_bvps_growth",
    "fin_earnings_yield",
    "fin_book_to_market",
    "fin_net_payout_yield",
    "fin_share_dilution",
)

lower_is_better = {
    "net_debt_assets",
    "debt_assets",
    "debt_equity",
    "cash_conversion_cycle",
    "price_earnings",
    "price_book",
    "enterprise_value_ebit",
    "enterprise_value_fcf",
    "fin_assets_equity",
    "fin_share_dilution",
}


def _require_fundamental_plotting():
    try:
        import matplotlib.pyplot as plt

        from quantfinlab.plotting import fundamentals as plots
    except ModuleNotFoundError as exc:  # pragma: no cover - optional plotting extra
        if exc.name and (exc.name == "matplotlib" or exc.name.startswith("matplotlib.")):
            raise ImportError(
                "fundamental_report figures require matplotlib. "
                "Install the plotting extra with `pip install quantfinlab[plotting]`."
            ) from exc
        raise
    return plt, plots


def _display_table(table: pd.DataFrame, *, round_digits: int) -> None:
    shown = table.round(round_digits)
    if ipy_display is not None:
        ipy_display(shown)
    else:  # pragma: no cover
        print(shown)


def _family(row: pd.Series) -> str:
    value = row.get("company_type", row.get("score_family", "corporate"))
    return "financial" if str(value).lower() == "financial" else "corporate"


def _issuer_rows(
    data: pd.DataFrame,
    *,
    ticker: str,
    cik: Any = None,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    values = data.copy()
    if cik is not None and "cik" in values:
        selected = values["cik"].eq(cik)
    else:
        selected = pd.Series(False, index=values.index)
    if "ticker" in values:
        selected |= values["ticker"].astype("string").str.upper().eq(ticker.upper())
    if "display_ticker" in values:
        selected |= values["display_ticker"].astype("string").str.upper().eq(ticker.upper())
    values = values[selected].copy()
    if "decision_date" in values:
        values["decision_date"] = pd.to_datetime(values["decision_date"])
        if asof is not None:
            values = values[values["decision_date"].le(asof)]
        values = values.sort_values("decision_date")
    return values


def _quarterly_history(company: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    if company.empty:
        return company.copy()
    history = company.copy()
    if "latest_quarter_end" in history and history["latest_quarter_end"].notna().any():
        history["latest_quarter_end"] = pd.to_datetime(history["latest_quarter_end"])
        history = (
            history.dropna(subset=["latest_quarter_end"])
            .drop_duplicates("latest_quarter_end", keep="last")
            .sort_values("latest_quarter_end")
            .set_index("latest_quarter_end")
        )
    elif "decision_date" in history:
        history = (
            history.drop_duplicates("decision_date", keep="last")
            .sort_values("decision_date")
            .set_index("decision_date")
        )
    return history.tail(max(int(periods), 1)).copy()


def _snapshot(current: pd.Series) -> pd.DataFrame:
    fields = (
        "ticker",
        "entity_name",
        "company_type",
        "score_family",
        "industry",
        "price",
        "market_cap",
        "enterprise_value",
        "filed_date",
        "latest_period_end",
    )
    values = {field: current[field] for field in fields if field in current.index}
    return pd.Series(values, name="value").to_frame()


def _statement_table(
    history: pd.DataFrame,
    *,
    kind: str,
    periods: int,
    scale: str,
    show_common_size: bool,
    show_growth: bool,
) -> pd.DataFrame:
    choices = {
        "income": {
            "revenue": ("revenue_q", "revenue", "revenue_ttm"),
            "gross_profit": ("gross_profit_q", "gross_profit", "gross_profit_ttm"),
            "operating_income": (
                "operating_income_q",
                "operating_income",
                "operating_income_ttm",
            ),
            "pretax_income": ("pretax_income_q", "pretax_income", "pretax_income_ttm"),
            "net_income": ("net_income_q", "net_income", "net_income_ttm"),
            "diluted_eps": ("eps_diluted_q", "eps_diluted", "eps_diluted_ttm"),
        },
        "cash_flow": {
            "operating_cash_flow": ("cfo_q", "cfo", "cfo_ttm"),
            "capital_expenditure": ("capex_q", "capex", "capex_ttm"),
            "free_cash_flow": ("free_cash_flow_q", "free_cash_flow", "free_cash_flow_ttm"),
            "dividends": ("dividends_q", "dividends", "dividends_ttm"),
            "repurchases": ("repurchases_q", "repurchases", "repurchases_ttm"),
            "share_issuance": ("share_issuance_q", "share_issuance", "share_issuance_ttm"),
        },
    }
    selected = {}
    for label, candidates in choices[kind].items():
        column = next(
            (
                candidate
                for candidate in candidates
                if candidate in history and history[candidate].notna().any()
            ),
            None,
        )
        if column is not None:
            selected[label] = column
    table = (
        history[list(selected.values())]
        .tail(max(int(periods), 1))
        .rename(columns={column: label for label, column in selected.items()})
    )

    if kind == "income" and show_common_size:
        for column in ("gross_margin", "operating_margin", "pretax_margin", "net_margin"):
            if column in history:
                table[column] = history.loc[table.index, column]
    if show_growth:
        growth_columns = (
            ("revenue_growth", "operating_income_growth", "net_income_growth")
            if kind == "income"
            else ("cfo_growth", "fcf_growth")
        )
        for column in growth_columns:
            if column in history:
                table[column] = history.loc[table.index, column]

    divisors = {"raw": 1.0, "millions": 1e6, "billions": 1e9}
    divisor = divisors.get(str(scale).lower(), 1.0)
    money_columns = [
        column
        for column in selected
        if "eps" not in column and column not in {"gross_margin", "operating_margin"}
    ]
    table[money_columns] = table[money_columns].apply(pd.to_numeric, errors="coerce").div(divisor)
    table.attrs["scale"] = str(scale).lower()
    return table


def _fundamental_summary(
    current: pd.Series,
    *,
    company_type: str,
    include: Mapping[str, bool],
) -> pd.DataFrame:
    sections = financial_sections if company_type == "financial" else corporate_sections
    rows = []
    used = set()
    for section, candidates in sections.items():
        if not include.get(section, False):
            continue
        for metric in candidates:
            if metric in current.index and metric not in used:
                rows.append({"section": section, "metric": metric, "value": current[metric]})
                used.add(metric)
    if not rows:
        return pd.DataFrame(columns=["section", "value"]).rename_axis("metric")
    return pd.DataFrame(rows).set_index("metric")


def _peer_comparison(
    metrics: pd.DataFrame,
    current: pd.Series,
    *,
    company_type: str,
    group: str,
    minimum_peers: int,
    percentiles: tuple[float, float, float],
    market_cap_band: tuple[float, float] | None,
) -> pd.DataFrame:
    if "decision_date" not in metrics:
        return pd.DataFrame()
    decision_date = pd.Timestamp(current["decision_date"])
    cross_section = metrics[pd.to_datetime(metrics["decision_date"]).eq(decision_date)].copy()
    family_column = "company_type" if "company_type" in cross_section else "score_family"
    if family_column in cross_section:
        cross_section = cross_section[
            cross_section[family_column].astype("string").str.lower().eq(company_type)
        ]

    peer_group = group if group in cross_section else "industry"
    peers = cross_section
    if (
        peer_group in cross_section
        and peer_group in current.index
        and pd.notna(current[peer_group])
    ):
        grouped = cross_section[cross_section[peer_group].eq(current[peer_group])]
        if len(grouped) >= minimum_peers:
            peers = grouped

    if (
        market_cap_band is not None
        and "market_cap" in peers
        and pd.notna(current.get("market_cap"))
    ):
        low, high = market_cap_band
        bounded = peers[
            peers["market_cap"].between(
                float(current["market_cap"]) * float(low),
                float(current["market_cap"]) * float(high),
            )
        ]
        if len(bounded) >= minimum_peers:
            peers = bounded

    candidates = financial_peer_metrics if company_type == "financial" else corporate_peer_metrics
    available = [
        metric
        for metric in candidates
        if metric in peers
        and metric in current.index
        and pd.to_numeric(peers[metric], errors="coerce").notna().any()
    ]
    rows = []
    quantile_labels = [f"peer {int(value * 100)}th percentile" for value in percentiles]
    for metric in available:
        values = pd.to_numeric(peers[metric], errors="coerce").dropna()
        company_value = pd.to_numeric(pd.Series([current[metric]]), errors="coerce").iloc[0]
        if values.empty or pd.isna(company_value):
            favorable = np.nan
        else:
            favorable = float(values.le(company_value).mean())
            if metric in lower_is_better:
                favorable = 1.0 - favorable
        quantiles = values.quantile(list(percentiles))
        row = {"metric": metric, "company": company_value}
        row.update(
            {label: quantiles.iloc[position] for position, label in enumerate(quantile_labels)}
        )
        row["favorable_percentile"] = favorable
        row["peer_count"] = len(values)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("metric")


def _traditional_models(current: pd.Series) -> pd.DataFrame:
    fields = {
        "Piotroski F-score": ("piotroski_f_score", "piotroski_score"),
        "Altman Z-score": ("altman_z",),
        "Altman class": ("altman_class",),
        "Beneish M-score": ("beneish_m",),
        "Beneish warning": ("beneish_warning", "warning_beneish_warning"),
        "weighted red-flag points": ("warning_penalty",),
    }
    rows = {}
    for label, candidates in fields.items():
        column = next((candidate for candidate in candidates if candidate in current.index), None)
        rows[label] = current[column] if column is not None else np.nan
    return pd.Series(rows, name="value").to_frame()


def _warning_table(
    current: pd.Series,
    company: pd.DataFrame,
    *,
    company_type: str,
    active_only: bool,
    minimum_severity: int,
    show_history: bool,
    show_penalty: bool,
) -> pd.DataFrame:
    prefix = "warning_fin_" if company_type == "financial" else "warning_"
    warning_columns = [
        column
        for column in warning_penalty_weights
        if column.startswith(prefix)
        and (company_type == "financial" or not column.startswith("warning_fin_"))
        and column in current.index
        and warning_penalty_weights[column] >= int(minimum_severity)
    ]
    rows = []
    for column in warning_columns:
        value = current[column]
        active = bool(value) if pd.notna(value) else False
        if active_only and not active:
            continue
        row = {
            "warning": column.removeprefix(prefix).replace("_", " "),
            "active": active,
            "severity": warning_penalty_weights[column],
        }
        if show_penalty:
            row["penalty_points"] = warning_penalty_weights[column] if active else 0
        if show_history:
            history = company[column].fillna(False).astype(bool)
            row["active_observations"] = int(history.sum())
            if history.any() and "decision_date" in company:
                row["last_active"] = pd.to_datetime(company.loc[history, "decision_date"]).max()
            else:
                row["last_active"] = pd.NaT
        rows.append(row)
    columns = ["warning", "active", "severity"]
    if show_penalty:
        columns.append("penalty_points")
    if show_history:
        columns.extend(["active_observations", "last_active"])
    if not rows:
        return pd.DataFrame(columns=columns).set_index("warning")
    return pd.DataFrame(rows).set_index("warning")


def _score_table(
    current: pd.Series,
    *,
    score: str,
    fundamental_score: str,
    momentum_score: str,
    show_blocks: bool,
    show_rank: bool,
) -> pd.DataFrame:
    columns = [score, fundamental_score, momentum_score, "fixed_score"]
    if show_rank:
        columns.extend(["score_rank", "selection_rank"])
    columns.extend(
        [
            "piotroski_penalty",
            "red_flag_penalty",
            "warning_penalty",
            "severe_warning_count",
        ]
    )
    if show_blocks:
        block_names = (
            "profitability",
            "cash",
            "stability",
            "growth",
            "strength",
            "efficiency",
            "capital_allocation",
            "valuation",
        )
        columns.extend(
            [
                column
                for column in current.index
                if column.endswith("_score") and any(name in column for name in block_names)
            ]
        )
    columns = list(dict.fromkeys(column for column in columns if column in current.index))
    return current[columns].rename("value").to_frame()


def _summary_text(
    current: pd.Series,
    score_current: pd.Series | None,
    warnings: pd.DataFrame | None,
    *,
    score_column: str,
) -> list[str]:
    ticker = str(current.get("ticker", "issuer"))
    date = pd.Timestamp(current["decision_date"]).date()
    lines = [f"{ticker} fundamental report as of {date}."]
    if score_current is not None and score_column in score_current.index:
        value = score_current[score_column]
        if pd.notna(value):
            lines.append(f"{score_column.replace('_', ' ')}: {float(value):.2f}.")
    if warnings is not None:
        lines.append(f"Active reported warnings: {int(warnings['active'].sum())}.")
    return lines


def fundamental_report(
    *,
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    ticker: str,
    asof: str | pd.Timestamp | None = None,
    include: Mapping[str, bool] | None = None,
    statement_settings: Mapping[str, Any] | None = None,
    history_settings: Mapping[str, Any] | None = None,
    peer_settings: Mapping[str, Any] | None = None,
    score_settings: Mapping[str, Any] | None = None,
    warning_settings: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
) -> FundamentalReportArtifacts:
    """Build a report from already-computed issuer metrics and scores.

    The report does not calculate accounting metrics or scores. It selects one
    issuer's existing history, prepares concise report tables, delegates each
    chart to an atomic ``quantfinlab.plotting.fundamentals`` function, and
    returns all artifacts for notebook or programmatic use. Set
    ``layout={"combine_figures": True}`` to place the enabled charts in one
    report-level grid.
    """

    include_cfg = {
        "snapshot": True,
        "statements": True,
        "profitability": True,
        "cash_quality": True,
        "growth": True,
        "financial_strength": True,
        "efficiency": True,
        "capital_allocation": True,
        "valuation": True,
        "dupont": True,
        "traditional_models": True,
        "warnings": True,
        "peer_comparison": True,
        "score": True,
        "score_history": True,
        "summary": True,
    }
    if include:
        include_cfg.update({str(key): bool(value) for key, value in include.items()})

    statement_cfg = {
        "scale": "billions",
        "periods": 8,
        "show_common_size": True,
        "show_growth": True,
    }
    if statement_settings:
        statement_cfg.update(dict(statement_settings))
    history_cfg = {
        "periods": 12,
        "frequency": "quarterly",
        "rolling_periods": 4,
        "show_latest_value": True,
    }
    if history_settings:
        history_cfg.update(dict(history_settings))
    peer_cfg = {
        "group": "industry_group",
        "minimum_peers": 10,
        "percentiles": (0.25, 0.50, 0.75),
        "market_cap_band": (0.25, 4.0),
    }
    if peer_settings:
        peer_cfg.update(dict(peer_settings))
    score_cfg = {
        "score": "selection_score",
        "fundamental_score": "final_score",
        "momentum_score": "momentum_score",
        "show_blocks": True,
        "show_rank": True,
        "show_weight_history": True,
    }
    if score_settings:
        score_cfg.update(dict(score_settings))
    warning_cfg = {
        "active_only": True,
        "minimum_severity": 1,
        "show_history": True,
        "show_penalty": True,
    }
    if warning_settings:
        warning_cfg.update(dict(warning_settings))
    layout_cfg = {
        "ncols": 2,
        "sharex": False,
        "sharey": False,
        "figure_width": 11.0,
        "panel_height": 3.2,
        "combine_figures": False,
    }
    if layout:
        layout_cfg.update(dict(layout))
    output_cfg = {
        "round_tables": 4,
        "display_tables": True,
        "display_table_keys": None,
        "show_figures": True,
        "display_figure_keys": None,
        "print_summary": True,
        "short_labels": False,
    }
    if output:
        output_cfg.update(dict(output))

    metrics_data = metrics.copy(deep=True)
    scores_data = scores.copy(deep=True)
    report_date = pd.Timestamp(asof) if asof is not None else None
    company = _issuer_rows(metrics_data, ticker=ticker, asof=report_date)
    if company.empty:
        raise ValueError(
            f"No fundamental metrics found for ticker {ticker!r} at the requested date."
        )
    current = company.iloc[-1].copy()
    current_date = pd.Timestamp(current["decision_date"])
    company_type = _family(current)
    cik = current.get("cik")
    company = _issuer_rows(metrics_data, ticker=ticker, cik=cik, asof=current_date)
    history = _quarterly_history(company, periods=int(history_cfg["periods"]))

    score_history = _issuer_rows(
        scores_data,
        ticker=ticker,
        cik=cik,
        asof=current_date,
    )
    score_current = score_history.iloc[-1].copy() if not score_history.empty else None

    tables: dict[str, pd.DataFrame] = {}
    figures: dict[str, list[Any]] = {}
    series: dict[str, Any] = {"metric_history": history.copy()}
    text: dict[str, list[str]] = {}

    if include_cfg["snapshot"]:
        tables["snapshot"] = _snapshot(current)
    if include_cfg["statements"]:
        tables["income_statement"] = _statement_table(
            history,
            kind="income",
            periods=int(statement_cfg["periods"]),
            scale=str(statement_cfg["scale"]),
            show_common_size=bool(statement_cfg["show_common_size"]),
            show_growth=bool(statement_cfg["show_growth"]),
        )
        tables["cash_flow"] = _statement_table(
            history,
            kind="cash_flow",
            periods=int(statement_cfg["periods"]),
            scale=str(statement_cfg["scale"]),
            show_common_size=False,
            show_growth=bool(statement_cfg["show_growth"]),
        )

    summary_sections = (
        "profitability",
        "cash_quality",
        "growth",
        "financial_strength",
        "efficiency",
        "capital_allocation",
        "valuation",
        "dupont",
    )
    if any(include_cfg[section] for section in summary_sections):
        tables["fundamental_summary"] = _fundamental_summary(
            current,
            company_type=company_type,
            include=include_cfg,
        )
    if include_cfg["traditional_models"]:
        tables["traditional_models"] = _traditional_models(current)

    warnings = None
    if include_cfg["warnings"]:
        warnings = _warning_table(
            current,
            company,
            company_type=company_type,
            active_only=bool(warning_cfg["active_only"]),
            minimum_severity=int(warning_cfg["minimum_severity"]),
            show_history=bool(warning_cfg["show_history"]),
            show_penalty=bool(warning_cfg["show_penalty"]),
        )
        tables["warnings"] = warnings

    peer_comparison = pd.DataFrame()
    if include_cfg["peer_comparison"]:
        percentiles = tuple(float(value) for value in peer_cfg["percentiles"])
        if len(percentiles) != 3:
            raise ValueError("peer_settings['percentiles'] must contain three values.")
        market_cap_band = peer_cfg.get("market_cap_band")
        peer_comparison = _peer_comparison(
            metrics_data,
            current,
            company_type=company_type,
            group=str(peer_cfg["group"]),
            minimum_peers=int(peer_cfg["minimum_peers"]),
            percentiles=percentiles,
            market_cap_band=tuple(market_cap_band) if market_cap_band is not None else None,
        )
        tables["peer_comparison"] = peer_comparison
        series["peer_percentiles"] = peer_comparison.get(
            "favorable_percentile", pd.Series(dtype=float)
        )

    if include_cfg["score"]:
        tables["score_summary"] = (
            _score_table(
                score_current,
                score=str(score_cfg["score"]),
                fundamental_score=str(score_cfg["fundamental_score"]),
                momentum_score=str(score_cfg["momentum_score"]),
                show_blocks=bool(score_cfg["show_blocks"]),
                show_rank=bool(score_cfg["show_rank"]),
            )
            if score_current is not None
            else pd.DataFrame(columns=["value"])
        )
    if include_cfg["score_history"]:
        series["score_history"] = score_history.copy()
    if include_cfg["summary"]:
        text["summary"] = _summary_text(
            current,
            score_current,
            warnings,
            score_column=str(score_cfg["score"]),
        )

    figure_sections = {
        "earnings": include_cfg["statements"],
        "cash_flow": include_cfg["statements"] or include_cfg["cash_quality"],
        "profitability": include_cfg["profitability"],
        "financial_position": include_cfg["financial_strength"],
        "peer_percentiles": include_cfg["peer_comparison"] and not peer_comparison.empty,
        "score_history": include_cfg["score_history"] and not score_history.empty,
    }
    if any(figure_sections.values()):
        plt, plots = _require_fundamental_plotting()
        plot_calls = {
            "earnings": (plots.plot_revenue_margin, history),
            "cash_flow": (plots.plot_cash_flow, history),
            "profitability": (plots.plot_profitability, history),
            "financial_position": (plots.plot_financial_position, history),
            "peer_percentiles": (plots.plot_peer_percentiles, peer_comparison),
            "score_history": (plots.plot_score_history, score_history),
        }
        figure_keys = [key for key, enabled in figure_sections.items() if enabled]

        if bool(layout_cfg["combine_figures"]):
            ncols = min(max(int(layout_cfg["ncols"]), 1), len(figure_keys))
            nrows = int(np.ceil(len(figure_keys) / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(
                    max(float(layout_cfg["figure_width"]), 4.8 * ncols),
                    max(float(layout_cfg["panel_height"]), 2.8) * nrows,
                ),
                sharex=bool(layout_cfg["sharex"]),
                sharey=bool(layout_cfg["sharey"]),
                squeeze=False,
            )
            axes = axes.reshape(-1)
            for key, ax in zip(figure_keys, axes, strict=False):
                function, data = plot_calls[key]
                plot_kwargs = (
                    {}
                    if key == "peer_percentiles"
                    else {"ticker": str(current.get("ticker", ticker))}
                )
                function(data, ax=ax, **plot_kwargs)
            for ax in axes[len(figure_keys) :]:
                ax.axis("off")
            fig.tight_layout()
            figures["overview"] = [fig]
        else:
            ncols = max(int(layout_cfg["ncols"]), 1)
            width = max(float(layout_cfg["figure_width"]) / ncols, 4.8)
            height = max(float(layout_cfg["panel_height"]), 2.8)
            for key in figure_keys:
                fig, ax = plt.subplots(figsize=(width, height))
                function, data = plot_calls[key]
                plot_kwargs = (
                    {}
                    if key == "peer_percentiles"
                    else {"ticker": str(current.get("ticker", ticker))}
                )
                function(data, ax=ax, **plot_kwargs)
                fig.tight_layout()
                figures[key] = [fig]

    if bool(output_cfg["display_tables"]):
        selected_tables = output_cfg.get("display_table_keys")
        table_keys = (
            list(tables)
            if selected_tables is None
            else [str(key) for key in selected_tables if str(key) in tables]
        )
        for key in table_keys:
            _display_table(tables[key], round_digits=int(output_cfg["round_tables"]))

    if bool(output_cfg["show_figures"]):
        selected_figures = output_cfg.get("display_figure_keys")
        figure_keys = (
            list(figures)
            if selected_figures is None
            else [str(key) for key in selected_figures if str(key) in figures]
        )
        for key in figure_keys:
            for figure in figures[key]:
                if ipy_display is not None:
                    ipy_display(figure)
                else:  # pragma: no cover
                    figure.show()
                plt.close(figure)

    if bool(output_cfg["print_summary"]) and text.get("summary"):
        for line in text["summary"]:
            print(line)

    return FundamentalReportArtifacts(
        tables=tables,
        figures=figures,
        series=series,
        text=text,
    )


__all__ = ["FundamentalReportArtifacts", "fundamental_report"]
