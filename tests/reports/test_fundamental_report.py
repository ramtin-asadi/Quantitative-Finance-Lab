from __future__ import annotations

import importlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from quantfinlab.common.contracts import FundamentalReportArtifacts
from quantfinlab.reports import fundamental_report

matplotlib.use("Agg", force=True)


include_keys = (
    "snapshot",
    "statements",
    "profitability",
    "cash_quality",
    "growth",
    "financial_strength",
    "efficiency",
    "capital_allocation",
    "valuation",
    "dupont",
    "traditional_models",
    "warnings",
    "peer_comparison",
    "score",
    "score_history",
    "summary",
)


class _plot_api:
    def __init__(self) -> None:
        self.called = []

    def _draw(self, name, data, *, ax, ticker=None) -> None:
        self.called.append((name, ticker, len(data)))
        ax.plot([0, 1], [0, 1])

    def plot_revenue_margin(self, data, *, ax, ticker=None, title=None) -> None:
        self._draw("earnings", data, ax=ax, ticker=ticker)

    def plot_cash_flow(self, data, *, ax, ticker=None, title=None) -> None:
        self._draw("cash_flow", data, ax=ax, ticker=ticker)

    def plot_profitability(self, data, *, ax, ticker=None, title=None) -> None:
        self._draw("profitability", data, ax=ax, ticker=ticker)

    def plot_financial_position(self, data, *, ax, ticker=None, title=None) -> None:
        self._draw("financial_position", data, ax=ax, ticker=ticker)

    def plot_peer_percentiles(self, data, *, ax, ticker=None, title=None) -> None:
        self._draw("peer_percentiles", data, ax=ax, ticker=ticker)

    def plot_score_history(self, data, *, ax, ticker=None, title=None) -> None:
        self._draw("score_history", data, ax=ax, ticker=ticker)


def _include(**enabled) -> dict[str, bool]:
    selected = {key: False for key in include_keys}
    selected.update(enabled)
    return selected


def _corporate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-29"])
    rows = []
    for position, date in enumerate(dates):
        rows.append(
            {
                "decision_date": date,
                "latest_quarter_end": date,
                "latest_period_end": date,
                "filed_date": date - pd.Timedelta(days=20),
                "ticker": "KO",
                "cik": 21344,
                "entity_name": "The Coca-Cola Company",
                "company_type": "corporate",
                "score_family": "corporate",
                "industry": "BEVERAGES",
                "industry_group": "Consumer Staples",
                "price": 55.0 + position,
                "market_cap": 240e9 + position * 2e9,
                "enterprise_value": 270e9 + position * 2e9,
                "revenue_q": 10.0e9 + position * 0.3e9,
                "gross_profit_q": 6.0e9 + position * 0.2e9,
                "operating_income_q": 3.0e9 + position * 0.1e9,
                "pretax_income_q": 2.8e9 + position * 0.1e9,
                "net_income_q": 2.4e9 + position * 0.1e9,
                "eps_diluted_q": 0.55 + position * 0.02,
                "cfo_q": 3.1e9 + position * 0.1e9,
                "capex_q": 0.5e9,
                "free_cash_flow_q": 2.6e9 + position * 0.1e9,
                "dividends_q": 1.9e9,
                "repurchases_q": 0.3e9,
                "share_issuance_q": 0.05e9,
                "total_assets": 95e9,
                "total_debt": 42e9,
                "common_equity": 27e9,
                "cash": 12e9,
                "gross_margin": 0.60,
                "operating_margin": 0.30 + position * 0.002,
                "pretax_margin": 0.28,
                "net_margin": 0.24,
                "fcf_margin": 0.25,
                "roa": 0.105 + position * 0.002,
                "roe": 0.39,
                "roic": 0.18,
                "cfo_assets": 0.13,
                "fcf_assets": 0.11,
                "cfo_net_income": 1.25,
                "fcf_conversion": 1.05,
                "total_accruals": -0.01,
                "positive_cfo_frequency": 1.0,
                "positive_fcf_frequency": 1.0,
                "revenue_growth": 0.05 + position * 0.005,
                "operating_income_growth": 0.06,
                "net_income_growth": 0.05,
                "cfo_growth": 0.04,
                "fcf_growth": 0.05,
                "current_ratio": 1.2,
                "cash_ratio": 0.4,
                "debt_equity": 1.55,
                "debt_assets": 0.44,
                "net_debt_assets": 0.31,
                "interest_coverage": 8.0,
                "asset_turnover": 0.48,
                "net_shareholder_yield": 0.035,
                "dividend_yield": 0.031,
                "repurchase_yield": 0.005,
                "issuance_yield": 0.001,
                "share_count_dilution": -0.004,
                "earnings_yield": 0.045,
                "fcf_yield": 0.05,
                "sales_yield": 0.18,
                "book_to_market": 0.11,
                "equity_multiplier": 3.5,
                "dupont_roe": 0.40,
                "dupont_gap": -0.01,
                "piotroski_f_score": 8.0,
                "altman_z": 3.4,
                "altman_class": "safe",
                "beneish_m": -2.3,
                "beneish_warning": False,
                "warning_penalty": 5 if position == 3 else 0,
                "warning_low_interest_coverage": position == 3,
            }
        )

    latest = dates[-1]
    for peer in range(8):
        rows.append(
            {
                **rows[-1],
                "ticker": f"P{peer}",
                "cik": 30000 + peer,
                "entity_name": f"Beverage Peer {peer}",
                "decision_date": latest,
                "latest_quarter_end": latest,
                "market_cap": 120e9 + peer * 20e9,
                "operating_margin": 0.20 + peer * 0.015,
                "roa": 0.06 + peer * 0.01,
                "cfo_assets": 0.08 + peer * 0.008,
                "revenue_growth": 0.01 + peer * 0.01,
                "net_debt_assets": 0.45 - peer * 0.03,
                "earnings_yield": 0.03 + peer * 0.005,
                "fcf_yield": 0.035 + peer * 0.005,
                "net_shareholder_yield": 0.015 + peer * 0.004,
                "warning_low_interest_coverage": False,
            }
        )

    scores = pd.DataFrame(
        {
            "decision_date": dates,
            "ticker": "KO",
            "cik": 21344,
            "selection_score": [63.0, 66.0, 70.0, 74.0],
            "final_score": [65.0, 68.0, 71.0, 75.0],
            "momentum_score": [45.0, 48.0, 60.0, 65.0],
            "fixed_score": [64.0, 67.0, 69.0, 72.0],
            "score_rank": [40, 32, 24, 16],
            "profitability_score": [70.0, 72.0, 74.0, 77.0],
            "cash_quality_score": [68.0, 70.0, 73.0, 75.0],
            "valuation_score": [55.0, 57.0, 58.0, 60.0],
            "piotroski_penalty": 0.0,
            "red_flag_penalty": [0.0, 0.0, 0.0, 0.5],
        }
    )
    return pd.DataFrame(rows), scores


def _financial_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-29", "2023-12-29"])
    rows = []
    for position, date in enumerate(dates):
        rows.append(
            {
                "decision_date": date,
                "latest_quarter_end": date,
                "latest_period_end": date,
                "filed_date": date - pd.Timedelta(days=18),
                "ticker": "JPM",
                "cik": 19617,
                "entity_name": "JPMorgan Chase",
                "company_type": "financial",
                "score_family": "financial",
                "industry": "NATIONAL COMMERCIAL BANKS",
                "industry_group": "Financials",
                "price": 145.0 + position,
                "market_cap": 430e9,
                "enterprise_value": 450e9,
                "revenue_q": float("nan"),
                "revenue": 150e9 + position * 2e9,
                "pretax_income_q": 14e9 + position * 0.5e9,
                "net_income_q": 11e9 + position * 0.4e9,
                "cfo_q": 12e9,
                "capex_q": 2e9,
                "free_cash_flow_q": 10e9,
                "total_assets": 3.9e12,
                "total_debt": 650e9,
                "common_equity": 330e9,
                "cash": 500e9,
                "fin_roa": 0.012 + position * 0.0002,
                "fin_roe": 0.15 + position * 0.002,
                "fin_equity_assets": 0.085,
                "fin_assets_equity": 11.8,
                "fin_bvps_growth": 0.08,
                "fin_earnings_yield": 0.09,
                "fin_book_to_market": 0.75,
                "fin_net_payout_yield": 0.045,
                "fin_share_dilution": -0.01,
                "fin_net_income_growth": 0.07,
                "fin_net_income_variability": 0.12,
                "warning_penalty": 4 if position == 3 else 0,
                "warning_fin_extreme_leverage": position == 3,
            }
        )
    latest = dates[-1]
    for peer in range(5):
        rows.append(
            {
                **rows[-1],
                "ticker": f"B{peer}",
                "cik": 50000 + peer,
                "entity_name": f"Bank Peer {peer}",
                "decision_date": latest,
                "market_cap": 200e9 + peer * 30e9,
                "fin_roa": 0.008 + peer * 0.001,
                "fin_roe": 0.10 + peer * 0.012,
                "fin_equity_assets": 0.07 + peer * 0.004,
                "fin_assets_equity": 14.0 - peer * 0.6,
                "fin_bvps_growth": 0.03 + peer * 0.01,
                "fin_earnings_yield": 0.06 + peer * 0.01,
                "fin_book_to_market": 0.55 + peer * 0.05,
                "fin_net_payout_yield": 0.02 + peer * 0.006,
                "fin_share_dilution": 0.02 - peer * 0.008,
                "warning_fin_extreme_leverage": False,
            }
        )
    scores = pd.DataFrame(
        {
            "decision_date": dates,
            "ticker": "JPM",
            "cik": 19617,
            "selection_score": [60.0, 64.0, 68.0, 72.0],
            "final_score": [62.0, 65.0, 69.0, 73.0],
            "momentum_score": [42.0, 55.0, 59.0, 63.0],
            "financial_profitability_score": [70.0, 72.0, 74.0, 76.0],
            "financial_strength_score": [66.0, 68.0, 69.0, 70.0],
        }
    )
    return pd.DataFrame(rows), scores


def test_fundamental_report_returns_selected_tables_figures_and_series(monkeypatch) -> None:
    metrics, scores = _corporate_data()
    plots = _plot_api()
    displayed = []
    module = importlib.import_module("quantfinlab.reports.fundamental_report")
    monkeypatch.setattr(module, "_require_fundamental_plotting", lambda: (plt, plots))
    monkeypatch.setattr(module, "ipy_display", displayed.append)

    report = fundamental_report(
        metrics=metrics,
        scores=scores,
        ticker="KO",
        peer_settings={"minimum_peers": 5},
        output={
            "display_tables": False,
            "show_figures": True,
            "display_figure_keys": ["earnings"],
            "print_summary": False,
        },
    )

    assert isinstance(report, FundamentalReportArtifacts)
    assert set(report.tables) == {
        "snapshot",
        "income_statement",
        "cash_flow",
        "fundamental_summary",
        "traditional_models",
        "warnings",
        "peer_comparison",
        "score_summary",
    }
    assert set(report.figures) == {
        "earnings",
        "cash_flow",
        "profitability",
        "financial_position",
        "peer_percentiles",
        "score_history",
    }
    assert [name for name, _, _ in plots.called] == list(report.figures)
    assert report.tables["snapshot"].loc["ticker", "value"] == "KO"
    assert report.tables["warnings"].index.tolist() == ["low interest coverage"]
    assert report.tables["peer_comparison"]["peer_count"].min() >= 5
    assert report.series["score_history"]["selection_score"].iloc[-1] == 74.0
    assert len(report.text["summary"]) == 3
    assert displayed == report.figures["earnings"]
    for figures in report.figures.values():
        for figure in figures:
            plt.close(figure)


def test_fundamental_report_can_combine_atomic_plots(monkeypatch) -> None:
    metrics, scores = _corporate_data()
    plots = _plot_api()
    displayed = []
    module = importlib.import_module("quantfinlab.reports.fundamental_report")
    monkeypatch.setattr(module, "_require_fundamental_plotting", lambda: (plt, plots))
    monkeypatch.setattr(module, "ipy_display", displayed.append)

    report = fundamental_report(
        metrics=metrics,
        scores=scores,
        ticker="KO",
        peer_settings={"minimum_peers": 5},
        layout={"ncols": 2, "combine_figures": True},
        output={
            "display_tables": False,
            "show_figures": True,
            "display_figure_keys": ["overview"],
            "print_summary": False,
        },
    )

    assert list(report.figures) == ["overview"]
    assert [name for name, _, _ in plots.called] == [
        "earnings",
        "cash_flow",
        "profitability",
        "financial_position",
        "peer_percentiles",
        "score_history",
    ]
    assert displayed == report.figures["overview"]
    assert len(report.figures["overview"][0].axes) == 6
    assert not plt.fignum_exists(report.figures["overview"][0].number)


def test_include_and_output_keys_limit_work_without_mutating_inputs(monkeypatch) -> None:
    metrics, scores = _corporate_data()
    metrics_before = metrics.copy(deep=True)
    scores_before = scores.copy(deep=True)
    displayed = []
    module = importlib.import_module("quantfinlab.reports.fundamental_report")
    monkeypatch.setattr(
        module,
        "_display_table",
        lambda table, *, round_digits: displayed.append(table),
    )

    report = fundamental_report(
        metrics=metrics,
        scores=scores,
        ticker="KO",
        include=_include(snapshot=True, warnings=True, summary=True),
        output={
            "display_tables": True,
            "display_table_keys": ["warnings"],
            "show_figures": False,
            "print_summary": False,
        },
    )

    assert set(report.tables) == {"snapshot", "warnings"}
    assert report.figures == {}
    assert displayed == [report.tables["warnings"]]
    pd.testing.assert_frame_equal(metrics, metrics_before)
    pd.testing.assert_frame_equal(scores, scores_before)


def test_financial_issuer_uses_financial_peers_warnings_and_revenue_fallback_input(
    monkeypatch,
) -> None:
    metrics, scores = _financial_data()
    plots = _plot_api()
    module = importlib.import_module("quantfinlab.reports.fundamental_report")
    monkeypatch.setattr(module, "_require_fundamental_plotting", lambda: (plt, plots))

    report = fundamental_report(
        metrics=metrics,
        scores=scores,
        ticker="JPM",
        peer_settings={"minimum_peers": 3},
        output={"display_tables": False, "show_figures": False, "print_summary": False},
    )

    assert report.tables["warnings"].index.tolist() == ["extreme leverage"]
    assert "fin_roa" in report.tables["peer_comparison"].index
    assert "fin_assets_equity" in report.tables["fundamental_summary"].index
    earnings_call = next(call for call in plots.called if call[0] == "earnings")
    assert earnings_call == ("earnings", "JPM", 4)
    assert report.series["metric_history"]["revenue_q"].isna().all()
    assert report.series["metric_history"]["revenue"].notna().all()
    for figures in report.figures.values():
        for figure in figures:
            plt.close(figure)
