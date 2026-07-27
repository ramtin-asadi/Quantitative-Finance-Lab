from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from quantfinlab.plotting import fundamentals as plots, plot_statement_coverage


def _assert_month_ticks(ax) -> None:
    labels = [label.get_text() for label in ax.get_xticklabels() if label.get_text()]
    assert len(labels) <= 6
    assert all(re.fullmatch(r"\d{4}-\d{2}", label) for label in labels)


def test_research_plots_are_atomic_sorted_and_preserve_rcparams() -> None:
    dates = pd.date_range("2023-01-31", periods=12, freq="ME")
    scores = pd.DataFrame(
        {
            "decision_date": np.repeat(dates, 4),
            "ticker": [f"T{number}" for number in range(4)] * len(dates),
            "company_type": ["corporate", "corporate", "financial", "financial"] * len(dates),
            "selection_score": np.linspace(20.0, 90.0, 4 * len(dates)),
        }
    )
    weights = {
        "corporate": pd.DataFrame(
            {
                "corporate_profitability_score": np.linspace(0.20, 0.30, len(dates)),
                "corporate_cash_quality_score": np.linspace(0.30, 0.20, len(dates)),
                "corporate_valuation_score": 0.50,
            },
            index=dates,
        )
    }
    validation = pd.DataFrame(
        {
            "model": ["selection_score"] * 4,
            "horizon_months": [1, 3, 6, 12],
            "mean_rank_ic": [0.01, 0.03, 0.04, 0.06],
        }
    )
    bucket_returns = pd.DataFrame(
        {
            "bucket": np.tile(np.arange(1, 6), 2),
            "horizon_months": np.repeat([6, 12], 5),
            "mean": np.r_[np.linspace(0.01, 0.05, 5), np.linspace(0.02, 0.10, 5)],
        }
    )
    selections = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-01-31"] * 2 + ["2024-02-29"] * 4),
            "top_n": [15] * 6,
            "ticker": ["OLD1", "OLD2", "A", "B", "C", "D"],
            "industry_group": [
                "Old",
                "Old",
                "Consumer",
                "Technology",
                "Technology",
                "Technology",
            ],
        }
    )
    coverage = pd.DataFrame(
        {"revenue_ttm": [0.8, 0.9], "net_income_ttm": [0.7, 0.85]},
        index=[2023, 2024],
    )
    sources = pd.DataFrame(
        {
            "four standalone quarters": [0.8, 0.6],
            "annual + current YTD - prior YTD": [0.2, 0.4],
        },
        index=["revenue", "net_income"],
    )

    colors = ["#069AF3", "#FE420F", "#00008B"]
    with matplotlib.rc_context(
        {
            "axes.facecolor": "#EFEFEF",
            "axes.grid": False,
            "axes.prop_cycle": cycler(color=colors),
        }
    ):
        before = (matplotlib.rcParams["axes.facecolor"], matplotlib.rcParams["axes.grid"])
        fig, axes = plt.subplots(4, 2, figsize=(12, 14))
        calls = [
            plot_statement_coverage(coverage, ax=axes[0, 0]),
            plots.plot_reconstruction_sources(sources, ax=axes[0, 1]),
            plots.plot_score_counts(scores, ax=axes[1, 0]),
            plots.plot_score_weights(
                weights,
                company_type="corporate",
                ax=axes[1, 1],
            ),
            plots.plot_rank_ic(validation, score="selection_score", ax=axes[2, 0]),
            plots.plot_bucket_returns(bucket_returns, horizon=12, ax=axes[2, 1]),
            plots.plot_selection_mix(
                selections,
                top_n=15,
                group="industry_group",
                ax=axes[3, 0],
            ),
        ]
        after = (matplotlib.rcParams["axes.facecolor"], matplotlib.rcParams["axes.grid"])

        assert before == after
        assert calls == [
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
            axes[2, 0],
            axes[2, 1],
            axes[3, 0],
        ]
        assert len(axes[0, 0].images) == 1
        assert axes[1, 0].lines[0].get_color() == colors[0]
        assert axes[1, 0].lines[1].get_color() == colors[1]
        _assert_month_ticks(axes[1, 0])
        _assert_month_ticks(axes[1, 1])
        assert [label.get_text() for label in axes[3, 0].get_yticklabels()] == [
            "Consumer",
            "Technology",
        ]
        assert len(axes[3, 0].patches) == 2
        plt.close(fig)


def test_issuer_plots_use_categorical_dates_and_return_supplied_axes() -> None:
    dates = pd.date_range("2022-03-31", periods=8, freq="QE")
    history = pd.DataFrame(
        {
            "latest_quarter_end": dates,
            "ticker": "NKE",
            "company_type": "corporate",
            "revenue_q": np.linspace(11e9, 13e9, len(dates)),
            "operating_margin": np.linspace(0.11, 0.14, len(dates)),
            "net_income_q": np.linspace(1.1e9, 1.4e9, len(dates)),
            "cfo_q": np.linspace(1.3e9, 1.7e9, len(dates)),
            "capex_q": np.linspace(0.2e9, 0.3e9, len(dates)),
            "roa": np.linspace(0.08, 0.10, len(dates)),
            "roe": np.linspace(0.25, 0.30, len(dates)),
            "total_debt": np.linspace(12e9, 10e9, len(dates)),
            "total_assets": np.linspace(38e9, 42e9, len(dates)),
            "common_equity": np.linspace(13e9, 16e9, len(dates)),
            "decision_date": dates + pd.offsets.MonthEnd(1),
            "selection_score": np.linspace(55.0, 80.0, len(dates)),
            "final_score": np.linspace(50.0, 78.0, len(dates)),
            "momentum_score": np.linspace(60.0, 82.0, len(dates)),
        }
    )
    peer = pd.DataFrame(
        {"favorable percentile": [0.72, 0.41, 0.65]},
        index=["operating_margin", "roa", "net_debt_assets"],
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    returned = [
        plots.plot_revenue_margin(history, ticker="NKE", ax=axes[0, 0]),
        plots.plot_cash_flow(history, ticker="NKE", ax=axes[0, 1]),
        plots.plot_profitability(history, ticker="NKE", ax=axes[0, 2]),
        plots.plot_financial_position(history, ticker="NKE", ax=axes[1, 0]),
        plots.plot_peer_percentiles(peer, ax=axes[1, 1]),
        plots.plot_score_history(history, ticker="NKE", ax=axes[1, 2]),
    ]

    assert returned == list(axes.reshape(-1))
    assert "Quarterly revenue" in axes[0, 0].get_ylabel()
    assert "quarterly revenue" in axes[0, 0].get_title().lower()
    assert len(axes[0, 1].patches) == 3 * len(history)
    assert len(axes[0, 2].lines) == 3
    assert len(axes[1, 0].lines) == 3
    assert axes[1, 1].get_xlim() == (0.0, 1.0)
    assert len(axes[1, 2].lines) == 3
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]):
        _assert_month_ticks(axis)
    plt.close(fig)


def test_financial_earnings_priority_and_labels_are_truthful() -> None:
    dates = pd.date_range("2022-03-31", periods=8, freq="QE")
    base = pd.DataFrame(
        {
            "latest_quarter_end": dates,
            "ticker": "GS",
            "score_family": "financial",
            "revenue_q": np.nan,
            "revenue_ttm": np.nan,
            "pretax_income_q": np.linspace(3e9, 4e9, len(dates)),
            "net_income_q": np.linspace(2e9, 3e9, len(dates)),
            "fin_roa": np.linspace(0.008, 0.012, len(dates)),
            "fin_roe": np.linspace(0.08, 0.12, len(dates)),
            "book_value_per_share": np.linspace(280.0, 330.0, len(dates)),
            "fin_tangible_bvps": np.linspace(260.0, 305.0, len(dates)),
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plots.plot_revenue_margin(base, ticker="GS", ax=axes[0])
    plots.plot_profitability(base, ticker="GS", ax=axes[1])
    plots.plot_financial_position(base, ticker="GS", ax=axes[2])

    assert "quarterly pretax income" in axes[0].get_title().lower()
    assert "Quarterly pretax income" in axes[0].get_ylabel()
    right_axis = next(axis for axis in fig.axes if axis not in axes)
    assert "Quarterly net income" in right_axis.get_ylabel()
    assert [line.get_label() for line in axes[1].lines] == ["ROA", "ROE"]
    assert [line.get_label() for line in axes[2].lines] == [
        "Book value per share",
        "Tangible book value per share",
    ]
    _assert_month_ticks(axes[0])
    plt.close(fig)

    ttm = base.copy()
    ttm["revenue_ttm"] = np.linspace(40e9, 48e9, len(ttm))
    quarterly = ttm.copy()
    quarterly.loc[:2, "revenue_q"] = [9e9, 10e9, 11e9]
    amount_only = base.drop(columns=["book_value_per_share", "fin_tangible_bvps"]).assign(
        total_debt=np.linspace(600e9, 550e9, len(base)),
        total_assets=np.linspace(3.5e12, 3.9e12, len(base)),
        common_equity=np.linspace(300e9, 330e9, len(base)),
    )
    fig, axes = plt.subplots(1, 3)
    plots.plot_revenue_margin(ttm, ticker="GS", ax=axes[0])
    plots.plot_revenue_margin(quarterly, ticker="GS", ax=axes[1])
    plots.plot_financial_position(amount_only, ticker="GS", ax=axes[2])
    assert "ttm revenue" in axes[0].get_title().lower()
    assert "quarterly revenue" in axes[1].get_title().lower()
    assert [line.get_label() for line in axes[2].lines] == [
        "Total debt",
        "Total assets",
        "Common equity",
    ]
    plt.close(fig)
