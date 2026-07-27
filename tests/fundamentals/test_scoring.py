from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fundamentals import scoring


def test_notebook_score_configurations_are_preserved() -> None:
    assert scoring.corporate_score["block_weights"] == {
        "profitability": 0.35,
        "cash_quality": 0.20,
        "growth": 0.15,
        "strength": 0.10,
        "efficiency": 0.05,
        "capital_allocation": 0.05,
        "valuation": 0.10,
    }
    assert scoring.corporate_score["minimum_metrics"] == 12
    assert scoring.corporate_score["one_of_blocks"] == (
        "cash_quality",
        "strength",
    )
    assert scoring.financial_score["block_weights"]["valuation_return"] == 0.30
    assert scoring.financial_score["minimum_metrics"] == 8
    assert scoring.financial_score["required_blocks"] == (
        "profitability",
        "capital_strength",
    )


def test_metric_scores_keep_peer_family_blend_and_direction() -> None:
    date = pd.Timestamp("2025-01-31")
    frame = pd.DataFrame(
        {
            "decision_date": [date] * 4,
            "score_family": ["corporate"] * 4,
            "industry": ["one", "one", "two", "two"],
            "quality": [1.0, 2.0, 3.0, 4.0],
            "risk": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = scoring.metric_percentile_scores(
        frame,
        ["quality", "risk"],
        lower_is_better=["risk"],
        minimum_peers=2,
        minimum_family=2,
        minimum_winsor_count=20,
    )

    assert result["quality_score"].tolist() == pytest.approx(
        [42.5, 85.0, 57.5, 100.0]
    )
    assert result["risk_score"].tolist() == pytest.approx(
        [57.5, 15.0, 42.5, 0.0]
    )


def test_block_and_family_scores_apply_available_weight_and_coverage() -> None:
    frame = pd.DataFrame(
        {
            "score_family": ["test", "test"],
            "a_score": [80.0, 80.0],
            "b_score": [60.0, np.nan],
            "c_score": [40.0, np.nan],
        }
    )
    config = {
        "family": "test",
        "blocks": {
            "first": {"a": 1.0},
            "second": {"b": 1.0},
            "third": {"c": 1.0},
        },
        "block_weights": {
            "first": 0.5,
            "second": 0.3,
            "third": 0.2,
        },
        "minimum_metrics": 2,
        "minimum_blocks": 2,
        "required_blocks": ("first",),
        "one_of_blocks": ("second", "third"),
    }

    blocks = scoring.block_scores(frame, config["blocks"], prefix="test")
    result = scoring.family_composite_score(frame, blocks, config)

    assert result.loc[0, "test_base_score"] == pytest.approx(66.0)
    assert result.loc[0, "test_valid_metrics"] == 3
    assert result.loc[0, "test_valid_blocks"] == 3
    assert np.isnan(result.loc[1, "test_base_score"])


def test_accounting_penalties_match_notebook_scale() -> None:
    frame = pd.DataFrame(
        {
            "score_family": ["corporate", "corporate", "financial"],
            "piotroski_f_score": [3.0, np.nan, 2.0],
            "warning_penalty": [20.0, np.nan, 5.0],
        }
    )

    piotroski = scoring.piotroski_penalty(frame)
    warnings = scoring.red_flag_penalty(frame)

    assert piotroski.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert warnings.tolist() == pytest.approx([2.0, 0.0, 0.5])


def _walkforward_research() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    tickers = [f"s{number}" for number in range(6)]
    secondary_signal = [0.0, 2.0, 1.0, 4.0, 3.0, 5.0]
    rows = []
    for date in dates:
        for rank, ticker in enumerate(tickers):
            signal = float(rank - 2.5)
            rows.append(
                {
                    "decision_date": date,
                    "ticker": ticker,
                    "score_family": "corporate",
                    "corporate_first_score": signal,
                    "corporate_second_score": -signal,
                    "corporate_third_score": secondary_signal[rank],
                    "forward_3m": signal,
                    "forward_6m": signal,
                    "forward_12m": signal,
                }
            )
    return pd.DataFrame(rows)


def test_walkforward_weights_are_lagged_capped_and_normalized() -> None:
    research = _walkforward_research()
    columns = [
        "corporate_first_score",
        "corporate_second_score",
        "corporate_third_score",
    ]
    prior = {"first": 0.50, "second": 0.30, "third": 0.20}

    capped = scoring.walkforward_block_weights(
        research,
        columns,
        prior,
        family="corporate",
        window=6,
        minimum_periods=1,
    )

    assert np.allclose(capped.sum(axis=1), 1.0)
    assert capped.max().max() <= 0.35 + 1e-12

    changed = research.copy()
    shock_date = research["decision_date"].drop_duplicates().iloc[15]
    shock = changed["decision_date"].eq(shock_date)
    changed.loc[shock, "forward_3m"] = [0.0, 2.0, 1.0, 4.0, 3.0, 5.0]

    original_weights = scoring.walkforward_block_weights(
        research,
        columns,
        prior,
        family="corporate",
        window=6,
        minimum_periods=1,
        weight_cap=0.60,
    )
    changed_weights = scoring.walkforward_block_weights(
        changed,
        columns,
        prior,
        family="corporate",
        window=6,
        minimum_periods=1,
        weight_cap=0.60,
    )
    dates = research["decision_date"].drop_duplicates().reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_weights.loc[: dates.iloc[17]],
        changed_weights.loc[: dates.iloc[17]],
    )
    assert not np.allclose(
        original_weights.loc[dates.iloc[18]],
        changed_weights.loc[dates.iloc[18]],
    )


def test_adaptive_and_definitive_scores_apply_penalties_and_family_momentum() -> None:
    date = pd.Timestamp("2025-01-31")
    frame = pd.DataFrame(
        {
            "decision_date": [date] * 4,
            "score_family": ["corporate", "corporate", "financial", "financial"],
            "final_score": [50.0] * 4,
            "momentum_6_1": [1.0, 2.0, 100.0, 200.0],
        }
    )
    selection = scoring.definitive_selection_score(frame)

    assert selection["momentum_score"].tolist() == pytest.approx(
        [50.0, 100.0, 50.0, 100.0]
    )
    assert selection["selection_score"].tolist() == pytest.approx(
        [50.0, 55.0, 50.0, 55.0]
    )

    corporate = pd.concat([frame.iloc[:2], frame.iloc[[0]]], ignore_index=True)
    corporate["corporate_first_score"] = [80.0, 60.0, 100.0]
    corporate["corporate_base_score"] = [70.0, 70.0, np.nan]
    blocks = corporate[["corporate_first_score"]]
    config = {
        "family": "corporate",
        "block_weights": {"first": 1.0},
    }
    weights = pd.DataFrame(
        {"corporate_first_score": [1.0]},
        index=[date],
    )
    adaptive = scoring.adaptive_family_score(
        corporate,
        blocks,
        weights,
        config,
        penalties=pd.Series([30.0, 0.0, 0.0], index=corporate.index),
    )

    assert adaptive["uncapped_score"].iloc[:2].tolist() == pytest.approx([50.0, 60.0])
    assert adaptive["final_score"].iloc[:2].tolist() == pytest.approx([50.0, 100.0])
    assert adaptive.iloc[2].isna().all()


def test_price_signals_rank_ic_and_bucket_returns_are_explicit() -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    prices = pd.DataFrame(
        {
            "a": [100.0, 110.0, 121.0, 133.1, 146.41, 161.051],
            "b": [100.0, 100.0, 99.0, 98.0, 97.0, 96.0],
        },
        index=dates,
    )
    signals = scoring.price_signal_frame(
        prices,
        horizons=(1, 3),
        momentum_horizons=(3,),
    )

    assert signals.loc[(dates[0], "a"), "forward_1m"] == pytest.approx(0.10)
    assert signals.loc[(dates[3], "a"), "momentum_3_1"] == pytest.approx(0.21)

    validation = pd.DataFrame(
        {
            "decision_date": np.repeat(dates[:2], 5),
            "ticker": [f"s{number}" for number in range(5)] * 2,
            "selection_score": list(range(5)) * 2,
            "forward_1m": list(np.linspace(-0.02, 0.02, 5)) * 2,
        }
    )
    rank_table = scoring.rank_ic_table(
        validation,
        score_columns=["selection_score"],
        horizons=(1,),
    )
    buckets = scoring.bucket_return_table(
        validation,
        horizons=(1,),
        buckets=5,
    )

    assert rank_table.loc[0, "mean_rank_ic"] == pytest.approx(1.0)
    assert rank_table.loc[0, "horizon_months"] == 1
    assert buckets["bucket"].tolist() == [1, 2, 3, 4, 5]
    assert buckets["horizon_months"].eq(1).all()


def test_stock_selection_and_investment_universes_respect_seasoning() -> None:
    daily_dates = pd.bdate_range("2025-01-02", periods=10)
    decision_dates = daily_dates[[4, 7]]
    score_rows = []
    for date in decision_dates:
        for ticker, score in zip(["a", "b", "c"], [90.0, 80.0, 70.0], strict=True):
            score_rows.append(
                {
                    "decision_date": date,
                    "ticker": ticker,
                    "selection_score": score,
                }
            )
    scores = pd.DataFrame(score_rows)
    selections = scoring.select_stocks(scores, top_n=(2, 3))

    counts = selections.groupby(["decision_date", "top_n"]).size()
    assert counts.eq(pd.Series([2, 3, 2, 3], index=counts.index)).all()
    assert selections.groupby(["decision_date", "top_n"])[
        "selection_rank"
    ].min().eq(1.0).all()

    monthly_universe = pd.DataFrame(
        {
            "decision_date": np.repeat(decision_dates, 3),
            "ticker": ["a", "b", "c"] * 2,
        }
    )
    returns = pd.DataFrame(
        {
            "a": 0.0,
            "b": 0.0,
            "c": [np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=daily_dates,
    )
    date_map = pd.DataFrame(
        {
            "decision_date": decision_dates,
            "execution_date": daily_dates[[5, 8]],
        }
    )
    top_two = selections[selections["top_n"].eq(2)].copy()
    top_two.loc[top_two["ticker"].eq("b"), "ticker"] = "c"
    universes = scoring.investment_universes(
        monthly_universe,
        top_two,
        returns,
        date_map,
        top_n=(2,),
        lookback=5,
        minimum_observations=4,
        minimum_assets=2,
    )

    assert set(universes) == {"full", "top2"}
    assert list(universes["full"]) == list(daily_dates[[5, 8]])
    assert list(universes["top2"]) == [daily_dates[8]]
    assert universes["top2"][daily_dates[8]]["tickers"] == ["a", "c"]


def test_investment_universes_can_match_price_observation_rule() -> None:
    dates = pd.bdate_range("2025-01-02", periods=7)
    decision_date = dates[5]
    execution_date = dates[6]
    monthly_universe = pd.DataFrame(
        {"decision_date": [decision_date], "ticker": ["a"]}
    )
    selections = pd.DataFrame(
        {
            "decision_date": [decision_date],
            "ticker": ["a"],
            "top_n": [1],
            "selection_rank": [1.0],
        }
    )
    returns = pd.DataFrame(
        {"a": [np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, np.nan]},
        index=dates,
    )
    prices = pd.DataFrame(
        {"a": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, np.nan]},
        index=dates,
    )
    date_map = pd.DataFrame(
        {
            "decision_date": [decision_date],
            "execution_date": [execution_date],
        }
    )

    universes = scoring.investment_universes(
        monthly_universe,
        selections,
        returns,
        date_map,
        top_n=(1,),
        lookback=5,
        minimum_observations=5,
        minimum_assets=1,
        prices=prices,
    )

    assert universes["full"][execution_date]["tickers"] == ["a"]
    assert universes["top1"][execution_date]["tickers"] == ["a"]
