from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fixed_income.bond_pricing import position_values_by_bucket
from quantfinlab.fixed_income.duration_overlay import compute_duration_gap, duration_overlay_trade
from quantfinlab.fixed_income.ladder import (
    choose_backtest_block,
    clone_positions,
    gap_safe_frame,
    initialize_ladder,
    ladder_nav,
    ladder_performance_table,
    ladder_returns,
    rebalance_ladder_to_buckets,
    roll_bucket_positions,
    split_contiguous_blocks,
)
from quantfinlab.fixed_income.scenarios import (
    key_rate_shock_scenarios,
    krd_approx_scenario_pnl,
    parallel_shift_scenarios,
    scenario_quantiles,
    strategy_scenario_summary,
)
from tests.synthetic.generators import flat_curve


def test_duration_overlay_trade_is_capped_by_available_sell_value() -> None:
    trade = duration_overlay_trade(
        effective_duration=7.0,
        target_duration=5.0,
        nav=100.0,
        duration_sell=8.0,
        duration_buy=2.0,
        sell_value_available=20.0,
    )

    assert compute_duration_gap(7.0, 5.0) == 2.0
    assert trade == pytest.approx(20.0)


def test_scenarios_and_ladder_helpers_produce_diagnostic_tables() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-06-30", "2024-07-31", "2024-08-31"])
    blocks = split_contiguous_blocks(dates, max_gap_days=45)
    chosen = choose_backtest_block(dates, max_gap_days=45, min_len=2)
    series = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-06-30"]))
    safe = gap_safe_frame(series, max_gap_days=45)
    strategy = pd.DataFrame(
        {"strategy": "ladder", "ret": [0.01, -0.005, 0.003], "nav": [1.01, 1.00495, 1.007965]},
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
    )

    parallel = parallel_shift_scenarios(shocks_bp=(-50, 50), maturities=(2, 5))
    key = key_rate_shock_scenarios(shock_bp=25, maturities=(2, 5))
    pnl = krd_approx_scenario_pnl(pd.Series({2: 3.0, 5: 4.0}), parallel)
    summary = strategy_scenario_summary({"A": pd.Series({2: 3.0, 5: 4.0})}, key)
    q = scenario_quantiles(np.arange(12).reshape(3, 4), quantiles=(0.5,))

    assert [len(b) for b in blocks] == [2, 3]
    assert len(chosen) == 3
    assert np.isnan(safe.iloc[2])
    assert ladder_returns(strategy).iloc[0] == 0.01
    assert ladder_nav(strategy).iloc[-1] == pytest.approx(1.007965)
    assert ladder_performance_table(strategy).index.tolist() == ["ladder"]
    assert clone_positions({2: {"units": 1.0}})[2]["units"] == 1.0
    assert pnl.loc["parallel +50 bp"] < 0.0
    assert summary.shape == (4, 1)
    assert q.shape == (1, 4)


def test_ladder_initialization_rebalance_and_roll_workflow_changes_positions() -> None:
    date = pd.Timestamp("2024-01-31")
    curve = flat_curve(0.04)

    def coupon_lookup(_: pd.Timestamp, maturity: int) -> float:
        return 0.035 + 0.001 * int(maturity)

    positions, cash = initialize_ladder(
        date,
        buckets=(2, 5),
        target_weights={2: 0.60, 5: 0.40},
        initial_nav=100.0,
        coupon_lookup=coupon_lookup,
    )
    before = position_values_by_bucket(positions, date, curve.df, buckets=(2, 5))
    positions, cash, trades = rebalance_ladder_to_buckets(
        positions,
        cash,
        date,
        curve.df,
        {2: 0.40, 5: 0.60},
        buckets=(2, 5),
        rebalance_band=0.01,
        coupon_lookup=coupon_lookup,
        trading_cost_bps=0.0,
    )
    after = position_values_by_bucket(positions, date, curve.df, buckets=(2, 5))
    pre_roll_count = len(positions)
    rolled, rolled_cash, roll_trades = roll_bucket_positions(
        positions,
        cash,
        date + pd.DateOffset(months=18),
        curve.df,
        buckets=(2, 5),
        bucket_floor={2: 1.0, 5: 4.0},
        trading_cost_bps=0.0,
    )

    assert cash == pytest.approx(0.0, abs=1e-8)
    assert before[2] > before[5]
    assert after[5] > after[2]
    assert {row["side"] for row in trades} == {"buy", "sell"}
    assert roll_trades
    assert rolled_cash > cash
    assert len(rolled) < pre_roll_count
