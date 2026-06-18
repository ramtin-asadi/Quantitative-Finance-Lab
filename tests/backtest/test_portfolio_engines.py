from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.backtest import costs, portfolio
from tests.synthetic.generators import return_panel


def test_cost_helpers_handle_arrays_series_and_frames() -> None:
    assert np.allclose(costs.bps_cost([100.0, 200.0], 5.0), [0.05, 0.10])
    assert costs.turnover_cost(pd.Series([-10.0, 5.0]), 10.0).tolist() == pytest.approx([0.010, 0.005])

    trades = pd.DataFrame({"AAA": [10.0, -5.0], "BBB": [-2.0, 3.0]})
    assert costs.turnover_cost(trades, 20.0).tolist() == pytest.approx([0.024, 0.016])


def test_run_weights_backtest_applies_timing_costs_and_normalization() -> None:
    returns = return_panel(n=20, assets=("AAA", "BBB", "CCC"))
    weights = pd.DataFrame(
        [[2.0, 1.0, -1.0], [0.10, 0.20, 0.70]],
        index=[returns.index[0], returns.index[8]],
        columns=returns.columns,
    )

    result = portfolio.run_weights_backtest(
        returns,
        weights,
        cost_bps=25.0,
        w_max=0.80,
        initial_value=100.0,
        name="synthetic",
    )

    assert result.gross_values.index[0] == returns.index[1]
    assert result.net_values.iloc[-1] < result.gross_values.iloc[-1]
    assert np.allclose(result.weights.sum(axis=1), 1.0)
    assert result.weights.max().max() <= 0.80 + 1e-12
    assert result.turnover.iloc[0] > 0.0
    assert result.metadata["strategy_name"] == "synthetic"


def test_run_many_weights_backtests_returns_named_results() -> None:
    returns = return_panel(n=18, assets=("AAA", "BBB"))
    first = returns.index[0]
    schedules = {
        "equal": pd.DataFrame([[0.5, 0.5]], index=[first], columns=returns.columns),
        "tilted": pd.DataFrame([[0.7, 0.3]], index=[first], columns=returns.columns),
    }

    results = portfolio.run_many_weights_backtests(schedules, returns=returns, cost_bps=0.0)

    assert set(results) == {"equal", "tilted"}
    assert results["tilted"].weights.iloc[0]["AAA"] == pytest.approx(0.7)


def test_run_rebalanced_portfolio_backtest_uses_cache_and_fallbacks() -> None:
    returns = return_panel(n=24, assets=("AAA", "BBB", "CCC"))
    rebalances = [returns.index[0], returns.index[10]]
    cache = {dt: {"tickers": ["AAA", "BBB", "CCC"]} for dt in rebalances}

    def weight_fn(dt: pd.Timestamp, state: dict, previous: np.ndarray) -> np.ndarray | None:
        if dt == rebalances[0]:
            return np.array([0.60, 0.30, 0.10])
        return None

    result = portfolio.run_rebalanced_portfolio_backtest(
        returns,
        rebalances,
        cache,
        weight_fn,
        cost_bps=10.0,
        fallback="previous",
        w_max=0.70,
    )

    assert len(result.weights) == 2
    assert result.fallbacks == 1
    assert np.allclose(result.weights.sum(axis=1), 1.0)
    assert result.net_values.iloc[-1] <= result.gross_values.iloc[-1]


def test_strategy_grid_backtests_share_the_same_engine() -> None:
    returns = return_panel(n=20, assets=("AAA", "BBB"))
    rebalances = [returns.index[0]]
    cache = {rebalances[0]: {"tickers": ["AAA", "BBB"]}}
    strategies = {
        "equal": lambda _dt, _state, _prev: np.array([0.5, 0.5]),
        "tilt": lambda _dt, _state, _prev: pd.Series({"AAA": 0.7, "BBB": 0.3}),
    }

    results = portfolio.run_strategy_grid_backtests(
        strategies,
        returns=returns,
        rebal_dates=rebalances,
        cache=cache,
        cost_bps=0.0,
        w_max=0.80,
    )

    assert set(results) == {"equal", "tilt"}
    assert results["tilt"].weights.iloc[0]["AAA"] == pytest.approx(0.7)
