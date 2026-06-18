from __future__ import annotations

import pandas as pd
import pytest

from quantfinlab.backtest import fixed_income, hedging
from quantfinlab.common.results import SimpleBacktestResult


def test_run_hedge_backtest_lags_beta_and_charges_turnover_costs() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    returns = pd.DataFrame(
        {
            "Target": [0.010, -0.005, 0.004, 0.002, -0.003, 0.006, 0.001, -0.002],
            "Hedge": [0.004, -0.002, 0.001, 0.003, -0.001, 0.002, 0.001, -0.001],
        },
        index=dates,
    )
    beta = pd.DataFrame({"Hedge": [0.4, 0.5, 0.6, 0.6]}, index=dates[::2])

    result = hedging.run_hedge_backtest(returns, beta, target="Target", hedges=["Hedge"], cost_bps=5.0, beta_lag=1)

    assert result.beta.index.min() > dates[0]
    assert (result.turnover >= 0.0).all()
    assert (result.cost >= 0.0).all()
    assert result.net_values.iloc[-1] <= result.gross_values.iloc[-1]
    assert result.weights.equals(-result.beta)
    assert result["metadata"]["target"] == "target"


def test_run_many_hedge_backtests_accepts_mapping_and_tuple_specs() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    returns = pd.DataFrame({"target": 0.001, "hedge": 0.0005}, index=dates)
    beta = pd.DataFrame({"hedge": [0.3, 0.4, 0.4]}, index=dates[:3])

    results = hedging.run_many_hedge_backtests(
        {
            "mapping": {"target": "target", "hedges": ["hedge"], "beta": beta},
            "tuple": ("target", ["hedge"], beta),
        },
        returns=returns,
        beta_lag=1,
    )

    assert set(results) == {"mapping", "tuple"}
    assert results["mapping"].metadata["strategy_name"] == "mapping"


def test_combine_ladder_with_return_overlay_aligns_nav_costs_and_diagnostics() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    base_returns = pd.Series([0.001, 0.002, -0.001, 0.001, 0.0005], index=dates, name="base")
    base = SimpleBacktestResult(
        nav=(100.0 * (1.0 + base_returns).cumprod()).rename("base"),
        returns=base_returns,
        weights=pd.DataFrame({"2Y": 0.5, "5Y": 0.5}, index=dates),
        trades=pd.DataFrame({"date": dates[:1], "action": ["rebalance"]}),
        diagnostics={"base": True},
    )
    overlay_returns = pd.Series([0.0005, -0.0002, 0.0001], index=dates[1:4])
    overlay_costs = pd.Series([0.01, 0.02, 0.01], index=dates[1:4])

    result = fixed_income.combine_ladder_with_return_overlay(
        base,
        overlay_returns,
        strategy_name="ladder_overlay",
        overlay_costs=overlay_costs,
        diagnostics={"overlay": True},
    )

    assert list(result.returns.index) == list(dates[1:4])
    assert result.nav.name == "ladder_overlay"
    assert result.costs.sum() == pytest.approx(0.04)
    assert result.diagnostics["base_result"] is base
    assert result.diagnostics["overlay"] is True
