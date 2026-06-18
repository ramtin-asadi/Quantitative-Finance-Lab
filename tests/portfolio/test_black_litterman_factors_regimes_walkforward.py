from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.common.contracts import BacktestResult
from quantfinlab.portfolio import black_litterman, factors, regimes, walkforward
from tests.synthetic.generators import price_panel, return_panel


def _result(returns: pd.Series, weights: pd.DataFrame) -> BacktestResult:
    nav = (1.0 + returns).cumprod()
    return BacktestResult(
        gross_values=nav,
        net_values=nav,
        gross_returns=returns,
        net_returns=returns,
        weights=weights,
        turnover=pd.Series(0.05, index=weights.index),
        costs=pd.Series(0.00005, index=weights.index),
        metadata={},
    )


def test_black_litterman_prior_posterior_weights_and_report_tables() -> None:
    pytest.importorskip("cvxpy")

    assets = pd.Index(["AAA", "BBB", "CCC"])
    cov_ann = pd.DataFrame(
        [[0.05, 0.01, 0.00], [0.01, 0.04, 0.005], [0.00, 0.005, 0.03]],
        index=assets,
        columns=assets,
    )
    benchmark = pd.Series([0.45, 0.35, 0.20], index=assets)
    prior, delta = black_litterman.prior_from_benchmark(cov_ann, benchmark, delta=2.0)
    view_p = np.array([[1.0, -1.0, 0.0]])
    omega, view_var = black_litterman.omega_from_confidence(view_p, cov_ann, [0.70], tau=0.05)
    posterior, post_cov, diag = black_litterman.posterior_returns(prior, cov_ann, view_p, np.array([0.04]), omega)
    weights, fallback, info = black_litterman.posterior_weights(
        posterior,
        post_cov,
        benchmark,
        settings=black_litterman.BLSettings(max_weight=0.75, active_weight_limit=0.40, active_weight_relaxed=0.60),
    )

    assert delta == pytest.approx(2.0)
    assert prior.index.equals(assets)
    assert omega.shape == (1, 1)
    assert view_var.shape == (1,)
    assert diag["n_views"] == 1
    assert weights.sum() == pytest.approx(1.0)
    assert not fallback
    assert "active_limit" in info

    view_table = black_litterman.latest_view_table(
        [{"view_family": "dual_momentum", "view_name": "AAA over BBB", "long_assets": ["AAA"], "short_assets": ["BBB"], "confidence": 0.6}]
    )
    weights_table = black_litterman.latest_weight_table(pd.DataFrame([weights], index=[pd.Timestamp("2024-01-31")]), benchmark)

    assert view_table.loc[0, "family"] == "dual_momentum"
    assert weights_table["abs_active_weight"].is_monotonic_decreasing


def test_black_litterman_diagnostics_tables_from_synthetic_results() -> None:
    prices = price_panel(n=40, assets=("AAA", "BBB", "CCC", "SPY"))
    weights = pd.DataFrame(0.25, index=prices.index[::10], columns=prices.columns)
    returns = prices.pct_change(fill_method=None).dropna()["AAA"]
    benchmark_returns = prices.pct_change(fill_method=None).dropna()["SPY"]
    strategy = _result(returns, weights.reindex(returns.index).ffill().dropna(how="all"))
    benchmark = _result(benchmark_returns, weights.reindex(benchmark_returns.index).ffill().dropna(how="all"))

    coverage = black_litterman.data_coverage_table(prices, tradable_assets=["AAA", "BBB"], signal_assets=["SPY"])
    specs = black_litterman.view_spec_table([{"family": "dual_momentum", "function": "rule"}], {"dual_momentum": 0.02})
    comparison = black_litterman.model_comparison_table({"Strategy": strategy, "Benchmark": benchmark}, benchmark_name="Benchmark")
    active = black_litterman.active_summary_table(strategy, benchmark)
    selection_summary = black_litterman.selection_summary_table(
        pd.DataFrame({"date": [prices.index[-1], prices.index[-1]], "scale_reason": ["kept", "below"], "view_family": ["a", "b"], "kept": [True, False]})
    )
    stress = black_litterman.stress_summary_table({"Strategy": strategy, "Benchmark": benchmark}, benchmark_name="Benchmark", windows={"sample": (str(returns.index[0].date()), str(returns.index[-1].date()))})

    assert set(coverage["ticker"]) == {"AAA", "BBB", "SPY"}
    assert specs.loc[0, "cap"] == pytest.approx(0.02)
    assert "Sharpe" in comparison.columns
    assert active.index[0] == "Learned-Confidence BL"
    assert "kept" in selection_summary.index
    assert set(stress["strategy"]) == {"Strategy", "Benchmark"}


def test_factor_pipeline_scores_validation_tables_and_exposures() -> None:
    dates = pd.date_range("2020-01-31", periods=90, freq="ME")
    assets = ["AAA", "BBB", "CCC", "DDD"]
    asset_returns = pd.DataFrame(
        {
            asset: 0.005 + 0.002 * i + 0.015 * np.sin(np.linspace(0.0, 5.0, len(dates)) + i)
            for i, asset in enumerate(assets)
        },
        index=dates,
    )
    factor_returns = pd.DataFrame(
        {
            "Mkt-RF": 0.004 + 0.010 * np.sin(np.linspace(0.0, 4.0, len(dates))),
            "SMB": 0.002 + 0.008 * np.cos(np.linspace(0.0, 3.0, len(dates))),
        },
        index=dates,
    )

    alpha, beta, r2, eps = factors.rolling_factor_fit(asset_returns, factor_returns, window=18)
    state = factors.factor_state(factor_returns, short_window=3, long_window=6, vol_window=12)
    scores = factors.factor_scores(beta, state)
    combined = factors.combine_scores({"factor": scores, "trend": factors.trend_strength(asset_returns, window=6)}, {"factor": 0.7, "trend": 0.3})
    weights = factors.soft_active_weights(combined.fillna(0.0), asset_returns, turnover_limit=0.20)
    validation_scores, component_weights = factors.validation_weighted_score(
        {"factor": scores.fillna(0.0), "trend": combined.fillna(0.0)},
        asset_returns.shift(-1),
        window=24,
        min_periods=12,
        top_n=2,
    )
    rank_ic = factors.rank_ic_table(combined, asset_returns.shift(-1))
    decay = factors.signal_decay_table(combined, asset_returns, horizons=(1, 3), top_n=2)
    exposure = factors.portfolio_factor_exposure(weights, beta)

    assert alpha.notna().any().any()
    assert r2.notna().any().any()
    assert eps.notna().any().any()
    assert scores.columns.tolist() == assets
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert validation_scores.notna().any().any()
    assert component_weights.notna().any().any()
    assert "rank_ic" in rank_ic.columns
    assert tuple(decay.index.get_level_values("horizon").unique()) == (1, 3)
    assert set(exposure.columns) == {"Mkt-RF", "SMB"}


def test_regime_helpers_blend_sleeves_and_measure_risky_allocation() -> None:
    returns = return_panel(n=80, assets=("AAA", "BBB", "CCC", "SHY"))
    proba = pd.DataFrame(
        {"risk_on": np.linspace(0.3, 0.8, len(returns)), "risk_off": np.linspace(0.7, 0.2, len(returns))},
        index=returns.index,
    )
    scores = regimes.regime_asset_scores(
        returns,
        proba,
        assets=["AAA", "BBB", "CCC"],
        trend_window=30,
        vol_window=30,
        drawdown_window=40,
    )
    sleeves = pd.DataFrame({state: regimes.sleeve_weights(scores.loc[state], assets=["AAA", "BBB", "CCC"], cash_ticker="SHY", top_n=2) for state in scores.index}).T
    blended = regimes.blend_sleeves(proba.iloc[-1], sleeves)
    hybrid = regimes.hybrid_weights(blended, pd.Series({"AAA": 0.3, "BBB": 0.3, "CCC": 0.2, "SHY": 0.2}), alpha=0.6)

    assert set(scores.columns) == {"AAA", "BBB", "CCC"}
    assert blended.sum() == pytest.approx(1.0)
    assert hybrid.sum() == pytest.approx(1.0)
    assert 0.0 <= regimes.risky_allocation(hybrid, cash_ticker="SHY") <= 1.0


def test_walkforward_result_contract_and_rebalance_frequency() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    result = walkforward.WalkForwardGridResult(
        results=pd.DataFrame({"Sharpe": [1.0]}, index=["EW"]),
        nav=pd.DataFrame(),
        returns=pd.DataFrame(),
        weights={},
        turnover=pd.DataFrame(),
        costs=pd.DataFrame(),
        diagnostics=pd.DataFrame(),
        cache={},
        backtests={},
        metadata={"sample": True},
    )

    assert walkforward.rebalances_per_year(dates) == pytest.approx(4.0)
    assert result["results"].loc["EW", "Sharpe"] == pytest.approx(1.0)
    assert result.as_dict()["metadata"] == {"sample": True}
    with pytest.raises(KeyError):
        _ = result["missing"]
