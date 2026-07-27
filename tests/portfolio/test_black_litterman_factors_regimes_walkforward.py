from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.common.contracts import BacktestResult
from quantfinlab.common.errors import InputError
from quantfinlab.portfolio import (
    black_litterman,
    covariance,
    expected_returns,
    factors,
    optimizers,
    regimes,
    walkforward,
)
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


def test_black_litterman_missing_data_tables_market_state_and_solver_fallback(monkeypatch) -> None:
    prices = price_panel(n=45, assets=("AAA", "SPY"))
    prices.loc[prices.index[5:10], "AAA"] = np.nan
    coverage = black_litterman.data_coverage_table(
        prices,
        tradable_assets=["AAA", "MISSING"],
        signal_assets=["SPY", "MISSING_SIGNAL"],
    ).set_index("ticker")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    early_state = black_litterman.market_state(
        returns=returns[["AAA"]],
        signal_returns=returns[["SPY"]],
        date=prices.index[0] - pd.Timedelta(days=5),
        roles={"assets": ["AAA"], "risky": ["AAA"]},
    )

    assets = pd.Index(["AAA", "BBB", "CCC"])
    mu = pd.Series([0.08, 0.05, 0.03], index=assets)
    cov_ann = pd.DataFrame(
        [[0.05, 0.01, 0.00], [0.01, 0.04, 0.005], [0.00, 0.005, 0.03]],
        index=assets,
        columns=assets,
    )
    benchmark = pd.Series([0.40, 0.35, 0.25], index=assets)
    previous = pd.Series([0.60, 0.30, 0.10], index=assets)
    monkeypatch.setattr(black_litterman, "cp", None)

    weights, fallback, info = black_litterman.posterior_weights(
        mu,
        cov_ann,
        benchmark,
        previous_weights=previous,
        settings=black_litterman.BLSettings(max_weight=0.80),
    )

    assert not bool(coverage.loc["MISSING", "included"])
    assert coverage.loc["MISSING", "observations"] == 0
    assert coverage.loc["MISSING_SIGNAL", "role"] == "signal"
    assert coverage.loc["AAA", "missing_pct_after_first_valid"] > 0.0
    assert early_state.signal_table.empty
    assert early_state.returns.empty
    assert fallback
    assert weights.sum() == pytest.approx(1.0)
    pd.testing.assert_series_equal(weights, previous.astype(float), check_names=False)
    assert np.isnan(info["te_gamma"])


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


def test_walkforward_grid_runs_from_cached_covariance_and_mu_state() -> None:
    pytest.importorskip("cvxpy")

    returns = return_panel(n=90, assets=("AAA", "BBB", "CCC", "DDD"))
    rebalance_dates = [returns.index[45], returns.index[65]]
    universe_by_date = {
        dt: {
            "tickers": list(returns.columns),
            "avg_dollar_volume": pd.Series(1_000_000.0, index=returns.columns),
        }
        for dt in rebalance_dates
    }
    cache = walkforward.build_rebalance_state_cache(
        returns=returns,
        rebalance_dates=rebalance_dates,
        universe_by_date=universe_by_date,
        cov_models={"Sample": covariance.sample_covariance},
        mu_models={"Momentum": expected_returns.momentum_mu},
        cov_lookback=35,
        mu_lookback=45,
        min_cov_observations=25,
        min_mu_observations=25,
    )
    specs = [
        {"name": "EW", "optimizer": "EW"},
        {"name": "MV sample momentum", "optimizer": "MV", "cov_model": "Sample", "mu_model": "Momentum"},
    ]
    grid = walkforward.run_walkforward_grid(
        returns=returns,
        rebalance_dates=rebalance_dates,
        cache=cache,
        strategy_specs=specs,
        max_weight=0.80,
        trading_cost_bps=1.0,
        turnover_penalty_bps=0.0,
        solver_order=("OSQP", "SCS"),
    )
    with_frontier = walkforward.append_frontiergrid_strategy(
        grid,
        cov_model="Sample",
        mu_model="Momentum",
        grid_n=5,
    )

    assert set(cache) == set(rebalance_dates)
    assert {"EW", "MV sample momentum"}.issubset(grid.backtests)
    assert grid.nav.notna().any().all()
    assert grid.diagnostics.loc["MV sample momentum", "Optimizer"] == "MV"
    assert len(with_frontier.backtests) == len(grid.backtests) + 1
    assert any("FrontierGrid" in name for name in with_frontier.backtests)


def test_equal_weight_walkforward_uses_ticker_only_state() -> None:
    returns = return_panel(n=70, assets=("AAA", "BBB", "CCC"))
    rebalance_dates = [returns.index[30], returns.index[50]]
    universe_by_date = {
        date: {"tickers": ["AAA", "BBB", "CCC"]}
        for date in rebalance_dates
    }

    cache = walkforward.build_universe_state_cache(
        returns=returns,
        rebalance_dates=rebalance_dates,
        universe_by_date=universe_by_date,
    )
    result = walkforward.run_equal_weight_walkforward(
        returns=returns,
        rebalance_dates=rebalance_dates,
        universe_by_date=universe_by_date,
        max_weight=0.50,
        trading_cost_bps=1.0,
    )

    assert set(cache) == set(rebalance_dates)
    assert all(set(state) == {"tickers"} for state in cache.values())
    assert result.metadata["optimizer"] == "EW"
    assert result.weights.loc[rebalance_dates[0]].sum() == pytest.approx(1.0)
    assert result.weights.loc[rebalance_dates[0]].nunique() == 1


def test_equal_weight_walkforward_matches_the_model_state_grid() -> None:
    returns = return_panel(n=90, assets=("AAA", "BBB", "CCC", "DDD"))
    rebalance_dates = [returns.index[45], returns.index[65]]
    universe_by_date = {
        date: {"tickers": list(returns.columns)}
        for date in rebalance_dates
    }
    model_cache = walkforward.build_rebalance_state_cache(
        returns=returns,
        rebalance_dates=rebalance_dates,
        universe_by_date=universe_by_date,
        cov_models={"Sample": covariance.sample_covariance},
        mu_models={"Momentum": expected_returns.momentum_mu},
        cov_lookback=35,
        mu_lookback=45,
        min_cov_observations=25,
        min_mu_observations=25,
    )
    grid = walkforward.run_walkforward_grid(
        returns=returns,
        rebalance_dates=rebalance_dates,
        cache=model_cache,
        optimizers={"EW": optimizers.equal_weight},
        strategy_specs=[{"optimizer": "EW"}],
        max_weight=0.80,
        trading_cost_bps=1.0,
    )
    direct = walkforward.run_equal_weight_walkforward(
        returns=returns,
        rebalance_dates=rebalance_dates,
        universe_by_date=universe_by_date,
        max_weight=0.80,
        trading_cost_bps=1.0,
    )

    pd.testing.assert_series_equal(
        direct.net_values,
        grid.backtests["EW"].net_values,
    )
    pd.testing.assert_frame_equal(
        direct.weights,
        grid.backtests["EW"].weights,
    )


def test_walkforward_rejects_empty_state_and_invalid_strategy_specs() -> None:
    returns = return_panel(n=70, assets=("AAA", "BBB", "CCC"))
    dt = returns.index[40]
    cache = {
        dt: {
            "tickers": ["AAA", "BBB"],
            "cov_ann_map": {"Sample": np.eye(2)},
            "mu_ann_map": {"Sample": {"Momentum": pd.Series([0.08, 0.04], index=["AAA", "BBB"])}},
        }
    }

    with pytest.raises(InputError):
        walkforward.build_rebalance_state_cache(returns=pd.DataFrame(), rebalance_dates=[])
    with pytest.raises(InputError):
        walkforward.build_rebalance_state_cache(returns=returns, rebalance_dates=[dt])
    with pytest.raises(InputError):
        walkforward.run_walkforward_grid(
            returns=returns,
            rebalance_dates=[dt],
            universe_by_date={},
        )
    with pytest.raises(InputError):
        walkforward.run_walkforward_grid(
            returns=returns[["AAA", "BBB"]],
            rebalance_dates=[dt],
            cache=cache,
            strategy_specs=[{"name": "bad", "optimizer": "unknown"}],
        )
    with pytest.raises(InputError):
        walkforward.run_walkforward_grid(
            returns=returns[["AAA", "BBB"]],
            rebalance_dates=[dt],
            cache=cache,
            strategy_specs=[{"name": "bad-mv", "optimizer": "MV", "cov_model": "Sample"}],
        )
    with pytest.raises(InputError):
        walkforward.run_walkforward_grid(
            returns=returns[["AAA", "BBB"]],
            rebalance_dates=[dt],
            cache=cache,
            strategy_specs=[
                {"name": "duplicate", "optimizer": "EW"},
                {"name": "duplicate", "optimizer": "EW"},
            ],
        )
