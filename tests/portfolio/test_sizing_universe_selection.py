from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.common.contracts import BacktestResult
from quantfinlab.common.errors import InputError
from quantfinlab.portfolio import selection, sizing, universe
from tests.synthetic.generators import price_panel, return_panel, volume_panel


def _backtest_result(returns: pd.Series, name: str) -> BacktestResult:
    nav = (1.0 + returns).cumprod()
    weights = pd.DataFrame(1.0 / 3.0, index=returns.index[::20], columns=["AAA", "BBB", "CCC"])
    turnover = pd.Series(0.10, index=weights.index)
    costs = pd.Series(0.0001, index=weights.index)
    return BacktestResult(
        gross_values=nav,
        net_values=nav * (1.0 - costs.reindex(nav.index).fillna(0.0).cumsum()),
        gross_returns=returns,
        net_returns=returns - costs.reindex(returns.index).fillna(0.0),
        weights=weights,
        turnover=turnover,
        costs=costs,
        fallbacks=1,
        metadata={"optimizer": name, "cov_model": "Sample", "mu_model": "Momentum"},
    )


def test_universe_cleaning_rebalances_and_liquidity_selection() -> None:
    close = price_panel(n=95, assets=("AAA", "BBB", "CCC", "DDD"))
    volume = volume_panel(n=95, assets=("AAA", "BBB", "CCC", "DDD"))
    volume["DDD"] = volume["DDD"] * 4.0

    clean_close, clean_volume = universe.clean_close_volume_panels(close, volume, start=close.index[0])
    returns = universe.prices_to_returns(clean_close)
    rebalances = universe.make_rebalance_dates(clean_close.index, freq="M", min_history_days=20)
    tickers, adv = universe.select_liquid_universe(
        clean_close.index[-1],
        close=clean_close,
        volume=clean_volume,
        top_n=2,
        liquidity_lookback=20,
        min_listing_days=20,
        min_obs=15,
    )
    by_date = universe.build_liquid_universe_by_date(
        close=clean_close,
        volume=clean_volume,
        rebalance_dates=rebalances[-2:],
        top_n=2,
        liquidity_lookback=20,
        min_listing_days=20,
        min_obs=15,
    )

    assert returns.shape[0] == clean_close.shape[0] - 1
    assert len(rebalances) >= 3
    assert tickers[0] == "DDD"
    assert len(adv) == 2
    assert by_date


def test_universe_handles_duplicate_tickers_empty_history_and_missing_blocks() -> None:
    dates = pd.bdate_range("2024-01-02", periods=35)
    close = pd.DataFrame(
        np.column_stack(
            [
                np.linspace(100.0, 110.0, len(dates)),
                np.linspace(200.0, 220.0, len(dates)),
                np.linspace(50.0, 55.0, len(dates)),
            ]
        ),
        index=dates,
        columns=["AAA", "AAA", "BBB"],
    )
    volume = pd.DataFrame(
        np.column_stack(
            [
                np.full(len(dates), 1_000.0),
                np.full(len(dates), 2_000.0),
                np.full(len(dates), 3_000.0),
            ]
        ),
        index=dates,
        columns=["AAA", "AAA", "BBB"],
    )

    clean_close, clean_volume = universe.clean_close_volume_panels(close, volume, start=dates[0])
    early_tickers, early_adv = universe.select_liquid_universe(
        dates[5],
        close=clean_close,
        volume=clean_volume,
        top_n=2,
        liquidity_lookback=10,
        min_listing_days=10,
        min_obs=8,
    )
    zero_liquidity_tickers, zero_liquidity_adv = universe.select_liquid_universe(
        dates[-1],
        close=clean_close,
        volume=clean_volume * 0.0,
        top_n=2,
        liquidity_lookback=10,
        min_listing_days=10,
        min_obs=8,
    )

    assert list(clean_close.columns) == ["AAA", "BBB"]
    assert clean_close["AAA"].iloc[0] == pytest.approx(200.0)
    assert early_tickers == []
    assert early_adv.empty
    assert zero_liquidity_tickers == []
    assert zero_liquidity_adv.empty
    with pytest.raises(InputError):
        universe.select_liquid_universe(dates[-1], close=clean_close, volume=clean_volume, top_n=0)
    with pytest.raises(InputError):
        universe.select_liquid_universe(dates[-1], close=clean_close, volume=None)
    with pytest.raises(InputError):
        universe.clean_close_volume_panels(
            clean_close.assign(BBB=np.nan),
            clean_volume.assign(BBB=np.nan),
            start=dates[0],
        )


def test_sizing_caps_smoothing_forecast_and_rank_weights() -> None:
    returns = return_panel(n=95, assets=("AAA", "BBB", "CCC", "CASH"))
    dates = returns.index[-3:]
    forecast = pd.DataFrame(
        [
            {"date": dt, "asset": asset, "score": score, "mu": score * 0.002, "sigma_21": 0.04, "c_total": 0.80}
            for dt in dates
            for asset, score in zip(("AAA", "BBB", "CCC"), (1.5, 0.8, -0.2), strict=False)
        ]
    )

    capped = sizing.cap_weights(pd.Series({"AAA": 0.90, "BBB": 0.10, "CCC": 0.0}), max_weight=0.60)
    smoothed = sizing.smooth_weights(pd.DataFrame([capped, capped[::-1]], index=dates[:2]))
    kelly = sizing.kelly_weight_vector(pd.Series([0.08, 0.04, -0.01], index=["AAA", "BBB", "CCC"]), np.eye(3) * 0.05)
    ranked = sizing.rank_signal_weight_frame(
        forecast,
        score_col="score",
        vol_col="sigma_21",
        rebalance_dates=dates,
        assets=["AAA", "BBB", "CCC"],
        top_k=2,
        max_weight=0.60,
        cash_asset="CASH",
        smooth=0.0,
    )
    kelly_frame = sizing.forecast_kelly_weight_frame(
        forecast,
        mu_col="mu",
        returns=returns,
        rebalance_dates=dates,
        assets=["AAA", "BBB", "CCC"],
        cash_asset="CASH",
        sigma_col="sigma_21",
        mu_is_z=False,
        lookback=70,
        horizon=21,
        max_weight=0.60,
        smooth=0.0,
    )

    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.60 + 1e-12
    assert np.allclose(smoothed.sum(axis=1), 1.0)
    assert kelly.sum() <= 1.0 + 1e-12
    assert np.allclose(ranked.sum(axis=1), 1.0)
    assert not kelly_frame.empty
    assert np.allclose(kelly_frame.sum(axis=1), 1.0)


def test_sizing_handles_empty_weights_missing_forecasts_cash_and_singular_covariance() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    aligned_empty = sizing.align_weight_frame(
        pd.DataFrame(),
        target_dates=dates,
        assets=["AAA", "BBB"],
        cash_asset="CASH",
        max_weight=0.70,
    )
    aligned_zero = sizing.align_weight_frame(
        pd.DataFrame([[0.0, 0.0]], index=[dates[0]], columns=["AAA", "BBB"]),
        target_dates=dates[:1],
        assets=["AAA", "BBB"],
    )
    missing_score = sizing.rank_signal_weight_frame(
        pd.DataFrame({"date": [dates[-1]], "asset": ["AAA"], "vol_63": [0.20]}),
        score_col="score",
        rebalance_dates=dates,
        assets=["AAA", "BBB"],
        cash_asset="CASH",
    )
    singular_kelly = sizing.kelly_weight_vector(
        pd.Series([0.08, 0.08, -0.02], index=["AAA", "BBB", "CCC"]),
        pd.DataFrame(
            [[0.04, 0.04, 0.01], [0.04, 0.04, 0.01], [0.01, 0.01, 0.03]],
            index=["AAA", "BBB", "CCC"],
            columns=["AAA", "BBB", "CCC"],
        ),
        max_weight=0.60,
    )
    negative_kelly = sizing.kelly_weight_vector(
        pd.Series([-0.01, -0.02], index=["AAA", "BBB"]),
        np.eye(2),
    )

    returns = return_panel(n=90, assets=("AAA", "BBB", "CASH"))
    rebalance = returns.index[-1]
    negative_forecast = pd.DataFrame(
        {
            "date": [rebalance, rebalance],
            "asset": ["AAA", "BBB"],
            "mu": [-0.01, -0.02],
        }
    )
    cash_weights = sizing.weights_from_forecasts(
        negative_forecast,
        date_col="date",
        asset_col="asset",
        mu_col="mu",
        returns=returns,
        rebalance_dates=[rebalance],
        cash_asset="CASH",
    )
    capped = sizing.cap_weights(pd.Series({"AAA": 0.90, "BBB": 0.10}), max_weight={"AAA": 0.10, "BBB": 0.10})

    assert np.allclose(aligned_empty[["AAA", "BBB"]], 0.50)
    assert (aligned_empty["CASH"] == 0.0).all()
    assert aligned_zero.iloc[0].to_numpy() == pytest.approx([0.50, 0.50])
    assert missing_score.empty
    assert singular_kelly.sum() <= 1.0 + 1e-12
    assert singular_kelly.max() <= 0.60 + 1e-12
    assert np.allclose(negative_kelly, 0.0)
    assert cash_weights.loc[rebalance, "CASH"] == pytest.approx(1.0)
    assert capped.sum() == pytest.approx(1.0)


def test_align_and_gated_blend_weight_frames() -> None:
    dates = pd.bdate_range("2024-01-02", periods=3)
    assets = ["AAA", "BBB", "CCC"]
    base = pd.DataFrame(1.0 / 3.0, index=dates[:2], columns=assets)
    overlay = pd.DataFrame([[0.55, 0.30, 0.15], [0.20, 0.50, 0.30]], index=dates[1:], columns=assets)
    forecast = pd.DataFrame(
        [{"date": dt, "asset": asset, "score": score} for dt in dates[1:] for asset, score in zip(assets, [3.0, 2.0, 1.0], strict=False)]
    )

    aligned = sizing.align_weight_frame(base.iloc[:1], target_dates=dates, assets=assets, max_weight=0.60)
    blended = sizing.gated_blend_weight_frame(base, overlay, forecast, score_col="score", assets=assets, max_weight=0.60)

    assert list(aligned.index) == list(dates)
    assert np.allclose(aligned.sum(axis=1), 1.0)
    assert list(blended.index) == list(dates[1:])
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_selection_tables_filter_and_rank_backtest_results() -> None:
    returns = return_panel(n=80, assets=("AAA", "BBB", "CCC"))
    res_a = _backtest_result(returns["AAA"], "EW")
    res_b = _backtest_result(returns["BBB"] + 0.0003, "MV")
    results = {"EW": res_a, "MV (Momentum, Sample)": res_b}

    metrics, trades = selection.summarize_results(results)
    summary = selection.build_strategy_summary(results)
    best, sharpes = selection.best_strategy_by_sharpe(results, min_obs=20)
    finalists = selection.select_finalists(summary, minvar_n=0, mv_n=1, ridge_n=0, maxsharpe_n=0, include_frontier=False)

    assert set(metrics.index) == set(results)
    assert "Effective N" in trades.columns
    assert selection.strategy_display_label("MV (Sample, Momentum)") == "MV [Sample, Momentum]"
    assert best in results
    assert set(sharpes) == set(results)
    assert "EW" in finalists
