from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.volatility import forecasting, har, realized
from tests.synthetic.generators import price_panel, return_panel


def _single_asset_returns(n: int = 140) -> pd.Series:
    returns = return_panel(n=n, assets=("AAA",))["AAA"]
    returns.name = "return"
    return returns


def test_realized_returns_volatility_and_iv_alignment() -> None:
    prices = price_panel(n=40, assets=("AAA",))["AAA"]
    simple = realized.simple_returns(prices)
    log_ret = realized.log_returns(prices)
    vol = realized.realized_volatility(log_ret.dropna())
    table = realized.realized_volatility_table(log_ret.dropna(), windows=(5, 10))
    options = pd.DataFrame(
        {
            "date": [prices.index[10], prices.index[20]],
            "iv_mid": [0.22, 0.25],
            "expiry": [prices.index[20], prices.index[30]],
        }
    )

    aligned = realized.align_realized_to_option_expiries(table[["rv_5"]], options)
    comparison = realized.compare_realized_implied_vol(table[["rv_5"]], options, iv_col="iv_mid")
    summary = realized.compare_realized_implied_vol_summary(comparison)

    assert simple.dropna().iloc[0] == pytest.approx(prices.iloc[1] / prices.iloc[0] - 1.0)
    assert np.isfinite(vol)
    assert {"rv_5", "rv_10", "rv_full"}.issubset(table.columns)
    assert aligned["rv_5"].notna().all()
    assert comparison["iv_minus_rv"].notna().all()
    assert summary.loc[0, "n"] == 2


def test_har_features_fit_predict_and_rolling_forecasts() -> None:
    returns = _single_asset_returns(150)
    rv_daily = returns.pow(2)
    target = rv_daily.shift(-1)

    features = har.make_har_features(rv_daily, weekly_window=5, monthly_window=22, use_log=True)
    fit = har.fit_har_rv(rv_daily, target, weekly_window=5, monthly_window=22, use_log=True)
    prediction = fit.predict(features.dropna().tail(2))
    rolling = har.rolling_har_forecasts(
        returns,
        horizons=(1, 5),
        train_window=55,
        refit_every=15,
        use_log=True,
    )

    assert all(col.startswith("log_") for col in features.columns)
    assert fit.n_obs > 30
    assert prediction.gt(0.0).all()
    assert set(rolling["horizon"]) == {1, 5}
    assert rolling["forecast_var_sum"].gt(0.0).all()


def test_forecast_scoring_selection_and_statistical_tables() -> None:
    returns = _single_asset_returns(80)
    targets = forecasting.future_realized_variance(returns, horizons=(1, 5))
    dates = targets.index[:35]
    rows: list[dict[str, object]] = []
    for date in dates:
        for horizon in (1, 5):
            realized_sum = float(targets.loc[date, f"realized_var_sum_{horizon}"])
            if not np.isfinite(realized_sum):
                continue
            for model, scale in (("har_rv", 1.05), ("rough_kernel", 0.85)):
                forecast_sum = max(realized_sum * scale + 1e-8, 1e-10)
                rows.append(
                    {
                        "date": date,
                        "model": model,
                        "horizon": horizon,
                        "forecast_var_sum": forecast_sum,
                        "forecast_var_daily": forecast_sum / horizon,
                        "forecast_var_ann": 252.0 * forecast_sum / horizon,
                        "forecast_vol_ann": np.sqrt(252.0 * forecast_sum / horizon),
                        "realized_var_sum": realized_sum,
                        "realized_var_ann": 252.0 * realized_sum / horizon,
                        "realized_vol_ann": np.sqrt(252.0 * realized_sum / horizon),
                    }
                )
    panel = pd.DataFrame(rows)

    scores = forecasting.score_forecasts_by_model(panel)
    mz = forecasting.mincer_zarnowitz_table(panel)
    dm = forecasting.diebold_mariano_table(panel, benchmark_model="har_rv")
    selected = forecasting.select_forecast_by_rolling_loss(panel, horizons=(1, 5), lookback=20, min_obs=5)
    dates_weekly = forecasting.make_weekly_signal_dates(returns.index, step=5)

    assert not scores.empty
    assert np.isfinite(forecasting.qlike_loss(panel["realized_var_sum"], panel["forecast_var_sum"]))
    assert not mz.empty
    assert not dm.empty
    assert not selected.empty
    assert set(selected["horizon"]).issubset({1, 5})
    assert len(dates_weekly) == pytest.approx(np.ceil(len(returns.index) / 5), abs=1)
