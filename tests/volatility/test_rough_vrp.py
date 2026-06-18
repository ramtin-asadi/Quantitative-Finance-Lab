from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.volatility import forecasting, rough, vrp
from tests.synthetic.generators import return_panel


def test_rough_variance_moment_scaling_and_forecast_frame() -> None:
    returns = return_panel(n=90, assets=("AAA",))["AAA"]
    daily_var = rough.daily_variance(returns, annualization=252.0)
    log_var = rough.log_variance(daily_var)
    cov = rough.fgn_covariance(0.2, 8)
    paths = rough.fbm_cholesky_paths(h_values=(0.2,), n_steps=24, n_paths=2, seed=3)
    scaling = rough.moment_scaling(log_var, q_values=(1.0, 2.0), lags=(1, 2, 4, 8))
    hurst = rough.hurst_from_moments(scaling)
    pooled = rough.hurst_from_moments_pooled(scaling, use_huber=False)
    weights = rough.rough_kernel_weights(0.1, lookback=20, horizon=5)
    forecasts = rough.rough_kernel_forecasts(daily_var, h=0.1, horizons=(1, 5), train_window=30, signal_step=10)
    targets = forecasting.future_realized_variance(returns, horizons=(1, 5))
    frame = rough.rough_forecast_frame(rough_fc=forecasts, rv_targets=targets, horizons=(1, 5))
    multi = rough.hurst_multi_window(returns, windows=(1, 2, 5), lags=(1, 2, 4), q_values=(1.0, 2.0), use_huber=False)

    assert daily_var.name == "daily_variance"
    assert log_var.notna().all()
    assert cov.shape == (8,)
    assert set(paths.columns) == {"h", "path", "t", "x"}
    assert not scaling.empty
    assert not hurst.empty
    assert pooled.loc[0, "n"] >= 4
    assert weights.sum() == pytest.approx(1.0)
    assert set(forecasts["horizon"]) == {1, 5}
    assert {"realized_var_sum", "realized_vol_ann"}.issubset(frame.columns)
    assert "main_H" in multi.columns


def test_vrp_interpolates_forecasts_to_option_dte_and_scores_premium() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    rows = []
    for i, date in enumerate(dates):
        for horizon in (5, 21, 42):
            forecast_sum = 0.00002 * horizon * (1.0 + 0.02 * i)
            rows.append(
                {
                    "date": date,
                    "model": "har_rv",
                    "selected_model": "har_rv",
                    "horizon": horizon,
                    "forecast_var_sum": forecast_sum,
                    "forecast_var_daily": forecast_sum / horizon,
                    "forecast_var_ann": 252.0 * forecast_sum / horizon,
                    "forecast_vol_ann": np.sqrt(252.0 * forecast_sum / horizon),
                }
            )
    forecasts = pd.DataFrame(rows)
    target = pd.DataFrame({"date": dates, "target_dte": np.linspace(8.0, 30.0, len(dates))})
    interpolated = vrp.interpolate_forecast_variance_to_dte(forecasts, target)
    iv_panel = pd.DataFrame(
        {
            "date": dates,
            "dte": np.linspace(12.0, 45.0, len(dates)),
            "atm_iv_mid": np.linspace(0.18, 0.24, len(dates)),
        }
    )
    panel = vrp.compute_vrp_panel(
        iv_panel,
        forecasts,
        dte_col="dte",
        iv_col="atm_iv_mid",
        z_window=4,
        rank_window=4,
        min_periods=2,
    )

    assert len(interpolated) == len(dates)
    assert interpolated["forecast_cum_var"].gt(0.0).all()
    assert not panel.empty
    assert {"vrp_var", "vol_spread", "vrp_z", "vrp_rank"}.issubset(panel.columns)
    assert panel["vrp_rank"].notna().sum() >= 1
