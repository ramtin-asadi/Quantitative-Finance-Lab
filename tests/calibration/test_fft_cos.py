from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.calibration import fft_cos
from quantfinlab.options.surface import fit_log_total_variance_surface
from tests.synthetic.generators import option_surface_quotes


def _surface_quotes() -> pd.DataFrame:
    return option_surface_quotes(
        dates=("2024-01-02", "2024-01-03"),
        tau_days=(21, 45, 75),
        k_values=(-0.20, -0.10, 0.0, 0.10, 0.20),
    )


def test_calibration_weights_groups_and_cos_price_residuals() -> None:
    quotes = _surface_quotes().head(20)
    weighted = fft_cos.calibration_weights(quotes)
    groups = fft_cos.cos_group_arrays(weighted.head(10))
    grouped_prices = fft_cos.cos_prices_grouped(
        "bsm",
        {"sigma": 0.22},
        weighted.head(10),
        engine="numpy",
        n_terms=64,
    )
    from_groups = fft_cos.cos_prices_from_groups(
        "bsm",
        {"sigma": 0.22},
        groups,
        len(weighted.head(10)),
        engine="numpy",
        n_terms=64,
    )
    residuals = fft_cos.price_residuals(weighted.head(10), "bsm", {"sigma": 0.22}, engine="numpy")

    assert weighted["calib_scale_px"].gt(0.0).all()
    assert weighted["obs_weight"].median() == pytest.approx(1.0)
    assert len(groups) >= 1
    assert np.allclose(grouped_prices, from_groups)
    assert np.isfinite(grouped_prices).all()
    assert {"model_price", "price_residual", "iv_residual"}.issubset(residuals.columns)
    assert residuals["model_price"].gt(0.0).all()


def test_calibration_grid_and_fourier_summary_tables_are_ranked() -> None:
    quotes = _surface_quotes()
    grid, steps = fft_cos.calibration_grid_quotes(
        quotes,
        min_quotes_per_expiry=4,
        min_expiries_per_date=2,
        min_quotes_per_date=8,
        max_quotes_per_date=12,
        return_steps=True,
    )
    fit_a = {
        "diag": pd.DataFrame([{"weighted_price_rmse": 1.0, "runtime": 0.20, "quotes": 10}]),
        "params": pd.DataFrame([{"p0": 0.20, "success": True}]),
        "elapsed_sec": 0.25,
    }
    fit_b = {
        "diag": pd.DataFrame([{"weighted_price_rmse": 0.8, "runtime": 0.40, "quotes": 10}]),
        "params": pd.DataFrame([{"p0": 0.30, "success": True}]),
        "elapsed_sec": 0.45,
    }
    comparison = fft_cos.compare_fourier_models(fits={"bsm": fit_a, "merton": fit_b})
    warm = fft_cos.warm_start_table({"bsm": fit_a})

    assert not grid.empty
    assert {"calib_scale_px", "obs_weight", "dte_bucket", "log_moneyness_bucket"}.issubset(grid.columns)
    assert steps.iloc[-1]["step"] == "daily grid"
    assert comparison.iloc[0]["model"] == "merton"
    assert fft_cos.family_winner(comparison) == "merton"
    assert warm.loc[0, "p0"] == pytest.approx(0.20)


def test_daily_success_and_residual_bucket_tables() -> None:
    daily = pd.DataFrame(
        {
            "model": ["bsm", "bsm", "vg"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
            "success": [True, False, True],
            "nfev": [3, 5, 4],
            "runtime_sec": [0.10, 0.20, 0.30],
            "quotes": [10, 9, 8],
            "weighted_price_rmse": [1.0, 2.0, 1.5],
            "median_abs_price_error": [0.1, 0.2, 0.3],
            "weighted_iv_rmse": [0.01, 0.02, 0.03],
            "bid_ask_hit_rate": [0.8, 0.7, 0.9],
            "otm_put_rmse": [1.0, 2.0, 3.0],
            "short_maturity_rmse": [1.0, 1.0, 1.0],
        }
    )
    quotes = fft_cos.calibration_weights(_surface_quotes().head(10))
    residuals = fft_cos.price_residuals(quotes, "bsm", {"sigma": 0.22}, engine="numpy").assign(model="bsm")
    model_table = fft_cos.compare_fourier_models(daily=daily)
    success = fft_cos.calibration_success_table(daily)
    bucket = fft_cos.residual_by_bucket(residuals)

    assert model_table.iloc[0]["model"] == "vg"
    assert success.loc[success["model"].eq("bsm"), "failures"].iloc[0] == 1
    assert not bucket.empty
    assert {"median_scaled_residual", "rows"}.issubset(bucket.columns)


def test_surface_target_grid_and_small_bsm_date_calibration_workflow() -> None:
    quotes = _surface_quotes()
    fits = {}
    for date, day in quotes.groupby("date"):
        fits[pd.Timestamp(date)] = fit_log_total_variance_surface(
            day,
            n_k_basis=5,
            n_tau_basis=4,
            degree=2,
            lambda_k=0.01,
            lambda_tau=0.01,
        )

    target, steps = fft_cos.surface_target_grid_quotes(
        quotes,
        fits,
        min_dte=14,
        max_dte=90,
        dte_targets=(21, 45, 75),
        k_targets=(-0.10, 0.0, 0.10),
        min_quotes_per_date=6,
        min_source_quotes_per_date=20,
        return_steps=True,
    )
    fit = fft_cos.fit_date_model(
        target.loc[target["date"].eq(target["date"].min())].head(8),
        "bsm",
        max_nfev=4,
        engine="numpy",
        n_terms=48,
    )
    daily = fft_cos.fit_daily_models(
        target,
        ["bsm"],
        min_quotes=6,
        max_nfev=3,
        engine="numpy",
        n_terms=48,
    )

    assert not target.empty
    assert steps.iloc[-1]["step"] == "balanced surface grid"
    assert {"iv_target_error", "quote_quality_weight", "obs_weight"}.issubset(target.columns)
    assert fit["row"]["quotes"] == 8
    assert fit["fit"]["model_price"].gt(0.0).all()
    assert not daily["params"].empty
    assert not daily["fit"].empty
