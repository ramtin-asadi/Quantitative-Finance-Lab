from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fixed_income.term_models import (
    ar1_expected_score,
    cir_expected_average,
    cir_yield_loading,
    estimate_g2_style,
    estimate_hw1f,
    estimate_mean_reversion,
    estimate_pca_curve,
    fit_cir_fast,
    fit_vasicek_ar,
    fit_vasicek_kalman,
    g2_loadings,
    hw1f_loading,
    model_curve_view,
    rolling_model_views,
    rolling_pca_diagnostics,
    simulate_cir_paths,
    simulate_g2_curves,
    simulate_hw1f_curves,
    simulate_vasicek_paths,
    vasicek_ab,
    vasicek_expected_average,
    vasicek_kalman,
    vasicek_loadings,
    vasicek_yield_loading,
)


def test_short_rate_simulations_are_seeded_and_shape_stable() -> None:
    vasicek_params = {"kappa": 0.7, "theta": 0.035, "sigma": 0.01}
    cir_params = {"kappa": 0.9, "theta": 0.04, "sigma": 0.08, "shift": 0.0}

    v1 = simulate_vasicek_paths(0.03, vasicek_params, years=1.0, steps_per_year=12, n_paths=5, seed=7)
    v2 = simulate_vasicek_paths(0.03, vasicek_params, years=1.0, steps_per_year=12, n_paths=5, seed=7)
    cir = simulate_cir_paths(0.03, cir_params, years=1.0, steps_per_year=12, n_paths=5, seed=8)

    assert v1.shape == (13, 5)
    assert v1.equals(v2)
    assert cir.shape == (13, 5)
    assert cir.min().min() >= 0.0
    assert vasicek_expected_average(0.03, vasicek_params, years=2.0) == pytest.approx(0.03195, abs=2e-3)
    assert np.all(vasicek_yield_loading(vasicek_params, np.asarray([1.0, 5.0])) > 0.0)


def test_pca_curve_estimator_orients_main_loadings() -> None:
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    maturities = np.asarray([0.5, 2.0, 5.0, 10.0])
    level = np.linspace(0.02, 0.04, len(idx))[:, None]
    slope = np.linspace(-0.002, 0.002, len(idx))[:, None] * maturities[None, :]
    history = pd.DataFrame(level + slope, index=idx, columns=maturities)

    fit = estimate_pca_curve(history, maturities, n_components=3)

    assert fit["loadings"].shape == (4, 3)
    assert fit["scores"].shape[1] == 3
    assert np.isfinite(fit["explained"]).all()
    assert fit["explained"].sum() > 0.90


def _monthly_curve_history(periods: int = 84) -> pd.DataFrame:
    idx = pd.date_range("2018-01-31", periods=periods, freq="ME")
    maturities = np.asarray([0.25, 1.0, 2.0, 5.0, 10.0], dtype=float)
    t = np.linspace(0.0, 1.0, periods)
    level = (0.025 + 0.010 * t)[:, None]
    slope = (0.004 * np.sin(2.0 * np.pi * t))[:, None] * np.log1p(maturities)[None, :] / np.log(11.0)
    curvature = (0.0015 * np.cos(4.0 * np.pi * t))[:, None] * np.exp(-((maturities - 2.0) ** 2))[None, :]
    return pd.DataFrame(level + slope + curvature, index=idx, columns=maturities)


def test_vasicek_cir_estimators_and_kalman_filter_return_usable_parameters() -> None:
    history = _monthly_curve_history()
    short = history[0.25]
    maturities = history.columns.to_numpy(dtype=float)

    ar_params = fit_vasicek_ar(short)
    kalman_params, filtered = fit_vasicek_kalman(history, maturities, maxiter=8)
    nll, states = vasicek_kalman(history.iloc[:12], maturities, kalman_params, state_hint=short.iloc[:12].to_numpy())
    cir_log: list[dict] = []
    cir_params = fit_cir_fast(short, maxiter=8, fit_log=cir_log)
    a, b = vasicek_ab(ar_params["kappa"], ar_params["theta"], ar_params["sigma"], np.asarray([1.0, 5.0]))
    affine, loadings = vasicek_loadings(ar_params, np.asarray([1.0, 5.0]))

    assert ar_params["kappa"] > 0.0
    assert kalman_params["sigma"] > 0.0
    assert filtered.index.equals(history.index)
    assert np.isfinite(nll)
    assert states.shape == (12,)
    assert cir_params["theta"] > 0.0
    assert cir_expected_average(short.iloc[-1], cir_params, years=3.0) > -0.01
    assert np.all(cir_yield_loading(cir_params, np.asarray([1.0, 5.0])) > 0.0)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(np.isfinite(affine)) and np.all(loadings > 0.0)


def test_hw_g2_pca_curve_views_and_simulations_are_shape_stable() -> None:
    history = _monthly_curve_history()
    maturities = history.columns.to_numpy(dtype=float)
    date = history.index[-1]
    params = pd.DataFrame(
        {
            "vasicek kappa": [0.7],
            "vasicek theta": [0.035],
            "vasicek sigma": [0.012],
            "cir kappa": [0.9],
            "cir theta": [0.035],
            "cir sigma": [0.04],
            "cir shift": [0.0],
        },
        index=[date],
    )

    hw = estimate_hw1f(history, maturities)
    g2 = estimate_g2_style(history, maturities)
    hw_curves = simulate_hw1f_curves(history.iloc[-1], hw, n_scenarios=6, seed=4)
    g2_curves = simulate_g2_curves(history.iloc[-1], g2, n_scenarios=6, seed=5)
    rolling = rolling_pca_diagnostics(history, maturities, dates=history.index[-3:], window=36)
    v_expected, v_cov = model_curve_view(
        "vasicek",
        date,
        maturities,
        zero_rates=history,
        short_rate=history[0.25],
        model_parameters=params,
        scenario_maturities=maturities,
        rolling_window=36,
    )
    c_expected, c_cov = model_curve_view(
        "cir",
        date,
        maturities,
        zero_rates=history,
        short_rate=history[0.25],
        model_parameters=params,
        scenario_maturities=maturities,
        rolling_window=36,
    )
    hw_expected, hw_cov = model_curve_view(
        "hw1f",
        date,
        maturities,
        zero_rates=history,
        short_rate=history[0.25],
        model_parameters=params,
        scenario_maturities=maturities,
        rolling_window=36,
    )
    views_table = rolling_model_views(
        [date],
        ["vasicek", "cir", "hw1f", "g2"],
        maturities,
        zero_rates=history,
        short_rate=history[0.25],
        model_parameters=params,
        scenario_maturities=maturities,
        rolling_window=36,
    )

    assert ar1_expected_score(np.linspace(-1.0, 1.0, 20))[1] <= 0.35
    assert estimate_mean_reversion(history[0.25]) > 0.0
    assert np.all(hw1f_loading(0.3, maturities) > 0.0)
    assert g2_loadings(0.4, 0.1, maturities).shape == (len(maturities), 2)
    assert hw["variance share"] >= 0.0
    assert g2["factor covariance"].shape == (2, 2)
    assert hw_curves.shape == (6, len(maturities))
    assert g2_curves.shape == (6, len(maturities))
    assert rolling["three pc total"].gt(0.0).all()
    assert v_expected.shape == c_expected.shape == hw_expected.shape == maturities.shape
    assert v_cov.shape == c_cov.shape == hw_cov.shape == (len(maturities), len(maturities))
    assert set(views_table.index.get_level_values("model")) == {"vasicek", "cir", "hw1f", "g2"}
