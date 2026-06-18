from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fixed_income.term_models import (
    estimate_pca_curve,
    simulate_cir_paths,
    simulate_vasicek_paths,
    vasicek_expected_average,
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
