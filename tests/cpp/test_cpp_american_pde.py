from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cpp


def _kernels():
    return pytest.importorskip("quantfinlab._kernels")


def test_cpp_pde_psor_returns_grid_residuals_and_early_exercise_premium() -> None:
    kernels = _kernels()
    european = kernels.american_pde_psor(100.0, 100.0, 0.03, 0.01, 0.22, 0.5, -1, 60, 40, 3.0, 1.3, 1e-6, 2000, False)
    american = kernels.american_pde_psor(100.0, 100.0, 0.03, 0.01, 0.22, 0.5, -1, 60, 40, 3.0, 1.3, 1e-6, 2000, True)

    assert {"price", "s_grid", "boundary", "residuals", "values"} == set(american)
    assert american["s_grid"].shape == (61,)
    assert american["values"].shape == (41, 61)
    assert american["boundary"].shape == (41,)
    assert np.isfinite(american["price"])
    assert american["price"] >= european["price"]
    assert np.nanmax(american["residuals"]) < 5e-6


def test_cpp_pde_batch_returns_prices_and_residual_summaries() -> None:
    kernels = _kernels()
    spot = np.array([100.0, 100.0, 100.0])
    strike = np.array([90.0, 100.0, 110.0])
    rate = np.full(3, 0.03)
    dividend = np.full(3, 0.01)
    sigma = np.full(3, 0.22)
    tau = np.full(3, 0.5)
    flags = np.array([1, -1, -1], dtype=np.int32)

    out = kernels.american_pde_psor_batch(spot, strike, rate, dividend, sigma, tau, flags, 50, 30, 3.0, 1.3, 1e-6, 2000, True)

    assert {"prices", "residuals"} == set(out)
    assert out["prices"].shape == (3,)
    assert out["residuals"].shape == (3,)
    assert np.isfinite(out["prices"]).all()
    assert np.nanmax(out["residuals"]) < 5e-6
