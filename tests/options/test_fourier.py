from __future__ import annotations

import numpy as np

from quantfinlab.options.bsm import bsm_price
from quantfinlab.options.fourier import (
    cos_density,
    cos_prices,
    direct_price,
    fft_grid,
    model_cf,
    risk_neutral_density,
    tail_probability,
)


def test_fourier_bsm_prices_are_close_to_closed_form() -> None:
    spot = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    rate = 0.035
    dividend = 0.01
    tau = 0.5
    params = {"sigma": 0.22}

    direct = direct_price("bsm", params, spot, strikes, rate, dividend, tau, option_type="call", engine="numpy", n=768)
    cos = cos_prices("bsm", params, strikes, tau, spot, rate, dividend, option_type="call", engine="numpy", n_terms=128)
    closed = bsm_price("call", spot, strikes, tau, params["sigma"], rate=rate, dividend_yield=dividend)

    assert model_cf("bsm", 0.0, params, spot, rate, dividend, tau) == 1.0 + 0.0j
    np.testing.assert_allclose(direct, closed, atol=0.08)
    np.testing.assert_allclose(cos, closed, atol=0.08)


def test_fft_grid_and_density_helpers_return_finite_outputs() -> None:
    x_grid = np.linspace(np.log(70.0), np.log(140.0), 151)

    fft = fft_grid("bsm", {"sigma": 0.20}, 100.0, 0.03, 0.0, 0.4, n=64, engine="numpy")
    density = cos_density("bsm", {"sigma": 0.20}, x_grid, 100.0, 0.03, 0.0, 0.4, n_terms=256)
    rn_density = risk_neutral_density("bsm", {"sigma": 0.20}, x_grid, 100.0, 0.03, 0.0, 0.4, n_terms=128)
    tail = tail_probability(x_grid, density, np.log(90.0))

    assert {"strike", "price"}.issubset(fft.columns)
    assert np.isfinite(fft["price"]).all()
    assert np.isclose(np.trapezoid(density, x_grid), 1.0, atol=1e-3)
    assert np.isfinite(rn_density).all()
    assert 0.0 <= tail <= 1.0
