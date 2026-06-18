from __future__ import annotations

import numpy as np

from quantfinlab.options.bates import bates_cf
from quantfinlab.options.bsm import forward_bsm_price
from quantfinlab.options.heston import heston_cf, heston_mc_price
from quantfinlab.options.merton import merton_cf, merton_price
from quantfinlab.options.variance_gamma import vg_cf


def test_jump_and_stochastic_characteristic_functions_are_normalized() -> None:
    spot = 100.0
    rate = 0.03
    div = 0.01
    tau = 0.5

    assert merton_cf(0.0, spot, rate, div, tau, 0.2, 0.3, -0.05, 0.15) == 1.0 + 0.0j
    assert heston_cf(0.0, spot, rate, div, tau, 0.04, 1.5, 0.04, 0.35, -0.4) == 1.0 + 0.0j
    assert bates_cf(0.0, spot, rate, div, tau, 0.04, 1.5, 0.04, 0.35, -0.4, 0.2, -0.05, 0.15) == 1.0 + 0.0j
    assert vg_cf(0.0, spot, rate, div, tau, 0.2, -0.05, 0.3) == 1.0 + 0.0j


def test_merton_zero_jump_intensity_matches_black76() -> None:
    forward = 101.0
    strike = np.array([95.0, 100.0, 105.0])
    tau = 0.4
    discount = np.exp(-0.03 * tau)
    sigma = 0.22

    merton = merton_price("call", forward, strike, tau, discount, sigma, lambda_jump=0.0, mu_jump=0.0, sigma_jump=0.2, engine="numpy")
    black = forward_bsm_price("call", forward, strike, tau, sigma, discount)

    np.testing.assert_allclose(merton, black, rtol=1e-10, atol=1e-10)


def test_heston_monte_carlo_small_run_returns_price_and_standard_error() -> None:
    tau = np.array([0.2, 0.2])
    forward = np.full(2, 100.0 * np.exp((0.03 - 0.01) * tau[0]))
    discount = np.exp(-0.03 * tau)
    price, standard_error = heston_mc_price(
        option_type=np.array(["call", "call"]),
        forward=forward,
        strike=np.array([95.0, 105.0]),
        tau=tau,
        discount_factor=discount,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.25,
        rho=-0.3,
        paths=200,
        steps_per_year=20,
        random_state=7,
        engine="numpy",
    )

    assert price.shape == (2,)
    assert standard_error.shape == (2,)
    assert np.isfinite(price).all()
    assert (standard_error >= 0).all()
