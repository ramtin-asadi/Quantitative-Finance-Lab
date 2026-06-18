from __future__ import annotations

import numpy as np

from quantfinlab.options import bsm


def test_forward_black_prices_respect_parity_bounds_and_time_value() -> None:
    forward = 102.0
    strikes = np.array([90.0, 100.0, 110.0])
    tau = 0.5
    sigma = 0.24
    discount = np.exp(-0.03 * tau)

    calls = bsm.forward_bsm_call(forward, strikes, tau, sigma, discount)
    puts = bsm.forward_bsm_put(forward, strikes, tau, sigma, discount)

    np.testing.assert_allclose(calls - puts, discount * (forward - strikes), rtol=1e-12, atol=1e-12)
    lower, upper = bsm.no_arbitrage_bounds(["call", "call", "call"], forward, strikes, discount)
    assert np.all(calls >= lower - 1e-12)
    assert np.all(calls <= upper + 1e-12)
    assert np.all(bsm.time_value(["call", "call", "call"], calls, forward, strikes, discount) >= -1e-12)


def test_spot_pricer_matches_forward_pricer_and_characteristic_function() -> None:
    spot = 100.0
    strike = 101.0
    tau = 0.75
    sigma = 0.21
    rate = 0.035
    dividend_yield = 0.012
    forward = spot * np.exp((rate - dividend_yield) * tau)
    discount = np.exp(-rate * tau)

    spot_price = bsm.bsm_price("call", spot, strike, tau, sigma, rate=rate, dividend_yield=dividend_yield)
    forward_price = bsm.forward_bsm_price("call", forward, strike, tau, sigma, discount)

    assert np.isclose(spot_price, forward_price, rtol=1e-12)
    assert bsm.bsm_cf(0.0, spot, rate, dividend_yield, tau, sigma) == 1.0 + 0.0j
