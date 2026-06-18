from __future__ import annotations

import numpy as np

from quantfinlab.options import bsm
from quantfinlab.options.greeks import (
    compute_greek_bands_from_iv_band,
    compute_greeks_numpy,
    forward_bsm_greeks_numpy,
    greek_summary_table,
)
from tests.synthetic.generators import option_surface_quotes


def test_forward_greeks_match_central_finite_difference_delta() -> None:
    forward = 101.0
    strike = 100.0
    tau = 0.4
    sigma = 0.22
    discount = np.exp(-0.03 * tau)
    bump = 1e-3

    greeks = forward_bsm_greeks_numpy("call", forward, strike, tau, sigma, discount)
    up = bsm.forward_bsm_price("call", forward + bump, strike, tau, sigma, discount)
    down = bsm.forward_bsm_price("call", forward - bump, strike, tau, sigma, discount)

    assert np.isclose(greeks.loc[0, "forward_delta"], (up - down) / (2.0 * bump), rtol=1e-5)
    assert greeks.loc[0, "vega"] > 0
    assert greeks.loc[0, "gamma"] > 0


def test_compute_greeks_and_iv_bands_produce_summary_columns() -> None:
    quotes = option_surface_quotes().head(12)

    greeks = compute_greeks_numpy(quotes)
    bands = compute_greek_bands_from_iv_band(quotes)
    summary = greek_summary_table(greeks, bands)

    assert greeks.attrs["greek_engine"] == "numpy"
    assert {"delta", "gamma", "vega", "delta_mid", "delta_numpy"}.issubset(greeks.columns)
    assert np.isfinite(greeks[["delta", "gamma", "vega"]]).all().all()
    assert (bands["vega_band"] >= 0).all()
    assert set(summary["greek"]) == {"delta", "gamma", "vega", "volga", "vanna", "theta", "rho"}
