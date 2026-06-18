from __future__ import annotations

import numpy as np

from quantfinlab.options.bsm import forward_bsm_price
from quantfinlab.options.iv import (
    compute_iv_table,
    implied_vol,
    iv_pricing_error_summary,
    iv_uncertainty_band,
    weighted_median,
)
from tests.synthetic.generators import option_surface_quotes


def test_implied_vol_solver_recovers_black_vols_and_reports_bounds() -> None:
    forward = 101.0
    strike = np.array([95.0, 101.0, 108.0])
    tau = 45.0 / 365.25
    sigma = np.array([0.23, 0.21, 0.25])
    discount = np.exp(-0.035 * tau)
    price = forward_bsm_price(["call", "call", "put"], forward, strike, tau, sigma, discount)

    recovered = implied_vol(["call", "call", "put"], price, forward, strike, tau, discount, engine="python")

    np.testing.assert_allclose(recovered, sigma, rtol=1e-7, atol=1e-7)
    iv, status, iterations = implied_vol("call", discount * forward + 1.0, forward, 100.0, tau, discount, engine="python", return_status=True)
    assert np.isnan(iv)
    assert status == 2
    assert iterations == 0


def test_compute_iv_table_adds_bands_solver_metadata_and_price_error_summary() -> None:
    quotes = option_surface_quotes().query("k.abs() <= 0.12").copy()

    table = compute_iv_table(quotes, engine="python", solver="newton_bisection")
    banded = iv_uncertainty_band(table)
    summary = iv_pricing_error_summary(banded)

    assert bool(table["iv_mid_success"].all())
    assert table.attrs["engine_used"] == "python"
    assert np.nanmedian(np.abs(table["iv_mid"] - quotes["iv_mid"])) < 1e-6
    assert (banded["iv_low"] <= banded["iv_mid"]).all()
    assert (banded["iv_high"] >= banded["iv_mid"]).all()
    assert summary.loc[0, "max_abs_error"] < 1e-6
    assert weighted_median([0.2, 0.3, 0.4], [1.0, 3.0, 1.0]) == 0.3
