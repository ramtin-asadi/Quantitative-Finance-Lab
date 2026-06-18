from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.options.rates_dividends import (
    add_discount_factors,
    attach_rates_to_options,
    infer_carry_from_forward,
    infer_dividend_yield_from_forward,
)
from tests.synthetic.generators import option_surface_quotes, zero_rate_panel


def test_attach_rates_from_constant_series_and_curve_panel_then_discount() -> None:
    quotes = option_surface_quotes().head(8).drop(columns=["rate", "discount_factor"])

    constant = attach_rates_to_options(quotes, constant_rate=0.04)
    from_series = attach_rates_to_options(quotes, rates=pd.Series([0.035], index=pd.to_datetime(["2024-01-01"])))
    from_curve = attach_rates_to_options(quotes, curve_panel=zero_rate_panel(0.03))
    discounted = add_discount_factors(constant)

    assert constant["rate"].eq(0.04).all()
    assert from_series["rate"].eq(0.035).all()
    assert np.isfinite(from_curve["rate"]).all()
    np.testing.assert_allclose(discounted["discount_factor"], np.exp(-0.04 * discounted["tau"]))
    with pytest.raises(ValueError, match="exactly one"):
        attach_rates_to_options(quotes, rates=pd.Series(dtype=float), constant_rate=0.04)


def test_carry_and_dividend_yield_are_inverse_forward_transforms() -> None:
    spot = np.array([100.0, 101.0])
    rate = np.array([0.04, 0.035])
    dividend = np.array([0.01, 0.012])
    tau = np.array([0.5, 1.0])
    forward = spot * np.exp((rate - dividend) * tau)

    carry = infer_carry_from_forward(spot, forward, tau)
    inferred_dividend = infer_dividend_yield_from_forward(spot, forward, rate, tau)
    frame = infer_dividend_yield_from_forward(pd.DataFrame({"spot": spot, "forward": forward, "rate": rate, "tau": tau}))

    np.testing.assert_allclose(carry, rate - dividend)
    np.testing.assert_allclose(inferred_dividend, dividend)
    np.testing.assert_allclose(frame["implied_dividend_yield"], dividend)
