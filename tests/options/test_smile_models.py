from __future__ import annotations

import numpy as np

from quantfinlab.options.sabr import fit_sabr_surface, sabr_hagan_iv, sabr_prices
from quantfinlab.options.ssvi import fit_ssvi_surface, ssvi_prices, ssvi_total_var
from quantfinlab.options.svi import fit_svi_surface, svi_iv, svi_prices, svi_total_var
from tests.synthetic.generators import option_surface_quotes


def test_svi_ssvi_and_sabr_formula_shapes_are_positive() -> None:
    k = np.array([-0.1, 0.0, 0.1])
    tau = np.array([0.2, 0.4, 0.6])

    assert np.all(svi_total_var(k, 0.01, 0.05, -0.4, 0.0, 0.2, engine="numpy") > 0)
    assert np.all(svi_iv(k, tau, 0.01, 0.05, -0.4, 0.0, 0.2, engine="numpy") > 0)
    assert np.all(ssvi_total_var(k, theta=0.04, rho=-0.4, eta=1.2, gamma=0.5) > 0)
    assert np.all(sabr_hagan_iv(100.0, 100.0 * np.exp(k), tau, 0.22, 1.0, -0.25, 0.8, engine="numpy") > 0)


def test_smile_surface_fits_return_params_and_model_prices() -> None:
    quotes = option_surface_quotes()

    svi_fit = fit_svi_surface(quotes, engine="numpy")
    ssvi_fit = fit_ssvi_surface(quotes, engine="numpy")
    sabr_fit = fit_sabr_surface(quotes, betas=(1.0,), primary_beta=1.0, engine="numpy")

    for fit, price_func in [(svi_fit, svi_prices), (ssvi_fit, ssvi_prices), (sabr_fit, sabr_prices)]:
        prices = price_func(quotes, fit, engine="numpy")
        assert not fit["params"].empty
        assert not fit["fit"].empty
        assert not prices.empty
        assert np.isfinite(prices["model_price"]).all()
