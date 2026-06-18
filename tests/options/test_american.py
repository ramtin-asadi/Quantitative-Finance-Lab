from __future__ import annotations

import numpy as np

from quantfinlab.options.american import (
    american_premium,
    assignment_risk,
    boundary_distance,
    european_tree_batch,
    model_disagreement,
    pde_boundary,
    pde_price,
    pricing_error,
    roll_signal,
    tree_boundary,
    tree_price,
)
from quantfinlab.options.bsm import bsm_price


def test_tree_prices_european_close_to_bsm_and_american_put_has_premium() -> None:
    spot = 100.0
    strike = 102.0
    rate = 0.035
    dividend = 0.0
    sigma = 0.22
    tau = 0.5

    euro_tree = tree_price(spot, strike, rate, dividend, sigma, tau, "put", steps=140, american=False, engine="numpy")
    euro_bsm = bsm_price("put", spot, strike, tau, sigma, rate=rate, dividend_yield=dividend)
    american_put = tree_price(spot, strike, rate, dividend, sigma, tau, "put", steps=140, american=True, engine="numpy")
    batch = european_tree_batch(np.array([spot, spot]), np.array([95.0, 105.0]), rate, dividend, sigma, tau, ["call", "put"], steps=80, engine="numpy")

    assert abs(euro_tree - euro_bsm) < 0.08
    assert american_put >= euro_tree
    assert batch.shape == (2,)
    assert tree_boundary(spot, strike, rate, dividend, sigma, tau, "put", steps=30, engine="numpy").shape[0] == 31


def test_pde_outputs_and_assignment_risk_helpers_are_well_formed() -> None:
    result = pde_price(100.0, 100.0, 0.03, 0.0, 0.22, 0.35, "put", s_steps=50, t_steps=35, max_iter=500, engine="numpy")
    boundary = pde_boundary(result, tau=0.35)
    quotes = boundary.assign(
        option_type="call",
        spot=110.0,
        strike=100.0,
        mid=10.5,
        next_dividend=1.0,
        boundary_distance=0.01,
        rel_spread=0.02,
        model_disagreement=0.10,
        dte_days=7,
    )
    risk = assignment_risk(quotes.head(3))
    signals = roll_signal(risk, threshold=0.5)

    assert result["engine_used"] == "numpy"
    assert result["price"] > 0
    assert boundary.shape[0] == len(result["boundary"])
    assert np.allclose(pricing_error([1.2, 1.4], [1.0, 1.5]), [0.2, -0.1])
    assert np.allclose(model_disagreement([1.0, 2.0], [1.4, 1.7]), [0.4, 0.3])
    assert np.isfinite(boundary_distance([100.0], [98.0], "put")).all()
    assert risk["assignment_risk"].between(0, 1).all()
    assert signals["roll_signal"].dtype == bool
    assert american_premium([4.0], "put", [100.0], [100.0], [0.25], [0.2], rate=0.03)[0] >= -1.0
