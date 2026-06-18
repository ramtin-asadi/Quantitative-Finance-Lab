from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.rough import (
    atm_skew_term_structure,
    forward_variance_curve,
    simulate_rbergomi,
    skew_power_law,
)
from tests.synthetic.generators import option_surface_quotes


def test_forward_variance_and_skew_term_structure_are_deterministic() -> None:
    quotes = option_surface_quotes()

    xi = forward_variance_curve(quotes, use_pchip=False)
    skew = atm_skew_term_structure(quotes)
    power = skew_power_law(skew)

    assert {"tau", "variance", "atm_iv"}.issubset(xi.columns)
    assert (xi["variance"] > 0).all()
    assert len(skew) >= 3
    assert power.loc[0, "n"] >= 3


def test_small_rbergomi_simulation_uses_antithetic_numpy_path() -> None:
    xi = pd.DataFrame({"tau": [0.05, 0.20, 0.50], "variance": [0.04, 0.042, 0.045]})

    sim = simulate_rbergomi(spot=100.0, xi=xi, h=0.12, nu=1.0, rho=-0.6, tau=0.2, paths=10, steps=5, seed=11, engine="numpy", antithetic=True)

    assert sim["spot"].shape == (10, 6)
    assert sim["variance"].shape == (10, 6)
    assert np.isfinite(sim["spot"]).all()
    assert (sim["variance"] > 0).all()
