from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.rough import (
    atm_skew_term_structure,
    compare_heston_rough_heston,
    forward_variance_curve,
    fractional_riccati,
    rbergomi_calibration,
    rbergomi_smile,
    riccati_convergence,
    rough_delta_grid,
    rough_heston_cf,
    rough_heston_cf_diagnostics,
    rough_heston_iv,
    rough_heston_prices,
    rough_heston_residuals,
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


def test_rbergomi_smile_and_calibration_workflow_uses_small_common_random_numbers() -> None:
    quotes = option_surface_quotes(tau_days=(21,), k_values=(-0.10, 0.0, 0.10))
    xi = pd.DataFrame({"tau": [0.05, 0.20, 0.50], "variance": [0.04, 0.042, 0.045]})

    smile = rbergomi_smile(
        quotes,
        xi=xi,
        params={"h": 0.12, "nu": 1.0, "rho": -0.6},
        maturity_days=(21,),
        paths=12,
        steps=5,
        seed=4,
        engine="numpy",
    )
    calibration = rbergomi_calibration(
        quotes,
        xi=xi,
        paths=12,
        steps=5,
        restarts=1,
        seed=4,
        engine="numpy",
        use_sobol=False,
    )

    assert smile.shape[0] == 31
    assert smile["price"].notna().all()
    assert {"h", "nu", "rho", "iv_rmse"}.issubset(calibration["params"])
    assert calibration["fit"]["iv_mse"].is_monotonic_increasing


def test_rough_heston_riccati_prices_residuals_and_delta_grid_are_finite() -> None:
    quotes = option_surface_quotes(tau_days=(21,), k_values=(-0.05, 0.0, 0.05))
    params = {"h": 0.12, "v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma_v": 0.4, "rho": -0.5}
    strikes = np.array([95.0, 100.0, 105.0])
    tau = np.full(3, 30.0 / 365.25)

    riccati = fractional_riccati(1.0 - 0.5j, params, tau[0], n_steps=16)
    cf, cf_diag = rough_heston_cf(
        np.asarray([0.0, 1.0, -1j]),
        params,
        100.0,
        0.03,
        0.0,
        tau[0],
        riccati_steps=16,
        allow_clip=True,
        return_diagnostics=True,
    )
    diagnostics = rough_heston_cf_diagnostics(params, 100.0, 0.03, 0.0, tau[0], riccati_steps=16)
    prices = rough_heston_prices(
        params,
        strikes,
        tau,
        100.0,
        0.03,
        0.0,
        engine="numpy",
        n_terms=32,
        riccati_steps=16,
        allow_cf_clip=True,
    )
    iv = rough_heston_iv(
        params,
        strikes,
        tau,
        100.0,
        0.03,
        0.0,
        engine="numpy",
        n_terms=32,
        riccati_steps=16,
        allow_cf_clip=True,
    )
    convergence = riccati_convergence(quotes.head(1), params=params, n_grid_values=(8, 16), n_terms=16, engine="numpy")
    fit = quotes.head(6).assign(model="rough_heston", price_residual=np.linspace(-0.05, 0.05, 6))
    residuals = rough_heston_residuals(fit)
    comparison = compare_heston_rough_heston(
        heston_daily=pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "quotes": [6],
                "success": [True],
                "weighted_price_rmse": [1.0],
                "weighted_iv_rmse": [0.02],
                "median_abs_price_error": [0.10],
                "runtime_sec": [0.2],
            }
        ),
        rough_daily=pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "quotes": [6],
                "success": [True],
                "weighted_price_rmse": [0.8],
                "weighted_iv_rmse": [0.015],
                "median_abs_price_error": [0.08],
                "runtime_sec": [0.3],
            }
        ),
    )
    deltas = rough_delta_grid(
        quotes,
        heston_params=params,
        rough_params=params,
        k_values=np.array([-0.02, 0.0]),
        tau_days=np.array([21.0]),
        n_terms=16,
        riccati_steps=8,
        engine="numpy",
    )

    assert riccati["abs"].iloc[-1] > 0.0
    assert cf_diag["cf_nonfinite_count"] == 0
    assert np.isfinite(cf).all()
    assert diagnostics.loc[0, "phi_zero_error"] < 1e-8
    assert np.isfinite(prices).all() and (prices >= 0.0).all()
    assert np.isfinite(iv).all()
    assert convergence["abs_change"].iloc[-1] >= 0.0
    assert not residuals.empty
    assert comparison.iloc[0]["model"] == "rough_heston"
    assert deltas.shape[0] == 2
    assert np.isfinite(deltas[["rough_heston_delta", "heston_delta", "rough_minus_heston"]]).all().all()
