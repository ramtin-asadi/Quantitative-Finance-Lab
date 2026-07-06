from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options import bsm
from quantfinlab.options.greeks import (
    compute_greek_bands_from_iv_band,
    compute_greeks_numpy,
    forward_bsm_greeks_numpy,
    greek_summary_table,
    surface_delta_gamma_grid,
    surface_delta_gamma_risk,
    surface_greek_risk_panel,
)
from quantfinlab.options.surface import fit_log_total_variance_surface
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


def test_surface_delta_gamma_risk_grid_uses_numpy_fallback_workflow() -> None:
    quotes = option_surface_quotes(dates=("2024-01-02", "2024-01-03"))
    one_day = quotes.loc[quotes["date"].eq(pd.Timestamp("2024-01-02"))].copy()
    fit = fit_log_total_variance_surface(
        one_day,
        n_k_basis=5,
        n_tau_basis=4,
        degree=2,
        lambda_k=0.01,
        lambda_tau=0.01,
    )

    grid = surface_delta_gamma_grid(
        fit,
        one_day,
        n_k=7,
        tau_days=[21, 60],
        spot_shock=0.01,
        engine="numpy",
    )
    risk = surface_delta_gamma_risk(grid)
    panel = surface_greek_risk_panel(
        quotes,
        fits={pd.Timestamp("2024-01-02"): fit},
        n_k=5,
        tau_days=[21, 60],
        engine="numpy",
    )

    assert grid.attrs["engine_used"] == "numpy"
    assert grid.shape[0] == 14
    assert np.isfinite(grid[["delta_surface", "gamma_surface", "delta_flat", "gamma_flat"]]).all().all()
    assert risk.loc[0, "total_greek_pnl_rms"] >= 0.0
    assert panel.loc[0, "engine_used"] == "numpy"
