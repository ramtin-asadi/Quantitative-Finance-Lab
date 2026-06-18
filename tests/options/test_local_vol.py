from __future__ import annotations

import numpy as np

from quantfinlab.options.local_vol import (
    carry_by_tau,
    curve_by_tau,
    dividend_yield_by_tau,
    dupire_grid_numpy,
    dupire_stress_panel,
    dupire_stress_summary,
    rate_by_tau,
)
from quantfinlab.options.surface import fit_log_total_variance_surface
from tests.synthetic.generators import option_surface_quotes


def test_curve_helpers_interpolate_rates_carry_and_dividend_yield() -> None:
    quotes = option_surface_quotes()
    tau_values = np.array([25.0, 60.0, 100.0]) / 365.25

    assert np.allclose(curve_by_tau(quotes, "rate", tau_values), rate_by_tau(quotes, tau_values))
    np.testing.assert_allclose(carry_by_tau(quotes, tau_values), 0.025, atol=1e-12)
    np.testing.assert_allclose(dividend_yield_by_tau(quotes, tau_values), 0.010, atol=1e-12)


def test_dupire_numpy_grid_returns_stress_summary_and_panel() -> None:
    quotes = option_surface_quotes()
    fit = fit_log_total_variance_surface(quotes, n_k_basis=5, n_tau_basis=4, degree=2, lambda_k=0.01, lambda_tau=0.01)

    lv = dupire_grid_numpy(fit, quotes, k_min=-0.16, k_max=0.10, tau_min=21 / 365.25, tau_max=105 / 365.25, n_k=7, n_tau=6)
    summary = dupire_stress_summary(lv)
    panel = dupire_stress_panel(quotes, fits={quotes["date"].iloc[0]: fit}, engine="numpy", n_k=7, n_tau=6)

    assert lv["engine_used"] == "numpy"
    assert lv["local_vol"].shape == (6, 7)
    assert np.isfinite(lv["iv"]).any()
    assert summary["engine_used"] == "numpy"
    assert panel.loc[0, "engine_used"] == "numpy"
