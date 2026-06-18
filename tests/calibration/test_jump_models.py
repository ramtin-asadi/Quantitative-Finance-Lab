from __future__ import annotations

import pandas as pd

from quantfinlab.calibration import jump_models
from tests.synthetic.generators import option_surface_quotes


def _quotes() -> pd.DataFrame:
    return option_surface_quotes(
        dates=("2024-01-02", "2024-01-03"),
        tau_days=(21, 45, 75),
        k_values=(-0.20, -0.10, 0.0, 0.10, 0.20),
    )


def test_tail_hedge_components_scores_candidates_and_schedule() -> None:
    quotes = _quotes()
    components = jump_models.hedge_score_components(quotes.head(8))
    scores = jump_models.tail_hedge_score(quotes.head(8))
    tail = jump_models.tail_hedge_candidates(quotes, top_n=3, top_n_per_date=4)
    fixed_delta = jump_models.fixed_delta_hedge_candidates(quotes, top_n=2)
    schedule = jump_models.tail_hedge_schedule(tail, max_entries=2, budget_notional=1_000_000.0, premium_budget_bps=100.0)

    assert {"expected_crash_efficiency", "fair_value_edge", "convexity_per_premium"}.issubset(components.columns)
    assert scores.index.equals(components.index)
    assert set(tail["option_type"]) == {"put"}
    assert tail["hedge_score"].is_monotonic_decreasing
    assert set(fixed_delta["option_type"]) == {"put"}
    assert fixed_delta["hedge_score"].notna().all()
    assert schedule.loc[0, "quantity"] > 0.0
    assert schedule.loc[0, "label"] == "tail_put"


def test_density_family_tables_and_winners() -> None:
    fit_a = {
        "diag": pd.DataFrame([{"weighted_price_rmse": 1.0, "runtime": 0.20, "quotes": 10}]),
        "params": pd.DataFrame([{"p0": 0.20, "success": True}]),
        "elapsed_sec": 0.25,
    }
    fit_b = {
        "diag": pd.DataFrame([{"weighted_price_rmse": 0.8, "runtime": 0.40, "quotes": 10}]),
        "params": pd.DataFrame([{"p0": 0.30, "success": True}]),
        "elapsed_sec": 0.45,
    }
    density = jump_models.density_summary(
        {
            "bsm": ("bsm", {"sigma": 0.20}),
            "merton": ("merton", {"sigma": 0.20, "lambda_jump": 0.30, "mu_jump": -0.05, "sigma_jump": 0.20}),
        },
        spot=100.0,
        rate=0.03,
        dividend_yield=0.01,
        tau=0.25,
    )
    jump_table = jump_models.jump_family_table(fit_a, fit_b)
    sv_table = jump_models.sv_family_table(fit_a, fit_b)

    assert density["left_tail_probability"].between(0.0, 1.0).all()
    assert density["density_peak"].gt(0.0).all()
    assert jump_models.family_winner(jump_table) == "vg"
    assert jump_table.iloc[0]["model"] == "vg"
    assert sv_table.iloc[0]["model"] == "bates"
