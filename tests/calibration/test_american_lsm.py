from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.calibration import american_numerics, lsm
from tests.synthetic.generators import option_surface_quotes, yield_curve_panel


def _prepared_quotes() -> pd.DataFrame:
    quotes = option_surface_quotes(
        dates=("2024-01-02", "2024-01-03"),
        tau_days=(21, 45, 75),
        k_values=(-0.20, -0.10, 0.0, 0.10, 0.20),
    )
    underlying = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "Dividends": [0.0, 0.5, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-20", "2024-02-20"]),
    )
    prepared, _, _ = american_numerics.prepare_american_quotes(
        quotes,
        underlying,
        yield_curve_panel(),
        max_rel_spread=0.50,
    )
    return prepared


def test_prepare_american_quotes_teaching_contracts_and_regime_grids() -> None:
    quotes = option_surface_quotes(
        dates=("2024-01-02", "2024-01-03"),
        tau_days=(21, 45, 75),
        k_values=(-0.20, -0.10, 0.0, 0.10, 0.20),
    )
    underlying = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0], "Dividends": [0.0, 0.5, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-20", "2024-02-20"]),
    )
    prepared, audit, dividends = american_numerics.prepare_american_quotes(
        quotes,
        underlying,
        yield_curve_panel(),
        max_rel_spread=0.50,
    )
    teaching = american_numerics.select_teaching_contracts(prepared)
    pde_grid = american_numerics.pde_regime_grid(prepared.head(20))
    lsm_grid = lsm.lsm_regime_grid(prepared.head(10))

    assert not prepared.empty
    assert audit.iloc[-1]["step"] == "final rows"
    assert dividends.loc[0, "dividend"] == pytest.approx(0.5)
    assert {"sigma_used", "dividend_yield", "time_value", "pv_dividends"}.issubset(prepared.columns)
    assert {"atm_put_30_60", "atm_call_30_60"}.issubset(set(teaching["contract_role"]))
    assert pde_grid["coverage_pct"].between(0.0, 1.0).all()
    assert lsm_grid["coverage_rows"].ge(1).all()


def test_american_scan_overlay_disagreement_and_speed_summaries() -> None:
    prepared = _prepared_quotes()
    scan = prepared.head(6).copy()
    scan["american_premium"] = [0.10, 0.20, 0.05, 0.06, 0.01, 0.02]
    scan["abs_pricing_error"] = [0.01, 0.02, 0.03, 0.01, 0.02, 0.04]
    scan["tree_contracts_per_sec"] = 1000.0
    scan["model_disagreement"] = [0.01, 0.02, 0.01, 0.02, 0.03, 0.01]
    comparison = american_numerics.method_comparison(scan[["american_premium", "model_disagreement"]].rename(columns={"american_premium": "tree_price"}))
    disagreement = american_numerics.method_disagreement_table(
        pd.DataFrame({"tree_price": [1.0, 2.0], "pde_price": [1.1, 1.8], "lsm_price": [0.9, 2.2]})
    )
    summary = american_numerics.american_scan_summary(scan)
    overlay = american_numerics.overlay_candidates(scan)
    method_summary = american_numerics.method_summary(comparison)
    speed = american_numerics.speed_table({"identity": lambda n: list(range(n))}, sizes=[2, 3], repeats=1)

    assert comparison["tree_price"].dtype.kind == "f"
    assert disagreement.loc[0, "model_disagreement"] == pytest.approx(0.20)
    assert summary.loc[0, "rows"] == 6
    assert {"assignment_risk", "roll_urgency", "candidate_score"}.issubset(overlay.columns)
    assert overlay["assignment_risk"].between(0.0, 1.0).all()
    assert method_summary.equals(comparison)
    assert speed["items_per_sec"].gt(0.0).all()


def test_full_chain_tree_scan_prices_small_synthetic_chain() -> None:
    prepared = _prepared_quotes().head(4)
    scan = american_numerics.full_chain_tree_scan(prepared, steps=15, engine="numpy", tree_type="crr")

    assert len(scan) == len(prepared)
    assert {"american_tree_price", "european_tree_price", "american_premium", "pricing_error"}.issubset(scan.columns)
    assert scan["american_tree_price"].ge(scan["european_tree_price"] - 1e-10).all()


def test_lsm_train_value_crossfit_boundaries_and_policy_gap() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0, 88.0],
            [100.0, 105.0, 98.0, 92.0],
            [100.0, 99.0, 101.0, 96.0],
            [100.0, 92.0, 94.0, 91.0],
        ]
    )
    basis = lsm.basis_matrix(np.array([0.0, 1.0]), degree=3)
    train = lsm.lsm_train(paths, 100.0, 0.03, 0.25, "put", degree=2, engine="numpy")
    value = lsm.lsm_value(paths, 100.0, 0.03, 0.25, "put", train["coefficients"], engine="numpy")
    crossfit = lsm.lsm_crossfit(
        100.0,
        100.0,
        0.03,
        0.0,
        0.20,
        0.25,
        "put",
        steps=5,
        paths=200,
        degree=2,
        seed=1,
        engine="numpy",
    )
    boundary = lsm.exercise_boundary_from_policy(train["coefficients"], 100.0, "put")

    assert np.allclose(basis, [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    assert train["price"] > 0.0
    assert value["price"] == pytest.approx(train["price"])
    assert crossfit["engine_used"] == "numpy"
    assert crossfit["coefficients"].shape == (6, 3)
    assert lsm.policy_gap(3.0, 3.5) == pytest.approx(0.5)
    assert lsm.lsm_boundary(train["coefficients"], 100.0, "put").equals(boundary)
    assert {"step", "boundary"}.issubset(boundary.columns)
