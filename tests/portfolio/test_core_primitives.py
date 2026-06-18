from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.portfolio import attribution, constraints, costs, covariance
from tests.synthetic.generators import return_panel


def test_constraints_normalize_and_validate_box_weights() -> None:
    assert constraints.constraints_feasible(4, w_max=0.30)
    assert not constraints.constraints_feasible(4, w_max=0.20)

    raw = pd.Series([2.0, -1.0, 1.0], index=["AAA", "BBB", "CCC"])
    weights = constraints.normalize_weights(raw, long_only=True, w_max=0.60)

    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights.max() <= 0.60 + 1e-12

    bounds = constraints.long_only_box_constraints(3, w_max=0.50)
    assert bounds == [(0.0, 0.5)] * 3
    assert np.allclose(constraints.coerce_prev_weights(None, 3), np.ones(3) / 3.0)


def test_turnover_costs_and_nav_adjustment() -> None:
    old = pd.Series({"AAA": 0.50, "BBB": 0.50})
    new = pd.Series({"AAA": 0.25, "BBB": 0.50, "CCC": 0.25})
    turnover = costs.portfolio_turnover(new, old)

    assert turnover == pytest.approx(0.25)
    assert costs.transaction_cost_from_turnover(turnover, bps=20.0) == pytest.approx(0.0005)

    nav = pd.Series([1.0, 1.02, 1.04], index=pd.bdate_range("2024-01-02", periods=3))
    adjusted = costs.apply_transaction_costs(nav, pd.Series({nav.index[1]: turnover}), bps=20.0)

    assert adjusted.iloc[0] == pytest.approx(nav.iloc[0])
    assert adjusted.iloc[1] < nav.iloc[1]
    assert adjusted.iloc[2] < nav.iloc[2]


def test_covariance_estimators_return_symmetric_psd_labeled_matrices() -> None:
    returns = return_panel(n=120, assets=("AAA", "BBB", "CCC", "DDD"))

    cov_map = covariance.estimate_covariance_map(
        returns,
        methods=("Sample", "LedoitWolf", "OAS", "EWMA"),
        annualization=252.0,
        return_df=True,
    )

    assert set(cov_map) == {"Sample", "LedoitWolf", "OAS", "EWMA"}
    for cov_df in cov_map.values():
        assert list(cov_df.index) == list(returns.columns)
        assert np.allclose(cov_df, cov_df.T)
        assert np.linalg.eigvalsh(cov_df.to_numpy()).min() >= -1e-8

    near_psd = covariance.make_psd(np.array([[1.0, 1.2], [1.2, 1.0]]))
    assert np.linalg.eigvalsh(near_psd).min() >= 0.0
    assert covariance.normalize_covariance_method("ledoit wolf") == "LedoitWolf"


def test_attribution_effective_n_concentration_and_risk_contribution() -> None:
    weights = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
    cov_ann = pd.DataFrame(
        [[0.040, 0.010, 0.005], [0.010, 0.030, 0.006], [0.005, 0.006, 0.025]],
        index=weights.index,
        columns=weights.index,
    )

    eff_n = attribution.effective_number_of_holdings(weights)
    hhi = attribution.concentration(weights)
    rc = attribution.risk_contribution(weights, cov_ann)
    port_vol = np.sqrt(float(weights.to_numpy() @ cov_ann.to_numpy() @ weights.to_numpy()))

    assert eff_n == pytest.approx(1.0 / hhi)
    assert attribution.max_weight(weights) == pytest.approx(0.50)
    assert rc.sum() == pytest.approx(port_vol)
    assert attribution.turnover_summary(pd.Series([0.10, 0.20]))["Total Turnover"] == pytest.approx(0.30)
