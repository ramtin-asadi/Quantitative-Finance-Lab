from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.portfolio import covariance, cvar, hrp, risk_parity, robust
from tests.synthetic.generators import return_panel


def _inputs() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    returns = return_panel(n=130, assets=("AAA", "BBB", "CCC", "DDD"))
    cov_ann = covariance.estimate_covariance(returns, method="LedoitWolf", return_df=True)
    mu = pd.Series([0.09, 0.06, 0.03, 0.04], index=returns.columns)
    return returns, mu, cov_ann


def _assert_weights(weights: pd.Series, *, cap: float = 0.70) -> None:
    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert weights.min() >= -1e-8
    assert weights.max() <= cap + 1e-7


def test_risk_parity_contributions_sum_to_portfolio_volatility() -> None:
    _, _, cov_ann = _inputs()
    weights = risk_parity.equal_risk_contribution_weights(cov_ann, tickers=cov_ann.index, w_max=0.70)
    table = risk_parity.risk_contribution_table(weights, cov_ann)

    _assert_weights(weights)
    assert table["percent_risk_contribution"].sum() == pytest.approx(1.0)
    assert (table["risk_contribution"] >= 0.0).all()


def test_hrp_and_nco_return_cluster_aware_weight_vectors() -> None:
    _, mu, cov_ann = _inputs()

    labels = hrp.cluster_labels(cov_ann, tickers=cov_ann.index, n_clusters=2)
    membership = hrp.cluster_membership_table(cov_ann, tickers=cov_ann.index, n_clusters=2)
    hrp_weights = hrp.hrp_weights(cov_ann, tickers=cov_ann.index, w_max=0.70)
    nco_weights = hrp.nco_mv_weights(cov_ann, mu, tickers=cov_ann.index, n_clusters=2, w_max=0.70)

    assert set(labels.index) == set(cov_ann.index)
    assert set(membership["asset"]) == set(cov_ann.index)
    _assert_weights(hrp_weights)
    _assert_weights(nco_weights)


def test_cvar_optimizers_bound_tail_loss_and_budget_path() -> None:
    pytest.importorskip("cvxpy")
    returns, mu, _ = _inputs()

    equal = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    min_weights = cvar.min_cvar_weights(returns, alpha=0.90, w_max=0.70)
    mean_weights = cvar.mean_cvar_weights(returns, mu, reference=equal, alpha=0.90, w_max=0.70)
    path = cvar.cvar_budget_path(returns, mu, budget_scales=(0.90, 1.00, 1.10), alpha=0.90, w_max=0.70)

    _assert_weights(min_weights)
    _assert_weights(mean_weights)
    assert cvar.portfolio_cvar_loss(returns, min_weights, alpha=0.90) <= cvar.portfolio_cvar_loss(returns, equal, alpha=0.90) + 1e-8
    assert list(path["budget_scale"]) == [0.90, 1.00, 1.10]
    assert path["cvar_loss"].notna().all()


def test_robust_allocators_and_radius_path_are_finite() -> None:
    pytest.importorskip("cvxpy")
    _, mu, cov_ann = _inputs()

    sqrt = robust.psd_sqrt(cov_ann)
    assert np.allclose(sqrt @ sqrt, covariance.make_psd(cov_ann), atol=1e-8)

    for func in (
        robust.box_robust_mv_weights,
        robust.ellipsoid_robust_mv_weights,
        robust.wasserstein_drmv_weights,
    ):
        weights = func(mu, cov_ann, n_mu_obs=120, radius=0.20, w_max=0.70)
        _assert_weights(weights)

    path = robust.robust_radius_path("box", mu, cov_ann, n_mu_obs=120, radii=(0.0, 0.2, 0.4), w_max=0.70)
    assert list(path["radius"]) == [0.0, 0.2, 0.4]
    assert path[["robust_return", "volatility", "effective_n"]].notna().all().all()

    w_path = robust.robust_radius_path("wasserstein", mu, cov_ann, n_mu_obs=120, radii=(0.0, 1.0), mv_lambda=1.5, w_max=0.70)
    assert list(w_path["radius"]) == [0.0, 1.0]
    assert w_path[["robust_volatility", "risk_penalty", "objective"]].notna().all().all()
    assert (w_path["robust_volatility"] >= w_path["volatility"] - 1e-10).all()


def test_robust_weight_frames_build_all_three_models_from_cache() -> None:
    pytest.importorskip("cvxpy")
    returns, mu, cov_ann = _inputs()
    rebalance_dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    cache = {
        dt: {
            "tickers": list(mu.index),
            "R_cov": returns,
            "R_mu": returns,
            "cov_ann_map": {"LedoitWolf": cov_ann},
            "mu_ann_map": {"LedoitWolf": {"Momentum": mu}},
        }
        for dt in rebalance_dates
    }

    frames = robust.robust_weight_frames(cache, rebalance_dates, w_max=0.70)

    assert set(frames) == {"Box Robust MV", "Ellipsoid Robust MV", "Wasserstein DRMV"}
    for frame in frames.values():
        assert list(frame.index) == list(rebalance_dates[:-1])
        assert set(frame.columns) == set(mu.index)
        for _, weights in frame.iterrows():
            _assert_weights(weights, cap=0.70)
