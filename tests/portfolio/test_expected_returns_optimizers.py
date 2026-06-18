from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.portfolio import covariance, expected_returns, optimizers
from tests.synthetic.generators import return_panel


def _assert_feasible_weights(weights: np.ndarray | pd.Series, *, upper: float) -> None:
    w = np.asarray(weights, dtype=float)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-7)
    assert w.min() >= -1e-8
    assert w.max() <= upper + 1e-7


def test_expected_return_models_emit_labeled_capped_mu_vectors() -> None:
    returns = return_panel(n=180, assets=("AAA", "BBB", "CCC", "DDD"))
    cov_ann = covariance.estimate_covariance(returns, method="LedoitWolf", return_df=True)

    raw_momentum = expected_returns.momentum_score_from_returns(returns, mode="6-1")
    z = expected_returns.winsorize_and_zscore(raw_momentum)
    scaled = expected_returns.scale_mu_to_target_sharpe(z, cov_ann, mu_cap_ann=0.20)

    assert raw_momentum.shape == (4,)
    assert np.nanmax(np.abs(scaled)) <= 0.20 + 1e-12

    for model in ("Momentum", "BayesStein", "BayesSteinMomentum"):
        mu, info = expected_returns.build_mu_excess_ann(
            returns,
            cov_ann=cov_ann,
            mu_model=model,
            return_info=True,
            return_series=True,
            mu_cap_ann=0.25,
        )
        assert list(mu.index) == list(returns.columns)
        assert np.isfinite(mu).all()
        assert info["mu_model"] == model
        assert float(mu.abs().max()) <= 0.25 + 1e-12


def test_mu_diagnostics_summarizes_cache_states() -> None:
    returns = return_panel(n=170, assets=("AAA", "BBB", "CCC"))
    cov_ann = covariance.estimate_covariance(returns, method="Sample", return_df=True)
    cache = {
        pd.Timestamp("2024-05-31"): {"R_mu": returns.iloc[:120], "cov_ann_map": {"Sample": cov_ann}},
        pd.Timestamp("2024-07-31"): {"R_mu": returns.iloc[20:], "cov_ann_map": {"Sample": cov_ann}},
    }

    diagnostics = expected_returns.mu_diagnostics(cache, cov_key="Sample")

    assert set(diagnostics["mu_model"]) == {"Momentum", "BayesStein", "BayesSteinMomentum"}
    assert (diagnostics["invalid_rebalances"] == 0).all()
    assert diagnostics["avg_max_abs_mu"].notna().all()


def test_optimizers_produce_feasible_long_only_portfolios() -> None:
    pytest.importorskip("cvxpy")

    labels = pd.Index(["AAA", "BBB", "CCC"])
    mu = pd.Series([0.10, 0.07, 0.04], index=labels)
    cov_ann = pd.DataFrame(
        [[0.050, 0.010, 0.005], [0.010, 0.035, 0.006], [0.005, 0.006, 0.025]],
        index=labels,
        columns=labels,
    )
    previous = np.array([1.0 / 3.0] * 3)

    equal = optimizers.equal_weight(labels, w_max=0.70, as_series=True)
    minvar = optimizers.minimum_variance(cov_ann=cov_ann, w_prev=previous, w_max=0.70)
    mv = optimizers.mean_variance(mu_excess_ann=mu, cov_ann=cov_ann, w_prev=previous, w_max=0.70)
    ridge = optimizers.ridge_mean_variance(mu_excess_ann=mu, cov_ann=cov_ann, w_prev=previous, w_max=0.70)
    sharpe = optimizers.max_sharpe_slsqp(mu_excess_ann=mu, cov_ann=cov_ann, w_prev=previous, w_max=0.70)
    frontier = optimizers.max_sharpe_frontier_grid(
        mu_excess_ann=mu,
        cov_ann=cov_ann,
        w_prev=previous,
        w_max=0.70,
        grid_n=6,
    )

    for weights in (equal, minvar, mv, ridge, sharpe, frontier):
        assert weights is not None
        _assert_feasible_weights(weights, upper=0.70)


def test_max_sharpe_falls_back_to_min_variance_for_zero_mu() -> None:
    pytest.importorskip("cvxpy")

    cov_ann = np.diag([0.06, 0.03, 0.02])
    weights = optimizers.max_sharpe_slsqp(mu_excess_ann=np.zeros(3), cov_ann=cov_ann, w_max=0.80)

    assert weights is not None
    _assert_feasible_weights(weights, upper=0.80)
