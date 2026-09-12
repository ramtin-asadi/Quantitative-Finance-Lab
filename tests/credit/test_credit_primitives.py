import numpy as np
import pandas as pd
import pytest

from quantfinlab.credit.curves import bootstrap_hazards, default_probability, hazards_from_pd
from quantfinlab.credit.events import event_labels
from quantfinlab.credit.market import premium_validation
from quantfinlab.credit.portfolio import expected_loss, portfolio_losses
from quantfinlab.credit.pricing import cds_cs01, cds_spread, cds_value, fit_hazard_multiplier
from quantfinlab.credit.structural import merton_assets
from quantfinlab.credit.structured import tranche_loss
from quantfinlab.numerics.copulas import gaussian_copula, student_t_copula


def test_hazard_roundtrip_and_crossing_rejection():
    T = np.array([.25, .5, 1, 2])
    p = np.array([[.01, .02, .04, .08], [.02, .03, .09, .15]])
    np.testing.assert_allclose(default_probability(T, T, hazards_from_pd(p, T)), p)
    with pytest.raises(ValueError):
        hazards_from_pd([.1, .05], [1, 2])


def test_cds_repricing_and_fixed_coupon_cs01():
    T, lam = np.array([1, 3, 5, 7, 10]), np.array([.01, .03, .02, .04, .025])
    def discount(t):
        return np.exp(-.03 * t)
    s = np.array([cds_spread(lam, T, discount, t) for t in T])
    fit = bootstrap_hazards(s, T, discount)
    np.testing.assert_allclose(fit["hazards"], lam, atol=1e-9)
    assert np.max(np.abs(fit["error_bp"])) < 1e-6
    assert abs(cds_value(s[2], lam, T, discount, 5)) < 1e-12
    assert cds_cs01(s, T, discount, 5, coupon=s[2], notional=1e6) > 0
    result = fit_hazard_multiplier(np.array([lam, 2 * lam]), T, discount, .03, [.3, .7])
    assert abs(result["error_bp"]) < 1e-6


def test_merton_equations_recover_asset_inputs():
    from scipy.stats import norm
    V, sigma, D, r = 180., .28, 100., .03
    d1 = (np.log(V / D) + r + .5 * sigma ** 2) / sigma
    E = V * norm.cdf(d1) - D * np.exp(-r) * norm.cdf(d1 - sigma)
    sigma_E = V * sigma * norm.cdf(d1) / E
    result = merton_assets([E], [sigma_E], [D], [r])
    np.testing.assert_allclose(result[["V", "sigma_V"]], [[V, sigma]], rtol=1e-6)
    assert result["residual"].iloc[0] < 1e-6


def test_censoring_and_exact_horizon_boundary():
    data = pd.DataFrame({"decision_date": pd.to_datetime(["2020-01-31"] * 3),
                         "strict_date": pd.to_datetime(["2020-02-29", None, "2020-03-01"]),
                         "last_observed": pd.to_datetime(["2020-02-29", "2020-02-15", "2020-03-31"])})
    labels = event_labels(data, horizons=[1], events={"event": "strict_date"})
    assert labels["event_1m"].tolist() == [True, False, False]
    assert labels["observed_event_1m"].tolist() == [True, False, True]


@pytest.mark.parametrize("draw", [gaussian_copula, student_t_copula])
def test_copula_marginals_and_expected_loss(draw):
    p, E = np.array([.1, .2, .05]), np.array([1., 2., 3.])
    U = draw(np.full((3, 3), .3) + .7 * np.eye(3), 80000, seed=42)
    np.testing.assert_allclose((U < p).mean(axis=0), p, atol=.004)
    losses = portfolio_losses(U, p, E)
    assert abs(losses.mean() - expected_loss(p, E).sum()) < .015


def test_exhaustive_tranches_reconcile_portfolio_loss():
    loss = np.linspace(0, 1, 101)
    bands = [(0, .03), (.03, .07), (.07, .3), (.3, 1)]
    recovered = sum(tranche_loss(loss, A, D) * (D - A) for A, D in bands)
    np.testing.assert_allclose(recovered, loss)


def test_premium_validation_preserves_regression_direction():
    data = pd.DataFrame({"estimate": [0., 1., 2., np.nan], "ebp": [1., 3., 5., 7.]})
    result = premium_validation(data, "estimate")
    assert result["months"] == 3
    assert result["intercept"] == pytest.approx(1)
    assert result["slope"] == pytest.approx(2)
    assert result["RMSE"] == pytest.approx(np.sqrt(14 / 3))
