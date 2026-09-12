import numpy as np
import pandas as pd
import pytest

from quantfinlab.credit.models import (
    default_contributions,
    default_design,
    fit_lightgbm,
    fit_logit,
    fit_spline,
    predict_default,
)


@pytest.mark.parametrize("method", ["logit", "spline", "lightgbm"])
def test_fitted_transforms_do_not_learn_from_future(method):
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["debt_assets", "cfo_debt", "cash_assets"])
    X.loc[::20, "cfo_debt"] = np.nan
    y = (rng.uniform(size=500) < .2).astype(int)
    if method == "logit":
        fit = fit_logit(X, y)
    elif method == "spline":
        fit = fit_spline(X, y)
    else:
        pytest.importorskip("lightgbm")
        fit = fit_lightgbm(X, y, trees=20, n_jobs=1)
    before = predict_default(fit, X.iloc[:10])
    altered = pd.concat([X.iloc[:10], X.iloc[[11]] * 1e8])
    after = predict_default(fit, altered)
    np.testing.assert_allclose(before, after[:10])
    assert np.all((after >= 0) & (after <= 1))
    assert len(default_design(fit, altered)) == 11
    if method == "lightgbm":
        contributions = default_contributions(fit, altered)
        raw = fit.model.booster_.predict(default_design(fit, altered), raw_score=True)
        np.testing.assert_allclose(contributions.sum(axis=1), raw, atol=1e-10)
