import numpy as np
import pandas as pd
import pytest

from quantfinlab.macro.dfm import dfm_news, filter_dfm, fit_dfm


def test_quarterly_observation_and_news_reconcile():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(14)
    dates = pd.date_range("2000-01-01", periods=144, freq="MS")
    f = np.zeros(len(dates))
    for i in range(1, len(f)):
        f[i] = .7 * f[i - 1] + rng.normal()
    monthly = pd.DataFrame({"production": f + rng.normal(scale=.2, size=len(f)),
                             "hours": .8 * f + rng.normal(scale=.3, size=len(f)),
                             "sales": 1.2 * f + rng.normal(scale=.3, size=len(f))}, index=dates)
    quarterly = monthly["production"].groupby(dates.to_period("Q")).mean().to_frame("gdp")
    factors = {name: ["activity"] for name in [*monthly.columns, "gdp"]}
    fit = fit_dfm(monthly.iloc[:120], quarterly.iloc[:40], factors=factors,
                   orders={"activity": 1}, anchors={"activity": "production"}, max_iter=100, tolerance=1e-3)
    before = monthly.copy()
    before.iloc[-3:] = np.nan
    q = quarterly.copy()
    q.iloc[-1] = np.nan
    after = before.copy()
    after.loc[dates[-3], "production"] = monthly.loc[dates[-3], "production"]
    first = filter_dfm(fit, before, q)
    second = filter_dfm(fit, after, q)
    result = dfm_news(first["result"], second["result"], variable="gdp", impact_date=dates[-1],
                       location=fit.quarterly_location["gdp"], scale=fit.quarterly_scale["gdp"])
    impact = result["impacts"].iloc[0]
    assert impact["estimate (new)"] - impact["estimate (prev)"] == pytest.approx(impact["total impact"], abs=1e-7)
    assert result["details"]["impact"].sum() == pytest.approx(impact["impact of news"], abs=1e-7)
