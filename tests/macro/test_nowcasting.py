import numpy as np
import pandas as pd
import pytest

from quantfinlab.fixed_income.overnight import compound_overnight, reference_window
from quantfinlab.macro.bridge import complete_months, quarterly_signal
from quantfinlab.macro.bvar import bvar_paths, fit_bvar
from quantfinlab.macro.midas import beta_weights, fit_beta_shape
from quantfinlab.macro.policy import policy_distribution_gap, policy_window_draws
from quantfinlab.macro.realtime import release_growth, vintage_asof
from quantfinlab.ml.combination import combine_forecasts, forecast_weights
from quantfinlab.ml.validation import available_labels


def test_vintage_and_growth_use_same_release_information():
    data = pd.DataFrame({"series_id": ["x"] * 4,
                         "observation_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]),
                         "available_at": pd.to_datetime(["2020-02-10", "2020-03-10", "2020-03-10", "2020-04-10"]),
                         "value": [100., 102., 103., 104.]})
    assert vintage_asof(data, "2020-02-15")["value"].tolist() == [100.]
    growth = release_growth(data, scale=100)
    assert growth["first"].iloc[1] == pytest.approx(100 * np.log(103 / 102))


def test_label_availability_is_not_origin_date():
    mask = available_labels(pd.to_datetime(["2020-01-01", "2020-01-02"]),
                            pd.to_datetime(["2021-01-01", "2020-01-03"]), "2020-06-01")
    assert mask.tolist() == [False, True]


def test_quarterly_release_delay_includes_quarter_length():
    data = pd.DataFrame({"series_id": ["gdp"] * 3,
                         "observation_date": pd.to_datetime(["2013-01-01", "2013-01-01", "2013-04-01"]),
                         "available_at": pd.to_datetime(["2013-05-31", "2013-08-30", "2013-08-30"]),
                         "value": [100., 101., 102.]})
    result = release_growth(data, frequency="Q", scale=400, max_delay=180)
    assert result["first"].iloc[-1] == pytest.approx(400 * np.log(102 / 101))
    result = release_growth(data, frequency="Q", method="percent", scale=100,
                             annualization=4, max_delay=180)
    assert result["first"].iloc[-1] == pytest.approx(100 * ((102 / 101) ** 4 - 1))


def test_quarter_completion_has_three_months_and_no_future_dependency():
    dates = pd.date_range("2017-01-01", "2020-01-01", freq="MS")
    values = pd.Series(100 * np.exp(.01 * np.arange(len(dates))), index=dates)
    result = quarterly_signal(values, "2020Q1")
    assert result["observed_months"] == 1
    assert result["completed"].notna().sum() == 3
    assert result["signal"] == pytest.approx(12)
    assert complete_months(values, pd.to_datetime(["2020-02-01"])).iloc[-1] > values.iloc[-1]


def test_midas_shape_recovery():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(500, 12))
    w = beta_weights(12, 1.4, 4.2)
    y = X @ w + rng.normal(0, .1, len(X))
    shape, _ = fit_beta_shape(X, y)
    assert np.corrcoef(w, beta_weights(12, *shape))[0, 1] > .98


def test_bvar_paths_and_seed():
    rng = np.random.default_rng(2)
    monthly = pd.DataFrame(rng.normal(size=(180, 3)),
                            index=pd.date_range("2000-01-01", periods=180, freq="MS"),
                            columns=["activity", "inflation", "policy"])
    fit = fit_bvar(monthly, persistent=["policy"])
    a = bvar_paths(fit, steps=6, draws=50, rng=np.random.default_rng(5))
    b = bvar_paths(fit, steps=6, draws=50, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 6, 3)


def test_compounding_weekend_and_window():
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    actual = compound_overnight([.05, .06], dates, "2024-01-09")
    expected = ((1 + .05 * 3 / 360) * (1 + .06 / 360) - 1) * 360 / 4
    assert actual == pytest.approx(expected)
    assert reference_window("2024-03-20")[1] == pd.Timestamp("2024-06-19")


def test_mixture_dispersion_and_unreleased_weight_exclusion():
    result = combine_forecasts([0, 2], [1, 1], [.5, .5])
    assert result["mean"] == 1
    assert result["sigma"] == pytest.approx(np.sqrt(2))
    data = pd.DataFrame({"model": ["a", "b"], "actual": [0, 0], "mean": [1, 0],
                         "release_date": pd.to_datetime(["2020-01-01", "2021-01-01"])})
    weights = forecast_weights(data, "2020-06-01", minimum=1)
    assert weights.to_dict() == {"a": 1.}


def test_policy_comparison_uses_discrete_mass_and_exact_fixings():
    result = policy_distribution_gap(np.array([.02, .02, .05, .05]), [.02, .05], [.5, .5])
    assert result["wasserstein"] == pytest.approx(0)
    dates = pd.to_datetime(["2024-01-05", "2024-01-08"])
    fixings = pd.Series([.05, .99], index=dates)
    paths = np.array([[.04], [.06]])
    draws = policy_window_draws(paths, pd.to_datetime(["2024-01-01"]), fixings,
                                 as_of="2024-01-06", start="2024-01-05", end="2024-01-09",
                                 calendar=dates, basis=.001, day_count=365)
    expected = compound_overnight([[.05, .041], [.05, .061]], dates, "2024-01-09", day_count=365)
    np.testing.assert_allclose(draws, expected)
