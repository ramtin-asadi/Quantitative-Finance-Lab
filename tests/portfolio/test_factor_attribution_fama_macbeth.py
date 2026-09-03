from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.portfolio import factors


def test_factor_attribution_recovers_ols_exposures_and_return_identity() -> None:
    rng = np.random.default_rng(21)
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    factor_returns = pd.DataFrame(
        rng.normal(0.004, 0.025, size=(len(dates), 3)),
        index=dates,
        columns=["Mkt-RF", "SMB", "MOM"],
    )
    factor_returns["RF"] = 0.001
    design = np.column_stack(
        [np.ones(len(dates)), factor_returns[["Mkt-RF", "SMB", "MOM"]]]
    )
    true_coefficients = np.array([0.002, 0.90, -0.25, 0.40])
    noise = rng.normal(0.0, 0.006, len(dates))
    portfolio_returns = pd.DataFrame(
        {
            "Strategy": design @ true_coefficients + noise + factor_returns["RF"],
        },
        index=dates,
    )

    result = factors.factor_attribution(
        portfolio_returns,
        factor_returns,
        factor_columns=["Mkt-RF", "SMB", "MOM"],
        hac_lags=3,
    )
    expected = np.linalg.lstsq(
        design,
        (portfolio_returns["Strategy"] - factor_returns["RF"]).to_numpy(),
        rcond=None,
    )[0]

    assert result.exposures.loc["Strategy", "annual_alpha"] == pytest.approx(
        expected[0] * 12.0
    )
    assert result.exposures.loc["Strategy", "Mkt-RF"] == pytest.approx(expected[1])
    assert result.exposures.loc["Strategy", "SMB"] == pytest.approx(expected[2])
    assert result.exposures.loc["Strategy", "MOM"] == pytest.approx(expected[3])
    assert np.isfinite(result.inference.loc["Strategy"]).all()
    assert result.attribution.loc[
        "Strategy", "fitted_excess_return"
    ] == pytest.approx(result.attribution.loc["Strategy", "realized_excess_return"])


def test_point_in_time_market_return_changes_membership_on_execution_date() -> None:
    dates = pd.date_range("2020-01-02", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "B": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "C": [0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        },
        index=dates,
    )
    issuers = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-04", "2020-01-04"]
            ),
            "ticker": ["A", "B", "B", "C"],
        }
    )
    date_map = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2020-01-01", "2020-01-04"]),
            "execution_date": pd.to_datetime(["2020-01-02", "2020-01-05"]),
        }
    )

    result = factors.point_in_time_market_return(returns, issuers, date_map)
    expected = pd.Series(
        [0.02, 0.03, 0.04, 0.07, 0.08, 0.09],
        index=dates,
        name="eligible_market_return",
    )
    pd.testing.assert_series_equal(result, expected)


def test_fama_macbeth_estimates_monthly_characteristic_premia_with_industries() -> None:
    rng = np.random.default_rng(22)
    dates = pd.date_range("2015-01-31", periods=84, freq="ME")
    assets = [f"S{asset:03d}" for asset in range(180)]
    characteristics = ["size", "momentum", "value", "beta", "volatility"]
    monthly_frames = []
    slopes = np.array([0.0030, 0.0040, -0.0020, 0.0015, -0.0010])
    industries = np.repeat([f"industry_{group}" for group in range(6)], 30)
    industry_effect = np.repeat(np.linspace(-0.004, 0.004, 6), 30)

    for date in dates:
        values = rng.normal(size=(len(assets), len(characteristics)))
        forward_return = (
            0.005
            + values @ slopes
            + industry_effect
            + rng.normal(0.0, 0.008, len(assets))
        )
        month = pd.DataFrame(values, columns=characteristics)
        month["decision_date"] = date
        month["ticker"] = assets
        month["industry"] = industries
        month["forward_return"] = forward_return
        monthly_frames.append(month)

    result = factors.fama_macbeth(
        pd.concat(monthly_frames, ignore_index=True),
        date_column="decision_date",
        return_column="forward_return",
        characteristics=characteristics,
        industry_column="industry",
        min_cross_section=120,
        min_industry_size=5,
        hac_lags=3,
    )

    assert result.summary.index.tolist() == characteristics
    assert result.summary["months"].eq(len(dates)).all()
    assert result.diagnostics["design_rank"].eq(
        result.diagnostics["design_columns"]
    ).all()
    assert result.summary.loc["size", "annualized_premium"] == pytest.approx(
        slopes[0] * 12.0,
        abs=0.01,
    )
    assert result.summary.loc["momentum", "annualized_premium"] > 0.0
    assert result.summary.loc["value", "annualized_premium"] < 0.0
    assert result.summary.loc["beta", "hac_t"] > 0.0
    assert result.summary.loc["volatility", "hac_t"] < 0.0
