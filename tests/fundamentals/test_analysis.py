from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fundamentals import analysis


def test_safe_ratio_uses_the_notebook_denominator_policy() -> None:
    numerator = pd.Series([4.0, 4.0, 4.0, np.nan])
    denominator = pd.Series([2.0, 0.0, -2.0, 2.0])

    result = analysis.safe_ratio(numerator, denominator)
    signed = analysis.safe_ratio(
        numerator,
        denominator,
        positive_denominator=False,
    )

    assert result.iloc[0] == pytest.approx(2.0)
    assert result.iloc[1:].isna().all()
    assert signed.iloc[2] == pytest.approx(-2.0)
    assert np.isfinite(result.dropna()).all()


def test_total_debt_requires_complete_components_when_reported_total_is_missing() -> None:
    result = analysis.total_debt(
        reported_total_debt=pd.Series([100.0, np.nan, np.nan, np.nan]),
        long_term_debt=pd.Series([80.0, 80.0, 80.0, 80.0]),
        current_debt=pd.Series([20.0, np.nan, np.nan, np.nan]),
        short_term_borrowings=pd.Series([np.nan, 20.0, np.nan, 0.0]),
    )

    expected = pd.Series([100.0, 100.0, np.nan, 80.0])
    pd.testing.assert_series_equal(result, expected)


def test_tangible_equity_is_strict_unless_approximation_is_requested() -> None:
    equity = pd.Series([100.0, 100.0, np.nan])
    goodwill = pd.Series([10.0, np.nan, 10.0])
    intangibles = pd.Series([5.0, 5.0, 5.0])

    strict = analysis.tangible_equity(equity, goodwill, intangibles)
    approximate = analysis.tangible_equity(
        equity,
        goodwill,
        intangibles,
        approximate=True,
    )

    pd.testing.assert_series_equal(strict, pd.Series([85.0, np.nan, np.nan]))
    pd.testing.assert_series_equal(
        approximate,
        pd.Series([85.0, 95.0, np.nan]),
    )


def test_net_payout_does_not_turn_unknown_components_into_zero() -> None:
    result = analysis.net_payout_amount(
        dividends=pd.Series([5.0, 0.0, np.nan]),
        repurchases=pd.Series([2.0, 0.0, 2.0]),
        share_issuance=pd.Series([1.0, 0.0, 1.0]),
    )

    pd.testing.assert_series_equal(result, pd.Series([6.0, 0.0, np.nan]))
    payout_yield = analysis.shareholder_yield(
        result,
        pd.Series([100.0, 100.0, 100.0]),
    )
    pd.testing.assert_series_equal(
        payout_yield,
        pd.Series([0.06, 0.0, np.nan]),
    )


def test_eps_prefers_reported_then_weighted_average_then_optional_approximation() -> None:
    reported = pd.Series([2.0, np.nan, np.nan, np.nan])
    income = pd.Series([100.0, 100.0, 100.0, 100.0])
    weighted_shares = pd.Series([50.0, 40.0, np.nan, np.nan])
    point_in_time_shares = pd.Series([60.0, 50.0, 25.0, np.nan])

    strict = analysis.earnings_per_share(
        reported,
        income,
        weighted_average_diluted_shares=weighted_shares,
    )
    with_approximation = analysis.earnings_per_share(
        reported,
        income,
        weighted_average_diluted_shares=weighted_shares,
        shares_outstanding=point_in_time_shares,
    )

    pd.testing.assert_series_equal(
        strict,
        pd.Series([2.0, 2.5, np.nan, np.nan]),
    )
    pd.testing.assert_series_equal(
        with_approximation,
        pd.Series([2.0, 2.5, 4.0, np.nan]),
    )


def test_profitability_cash_quality_and_roic_formulas() -> None:
    revenue = pd.Series([100.0])
    gross_profit = pd.Series([40.0])
    operating_income = pd.Series([20.0])
    net_income = pd.Series([12.0])
    cfo = pd.Series([15.0])
    assets = pd.Series([120.0])
    equity = pd.Series([60.0])
    tax_rate = analysis.effective_tax_rate(
        pd.Series([5.0]),
        pd.Series([20.0]),
    )
    nopat = analysis.net_operating_profit_after_tax(
        operating_income,
        tax_rate,
    )
    capital = analysis.invested_capital(
        pd.Series([50.0]),
        equity,
        pd.Series([10.0]),
    )

    assert analysis.gross_margin(gross_profit, revenue).iloc[0] == pytest.approx(0.4)
    assert analysis.operating_margin(operating_income, revenue).iloc[0] == pytest.approx(0.2)
    assert analysis.return_on_assets(net_income, assets).iloc[0] == pytest.approx(0.1)
    assert analysis.return_on_equity(net_income, equity).iloc[0] == pytest.approx(0.2)
    assert nopat.iloc[0] == pytest.approx(15.0)
    assert analysis.return_on_invested_capital(nopat, capital).iloc[0] == pytest.approx(0.15)
    assert analysis.total_accruals(net_income, cfo, assets).iloc[0] == pytest.approx(-0.025)


def test_growth_efficiency_capital_allocation_and_valuation_formulas() -> None:
    groups = pd.Series(["one", "one", "one"])
    revenue = pd.Series([100.0, 110.0, 121.0])
    growth = analysis.annual_growth(revenue, groups, periods=1)
    receivable_days = analysis.days_sales_outstanding(
        pd.Series([10.0]),
        pd.Series([100.0]),
    )
    inventory_days = analysis.inventory_days(
        pd.Series([20.0]),
        pd.Series([80.0]),
    )
    payable_days = analysis.payable_days(
        pd.Series([8.0]),
        pd.Series([80.0]),
    )

    assert growth.iloc[1] == pytest.approx(0.10)
    assert growth.iloc[2] == pytest.approx(0.10)
    assert receivable_days.iloc[0] == pytest.approx(36.5)
    assert inventory_days.iloc[0] == pytest.approx(91.25)
    assert payable_days.iloc[0] == pytest.approx(36.5)
    cycle = analysis.cash_conversion_cycle(
        receivable_days,
        inventory_days,
        payable_days,
    )
    assert cycle.iloc[0] == pytest.approx(91.25)
    reinvestment = analysis.reinvestment_rate(
        pd.Series([20.0]),
        pd.Series([8.0]),
        pd.Series([3.0]),
        pd.Series([15.0]),
    )
    assert reinvestment.iloc[0] == pytest.approx(1.0)
    assert analysis.price_to_earnings(
        pd.Series([200.0]),
        pd.Series([20.0]),
    ).iloc[0] == pytest.approx(10.0)
    assert analysis.enterprise_value_to_sales(
        pd.Series([240.0]),
        pd.Series([120.0]),
    ).iloc[0] == pytest.approx(2.0)


def test_financial_company_stability_metrics_preserve_group_history() -> None:
    groups = pd.Series(["bank", "bank", "bank", "bank"])
    earnings = pd.Series([10.0, 12.0, -2.0, 14.0])
    variability = analysis.relative_variability(
        earnings,
        groups,
        window=3,
        min_periods=2,
    )
    positive = analysis.positive_earnings_frequency(
        earnings,
        groups,
        window=3,
        min_periods=2,
    )

    assert analysis.equity_to_assets(
        pd.Series([10.0]),
        pd.Series([100.0]),
    ).iloc[0] == pytest.approx(0.1)
    assert analysis.assets_to_equity(
        pd.Series([100.0]),
        pd.Series([10.0]),
    ).iloc[0] == pytest.approx(10.0)
    assert variability.iloc[-1] > 0
    assert positive.iloc[-1] == pytest.approx(2.0 / 3.0)


def test_piotroski_altman_and_beneish_match_notebook_formulas() -> None:
    score = analysis.piotroski_score(
        roa=pd.Series([0.10, 0.10]),
        cfo=pd.Series([12.0, 12.0]),
        net_income=pd.Series([10.0, 10.0]),
        prior_roa=pd.Series([0.08, 0.08]),
        debt_assets=pd.Series([0.30, 0.30]),
        prior_debt_assets=pd.Series([0.40, 0.40]),
        current_ratio_value=pd.Series([1.5, 1.5]),
        prior_current_ratio=pd.Series([1.2, 1.2]),
        share_dilution=pd.Series([0.0, 0.0]),
        gross_margin_value=pd.Series([0.40, 0.40]),
        prior_gross_margin=pd.Series([0.35, 0.35]),
        asset_turnover_value=pd.Series([1.2, 1.2]),
        prior_asset_turnover=pd.Series([1.0, np.nan]),
    )
    altman = analysis.altman_z_score(
        working_capital_value=pd.Series([20.0]),
        retained_earnings=pd.Series([10.0]),
        operating_income=pd.Series([15.0]),
        market_cap=pd.Series([60.0]),
        total_liabilities=pd.Series([50.0]),
        revenue=pd.Series([100.0]),
        total_assets=pd.Series([100.0]),
    )
    beneish = analysis.beneish_m_score(
        revenue=pd.Series([100.0]),
        prior_revenue=pd.Series([100.0]),
        receivables=pd.Series([10.0]),
        prior_receivables=pd.Series([10.0]),
        gross_margin_value=pd.Series([0.40]),
        prior_gross_margin=pd.Series([0.40]),
        current_assets=pd.Series([40.0]),
        prior_current_assets=pd.Series([40.0]),
        ppe=pd.Series([30.0]),
        prior_ppe=pd.Series([30.0]),
        total_assets=pd.Series([100.0]),
        prior_total_assets=pd.Series([100.0]),
        depreciation=pd.Series([10.0]),
        prior_depreciation=pd.Series([10.0]),
        sga_expense=pd.Series([15.0]),
        prior_sga_expense=pd.Series([15.0]),
        debt_assets=pd.Series([0.30]),
        prior_debt_assets=pd.Series([0.30]),
        net_income=pd.Series([10.0]),
        cfo=pd.Series([10.0]),
    )

    assert score.iloc[0] == pytest.approx(9.0)
    assert np.isnan(score.iloc[1])
    assert altman.iloc[0] == pytest.approx(2.595)
    assert beneish.iloc[0] == pytest.approx(-2.48)


def test_red_flag_penalties_keep_unknown_flags_missing_and_weight_active_flags() -> None:
    metrics = pd.DataFrame(
        {
            "score_family": ["corporate"],
            "net_income": [10.0],
            "cfo": [-1.0],
            "free_cash_flow": [-2.0],
            "positive_cfo_frequency": [0.10],
            "positive_fcf_frequency": [0.10],
            "total_accruals": [0.20],
            "interest_coverage": [0.50],
            "debt_assets": [0.50],
            "debt_assets_prior": [0.40],
            "cfo_growth": [-0.10],
            "common_equity": [-1.0],
            "share_count_dilution": [0.20],
            "dividends": [5.0],
            "altman_z": [1.0],
            "beneish_m": [-1.0],
            "operating_margin_change": [-0.06],
            "roa_change": [0.0],
        }
    )

    warnings = analysis.red_flag_penalties(metrics)

    assert bool(warnings.loc[0, "warning_earnings_without_cash"])
    assert bool(warnings.loc[0, "warning_altman_distress"])
    assert pd.isna(warnings.loc[0, "warning_fin_negative_earnings"])
    assert warnings.loc[0, "warning_penalty"] == 48
    assert warnings.loc[0, "severe_warning_count"] == 8
