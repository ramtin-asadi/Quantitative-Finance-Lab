"""Atomic fundamental-analysis formulas used by the research notebooks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _numeric(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").astype(float)
    return pd.to_numeric(pd.Series(values), errors="coerce").astype(float)


def _optional_numeric(values, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return _numeric(values).reindex(index)


def _eligible(reference: pd.Series, eligible) -> pd.Series:
    if eligible is None:
        return pd.Series(True, index=reference.index, dtype=bool)
    return pd.Series(eligible, index=reference.index).fillna(False).astype(bool)


def safe_ratio(
    numerator,
    denominator,
    *,
    positive_denominator: bool = True,
) -> pd.Series:
    """Divide two series while excluding missing, zero, and invalid denominators."""

    numerator = _numeric(numerator)
    denominator = _numeric(denominator)
    numerator, denominator = numerator.align(denominator)
    if positive_denominator:
        valid = denominator.gt(0)
    else:
        valid = denominator.abs().gt(1e-12)
    return (numerator / denominator).where(valid).replace([np.inf, -np.inf], np.nan)


def free_cash_flow(cfo, capex) -> pd.Series:
    return _numeric(cfo) - _numeric(capex)


def total_debt(
    reported_total_debt,
    long_term_debt,
    current_debt,
    short_term_borrowings,
) -> pd.Series:
    """Use reported total debt, then complete long- and short-term components."""

    reported = _numeric(reported_total_debt)
    long_term = _numeric(long_term_debt)
    current = _numeric(current_debt).combine_first(_numeric(short_term_borrowings))
    components = (long_term + current).where(long_term.notna() & current.notna())
    return reported.combine_first(components)


def tangible_equity(
    common_equity,
    goodwill,
    intangibles,
    *,
    approximate: bool = False,
) -> pd.Series:
    """Subtract goodwill and intangibles without treating missing values as zero."""

    equity = _numeric(common_equity)
    goodwill = _numeric(goodwill)
    intangibles = _numeric(intangibles)
    if approximate:
        return (
            equity - goodwill.fillna(0.0) - intangibles.fillna(0.0)
        ).where(equity.notna())
    known = equity.notna() & goodwill.notna() & intangibles.notna()
    return (equity - goodwill - intangibles).where(known)


def enterprise_value(market_cap, debt, cash) -> pd.Series:
    return _numeric(market_cap) + _numeric(debt) - _numeric(cash)


def net_debt(debt, cash) -> pd.Series:
    return _numeric(debt) - _numeric(cash)


def working_capital(current_assets, current_liabilities) -> pd.Series:
    return _numeric(current_assets) - _numeric(current_liabilities)


def net_payout_amount(dividends, repurchases, share_issuance) -> pd.Series:
    """Combine payout components only when every component is observed."""

    dividends = _numeric(dividends)
    repurchases = _numeric(repurchases)
    issuance = _numeric(share_issuance)
    known = dividends.notna() & repurchases.notna() & issuance.notna()
    return (dividends + repurchases - issuance).where(known)


def average_balance(current, prior) -> pd.Series:
    return (_numeric(current) + _numeric(prior)) / 2.0


def earnings_per_share(
    reported_diluted_eps,
    net_income,
    weighted_average_diluted_shares=None,
    shares_outstanding=None,
) -> pd.Series:
    """Apply the diluted-EPS hierarchy and keep point-in-time shares optional."""

    reported = _numeric(reported_diluted_eps)
    income = _numeric(net_income).reindex(reported.index)
    weighted_shares = _optional_numeric(
        weighted_average_diluted_shares,
        reported.index,
    )
    point_in_time_shares = _optional_numeric(shares_outstanding, reported.index)
    weighted_eps = safe_ratio(income, weighted_shares)
    approximate_eps = safe_ratio(income, point_in_time_shares)
    return reported.combine_first(weighted_eps).combine_first(approximate_eps)


def revenue_per_share(revenue, shares) -> pd.Series:
    return safe_ratio(revenue, shares)


def fcf_per_share(fcf, shares) -> pd.Series:
    return safe_ratio(fcf, shares)


def book_value_per_share(common_equity, shares) -> pd.Series:
    return safe_ratio(common_equity, shares)


def tangible_book_value_per_share(tangible_equity_value, shares) -> pd.Series:
    return safe_ratio(tangible_equity_value, shares)


def shareholder_yield(net_payout, market_cap) -> pd.Series:
    return safe_ratio(net_payout, market_cap)


def gross_margin(gross_profit, revenue) -> pd.Series:
    return safe_ratio(gross_profit, revenue)


def operating_margin(operating_income, revenue) -> pd.Series:
    return safe_ratio(operating_income, revenue)


def pretax_margin(pretax_income, revenue) -> pd.Series:
    return safe_ratio(pretax_income, revenue)


def net_margin(net_income, revenue) -> pd.Series:
    return safe_ratio(net_income, revenue)


def fcf_margin(fcf, revenue) -> pd.Series:
    return safe_ratio(fcf, revenue)


def gross_profitability_assets(gross_profit, average_assets) -> pd.Series:
    return safe_ratio(gross_profit, average_assets)


def return_on_assets(net_income, average_assets) -> pd.Series:
    return safe_ratio(net_income, average_assets)


def return_on_equity(net_income, average_equity) -> pd.Series:
    return safe_ratio(net_income, average_equity)


def return_on_tangible_equity(net_income, average_tangible_equity) -> pd.Series:
    return safe_ratio(net_income, average_tangible_equity)


def effective_tax_rate(
    tax_expense,
    pretax_income,
    *,
    lower: float = 0.0,
    upper: float = 0.50,
) -> pd.Series:
    return safe_ratio(tax_expense, pretax_income).clip(lower, upper)


def net_operating_profit_after_tax(
    operating_income,
    tax_rate,
    *,
    fallback_rate: float = 0.21,
) -> pd.Series:
    rate = _numeric(tax_rate).fillna(float(fallback_rate))
    return _numeric(operating_income) * (1.0 - rate)


def invested_capital(average_debt, average_equity, cash) -> pd.Series:
    return _numeric(average_debt) + _numeric(average_equity) - _numeric(cash)


def return_on_invested_capital(nopat, capital) -> pd.Series:
    return safe_ratio(nopat, capital)


def asset_turnover(revenue, average_assets) -> pd.Series:
    return safe_ratio(revenue, average_assets)


def equity_multiplier(average_assets, average_equity) -> pd.Series:
    return safe_ratio(average_assets, average_equity)


def dupont_return_on_equity(
    margin,
    turnover,
    multiplier,
) -> pd.Series:
    return _numeric(margin) * _numeric(turnover) * _numeric(multiplier)


def dupont_gap(roe, dupont_roe) -> pd.Series:
    return _numeric(roe) - _numeric(dupont_roe)


def cfo_to_assets(cfo, average_assets) -> pd.Series:
    return safe_ratio(cfo, average_assets)


def fcf_to_assets(fcf, average_assets) -> pd.Series:
    return safe_ratio(fcf, average_assets)


def cfo_to_net_income(cfo, net_income) -> pd.Series:
    return safe_ratio(cfo, net_income)


def fcf_conversion(fcf, net_income) -> pd.Series:
    return safe_ratio(fcf, net_income)


def total_accruals(net_income, cfo, average_assets) -> pd.Series:
    return safe_ratio(_numeric(net_income) - _numeric(cfo), average_assets)


def working_capital_accruals(working_capital_change, average_assets) -> pd.Series:
    return safe_ratio(working_capital_change, average_assets)


def cash_earnings_gap(net_income, cfo) -> pd.Series:
    return _numeric(net_income) - _numeric(cfo)


def positive_frequency(
    values,
    groups,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    values = _numeric(values)
    positive = values.gt(0).where(values.notna()).astype(float)
    return positive.groupby(groups, sort=False).transform(
        lambda series: series.rolling(window, min_periods=min_periods).mean()
    )


def positive_cfo_frequency(
    cfo,
    groups,
    *,
    window: int = 24,
    min_periods: int = 12,
) -> pd.Series:
    return positive_frequency(
        cfo,
        groups,
        window=window,
        min_periods=min_periods,
    )


def positive_fcf_frequency(
    fcf,
    groups,
    *,
    window: int = 24,
    min_periods: int = 12,
) -> pd.Series:
    return positive_frequency(
        fcf,
        groups,
        window=window,
        min_periods=min_periods,
    )


def annual_lag(values, groups, *, periods: int = 12) -> pd.Series:
    return _numeric(values).groupby(groups, sort=False).shift(periods)


def annual_growth(values, groups, *, periods: int = 12) -> pd.Series:
    values = _numeric(values)
    return safe_ratio(values, annual_lag(values, groups, periods=periods)) - 1.0


def annual_change(values, groups, *, periods: int = 12) -> pd.Series:
    values = _numeric(values)
    return values - annual_lag(values, groups, periods=periods)


def current_ratio(current_assets, current_liabilities) -> pd.Series:
    return safe_ratio(current_assets, current_liabilities)


def cash_ratio(cash, current_liabilities) -> pd.Series:
    return safe_ratio(cash, current_liabilities)


def debt_to_equity(debt, common_equity) -> pd.Series:
    return safe_ratio(debt, common_equity)


def debt_to_assets(debt, total_assets) -> pd.Series:
    return safe_ratio(debt, total_assets)


def net_debt_to_assets(net_debt_value, total_assets) -> pd.Series:
    return safe_ratio(net_debt_value, total_assets)


def liabilities_to_assets(total_liabilities, total_assets) -> pd.Series:
    return safe_ratio(total_liabilities, total_assets)


def interest_coverage(operating_income, interest_expense) -> pd.Series:
    return safe_ratio(operating_income, interest_expense)


def cfo_to_debt(cfo, debt) -> pd.Series:
    return safe_ratio(cfo, debt)


def fcf_to_debt(fcf, debt) -> pd.Series:
    return safe_ratio(fcf, debt)


def cash_to_assets(cash, total_assets) -> pd.Series:
    return safe_ratio(cash, total_assets)


def tangible_equity_to_assets(tangible_equity_value, total_assets) -> pd.Series:
    return safe_ratio(tangible_equity_value, total_assets)


def leverage_improvement(
    net_debt_assets,
    groups,
    *,
    periods: int = 12,
) -> pd.Series:
    return -annual_change(net_debt_assets, groups, periods=periods)


def receivables_turnover(revenue, average_receivables) -> pd.Series:
    return safe_ratio(revenue, average_receivables)


def days_sales_outstanding(average_receivables, revenue) -> pd.Series:
    return 365.0 * safe_ratio(average_receivables, revenue)


def inventory_turnover(cost_of_revenue, average_inventory) -> pd.Series:
    return safe_ratio(cost_of_revenue, average_inventory)


def inventory_days(average_inventory, cost_of_revenue) -> pd.Series:
    return 365.0 * safe_ratio(average_inventory, cost_of_revenue)


def payable_days(average_payables, cost_of_revenue) -> pd.Series:
    return 365.0 * safe_ratio(average_payables, cost_of_revenue)


def cash_conversion_cycle(
    days_receivable,
    days_inventory,
    days_payable,
) -> pd.Series:
    return (
        _numeric(days_receivable)
        + _numeric(days_inventory)
        - _numeric(days_payable)
    )


def working_capital_to_revenue(working_capital_value, revenue) -> pd.Series:
    return safe_ratio(working_capital_value, revenue)


def capex_to_revenue(capex, revenue) -> pd.Series:
    return safe_ratio(capex, revenue)


def capex_to_depreciation(capex, depreciation) -> pd.Series:
    return safe_ratio(capex, depreciation)


def research_to_revenue(research_expense, revenue) -> pd.Series:
    return safe_ratio(research_expense, revenue)


def reinvestment_rate(
    capex,
    depreciation,
    working_capital_change,
    nopat,
) -> pd.Series:
    reinvestment = (
        _numeric(capex)
        - _numeric(depreciation)
        + _numeric(working_capital_change)
    )
    return safe_ratio(reinvestment, nopat)


def dividend_payout_ratio(dividends, net_income) -> pd.Series:
    return safe_ratio(dividends, net_income)


def dividend_coverage_cfo(cfo, dividends) -> pd.Series:
    return safe_ratio(cfo, dividends)


def dividend_coverage_fcf(fcf, dividends) -> pd.Series:
    return safe_ratio(fcf, dividends)


def repurchase_yield(repurchases, market_cap) -> pd.Series:
    return safe_ratio(repurchases, market_cap)


def issuance_yield(share_issuance, market_cap) -> pd.Series:
    return safe_ratio(share_issuance, market_cap)


def dividend_yield(dividends, market_cap) -> pd.Series:
    return safe_ratio(dividends, market_cap)


def share_count_dilution(shares, groups, *, periods: int = 12) -> pd.Series:
    return annual_growth(shares, groups, periods=periods)


def growth_spread(per_share_growth, aggregate_growth) -> pd.Series:
    return _numeric(per_share_growth) - _numeric(aggregate_growth)


def reinvestment_quality(reinvestment) -> pd.Series:
    reinvestment = _numeric(reinvestment)
    return reinvestment.where(reinvestment.between(-1.0, 3.0))


def earnings_yield(net_income, market_cap) -> pd.Series:
    return safe_ratio(net_income, market_cap)


def price_to_earnings(market_cap, net_income) -> pd.Series:
    return safe_ratio(market_cap, net_income)


def fcf_yield(fcf, market_cap) -> pd.Series:
    return safe_ratio(fcf, market_cap)


def price_to_fcf(market_cap, fcf) -> pd.Series:
    return safe_ratio(market_cap, fcf)


def sales_yield(revenue, market_cap) -> pd.Series:
    return safe_ratio(revenue, market_cap)


def price_to_sales(market_cap, revenue) -> pd.Series:
    return safe_ratio(market_cap, revenue)


def book_to_market(common_equity, market_cap) -> pd.Series:
    return safe_ratio(common_equity, market_cap)


def price_to_book(market_cap, common_equity) -> pd.Series:
    return safe_ratio(market_cap, common_equity)


def tangible_book_to_market(tangible_equity_value, market_cap) -> pd.Series:
    return safe_ratio(tangible_equity_value, market_cap)


def price_to_tangible_book(market_cap, tangible_equity_value) -> pd.Series:
    return safe_ratio(market_cap, tangible_equity_value)


def ebit_to_enterprise_value(operating_income, enterprise_value_value) -> pd.Series:
    return safe_ratio(operating_income, enterprise_value_value)


def enterprise_value_to_ebit(enterprise_value_value, operating_income) -> pd.Series:
    return safe_ratio(enterprise_value_value, operating_income)


def enterprise_value_to_sales(enterprise_value_value, revenue) -> pd.Series:
    return safe_ratio(enterprise_value_value, revenue)


def sales_to_enterprise_value(revenue, enterprise_value_value) -> pd.Series:
    return safe_ratio(revenue, enterprise_value_value)


def enterprise_value_to_fcf(enterprise_value_value, fcf) -> pd.Series:
    return safe_ratio(enterprise_value_value, fcf)


def pretax_return_on_assets(pretax_income, average_assets) -> pd.Series:
    return safe_ratio(pretax_income, average_assets)


def equity_to_assets(common_equity, total_assets) -> pd.Series:
    return safe_ratio(common_equity, total_assets)


def assets_to_equity(total_assets, common_equity) -> pd.Series:
    return safe_ratio(total_assets, common_equity)


def operating_expense_ratio(operating_expenses, revenue) -> pd.Series:
    return safe_ratio(operating_expenses, revenue)


def revenue_to_assets(revenue, average_assets) -> pd.Series:
    return safe_ratio(revenue, average_assets)


def relative_variability(
    values,
    groups,
    *,
    window: int = 36,
    min_periods: int = 24,
) -> pd.Series:
    values = _numeric(values)
    rolling_std = values.groupby(groups, sort=False).transform(
        lambda series: series.rolling(window, min_periods=min_periods).std()
    )
    rolling_scale = values.abs().groupby(groups, sort=False).transform(
        lambda series: series.rolling(window, min_periods=min_periods).mean()
    )
    return safe_ratio(rolling_std, rolling_scale)


def positive_earnings_frequency(
    net_income,
    groups,
    *,
    window: int = 36,
    min_periods: int = 24,
) -> pd.Series:
    return positive_frequency(
        net_income,
        groups,
        window=window,
        min_periods=min_periods,
    )


def _binary_component(valid, condition) -> pd.Series:
    valid = pd.Series(valid).fillna(False).astype(bool)
    condition = pd.Series(condition, index=valid.index).fillna(False).astype(bool)
    result = pd.Series(np.nan, index=valid.index, dtype=float)
    result.loc[valid] = condition.loc[valid].astype(int)
    return result


def piotroski_score(
    *,
    roa,
    cfo,
    net_income,
    prior_roa,
    debt_assets,
    prior_debt_assets,
    current_ratio_value,
    prior_current_ratio,
    share_dilution,
    gross_margin_value,
    prior_gross_margin,
    asset_turnover_value,
    prior_asset_turnover,
    eligible=None,
) -> pd.Series:
    """Calculate the nine-component Piotroski F-score."""

    roa = _numeric(roa)
    cfo = _numeric(cfo).reindex(roa.index)
    net_income = _numeric(net_income).reindex(roa.index)
    prior_roa = _numeric(prior_roa).reindex(roa.index)
    debt_assets = _numeric(debt_assets).reindex(roa.index)
    prior_debt_assets = _numeric(prior_debt_assets).reindex(roa.index)
    current_ratio_value = _numeric(current_ratio_value).reindex(roa.index)
    prior_current_ratio = _numeric(prior_current_ratio).reindex(roa.index)
    share_dilution = _numeric(share_dilution).reindex(roa.index)
    gross_margin_value = _numeric(gross_margin_value).reindex(roa.index)
    prior_gross_margin = _numeric(prior_gross_margin).reindex(roa.index)
    asset_turnover_value = _numeric(asset_turnover_value).reindex(roa.index)
    prior_asset_turnover = _numeric(prior_asset_turnover).reindex(roa.index)
    allowed = _eligible(roa, eligible)

    components = pd.DataFrame(
        {
            "roa_positive": _binary_component(
                allowed & roa.notna(),
                roa.gt(0),
            ),
            "cfo_positive": _binary_component(
                allowed & cfo.notna(),
                cfo.gt(0),
            ),
            "roa_improved": _binary_component(
                allowed & roa.notna() & prior_roa.notna(),
                roa.gt(prior_roa),
            ),
            "accrual_quality": _binary_component(
                allowed & cfo.notna() & net_income.notna(),
                cfo.gt(net_income),
            ),
            "leverage_down": _binary_component(
                allowed & debt_assets.notna() & prior_debt_assets.notna(),
                debt_assets.lt(prior_debt_assets),
            ),
            "liquidity_up": _binary_component(
                allowed
                & current_ratio_value.notna()
                & prior_current_ratio.notna(),
                current_ratio_value.gt(prior_current_ratio),
            ),
            "no_dilution": _binary_component(
                allowed & share_dilution.notna(),
                share_dilution.le(0),
            ),
            "margin_up": _binary_component(
                allowed & gross_margin_value.notna() & prior_gross_margin.notna(),
                gross_margin_value.gt(prior_gross_margin),
            ),
            "turnover_up": _binary_component(
                allowed
                & asset_turnover_value.notna()
                & prior_asset_turnover.notna(),
                asset_turnover_value.gt(prior_asset_turnover),
            ),
        },
        index=roa.index,
    )
    return components.sum(axis=1, min_count=len(components.columns))


def altman_z_score(
    *,
    working_capital_value,
    retained_earnings,
    operating_income,
    market_cap,
    total_liabilities,
    revenue,
    total_assets,
    eligible=None,
) -> pd.Series:
    """Calculate the original five-factor Altman Z-score."""

    assets = _numeric(total_assets)
    parts = pd.DataFrame(
        {
            "working_capital_assets": safe_ratio(
                working_capital_value,
                assets,
            ),
            "retained_earnings_assets": safe_ratio(
                retained_earnings,
                assets,
            ),
            "ebit_assets": safe_ratio(operating_income, assets),
            "market_equity_liabilities": safe_ratio(
                market_cap,
                total_liabilities,
            ),
            "sales_assets": safe_ratio(revenue, assets),
        },
        index=assets.index,
    )
    valid = parts.notna().all(axis=1) & _eligible(assets, eligible)
    score = (
        1.2 * parts["working_capital_assets"]
        + 1.4 * parts["retained_earnings_assets"]
        + 3.3 * parts["ebit_assets"]
        + 0.6 * parts["market_equity_liabilities"]
        + parts["sales_assets"]
    )
    return score.where(valid)


def beneish_m_score(
    *,
    revenue,
    prior_revenue,
    receivables,
    prior_receivables,
    gross_margin_value,
    prior_gross_margin,
    current_assets,
    prior_current_assets,
    ppe,
    prior_ppe,
    total_assets,
    prior_total_assets,
    depreciation,
    prior_depreciation,
    sga_expense,
    prior_sga_expense,
    debt_assets,
    prior_debt_assets,
    net_income,
    cfo,
    eligible=None,
) -> pd.Series:
    """Calculate the eight-variable Beneish M-score."""

    revenue = _numeric(revenue)
    dsri = safe_ratio(
        safe_ratio(receivables, revenue),
        safe_ratio(prior_receivables, prior_revenue),
    )
    gmi = safe_ratio(prior_gross_margin, gross_margin_value)
    aqi = safe_ratio(
        1.0 - safe_ratio(_numeric(current_assets) + _numeric(ppe), total_assets),
        1.0
        - safe_ratio(
            _numeric(prior_current_assets) + _numeric(prior_ppe),
            prior_total_assets,
        ),
    )
    sgi = safe_ratio(revenue, prior_revenue)
    depi = safe_ratio(
        safe_ratio(
            prior_depreciation,
            _numeric(prior_depreciation) + _numeric(prior_ppe),
        ),
        safe_ratio(
            depreciation,
            _numeric(depreciation) + _numeric(ppe),
        ),
    )
    sgai = safe_ratio(
        safe_ratio(sga_expense, revenue),
        safe_ratio(prior_sga_expense, prior_revenue),
    )
    lvgi = safe_ratio(debt_assets, prior_debt_assets)
    tata = safe_ratio(_numeric(net_income) - _numeric(cfo), total_assets)
    parts = pd.DataFrame(
        {
            "dsri": dsri,
            "gmi": gmi,
            "aqi": aqi,
            "sgi": sgi,
            "depi": depi,
            "sgai": sgai,
            "lvgi": lvgi,
            "tata": tata,
        },
        index=revenue.index,
    )
    valid = parts.notna().all(axis=1) & _eligible(revenue, eligible)
    score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
    return score.where(valid)


warning_penalty_weights = {
    "warning_altman_distress": 6,
    "warning_beneish_warning": 5,
    "warning_earnings_without_cash": 4,
    "warning_persistent_negative_cfo": 4,
    "warning_persistent_negative_fcf": 3,
    "warning_extreme_accruals": 3,
    "warning_low_interest_coverage": 5,
    "warning_rising_debt_falling_cash": 4,
    "warning_negative_equity": 4,
    "warning_severe_dilution": 4,
    "warning_unsupported_payout": 3,
    "warning_profitability_deterioration": 3,
    "warning_fin_negative_earnings": 5,
    "warning_fin_roa_deterioration": 4,
    "warning_fin_roe_deterioration": 4,
    "warning_fin_capital_deterioration": 5,
    "warning_fin_nonpositive_tangible_equity": 6,
    "warning_fin_extreme_leverage": 4,
    "warning_fin_severe_dilution": 4,
    "warning_fin_falling_bvps": 4,
    "warning_fin_weak_payout_capital": 3,
    "warning_fin_earnings_instability": 3,
}


def _metric(metrics: pd.DataFrame, name: str) -> pd.Series:
    if name not in metrics:
        return pd.Series(np.nan, index=metrics.index, dtype=float)
    return _numeric(metrics[name]).reindex(metrics.index)


def _warning(valid, condition, index: pd.Index) -> pd.Series:
    valid = pd.Series(valid, index=index).fillna(False).astype(bool)
    condition = pd.Series(condition, index=index).fillna(False).astype(bool)
    result = pd.Series(pd.NA, index=index, dtype="boolean")
    result.loc[valid] = condition.loc[valid]
    return result


def red_flag_penalties(
    metrics: pd.DataFrame,
    *,
    family_column: str = "score_family",
    date_column: str = "decision_date",
    penalty_weights: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Evaluate the notebook's corporate and financial-company warning rules."""

    family = metrics[family_column].astype("string")
    corporate = family.eq("corporate")
    financial = family.eq("financial")
    index = metrics.index

    net_income = _metric(metrics, "net_income")
    cfo = _metric(metrics, "cfo")
    free_cash_flow_value = _metric(metrics, "free_cash_flow")
    positive_cfo = _metric(metrics, "positive_cfo_frequency")
    positive_fcf = _metric(metrics, "positive_fcf_frequency")
    accruals = _metric(metrics, "total_accruals")
    coverage = _metric(metrics, "interest_coverage")
    debt_assets = _metric(metrics, "debt_assets")
    prior_debt_assets = _metric(metrics, "debt_assets_prior")
    cfo_growth = _metric(metrics, "cfo_growth")
    common_equity = _metric(metrics, "common_equity")
    dilution = _metric(metrics, "share_count_dilution")
    dividends = _metric(metrics, "dividends")
    altman = _metric(metrics, "altman_z")
    beneish = _metric(metrics, "beneish_m")
    margin_change = _metric(metrics, "operating_margin_change")
    roa_change = _metric(metrics, "roa_change")
    fin_positive_earnings = _metric(
        metrics,
        "fin_positive_earnings_frequency",
    )
    fin_roa_change = _metric(metrics, "fin_roa_change")
    fin_roe_change = _metric(metrics, "fin_roe_change")
    fin_equity_assets_change = _metric(metrics, "fin_equity_assets_change")
    tangible_equity_value = _metric(metrics, "tangible_equity")
    fin_assets_equity = _metric(metrics, "fin_assets_equity")
    fin_share_dilution = _metric(metrics, "fin_share_dilution")
    fin_bvps_growth = _metric(metrics, "fin_bvps_growth")
    fin_net_payout_yield = _metric(metrics, "fin_net_payout_yield")
    fin_income_variability = _metric(metrics, "fin_net_income_variability")

    if date_column in metrics:
        leverage_cutoff = fin_assets_equity.groupby(
            metrics[date_column],
            sort=False,
        ).transform(lambda values: values.quantile(0.95))
    else:
        leverage_cutoff = pd.Series(
            fin_assets_equity.quantile(0.95),
            index=index,
            dtype=float,
        )

    warnings = {
        "warning_earnings_without_cash": _warning(
            corporate & net_income.notna() & cfo.notna(),
            net_income.gt(0) & cfo.lt(0),
            index,
        ),
        "warning_persistent_negative_cfo": _warning(
            corporate & positive_cfo.notna(),
            positive_cfo.lt(0.25),
            index,
        ),
        "warning_persistent_negative_fcf": _warning(
            corporate & positive_fcf.notna(),
            positive_fcf.lt(0.25),
            index,
        ),
        "warning_extreme_accruals": _warning(
            corporate & accruals.notna(),
            accruals.abs().gt(0.10),
            index,
        ),
        "warning_low_interest_coverage": _warning(
            corporate & coverage.notna(),
            coverage.lt(1.0),
            index,
        ),
        "warning_rising_debt_falling_cash": _warning(
            corporate
            & debt_assets.notna()
            & prior_debt_assets.notna()
            & cfo_growth.notna(),
            debt_assets.gt(prior_debt_assets + 0.03) & cfo_growth.lt(0),
            index,
        ),
        "warning_negative_equity": _warning(
            corporate & common_equity.notna(),
            common_equity.le(0),
            index,
        ),
        "warning_severe_dilution": _warning(
            corporate & dilution.notna(),
            dilution.gt(0.10),
            index,
        ),
        "warning_unsupported_payout": _warning(
            corporate
            & dividends.notna()
            & cfo.notna()
            & free_cash_flow_value.notna(),
            dividends.gt(cfo.clip(lower=0))
            | dividends.gt(free_cash_flow_value.clip(lower=0)),
            index,
        ),
        "warning_altman_distress": _warning(
            corporate & altman.notna(),
            altman.lt(1.81),
            index,
        ),
        "warning_beneish_warning": _warning(
            corporate & beneish.notna(),
            beneish.gt(-1.78),
            index,
        ),
        "warning_profitability_deterioration": _warning(
            corporate & margin_change.notna() & roa_change.notna(),
            margin_change.lt(-0.05) | roa_change.lt(-0.03),
            index,
        ),
        "warning_fin_negative_earnings": _warning(
            financial & net_income.notna() & fin_positive_earnings.notna(),
            net_income.lt(0) & fin_positive_earnings.lt(0.50),
            index,
        ),
        "warning_fin_roa_deterioration": _warning(
            financial & fin_roa_change.notna(),
            fin_roa_change.lt(-0.01),
            index,
        ),
        "warning_fin_roe_deterioration": _warning(
            financial & fin_roe_change.notna(),
            fin_roe_change.lt(-0.05),
            index,
        ),
        "warning_fin_capital_deterioration": _warning(
            financial & fin_equity_assets_change.notna(),
            fin_equity_assets_change.lt(-0.02),
            index,
        ),
        "warning_fin_nonpositive_tangible_equity": _warning(
            financial & tangible_equity_value.notna(),
            tangible_equity_value.le(0),
            index,
        ),
        "warning_fin_extreme_leverage": _warning(
            financial & fin_assets_equity.notna(),
            fin_assets_equity.gt(leverage_cutoff),
            index,
        ),
        "warning_fin_severe_dilution": _warning(
            financial & fin_share_dilution.notna(),
            fin_share_dilution.gt(0.10),
            index,
        ),
        "warning_fin_falling_bvps": _warning(
            financial & fin_bvps_growth.notna(),
            fin_bvps_growth.lt(-0.10),
            index,
        ),
        "warning_fin_weak_payout_capital": _warning(
            financial
            & fin_net_payout_yield.notna()
            & fin_equity_assets_change.notna(),
            fin_net_payout_yield.gt(0.08) & fin_equity_assets_change.lt(0),
            index,
        ),
        "warning_fin_earnings_instability": _warning(
            financial & fin_income_variability.notna(),
            fin_income_variability.gt(1.5),
            index,
        ),
    }
    result = pd.DataFrame(warnings, index=index)
    weights = dict(penalty_weights or warning_penalty_weights)
    penalty = pd.Series(0, index=index, dtype=int)
    severe_count = pd.Series(0, index=index, dtype=int)
    for name, weight in weights.items():
        active = result[name].fillna(False).astype(int)
        penalty = penalty + active * int(weight)
        if weight >= 4:
            severe_count = severe_count + active
    result["warning_penalty"] = penalty
    result["severe_warning_count"] = severe_count
    return result


__all__ = [
    "altman_z_score",
    "annual_change",
    "annual_growth",
    "annual_lag",
    "asset_turnover",
    "assets_to_equity",
    "average_balance",
    "beneish_m_score",
    "book_to_market",
    "book_value_per_share",
    "capex_to_depreciation",
    "capex_to_revenue",
    "cash_conversion_cycle",
    "cash_earnings_gap",
    "cash_ratio",
    "cash_to_assets",
    "cfo_to_assets",
    "cfo_to_debt",
    "cfo_to_net_income",
    "current_ratio",
    "days_sales_outstanding",
    "debt_to_assets",
    "debt_to_equity",
    "dividend_coverage_cfo",
    "dividend_coverage_fcf",
    "dividend_payout_ratio",
    "dividend_yield",
    "dupont_gap",
    "dupont_return_on_equity",
    "earnings_per_share",
    "earnings_yield",
    "ebit_to_enterprise_value",
    "effective_tax_rate",
    "enterprise_value",
    "enterprise_value_to_ebit",
    "enterprise_value_to_fcf",
    "enterprise_value_to_sales",
    "equity_multiplier",
    "equity_to_assets",
    "fcf_conversion",
    "fcf_margin",
    "fcf_per_share",
    "fcf_to_assets",
    "fcf_to_debt",
    "fcf_yield",
    "free_cash_flow",
    "gross_margin",
    "gross_profitability_assets",
    "growth_spread",
    "interest_coverage",
    "inventory_days",
    "inventory_turnover",
    "invested_capital",
    "issuance_yield",
    "leverage_improvement",
    "liabilities_to_assets",
    "net_debt",
    "net_debt_to_assets",
    "net_margin",
    "net_operating_profit_after_tax",
    "net_payout_amount",
    "operating_expense_ratio",
    "operating_margin",
    "payable_days",
    "piotroski_score",
    "positive_cfo_frequency",
    "positive_earnings_frequency",
    "positive_fcf_frequency",
    "positive_frequency",
    "pretax_margin",
    "pretax_return_on_assets",
    "price_to_book",
    "price_to_earnings",
    "price_to_fcf",
    "price_to_sales",
    "price_to_tangible_book",
    "receivables_turnover",
    "red_flag_penalties",
    "relative_variability",
    "reinvestment_quality",
    "reinvestment_rate",
    "repurchase_yield",
    "research_to_revenue",
    "return_on_assets",
    "return_on_equity",
    "return_on_invested_capital",
    "return_on_tangible_equity",
    "revenue_per_share",
    "revenue_to_assets",
    "safe_ratio",
    "sales_to_enterprise_value",
    "sales_yield",
    "share_count_dilution",
    "shareholder_yield",
    "tangible_book_to_market",
    "tangible_book_value_per_share",
    "tangible_equity",
    "tangible_equity_to_assets",
    "total_accruals",
    "total_debt",
    "warning_penalty_weights",
    "working_capital",
    "working_capital_accruals",
    "working_capital_to_revenue",
]
