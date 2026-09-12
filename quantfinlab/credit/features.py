"""Accounting ratios, maturity measures and filing-based credit warnings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.fundamentals import (
    annual_change,
    annual_growth,
    cash_to_assets,
    cfo_to_debt,
    debt_to_assets,
    fcf_to_assets,
    free_cash_flow,
    operating_margin,
    return_on_assets,
    safe_ratio,
    total_accruals,
    total_debt,
    working_capital,
)

credit_concepts = {
    "duration": {
        "revenue": ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                    "us-gaap:Revenues", "us-gaap:SalesRevenueNet"),
        "operating_income": ("us-gaap:OperatingIncomeLoss",),
        "net_income": ("us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"),
        "interest_expense": ("us-gaap:InterestExpense", "us-gaap:InterestExpenseNonoperating",
                             "us-gaap:InterestExpenseDebt"),
        "cfo": ("us-gaap:NetCashProvidedByUsedInOperatingActivities",
                "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        "capex": ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
                  "us-gaap:PaymentsToAcquireProductiveAssets"),
        "depreciation": ("us-gaap:DepreciationDepletionAndAmortization", "us-gaap:Depreciation"),
        "interest_paid": ("us-gaap:InterestPaidNet",),
        "debt_repaid": ("us-gaap:RepaymentsOfLongTermDebt",),
        "debt_issued": ("us-gaap:ProceedsFromIssuanceOfLongTermDebt",),
        "restructuring_charge": ("us-gaap:RestructuringCharges",),
        "impairment_charge": ("us-gaap:AssetImpairmentCharges", "us-gaap:GoodwillImpairmentLoss",
                              "us-gaap:TangibleAssetImpairmentCharges")},
    "instant": {
        "cash": ("us-gaap:CashAndCashEquivalentsAtCarryingValue",
                 "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        "short_investments": ("us-gaap:ShortTermInvestments", "us-gaap:MarketableSecuritiesCurrent"),
        "receivables": ("us-gaap:AccountsReceivableNetCurrent",),
        "inventory": ("us-gaap:InventoryNet",),
        "current_assets": ("us-gaap:AssetsCurrent",),
        "total_assets": ("us-gaap:Assets",),
        "current_liabilities": ("us-gaap:LiabilitiesCurrent",),
        "total_liabilities": ("us-gaap:Liabilities",),
        "common_equity": ("us-gaap:StockholdersEquity",),
        "retained_earnings": ("us-gaap:RetainedEarningsAccumulatedDeficit",),
        "total_debt_reported": ("us-gaap:DebtLongtermAndShorttermCombinedAmount",
                                "us-gaap:LongTermDebt"),
        "long_term_debt_noncurrent": ("us-gaap:LongTermDebtNoncurrent",),
        "debt_current": ("us-gaap:DebtCurrent", "us-gaap:LongTermDebtCurrent",
                         "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent"),
        "short_term_borrowings": ("us-gaap:ShortTermBorrowings", "us-gaap:CommercialPaper"),
        "operating_lease_current": ("us-gaap:OperatingLeaseLiabilityCurrent",),
        "operating_lease_noncurrent": ("us-gaap:OperatingLeaseLiabilityNoncurrent",),
        "credit_available": ("us-gaap:LineOfCreditFacilityRemainingBorrowingCapacity",),
        "credit_drawn": ("us-gaap:LineOfCreditFacilityAmountOutstanding",),
        "debt_due_1y": ("us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",),
        "debt_due_2y": ("us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",),
        "debt_due_3y": ("us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",),
        "debt_due_4y": ("us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",),
        "debt_due_5y": ("us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",),
        "debt_due_after_5y": ("us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",)}}

accounting_features = [
    "log_assets", "liabilities_assets", "debt_assets", "debt_capital", "debt_ebitda",
    "interest_coverage", "cfo_debt", "cash_assets", "current_ratio", "wc_assets",
    "fcf_assets", "roa", "operating_margin", "sales_assets", "accruals", "maturity_2y",
    "lease_assets", "restructuring_assets", "impairment_assets", "d_debt_assets",
    "d_interest_coverage", "d_cash_assets", "d_cfo_debt", "d_fcf_assets", "d_roa",
    "d_operating_margin", "revenue_growth", "asset_growth", "deterioration_2y",
    "roa_stability", "coverage_stability", "statement_age"
]

indicator_features = [
    "negative_ebitda", "missing_ebitda", "negative_ebit", "no_interest_expense",
    "missing_interest_expense", "negative_working_capital", "stale_statement",
    "item_204_12m", "late_filing_12m", "delisting_12m", "nonreliance_12m",
    "distress_index", *[f"sic_{division}" for division in range(10)]
]

time_controls = ["calendar_time", "calendar_time_sq", "month_sin", "month_cos"]

linear_features = [*accounting_features, *indicator_features]

model_features = [*linear_features, *time_controls]

def credit_ratios(statements: pd.DataFrame) -> pd.DataFrame:
    """Credit ratios with economically defined denominators and explicit missing/negative flags."""
    credit = statements.copy()
    expected = [*credit_concepts["instant"], *[f"{name}_ttm" for name in credit_concepts["duration"]]]
    credit = credit.reindex(columns=list(dict.fromkeys([*credit.columns, *expected])))
    short_debt = credit["debt_current"].combine_first(credit["short_term_borrowings"])
    long_debt = credit["long_term_debt_noncurrent"]
    debt = total_debt(credit["total_debt_reported"], long_debt,
                      credit["debt_current"], credit["short_term_borrowings"])
    long_debt = long_debt.combine_first((debt - short_debt).where(debt.gt(short_debt)))
    debt = total_debt(credit["total_debt_reported"], long_debt,
                      credit["debt_current"], credit["short_term_borrowings"])

    levels = pd.DataFrame({
        "short_debt": short_debt,
        "long_debt": long_debt,
        "debt": debt,
        "lease_debt": credit[["operating_lease_current", "operating_lease_noncurrent"]].sum(
            axis=1, min_count=2),
        "wc": working_capital(credit["current_assets"], credit["current_liabilities"]),
        "fcf": free_cash_flow(credit["cfo_ttm"], credit["capex_ttm"]),
        "ebitda": (credit["operating_income_ttm"] + credit["depreciation_ttm"]).where(
            credit[["operating_income_ttm", "depreciation_ttm"]].notna().all(axis=1)),
        "statement_age": (credit["decision_date"] - credit["filed_date"]).dt.days.div(30.4375)
    })
    levels["log_assets"] = np.log(credit["total_assets"].where(credit["total_assets"].gt(0)))
    levels["invalid_assets"] = credit["total_assets"].le(0) | credit["total_assets"].isna()
    levels["invalid_debt"] = debt.le(0) | debt.isna()
    levels["stale_statement"] = levels["statement_age"].gt(9)

    credit = pd.concat([credit, levels], axis=1)
    assets = credit["total_assets"].abs()
    ebitda_known = credit["ebitda"].notna()
    positive_ebitda = credit["ebitda"].gt(1e-6 * assets)
    interest_known = credit["interest_expense_ttm"].notna()
    positive_interest = credit["interest_expense_ttm"].gt(1e-6 * assets)
    positive_current_liabilities = credit["current_liabilities"].gt(1e-6 * assets)

    ratios = pd.DataFrame({
        "liabilities_assets": safe_ratio(credit["total_liabilities"], credit["total_assets"]),
        "debt_assets": debt_to_assets(credit["debt"], credit["total_assets"]),
        "debt_capital": safe_ratio(credit["debt"], credit["debt"] + credit["common_equity"]),
        "debt_ebitda": (credit["debt"] / credit["ebitda"]).where(positive_ebitda),
        "interest_coverage": (
            credit["operating_income_ttm"] / credit["interest_expense_ttm"]).where(positive_interest),
        "cfo_debt": cfo_to_debt(credit["cfo_ttm"], credit["debt"]),
        "cash_assets": cash_to_assets(credit["cash"], credit["total_assets"]),
        "current_ratio": (
            credit["current_assets"] / credit["current_liabilities"]).where(
                positive_current_liabilities),
        "wc_assets": safe_ratio(credit["wc"], credit["total_assets"]),
        "fcf_assets": fcf_to_assets(credit["fcf"], credit["total_assets"]),
        "roa": return_on_assets(credit["net_income_ttm"], credit["total_assets"]),
        "operating_margin": operating_margin(
            credit["operating_income_ttm"], credit["revenue_ttm"]),
        "sales_assets": safe_ratio(credit["revenue_ttm"], credit["total_assets"]),
        "accruals": total_accruals(
            credit["net_income_ttm"], credit["cfo_ttm"], credit["total_assets"]),
        "maturity_2y": safe_ratio(
            credit["debt_due_1y"] + credit["debt_due_2y"], credit["debt"]),
        "lease_assets": safe_ratio(credit["lease_debt"], credit["total_assets"]),
        "restructuring_assets": safe_ratio(
            credit["restructuring_charge_ttm"], credit["total_assets"]),
        "impairment_assets": safe_ratio(
            credit["impairment_charge_ttm"], credit["total_assets"])
    })
    ratio_flags = pd.DataFrame({
        "negative_ebitda": (credit["ebitda"].le(0) & ebitda_known).astype(float),
        "missing_ebitda": (~ebitda_known).astype(float),
        "negative_ebit": (
            credit["operating_income_ttm"].le(0)
            & credit["operating_income_ttm"].notna()).astype(float),
        "no_interest_expense": (interest_known & ~positive_interest).astype(float),
        "missing_interest_expense": (~interest_known).astype(float),
        "negative_working_capital": (
            credit["wc"].lt(0) & credit["wc"].notna()).astype(float)
    })


    return pd.concat([credit, ratios, ratio_flags], axis=1)


def ratio_changes(credit: pd.DataFrame) -> pd.DataFrame:
    """Annual changes and deterioration measures on a complete issuer-month grid."""
    change_fields = ["debt_assets", "interest_coverage", "cash_assets", "cfo_debt",
                     "fcf_assets", "roa", "operating_margin"]
    changes = pd.DataFrame({
        f"d_{name}": annual_change(credit[name], credit["cik"]) for name in change_fields
    })
    changes["revenue_growth"] = annual_growth(credit["revenue_ttm"], credit["cik"])
    changes["asset_growth"] = annual_growth(credit["total_assets"], credit["cik"])

    bad = pd.DataFrame(index=credit.index)
    bad["leverage"] = changes["d_debt_assets"].gt(0).where(changes["d_debt_assets"].notna())
    for name in ["interest_coverage", "cash_assets", "cfo_debt", "fcf_assets", "roa",
                 "operating_margin"]:
        change = changes[f"d_{name}"]
        bad[name] = change.lt(0).where(change.notna())
    changes["deterioration"] = bad.sum(axis=1, min_count=4)
    changes["deterioration_2y"] = pd.concat([
        changes["deterioration"],
        changes.groupby(credit["cik"])["deterioration"].shift(12)
    ], axis=1).sum(axis=1, min_count=2)
    changes["roa_stability"] = credit.groupby("cik")["roa"].transform(
        lambda x: x.rolling(24, min_periods=12).std())
    changes["coverage_stability"] = credit.groupby("cik")["interest_coverage"].transform(
        lambda x: x.rolling(24, min_periods=12).std())

    return changes



def distress_index(counts: pd.DataFrame, weights=None) -> pd.Series:
    """Weighted trailing filing-event counts, with caller-overridable weights."""
    weights = weights or {"item_204_12m": 3.0, "delisting_12m": 2.0,
                          "late_filing_12m": 1.5, "nonreliance_12m": 1.5,
                          "impairment_filing_12m": 1.0, "exit_disposal_12m": 1.0}
    return counts[list(weights)].mul(pd.Series(weights)).sum(axis=1, min_count=len(weights))


def debt_maturities(statements: pd.DataFrame) -> pd.DataFrame:
    """Debt maturity amounts and fractions, retaining unknown contractual amounts."""
    columns = [f"debt_due_{year}y" for year in range(1, 6)] + ["debt_due_after_5y"]
    result = statements.reindex(columns=columns).copy()
    for name in columns:
        result[name + "_share"] = safe_ratio(result[name], statements["debt"])
    return result


def credit_controls(dates, sic, *, start_year=2012) -> pd.DataFrame:
    """Calendar and industry controls; elapsed sample time is never issuer age."""
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    time = dates.dt.year + (dates.dt.month - 0.5) / 12 - start_year
    result = pd.DataFrame({"calendar_time": time, "calendar_time_sq": time ** 2,
                           "month_sin": np.sin(2 * np.pi * dates.dt.month / 12),
                           "month_cos": np.cos(2 * np.pi * dates.dt.month / 12)})
    division = pd.Series(sic).reset_index(drop=True).floordiv(1000).astype("Int64")
    for i in range(10):
        result[f"sic_{i}"] = division.eq(i).astype(float)
    return result
