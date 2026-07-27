"""Point-in-time SEC statement reconstruction.

The functions in this module mirror the accounting steps used by the
fundamental-analysis notebook.  They intentionally remain separate so a
research notebook can inspect the facts, reconstructed quarters, duration
values, and instant values before building a monthly data set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

statement_concepts = {
    "duration": {
        "revenue": (
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:Revenues",
            "us-gaap:SalesRevenueNet",
            "us-gaap:SalesRevenueGoodsNet",
            "us-gaap:SalesRevenueServicesNet",
        ),
        "cost_of_revenue": (
            "us-gaap:CostOfRevenue",
            "us-gaap:CostOfGoodsAndServicesSold",
            "us-gaap:CostOfGoodsSold",
        ),
        "gross_profit": ("us-gaap:GrossProfit",),
        "operating_income": ("us-gaap:OperatingIncomeLoss",),
        "pretax_income": (
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
        "net_income": (
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss",
        ),
        "interest_expense": (
            "us-gaap:InterestExpense",
            "us-gaap:InterestExpenseNonoperating",
            "us-gaap:InterestExpenseDebt",
        ),
        "tax_expense": ("us-gaap:IncomeTaxExpenseBenefit",),
        "eps_diluted": ("us-gaap:EarningsPerShareDiluted",),
        "cfo": (
            "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        "capex": (
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            "us-gaap:PaymentsToAcquireProductiveAssets",
        ),
        "depreciation": (
            "us-gaap:DepreciationDepletionAndAmortization",
            "us-gaap:Depreciation",
        ),
        "dividends": (
            "us-gaap:PaymentsOfDividendsCommonStock",
            "us-gaap:PaymentsOfDividends",
            "us-gaap:PaymentsOfOrdinaryDividends",
        ),
        "repurchases": ("us-gaap:PaymentsForRepurchaseOfCommonStock",),
        "share_issuance": (
            "us-gaap:ProceedsFromIssuanceOfCommonStock",
            "us-gaap:ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
            "us-gaap:ProceedsFromStockOptionsExercised",
        ),
        "rd_expense": ("us-gaap:ResearchAndDevelopmentExpense",),
        "operating_expenses": (
            "us-gaap:OperatingExpenses",
            "us-gaap:NoninterestExpense",
        ),
        "sga_expense": (
            "us-gaap:SellingGeneralAndAdministrativeExpense",
            "us-gaap:GeneralAndAdministrativeExpense",
        ),
        "credit_loss_provision": (
            "us-gaap:ProvisionForLoanLeaseAndOtherLosses",
            "us-gaap:ProvisionForLoanAndLeaseLosses",
            "us-gaap:ProvisionForLoanLossesExpensed",
        ),
    },
    "instant": {
        "cash": (
            "us-gaap:CashAndCashEquivalentsAtCarryingValue",
            "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        "receivables": ("us-gaap:AccountsReceivableNetCurrent",),
        "inventory": ("us-gaap:InventoryNet",),
        "current_assets": ("us-gaap:AssetsCurrent",),
        "total_assets": ("us-gaap:Assets",),
        "current_liabilities": ("us-gaap:LiabilitiesCurrent",),
        "accounts_payable": ("us-gaap:AccountsPayableCurrent",),
        "total_debt_reported": ("us-gaap:LongTermDebtAndCapitalLeaseObligations",),
        "long_term_debt_noncurrent": ("us-gaap:LongTermDebtNoncurrent",),
        "debt_current": (
            "us-gaap:DebtCurrent",
            "us-gaap:LongTermDebtCurrent",
            "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent",
        ),
        "short_term_borrowings": (
            "us-gaap:ShortTermBorrowings",
            "us-gaap:CommercialPaper",
        ),
        "total_liabilities": ("us-gaap:Liabilities",),
        "common_equity": ("us-gaap:StockholdersEquity",),
        "retained_earnings": ("us-gaap:RetainedEarningsAccumulatedDeficit",),
        "goodwill": ("us-gaap:Goodwill",),
        "intangibles": (
            "us-gaap:IntangibleAssetsNetExcludingGoodwill",
            "us-gaap:FiniteLivedIntangibleAssetsNet",
            "us-gaap:IndefiniteLivedIntangibleAssetsExcludingGoodwill",
        ),
        "shares_outstanding": (
            "dei:EntityCommonStockSharesOutstanding",
            "us-gaap:CommonStockSharesOutstanding",
        ),
        "ppe": ("us-gaap:PropertyPlantAndEquipmentNet",),
        "loans": (
            "us-gaap:LoansAndLeasesReceivableNetReportedAmount",
            "us-gaap:LoansAndLeasesReceivableNetOfDeferredIncome",
            "us-gaap:LoansReceivableNet",
        ),
        "deposits": ("us-gaap:Deposits",),
    },
}

periodic_forms = (
    "10-Q",
    "10-Q/A",
    "10-K",
    "10-K/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
    "6-K",
    "6-K/A",
)

financial_industries = {
    "ACCIDENT & HEALTH INSURANCE",
    "FINANCE SERVICES",
    "FIRE, MARINE & CASUALTY INSURANCE",
    "HOSPITAL & MEDICAL SERVICE PLANS",
    "INSURANCE AGENTS, BROKERS & SERVICE",
    "INSURANCE CARRIERS, NEC",
    "INVESTMENT ADVICE",
    "LIFE INSURANCE",
    "NATIONAL COMMERCIAL BANKS",
    "PERSONAL CREDIT INSTITUTIONS",
    "SAVINGS INSTITUTIONS, NOT FEDERALLY CHARTERED",
    "SECURITY & COMMODITY BROKERS, DEALERS, EXCHANGES & SERVICES",
    "SECURITY BROKERS, DEALERS & FLOTATION COMPANIES",
    "STATE COMMERCIAL BANKS",
}

reit_industries = {"REAL ESTATE INVESTMENT TRUSTS"}

ticker_renames = {
    "ABC": "COR",
    "ANTM": "ELV",
    "BLL": "BALL",
    "CTL": "LUMN",
    "FB": "META",
    "HRS": "LHX",
    "KORS": "CPRI",
    "LB": "BBWI",
    "MYL": "VTRS",
    "PKI": "RVTY",
    "RE": "EG",
    "UTX": "RTX",
    "WLTW": "WTW",
}

dual_class_preferences = {
    frozenset({"GOOG", "GOOGL"}): "GOOG",
    frozenset({"FOX", "FOXA"}): "FOXA",
    frozenset({"NWS", "NWSA"}): "NWSA",
    frozenset({"UA", "UAA"}): "UAA",
    frozenset({"CPRI", "KORS"}): "KORS",
}


def issuer_universe(
    market: pd.DataFrame,
    ticker_mappings: pd.DataFrame,
    date_map: pd.DataFrame,
    *,
    exclude_reits: bool = True,
) -> pd.DataFrame:
    """Map month-end index members to one valid ticker for each SEC issuer."""

    monthly = market.copy()
    if "decision_date" not in monthly:
        monthly = monthly.rename(columns={"date": "decision_date"})
    monthly["decision_date"] = pd.to_datetime(monthly["decision_date"])
    decision_dates = pd.DatetimeIndex(pd.to_datetime(date_map["decision_date"])).unique()
    monthly = monthly[monthly["decision_date"].isin(decision_dates)]
    if "is_sp500_member" in monthly:
        monthly = monthly[monthly["is_sp500_member"]]
    if "adj_close" in monthly and "price" not in monthly:
        monthly = monthly.rename(columns={"adj_close": "price"})

    mappings = ticker_mappings.copy().drop_duplicates()
    for column in ("mapping_valid_from", "mapping_valid_to"):
        mappings[column] = pd.to_datetime(mappings[column])
    monthly = monthly.merge(mappings, on="ticker", how="left")
    valid_mapping = (
        monthly["cik"].notna()
        & monthly["mapping_valid_from"].le(monthly["decision_date"])
        & monthly["mapping_valid_to"].ge(monthly["decision_date"])
    )
    monthly = monthly[valid_mapping].copy()
    monthly["cik"] = monthly["cik"].astype("int64")

    route = pd.Series("corporate", index=monthly.index, dtype="object")
    route.loc[monthly["industry"].isin(financial_industries)] = "financial"
    route.loc[monthly["industry"].isin(reit_industries)] = "reit"
    monthly["score_family"] = route
    monthly["company_type"] = route

    duplicate = monthly.duplicated(["decision_date", "cik"], keep=False)
    monthly["keep_ticker"] = True
    for _, group in monthly[duplicate].groupby(["decision_date", "cik"], sort=False):
        tickers = frozenset(group["ticker"])
        preferred = dual_class_preferences.get(tickers, sorted(tickers)[0])
        monthly.loc[group.index, "keep_ticker"] = group["ticker"].eq(preferred)
    monthly = monthly[monthly["keep_ticker"]].drop(columns="keep_ticker")

    if exclude_reits:
        monthly = monthly[monthly["score_family"].ne("reit")]
    monthly["display_ticker"] = monthly["ticker"].replace(ticker_renames)
    monthly["issuer_label"] = monthly["entity_name"].astype("string").str.title()
    return monthly.sort_values(["decision_date", "cik"]).reset_index(drop=True)


def classify_duration_facts(
    facts: pd.DataFrame,
    *,
    concepts: dict = statement_concepts,
    forms: tuple[str, ...] = periodic_forms,
) -> pd.DataFrame:
    """Map XBRL concepts to statement fields and classify their time spans."""

    concept_lookup = {
        concept: (field, priority)
        for section in concepts.values()
        for field, candidates in section.items()
        for priority, concept in enumerate(candidates)
    }
    classified = facts[facts["concept"].isin(concept_lookup)].copy()
    classified["field"] = classified["concept"].map(lambda concept: concept_lookup[concept][0])
    classified["concept_priority"] = (
        classified["concept"].map(lambda concept: concept_lookup[concept][1]).astype("int16")
    )

    for column in ("period_start", "period_end", "filed_date"):
        classified[column] = pd.to_datetime(classified[column])
    classified["form_type"] = classified["form_type"].str.upper().str.strip()
    classified["period_type"] = classified["period_type"].str.lower().str.strip()
    classified["value"] = pd.to_numeric(classified["value"], errors="coerce")
    classified = classified[classified["form_type"].isin(forms)]

    valid_unit = (
        classified["field"].eq("shares_outstanding") & classified["unit"].eq("shares")
        | classified["field"].eq("eps_diluted")
        & classified["unit"].isin(["USD per share", "USD/shares"])
        | ~classified["field"].isin(["shares_outstanding", "eps_diluted"])
        & classified["unit"].eq("USD")
    )
    classified = classified[
        np.isfinite(classified["value"])
        & classified["filed_date"].ge(classified["period_end"])
        & valid_unit
    ].copy()

    fact_version_key = ["cik", "concept", "unit", "period_start", "period_end"]
    exact_version_key = fact_version_key + ["filed_date", "accession", "filing_version"]
    classified = (
        classified.sort_values(exact_version_key)
        .drop_duplicates(exact_version_key, keep="last")
        .reset_index(drop=True)
    )

    duration = classified["period_type"].eq("duration") & classified["period_start"].notna()
    classified["period_days"] = (classified["period_end"] - classified["period_start"]).dt.days + 1
    classified["duration_class"] = "instant"
    classified.loc[duration & classified["period_days"].between(70, 120), "duration_class"] = (
        "quarter"
    )
    classified.loc[duration & classified["period_days"].between(150, 220), "duration_class"] = (
        "six_month_ytd"
    )
    classified.loc[duration & classified["period_days"].between(230, 310), "duration_class"] = (
        "nine_month_ytd"
    )
    classified.loc[duration & classified["period_days"].between(320, 410), "duration_class"] = (
        "annual"
    )
    classified.loc[duration & ~classified["period_days"].between(70, 410), "duration_class"] = (
        "irregular"
    )
    return classified.sort_values(["filed_date", "filing_version", "accession"]).reset_index(
        drop=True
    )


def select_filing_facts(
    facts: pd.DataFrame,
    *,
    decision_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Select the best known concept version strictly before a decision date."""

    available = facts.copy()
    if decision_date is not None:
        available = available[available["filed_date"].lt(pd.Timestamp(decision_date))]
    available = available.sort_values(["filed_date", "filing_version", "accession"])
    fact_version_key = ["cik", "concept", "unit", "period_start", "period_end"]
    economic_key = ["cik", "field", "unit", "period_start", "period_end"]
    latest = available.drop_duplicates(fact_version_key, keep="last")
    best_priority = latest.groupby(economic_key, sort=False, dropna=False)[
        "concept_priority"
    ].transform("min")
    return latest[latest["concept_priority"].eq(best_priority)].drop_duplicates(
        economic_key, keep="last"
    )


def reconstruct_quarters(
    selected_facts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover standalone quarters from direct and cumulative duration facts."""

    duration = selected_facts[
        selected_facts["period_type"].eq("duration")
        & selected_facts["duration_class"].isin(
            ["quarter", "six_month_ytd", "nine_month_ytd", "annual"]
        )
    ]
    columns = [
        "cik",
        "field",
        "unit",
        "fiscal_year",
        "period_start",
        "period_end",
        "value",
        "filed_date",
        "concept",
        "source_class",
    ]
    direct = duration[duration["duration_class"].eq("quarter")].copy()
    direct["source_class"] = "direct"
    candidates = [direct[columns]]

    previous_class = {
        "six_month_ytd": "quarter",
        "nine_month_ytd": "six_month_ytd",
        "annual": "nine_month_ytd",
    }
    join_columns = ["cik", "field", "unit", "fiscal_year", "period_start"]
    for current_class, prior_class in previous_class.items():
        current = duration[duration["duration_class"].eq(current_class)]
        prior = duration[duration["duration_class"].eq(prior_class)]
        pairs = current.merge(
            prior[join_columns + ["period_end", "value", "filed_date"]],
            on=join_columns,
            how="inner",
            suffixes=("", "_prior"),
        )
        pairs = pairs[pairs["period_end_prior"].lt(pairs["period_end"])]
        pairs = pairs.sort_values(
            join_columns + ["period_end", "period_end_prior"]
        ).drop_duplicates(join_columns + ["period_end"], keep="last")
        quarter_gap = (pairs["period_end"] - pairs["period_end_prior"]).dt.days
        quarter_value = pairs["value"] - pairs["value_prior"]
        scale = pairs[["value", "value_prior"]].abs().max(axis=1).clip(lower=1.0)
        pairs = pairs[quarter_gap.between(55, 140) & quarter_value.abs().le(10.0 * scale)].copy()
        pairs["period_start"] = pairs["period_end_prior"] + pd.Timedelta(days=1)
        pairs["value"] = quarter_value.loc[pairs.index]
        pairs["filed_date"] = pairs[["filed_date", "filed_date_prior"]].max(axis=1)
        pairs["source_class"] = f"reconstructed_{current_class}"
        candidates.append(pairs[columns])

    quarter_candidates = pd.concat(candidates, ignore_index=True)
    quarter_candidates["source_rank"] = quarter_candidates["source_class"].ne("direct")
    quarterly = (
        quarter_candidates.sort_values(["cik", "field", "period_end", "source_rank", "filed_date"])
        .drop_duplicates(["cik", "field", "period_end"], keep="first")
        .drop(columns="source_rank")
    )
    return quarterly, quarter_candidates


def duration_ttm_values(selected_facts: pd.DataFrame) -> pd.DataFrame:
    """Build annual and annual-plus-YTD trailing-twelve-month candidates."""

    duration = selected_facts[
        selected_facts["period_type"].eq("duration")
        & selected_facts["duration_class"].isin(
            ["quarter", "six_month_ytd", "nine_month_ytd", "annual"]
        )
    ]
    keys = ["cik", "field", "unit"]
    annual = (
        duration[duration["duration_class"].eq("annual")]
        .sort_values("period_end")
        .groupby(keys, sort=False)
        .tail(1)
    )
    annual = annual[keys + ["period_end", "value"]].rename(
        columns={"period_end": "annual_end", "value": "annual_value"}
    )

    current = duration[
        duration["duration_class"].isin(["quarter", "six_month_ytd", "nine_month_ytd"])
    ].merge(annual, on=keys, how="inner")
    distance = (current["period_end"] - current["annual_end"]).dt.days
    current = current[current["period_end"].gt(current["annual_end"]) & distance.between(40, 320)]
    current = (
        current.sort_values("period_end").groupby(keys + ["duration_class"], sort=False).tail(1)
    )
    current = current[
        keys
        + [
            "duration_class",
            "period_end",
            "value",
            "annual_end",
            "annual_value",
        ]
    ].rename(columns={"period_end": "current_end", "value": "current_value"})

    prior = duration[
        duration["duration_class"].isin(["quarter", "six_month_ytd", "nine_month_ytd"])
    ][keys + ["duration_class", "period_end", "value"]].rename(
        columns={"period_end": "prior_end", "value": "prior_value"}
    )
    updates = current.merge(prior, on=keys + ["duration_class"], how="inner")
    year_gap = (updates["current_end"] - updates["prior_end"]).dt.days
    updates = updates[year_gap.between(300, 430)]
    updates = (
        updates.sort_values("prior_end")
        .groupby(keys + ["duration_class", "current_end"], sort=False)
        .tail(1)
    )
    updates["ttm_fallback"] = (
        updates["annual_value"] + updates["current_value"] - updates["prior_value"]
    )
    updates["period_end"] = updates["current_end"]
    updates["ttm_fallback_method"] = "annual + current YTD - prior YTD"

    candidates = annual.rename(columns={"annual_end": "period_end", "annual_value": "ttm_fallback"})
    candidates["ttm_fallback_method"] = "annual"
    candidates = pd.concat(
        [
            candidates[keys + ["period_end", "ttm_fallback", "ttm_fallback_method"]],
            updates[keys + ["period_end", "ttm_fallback", "ttm_fallback_method"]],
        ],
        ignore_index=True,
    )
    chosen = candidates.sort_values("period_end").groupby(["cik", "field"], sort=False).tail(1)
    if chosen.empty:
        return pd.DataFrame(columns=["cik"])

    amount = chosen.pivot(index="cik", columns="field", values="ttm_fallback")
    amount.columns = [f"{field}_ttm_fallback" for field in amount]
    method = chosen.pivot(index="cik", columns="field", values="ttm_fallback_method")
    method.columns = [f"{field}_ttm_fallback_method" for field in method]
    result = pd.concat([amount, method], axis=1).reset_index()
    duration_end = chosen.groupby("cik")["period_end"].max()
    result["latest_duration_end"] = duration_end.reindex(result["cik"]).to_numpy()
    return result


def duration_statement_values(
    quarters: pd.DataFrame,
    selected_facts: pd.DataFrame,
    *,
    concepts: dict = statement_concepts,
) -> pd.DataFrame:
    """Calculate quarterly growth and TTM values, then apply the TTM fallback."""

    values = (
        quarters.sort_values(["cik", "field", "period_end", "filed_date"])
        .drop_duplicates(["cik", "field", "period_end"], keep="last")
        .copy()
    )
    if values.empty:
        result = pd.DataFrame(columns=["cik"])
    else:
        grouped = values.groupby(["cik", "field"], sort=False)
        previous_value = grouped["value"].shift(1)
        previous_end = grouped["period_end"].shift(1)
        year_ago_value = grouped["value"].shift(4)
        year_ago_end = grouped["period_end"].shift(4)
        gap = (values["period_end"] - previous_end).dt.days
        year_gap = (values["period_end"] - year_ago_end).dt.days

        values["q"] = values["value"]
        values["qoq"] = (
            values["value"].div(previous_value.where(previous_value.abs() > 1e-12)) - 1.0
        )
        values.loc[~gap.between(55, 140), "qoq"] = np.nan
        values["q_yoy"] = (
            values["value"].div(year_ago_value.where(year_ago_value.abs() > 1e-12)) - 1.0
        )
        values.loc[~year_gap.between(300, 430), "q_yoy"] = np.nan
        values["ttm"] = (
            grouped["value"].rolling(4, min_periods=4).sum().reset_index(level=[0, 1], drop=True)
        )
        valid_ttm = (
            gap.between(55, 140)
            .groupby([values["cik"], values["field"]])
            .rolling(3, min_periods=3)
            .sum()
            .reset_index(level=[0, 1], drop=True)
            .eq(3)
        )
        values.loc[~valid_ttm.fillna(False), "ttm"] = np.nan
        prior_ttm = values.groupby(["cik", "field"], sort=False)["ttm"].shift(4)
        values["ttm_yoy"] = values["ttm"].div(prior_ttm.where(prior_ttm.abs() > 1e-12)) - 1.0
        values.loc[~year_gap.between(300, 430), "ttm_yoy"] = np.nan

        latest = values.groupby(["cik", "field"], sort=False).tail(1)
        measures = ["q", "qoq", "q_yoy", "ttm", "ttm_yoy"]
        result = latest.pivot(index="cik", columns="field", values=measures)
        result.columns = [f"{field}_{measure}" for measure, field in result.columns]
        result = result.reset_index()
        quarter_end = latest.groupby("cik")["period_end"].max()
        result["latest_quarter_end"] = quarter_end.reindex(result["cik"]).to_numpy()
        revenue_periods = (
            values[values["field"].eq("revenue")]
            .groupby("cik", sort=False)
            .tail(4)
            .groupby("cik")["period_end"]
            .agg(lambda dates: "|".join(dates.dt.strftime("%Y-%m-%d")))
        )
        result["ttm_quarter_ends"] = revenue_periods.reindex(result["cik"]).to_numpy(dtype=object)
        result["ttm_quarter_ends"] = result["ttm_quarter_ends"].astype("string")

    fallback = duration_ttm_values(selected_facts)
    result = result.merge(fallback, on="cik", how="outer")
    for field in concepts["duration"]:
        ttm = f"{field}_ttm"
        fallback_value = f"{field}_ttm_fallback"
        fallback_method = f"{field}_ttm_fallback_method"
        method = f"{field}_ttm_method"
        if ttm not in result:
            result[ttm] = np.nan
        if fallback_value not in result:
            result[method] = pd.Series(
                np.where(result[ttm].notna(), "four standalone quarters", pd.NA),
                index=result.index,
                dtype="string",
            )
            continue
        scale = result[[ttm, fallback_value]].abs().max(axis=1).clip(lower=1.0)
        result[f"{field}_ttm_reconciliation"] = (
            (result[ttm] - result[fallback_value])
            .div(scale)
            .where(result[ttm].notna() & result[fallback_value].notna())
        )
        result[method] = pd.Series(
            np.where(
                result[ttm].notna(),
                "four standalone quarters",
                result[fallback_method],
            ),
            index=result.index,
            dtype="string",
        )
        result[ttm] = result[ttm].combine_first(result[fallback_value])
    return result.drop(columns=[column for column in result if "_ttm_fallback" in column]).copy()


def instant_statement_values(selected_facts: pd.DataFrame) -> pd.DataFrame:
    """Return each issuer's latest and prior instant statement observations."""

    instant = selected_facts[selected_facts["period_type"].eq("instant")].sort_values(
        ["cik", "field", "period_end", "filed_date"]
    )
    if instant.empty:
        return pd.DataFrame(columns=["cik"])
    grouped = instant.groupby(["cik", "field"], sort=False)
    instant = instant.copy()
    instant["prior_value"] = grouped["value"].shift(1)
    instant["prior_period_end"] = grouped["period_end"].shift(1)
    latest = instant.groupby(["cik", "field"], sort=False).tail(1)
    values = ["value", "period_end", "prior_value", "prior_period_end"]
    result = latest.pivot(index="cik", columns="field", values=values)
    names = {
        "value": lambda field: field,
        "period_end": lambda field: f"{field}_period_end",
        "prior_value": lambda field: f"{field}_prior",
        "prior_period_end": lambda field: f"{field}_prior_period_end",
    }
    result.columns = [names[measure](field) for measure, field in result.columns]
    result = result.reset_index()
    balance_end = latest.groupby("cik")["period_end"].max()
    result["latest_balance_end"] = balance_end.reindex(result["cik"]).to_numpy()
    return result


def monthly_statement_values(
    facts: pd.DataFrame,
    monthly_universe: pd.DataFrame,
    *,
    cache: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Reconstruct the latest strictly available statement values each month.

    ``facts`` must already have passed through :func:`classify_duration_facts`.
    The returned frame contains statement values only; merge it with
    ``monthly_universe`` to attach prices, industries, and issuer labels.
    """

    cache_path = Path(cache) if cache is not None else None
    if cache_path is not None and cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    eligible_ciks = monthly_universe["cik"].dropna().astype("int64").unique()
    facts_by_filing = (
        facts[facts["cik"].isin(eligible_ciks)]
        .sort_values(["filed_date", "filing_version", "accession"])
        .reset_index(drop=True)
    )
    filing_dates = pd.DatetimeIndex(facts_by_filing["filed_date"])
    decision_dates = pd.DatetimeIndex(
        pd.to_datetime(monthly_universe["decision_date"].dropna().unique())
    ).sort_values()

    latest_records = pd.DataFrame()
    monthly_records = []
    cursor = 0
    for decision_date in decision_dates:
        stop = filing_dates.searchsorted(decision_date, side="left")
        new_facts = facts_by_filing.iloc[cursor:stop]
        cursor = stop
        changed_ciks = new_facts["cik"].drop_duplicates()
        if not changed_ciks.empty:
            oldest_period = decision_date - pd.DateOffset(years=3)
            available = facts_by_filing.iloc[:stop]
            available = available[
                available["cik"].isin(changed_ciks) & available["period_end"].ge(oldest_period)
            ]
            selected = select_filing_facts(available)
            quarters, _ = reconstruct_quarters(selected)
            duration_values = duration_statement_values(quarters, selected)
            instant_values = instant_statement_values(selected)
            current = duration_values.merge(instant_values, on="cik", how="outer")
            latest_filed = available.groupby("cik")["filed_date"].max()
            current["filed_date"] = latest_filed.reindex(current["cik"]).to_numpy()
            period_dates = current.reindex(
                columns=[
                    "latest_quarter_end",
                    "latest_duration_end",
                    "latest_balance_end",
                ]
            ).apply(pd.to_datetime)
            current["latest_period_end"] = period_dates.max(axis=1)
            current = current.set_index("cik")
            latest_records = pd.concat(
                [
                    latest_records.drop(index=changed_ciks, errors="ignore"),
                    current,
                ]
            ).copy()
            latest_records.index.name = "cik"

        current_ciks = pd.Index(
            monthly_universe.loc[
                monthly_universe["decision_date"].eq(decision_date), "cik"
            ].unique()
        )
        available_ciks = current_ciks.intersection(latest_records.index)
        month = latest_records.loc[available_ciks].rename_axis("cik").reset_index()
        month["decision_date"] = decision_date
        monthly_records.append(month)

    if monthly_records:
        monthly = pd.concat(monthly_records, ignore_index=True).sort_values(
            ["decision_date", "cik"]
        )
    else:
        monthly = pd.DataFrame(columns=["decision_date", "cik"])
    monthly = monthly.reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_parquet(cache_path, index=False)
    return monthly


def statement_reconstruction_checks(
    quarter_candidates: pd.DataFrame,
    monthly_values: pd.DataFrame,
    *,
    coverage_fields: tuple[str, ...] = (
        "revenue_ttm",
        "operating_income_ttm",
        "net_income_ttm",
        "cfo_ttm",
        "total_assets",
        "common_equity",
        "shares_outstanding",
    ),
) -> dict[str, pd.DataFrame]:
    """Summarize reconstruction agreement, coverage, and TTM source methods."""

    source_mix = quarter_candidates.groupby(["field", "source_class"]).size().unstack(fill_value=0)
    paired = quarter_candidates[
        quarter_candidates.duplicated(["cik", "field", "period_end"], keep=False)
    ].pivot_table(
        index=["cik", "field", "period_end"],
        columns="source_class",
        values="value",
        aggfunc="last",
    )
    rows = []
    reconstructed = [column for column in paired if str(column).startswith("reconstructed_")]
    for column in reconstructed:
        if "direct" not in paired:
            continue
        values = paired[["direct", column]].dropna()
        scale = values.abs().max(axis=1).clip(lower=1.0)
        error = (values["direct"] - values[column]).div(scale)
        rows.append(
            {
                "comparison": f"direct vs {column.replace('reconstructed_', '')}",
                "pairs": len(values),
                "median_abs_error": error.abs().median(),
                "p90_abs_error": error.abs().quantile(0.90),
            }
        )
    reconciliation = pd.DataFrame(
        rows,
        columns=["comparison", "pairs", "median_abs_error", "p90_abs_error"],
    ).set_index("comparison")

    fields = [field for field in coverage_fields if field in monthly_values]
    if fields and "decision_date" in monthly_values:
        coverage = (
            monthly_values.assign(year=pd.to_datetime(monthly_values["decision_date"]).dt.year)
            .groupby("year")[fields]
            .agg(lambda values: values.notna().mean())
        )
    else:
        coverage = pd.DataFrame(columns=fields)

    method_columns = [column for column in monthly_values if column.endswith("_ttm_method")]
    if method_columns and not monthly_values.empty:
        latest_date = pd.to_datetime(monthly_values["decision_date"]).max()
        latest = monthly_values[pd.to_datetime(monthly_values["decision_date"]).eq(latest_date)]
        ttm_sources = pd.DataFrame(
            {
                column.removesuffix("_ttm_method"): latest[column].value_counts(normalize=True)
                for column in method_columns
            }
        ).T.fillna(0.0)
    else:
        ttm_sources = pd.DataFrame()

    filed = pd.to_datetime(monthly_values.get("filed_date", pd.Series(dtype="datetime64[ns]")))
    decision = pd.to_datetime(
        monthly_values.get("decision_date", pd.Series(dtype="datetime64[ns]"))
    )
    violations = int((filed.notna() & filed.ge(decision)).sum())
    summary = pd.DataFrame(
        {
            "value": {
                "quarter candidates": len(quarter_candidates),
                "monthly statement rows": len(monthly_values),
                "issuers": monthly_values["cik"].nunique() if "cik" in monthly_values else 0,
                "first decision date": decision.min(),
                "last decision date": decision.max(),
                "point-in-time violations": violations,
            }
        }
    )
    return {
        "summary": summary,
        "reconciliation": reconciliation,
        "quarter_sources": source_mix,
        "coverage": coverage,
        "ttm_sources": ttm_sources,
    }


__all__ = [
    "classify_duration_facts",
    "duration_statement_values",
    "duration_ttm_values",
    "instant_statement_values",
    "issuer_universe",
    "monthly_statement_values",
    "periodic_forms",
    "reconstruct_quarters",
    "select_filing_facts",
    "statement_concepts",
    "statement_reconstruction_checks",
]
