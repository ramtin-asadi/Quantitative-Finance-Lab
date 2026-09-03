from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import orjson
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
COMPANYFACTS = ROOT / "data" / "sp500_fundamentals" / "cache" / "edgar" / "companyfacts"
FACT_CACHE = CACHE / "companyfacts"
MANIFEST = CACHE / "companyfacts_manifest.parquet"
DEFAULT_START = "2012-01-01"

PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
FACT_FORMS = PERIODIC_FORMS | {"8-K", "8-K/A"}
ANNUAL_FORMS = {"10-K", "10-K/A"}

CONCEPT_GROUPS = {
    "Assets": "balance_sheet",
    "AssetsCurrent": "balance_sheet",
    "CashAndCashEquivalentsAtCarryingValue": "liquidity",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents": "liquidity",
    "ShortTermInvestments": "liquidity",
    "MarketableSecuritiesCurrent": "liquidity",
    "AccountsReceivableNet": "working_capital",
    "AccountsReceivableNetCurrent": "working_capital",
    "AllowanceForDoubtfulAccountsReceivableCurrent": "working_capital",
    "InventoryNet": "working_capital",
    "AccountsPayableCurrent": "working_capital",
    "AccountsPayableAndAccruedLiabilitiesCurrent": "working_capital",
    "AccruedLiabilitiesCurrent": "working_capital",
    "WorkingCapital": "working_capital",
    "PropertyPlantAndEquipmentNet": "asset_base",
    "Goodwill": "asset_base",
    "IntangibleAssetsNetExcludingGoodwill": "asset_base",
    "Liabilities": "balance_sheet",
    "LiabilitiesCurrent": "balance_sheet",
    "LiabilitiesAndStockholdersEquity": "balance_sheet",
    "StockholdersEquity": "balance_sheet",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "balance_sheet",
    "OtherLiabilitiesCurrent": "balance_sheet",
    "OtherLiabilitiesNoncurrent": "balance_sheet",
    "LongTermDebt": "debt",
    "LongTermDebtCurrent": "debt",
    "LongTermDebtNoncurrent": "debt",
    "ShortTermBorrowings": "debt",
    "ShortTermBankLoansAndNotesPayable": "debt",
    "CommercialPaper": "debt",
    "DebtCurrent": "debt",
    "DebtLongtermAndShorttermCombinedAmount": "debt",
    "LongTermDebtAndFinanceLeaseObligations": "debt",
    "LongTermDebtAndFinanceLeaseObligationsCurrent": "debt",
    "LongTermDebtAndFinanceLeaseObligationsNoncurrent": "debt",
    "SecuredDebt": "debt",
    "UnsecuredDebt": "debt",
    "OperatingLeaseLiability": "leases",
    "OperatingLeaseLiabilityCurrent": "leases",
    "OperatingLeaseLiabilityNoncurrent": "leases",
    "FinanceLeaseLiability": "leases",
    "FinanceLeaseLiabilityCurrent": "leases",
    "FinanceLeaseLiabilityNoncurrent": "leases",
    "LineOfCreditFacilityAmountOutstanding": "credit_facility",
    "LineOfCreditFacilityRemainingBorrowingCapacity": "credit_facility",
    "Revenues": "income_statement",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "income_statement",
    "SalesRevenueNet": "income_statement",
    "CostOfRevenue": "income_statement",
    "GrossProfit": "income_statement",
    "SellingGeneralAndAdministrativeExpense": "income_statement",
    "OperatingIncomeLoss": "income_statement",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "income_statement",
    "IncomeTaxExpenseBenefit": "income_statement",
    "NetIncomeLoss": "income_statement",
    "InterestExpense": "interest",
    "InterestExpenseNonOperating": "interest",
    "InterestExpenseDebt": "interest",
    "InterestIncomeExpenseNonoperatingNet": "interest",
    "InterestPaidNet": "interest",
    "NetCashProvidedByUsedInOperatingActivities": "cash_flow",
    "PaymentsToAcquirePropertyPlantAndEquipment": "cash_flow",
    "DepreciationDepletionAndAmortization": "cash_flow",
    "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment": "cash_flow",
    "ShareBasedCompensation": "cash_flow",
    "PaymentsOfDividends": "cash_outflow",
    "PaymentsForRepurchaseOfCommonStock": "cash_outflow",
    "ProceedsFromIssuanceOfLongTermDebt": "debt_flow",
    "RepaymentsOfLongTermDebt": "debt_flow",
    "ProceedsFromShortTermDebt": "debt_flow",
    "RepaymentsOfShortTermDebt": "debt_flow",
    "RepaymentsOfLinesOfCredit": "debt_flow",
    "PaymentsOfDebtExtinguishmentCosts": "debt_flow",
    "AssetImpairmentCharges": "distress_charge",
    "TangibleAssetImpairmentCharges": "distress_charge",
    "GoodwillImpairmentLoss": "distress_charge",
    "RestructuringCharges": "distress_charge",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths": "debt_maturity",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo": "debt_maturity",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree": "debt_maturity",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour": "debt_maturity",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive": "debt_maturity",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive": "debt_maturity",
    "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths": "lease_maturity",
    "LesseeOperatingLeaseLiabilityPaymentsDueYearTwo": "lease_maturity",
    "LesseeOperatingLeaseLiabilityPaymentsDueYearThree": "lease_maturity",
    "LesseeOperatingLeaseLiabilityPaymentsDueYearFour": "lease_maturity",
    "LesseeOperatingLeaseLiabilityPaymentsDueYearFive": "lease_maturity",
    "LesseeOperatingLeaseLiabilityPaymentsDueAfterYearFive": "lease_maturity",
}

LIABILITY_EQUITY_CONCEPTS = {
    "us-gaap:LiabilitiesAndStockholdersEquity",
    "us-gaap:Liabilities",
    "us-gaap:StockholdersEquity",
    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
}

FACT_COLUMNS = [
    "cik",
    "entity_name",
    "credit_group",
    "concept",
    "label",
    "value",
    "unit",
    "period_type",
    "period_start",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "filed_date",
    "form_type",
    "accession",
    "statement_type",
    "taxonomy",
    "data_quality",
    "confidence_score",
    "is_annual_filing",
    "is_amendment",
    "filing_version",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fact_cache_path(cik: int) -> Path:
    return FACT_CACHE / f"CIK{int(cik):010d}.parquet"


def read_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        return pd.DataFrame()
    return pd.read_parquet(MANIFEST)


def eligible_ciks() -> list[int]:
    manifest = read_manifest()
    if manifest.empty:
        raise FileNotFoundError(f"{MANIFEST} is missing. Run data/sec_credit/download.py first.")
    return sorted(manifest.loc[manifest["eligible"], "cik"].astype(int).unique())


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def safe_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    parsed = parsed.where(parsed.between(pd.Timestamp("1900-01-01"), pd.Timestamp("2100-12-31")))
    return parsed.astype("datetime64[ns]")


def _fact_rows(payload: dict[str, Any], start: str, expected_cik: int) -> pd.DataFrame:
    payload_cik = payload.get("cik")
    if payload_cik is not None and int(payload_cik) != expected_cik:
        raise ValueError(
            f"Company Facts CIK mismatch: file CIK {expected_cik}, payload CIK {payload_cik}"
        )
    cik = expected_cik
    entity_name = str(payload.get("entityName", ""))
    gaap = payload.get("facts", {}).get("us-gaap", {})
    records: list[dict[str, Any]] = []
    for base_concept, group in CONCEPT_GROUPS.items():
        source = gaap.get(base_concept)
        if not source:
            continue
        label = str(source.get("label", ""))
        for unit, observations in source.get("units", {}).items():
            for observation in observations:
                form = str(observation.get("form", "")).upper()
                period_end = str(observation.get("end", ""))
                filed_date = str(observation.get("filed", ""))
                accession = str(observation.get("accn", ""))
                value = observation.get("val")
                if (
                    form not in FACT_FORMS
                    or period_end < start
                    or filed_date < start
                    or not accession
                    or isinstance(value, bool)
                ):
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not np.isfinite(numeric_value):
                    continue
                records.append(
                    {
                        "cik": cik,
                        "entity_name": entity_name,
                        "credit_group": group,
                        "concept": f"us-gaap:{base_concept}",
                        "label": label,
                        "value": numeric_value,
                        "unit": str(unit),
                        "period_type": "duration" if observation.get("start") else "instant",
                        "period_start": observation.get("start"),
                        "period_end": period_end,
                        "fiscal_year": observation.get("fy"),
                        "fiscal_period": str(observation.get("fp", "")),
                        "filed_date": filed_date,
                        "form_type": form,
                        "accession": accession,
                        "statement_type": "",
                        "taxonomy": "us-gaap",
                        "data_quality": "sec_companyfacts_source",
                        "confidence_score": np.float32(1.0),
                        "is_annual_filing": form in ANNUAL_FORMS,
                        "is_amendment": form.endswith("/A"),
                    }
                )
    if not records:
        return pd.DataFrame(columns=FACT_COLUMNS)

    frame = pd.DataFrame.from_records(records)
    for column in ["period_start", "period_end", "filed_date"]:
        frame[column] = safe_datetime(frame[column])
    frame = frame.loc[
        frame["period_end"].notna()
        & frame["filed_date"].notna()
        & frame["period_end"].ge(pd.Timestamp(start))
        & frame["filed_date"].ge(frame["period_end"])
        & (frame["period_start"].isna() | frame["period_start"].le(frame["period_end"]))
    ].copy()
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int32")
    key = [
        "concept",
        "unit",
        "period_start",
        "period_end",
        "filed_date",
        "form_type",
        "accession",
        "value",
    ]
    frame = frame.drop_duplicates(key, keep="last")
    version_key = ["concept", "unit", "period_start", "period_end"]
    frame = frame.sort_values(version_key + ["filed_date", "accession"])
    frame["filing_version"] = (
        frame.groupby(version_key, dropna=False, observed=True).cumcount() + 1
    ).astype("int32")
    return frame[FACT_COLUMNS].reset_index(drop=True)


def _requirements(
    facts: pd.DataFrame, min_periodic_filings: int, min_history_years: float
) -> dict[str, Any]:
    periodic = facts.loc[facts["form_type"].isin(PERIODIC_FORMS)]
    filings = periodic[["accession", "form_type", "filed_date"]].drop_duplicates()
    annual = filings.loc[filings["form_type"].isin(ANNUAL_FORMS), "filed_date"]
    first_filed = filings["filed_date"].min()
    last_filed = filings["filed_date"].max()
    history_years = (
        (last_filed - first_filed).days / 365.25
        if pd.notna(first_filed) and pd.notna(last_filed)
        else 0.0
    )
    asset_periods = facts.loc[
        facts["concept"].eq("us-gaap:Assets"), "period_end"
    ].nunique()
    liability_equity_periods = facts.loc[
        facts["concept"].isin(LIABILITY_EQUITY_CONCEPTS), "period_end"
    ].nunique()
    first_annual = annual.min()
    has_later_periodic = bool(pd.notna(first_annual) and (filings["filed_date"] > first_annual).any())
    eligible = bool(
        len(filings) >= min_periodic_filings
        and history_years >= min_history_years
        and asset_periods >= 4
        and liability_equity_periods >= 4
        and not annual.empty
        and has_later_periodic
    )
    return {
        "eligible": eligible,
        "periodic_filings": int(len(filings)),
        "history_years": float(history_years),
        "asset_periods": int(asset_periods),
        "liability_equity_periods": int(liability_equity_periods),
        "has_10k": bool(not annual.empty),
        "has_later_periodic": has_later_periodic,
    }


def prepare_companyfacts(
    start: str = DEFAULT_START,
    ciks: Iterable[int] | None = None,
    min_periodic_filings: int = 8,
    min_history_years: float = 2.0,
    force: bool = False,
) -> dict[str, int]:
    pd.Timestamp(start)
    files = sorted(COMPANYFACTS.glob("CIK*.json"))
    if len(files) < 10_000:
        raise FileNotFoundError(
            f"The existing Company Facts cache is incomplete ({len(files):,} files). "
            "Run data/sp500_fundamentals/download.py once first."
        )
    selected = {int(cik) for cik in ciks} if ciks is not None else None
    if selected is not None:
        files = [path for path in files if int(path.stem[3:]) in selected]

    old = read_manifest()
    prior = {int(row.cik): row._asdict() for row in old.itertuples(index=False)}
    records = prior.copy()
    stats = {"files": len(files), "parsed": 0, "unchanged": 0, "eligible": 0, "fact_rows": 0}
    started = time.monotonic()
    for number, path in enumerate(files, start=1):
        cik = int(path.stem[3:])
        file_stat = path.stat()
        previous = prior.get(cik, {})
        cache_exists = fact_cache_path(cik).exists()
        unchanged = (
            not force
            and previous
            and str(previous.get("start", "")) == start
            and int(previous.get("source_size", -1)) == file_stat.st_size
            and int(previous.get("source_mtime_ns", -1)) == file_stat.st_mtime_ns
            and (not bool(previous.get("eligible", False)) or cache_exists)
        )
        if unchanged:
            stats["unchanged"] += 1
            stats["eligible"] += int(bool(previous.get("eligible", False)))
            stats["fact_rows"] += int(previous.get("fact_rows", 0))
            continue

        content = path.read_bytes()
        try:
            payload = orjson.loads(content)
            facts = _fact_rows(payload, start, cik)
            requirements = _requirements(facts, min_periodic_filings, min_history_years)
            status = "eligible" if requirements["eligible"] else "ineligible"
            entity_name = str(payload.get("entityName", ""))
        except (orjson.JSONDecodeError, TypeError, ValueError, OSError) as exc:
            facts = pd.DataFrame(columns=FACT_COLUMNS)
            requirements = {
                "eligible": False,
                "periodic_filings": 0,
                "history_years": 0.0,
                "asset_periods": 0,
                "liability_equity_periods": 0,
                "has_10k": False,
                "has_later_periodic": False,
            }
            status = f"parse_error:{type(exc).__name__}"
            entity_name = ""

        if requirements["eligible"]:
            _atomic_parquet(facts, fact_cache_path(cik))
        elif fact_cache_path(cik).exists():
            fact_cache_path(cik).unlink()
        records[cik] = {
            "cik": cik,
            "entity_name": entity_name,
            "status": status,
            **requirements,
            "fact_rows": int(len(facts)) if requirements["eligible"] else 0,
            "source_size": int(file_stat.st_size),
            "source_mtime_ns": int(file_stat.st_mtime_ns),
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "start": start,
            "prepared_at_utc": utc_now(),
        }
        stats["parsed"] += 1
        stats["eligible"] += int(requirements["eligible"])
        stats["fact_rows"] += int(len(facts)) if requirements["eligible"] else 0
        if number % 100 == 0 or number == len(files):
            manifest = pd.DataFrame.from_records(list(records.values())).sort_values("cik")
            _atomic_parquet(manifest, MANIFEST)
            print(
                f"companyfacts {number:,}/{len(files):,} eligible={stats['eligible']:,} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )

    manifest = pd.DataFrame.from_records(list(records.values())).sort_values("cik")
    _atomic_parquet(manifest, MANIFEST)
    return stats
