"""Filtered readers for saved SEC and credit-market source files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_sec_filings(path: str | Path, *, ciks=None) -> pd.DataFrame:
    """Read filing records without materializing the much larger CompanyFacts slice."""
    columns = ["record_type", "cik", "ticker", "submission_tickers", "entity_name", "sec_sic",
               "sec_industry", "entity_type", "mapping_valid_from", "mapping_valid_to", "accepted_at",
               "filed_date", "report_date", "form_type", "accession", "form_items", "bankruptcy_scope",
               "bankruptcy_review_note", "is_bankruptcy_or_receivership", "is_registrant_bankruptcy_event",
               "is_financial_obligation_trigger", "is_exit_or_disposal", "is_material_impairment",
               "is_delisting_or_listing_failure", "is_nonreliance", "is_late_filing", "is_deregistration"]
    filters = [("record_type", "==", "filing")]
    if ciks is not None:
        filters.append(("cik", "in", list(ciks)))
    data = pd.read_parquet(path, columns=columns, filters=filters)
    for name in ["mapping_valid_from", "mapping_valid_to", "accepted_at", "filed_date", "report_date"]:
        data[name] = pd.to_datetime(data[name])
    if data.duplicated(["cik", "accession"]).any() or data["accepted_at"].isna().any():
        raise ValueError("SEC filing keys or acceptance dates are invalid.")
    return data.sort_values(["accepted_at", "cik"]).reset_index(drop=True)


def read_credit_facts(path: str | Path, *, ciks=None, concepts=None, start="2009-01-01") -> pd.DataFrame:
    """Read filtered facts with acceptance timestamps and original filing dates intact."""
    from .fundamentals import read_sec_facts

    columns = ["cik", "concept", "value", "unit", "period_type", "period_start", "period_end",
               "fiscal_year", "fiscal_period", "filed_date", "accepted_at", "form_type", "accession", "filing_version"]
    facts = read_sec_facts(path, concepts=concepts, ciks=ciks, period_start=start,
                          forms=["10-Q", "10-Q/A", "10-K", "10-K/A"], columns=columns,
                          record_type="fact", validate=False)
    return facts[facts["accepted_at"].notna()].reset_index(drop=True)


def read_credit_market(path: str | Path) -> pd.DataFrame:
    """Read a saved credit benchmark or FINRA table without economic aggregation."""
    data = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    for name in ["date", "report_date", "trade_date"]:
        if name in data:
            data[name] = pd.to_datetime(data[name])
    return data


def read_structured_credit(path: str | Path, *, sheet=None, start=None) -> pd.DataFrame:
    """Read structured-product source rows, retaining status, categories and vintages."""
    filters = []
    if sheet is not None:
        filters.append(("sheet", "==", sheet))
    if start is not None:
        filters.append(("report_date", ">=", pd.Timestamp(start)))
    return pd.read_parquet(path, filters=filters or None)
