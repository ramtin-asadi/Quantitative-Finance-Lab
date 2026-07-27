from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fundamentals.statements import (
    classify_duration_facts,
    duration_statement_values,
    duration_ttm_values,
    instant_statement_values,
    issuer_universe,
    monthly_statement_values,
    reconstruct_quarters,
    select_filing_facts,
    statement_reconstruction_checks,
)


def _fact(
    concept: str,
    value: float,
    period_end: str,
    filed_date: str,
    *,
    period_start: str | None = None,
    period_type: str = "duration",
    unit: str = "USD",
    fiscal_year: int = 2024,
    accession: str = "a",
    filing_version: int = 1,
    form_type: str = "10-Q",
) -> dict:
    return {
        "cik": 1,
        "concept": concept,
        "label": concept,
        "value": value,
        "unit": unit,
        "period_type": period_type,
        "period_start": period_start,
        "period_end": period_end,
        "fiscal_year": fiscal_year,
        "fiscal_period": "Q1",
        "filed_date": filed_date,
        "form_type": form_type,
        "accession": accession,
        "statement_type": "income",
        "taxonomy": "us-gaap",
        "is_annual_filing": form_type.startswith("10-K"),
        "is_amendment": form_type.endswith("/A"),
        "filing_version": filing_version,
    }


def _classified(rows: list[dict]) -> pd.DataFrame:
    return classify_duration_facts(pd.DataFrame(rows))


def test_issuer_universe_keeps_preferred_share_class_and_financials() -> None:
    date = pd.Timestamp("2024-01-31")
    market = pd.DataFrame(
        {
            "date": [date] * 5,
            "ticker": ["GOOG", "GOOGL", "JPM", "PLD", "OLD"],
            "adj_close": [140.0, 139.0, 170.0, 120.0, 10.0],
            "is_sp500_member": [True] * 5,
            "industry": [
                "SERVICES-COMPUTER PROGRAMMING, DATA PROCESSING, ETC.",
                "SERVICES-COMPUTER PROGRAMMING, DATA PROCESSING, ETC.",
                "NATIONAL COMMERCIAL BANKS",
                "REAL ESTATE INVESTMENT TRUSTS",
                "RETAIL",
            ],
        }
    )
    mappings = pd.DataFrame(
        {
            "ticker": ["GOOG", "GOOGL", "JPM", "PLD", "OLD"],
            "cik": [10, 10, 20, 30, 40],
            "entity_name": ["Alphabet", "Alphabet", "JPMorgan", "Prologis", "Old"],
            "mapping_valid_from": pd.to_datetime(["2020-01-01"] * 5),
            "mapping_valid_to": pd.to_datetime(
                ["2030-01-01", "2030-01-01", "2030-01-01", "2030-01-01", "2023-12-31"]
            ),
        }
    )

    universe = issuer_universe(
        market,
        mappings,
        pd.DataFrame({"decision_date": [date]}),
    )

    assert set(universe["ticker"]) == {"GOOG", "JPM"}
    assert not universe.duplicated(["decision_date", "cik"]).any()
    assert universe.set_index("ticker").loc["JPM", "company_type"] == "financial"


def test_classify_duration_facts_maps_concepts_spans_and_units() -> None:
    revenue = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    rows = [
        _fact(revenue, 10.0, "2024-03-31", "2024-04-25", period_start="2024-01-01"),
        _fact(revenue, 21.0, "2024-06-30", "2024-07-25", period_start="2024-01-01"),
        _fact(revenue, 33.0, "2024-09-30", "2024-10-25", period_start="2024-01-01"),
        _fact(
            revenue,
            45.0,
            "2024-12-31",
            "2025-02-20",
            period_start="2024-01-01",
            form_type="10-K",
        ),
        _fact(revenue, 2.0, "2024-01-31", "2024-02-05", period_start="2024-01-01"),
        _fact(
            "us-gaap:Assets",
            100.0,
            "2024-03-31",
            "2024-04-25",
            period_type="instant",
        ),
        _fact(revenue, 99.0, "2024-03-31", "2024-04-25", period_start="2024-01-01", unit="EUR"),
        _fact(
            revenue,
            99.0,
            "2024-03-31",
            "2024-04-25",
            period_start="2024-01-01",
            form_type="8-K",
        ),
    ]

    facts = _classified(rows)

    assert list(facts["duration_class"]) == [
        "irregular",
        "instant",
        "quarter",
        "six_month_ytd",
        "nine_month_ytd",
        "annual",
    ]
    assert set(facts["field"]) == {"revenue", "total_assets"}
    assert len(facts) == 6


def test_select_filing_facts_is_strict_and_prefers_the_primary_concept() -> None:
    primary = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    rows = [
        _fact(
            "us-gaap:Revenues",
            99.0,
            "2024-03-31",
            "2024-04-20",
            period_start="2024-01-01",
            accession="lower-priority",
        ),
        _fact(
            primary,
            10.0,
            "2024-03-31",
            "2024-04-25",
            period_start="2024-01-01",
            accession="first",
        ),
        _fact(
            primary,
            11.0,
            "2024-03-31",
            "2024-05-15",
            period_start="2024-01-01",
            accession="amended",
            filing_version=2,
        ),
    ]
    facts = _classified(rows)

    before_amendment = select_filing_facts(facts, decision_date="2024-05-15")
    after_amendment = select_filing_facts(facts, decision_date="2024-05-16")

    assert before_amendment["value"].tolist() == [10.0]
    assert after_amendment["value"].tolist() == [11.0]


def test_reconstruct_quarters_subtracts_ytd_and_prefers_direct_facts() -> None:
    revenue = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    rows = [
        _fact(revenue, 30.0, "2024-03-31", "2024-04-25", period_start="2024-01-01"),
        _fact(revenue, 70.0, "2024-06-30", "2024-07-25", period_start="2024-01-01"),
        _fact(revenue, 42.0, "2024-06-30", "2024-07-25", period_start="2024-04-01"),
        _fact(revenue, 110.0, "2024-09-30", "2024-10-25", period_start="2024-01-01"),
        _fact(
            revenue,
            160.0,
            "2024-12-31",
            "2025-02-20",
            period_start="2024-01-01",
            form_type="10-K",
        ),
    ]
    selected = select_filing_facts(_classified(rows))

    quarters, candidates = reconstruct_quarters(selected)
    values = quarters.sort_values("period_end")["value"].tolist()
    june = quarters[quarters["period_end"].eq(pd.Timestamp("2024-06-30"))].iloc[0]
    reconstructed_june = candidates[
        candidates["period_end"].eq(pd.Timestamp("2024-06-30"))
        & candidates["source_class"].eq("reconstructed_six_month_ytd")
    ]

    assert values == [30.0, 42.0, 40.0, 50.0]
    assert june["source_class"] == "direct"
    assert reconstructed_june["value"].iloc[0] == 40.0


def test_duration_values_use_four_quarters_and_reconcile_to_ytd_fallback() -> None:
    revenue = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    rows = [
        _fact(
            revenue,
            20.0,
            "2023-09-30",
            "2023-10-25",
            period_start="2023-07-01",
            fiscal_year=2023,
        ),
        _fact(
            revenue,
            25.0,
            "2023-12-31",
            "2024-02-20",
            period_start="2023-10-01",
            fiscal_year=2023,
            form_type="10-K",
        ),
        _fact(
            revenue,
            30.0,
            "2024-03-31",
            "2024-04-25",
            period_start="2024-01-01",
        ),
        _fact(
            revenue,
            35.0,
            "2024-06-30",
            "2024-07-25",
            period_start="2024-04-01",
        ),
        _fact(
            revenue,
            100.0,
            "2023-12-31",
            "2024-02-20",
            period_start="2023-01-01",
            fiscal_year=2023,
            form_type="10-K",
            accession="annual",
        ),
        _fact(
            revenue,
            60.0,
            "2023-06-30",
            "2023-07-25",
            period_start="2023-01-01",
            fiscal_year=2023,
            accession="prior-ytd",
        ),
        _fact(
            revenue,
            70.0,
            "2024-06-30",
            "2024-07-25",
            period_start="2024-01-01",
            accession="current-ytd",
        ),
    ]
    selected = select_filing_facts(_classified(rows))
    fallback = duration_ttm_values(selected)
    quarters, _ = reconstruct_quarters(selected)
    values = duration_statement_values(quarters, selected).iloc[0]

    assert fallback.loc[0, "revenue_ttm_fallback"] == 110.0
    assert fallback.loc[0, "revenue_ttm_fallback_method"] == ("annual + current YTD - prior YTD")
    assert values["revenue_q"] == 35.0
    assert values["revenue_qoq"] == pytest.approx(35.0 / 30.0 - 1.0)
    assert values["revenue_ttm"] == 110.0
    assert values["revenue_ttm_reconciliation"] == pytest.approx(0.0)
    assert values["revenue_ttm_method"] == "four standalone quarters"


def test_instant_statement_values_keep_latest_and_comparable_prior() -> None:
    rows = [
        _fact(
            "us-gaap:Assets",
            100.0,
            "2023-12-31",
            "2024-02-20",
            period_type="instant",
            fiscal_year=2023,
        ),
        _fact(
            "us-gaap:Assets",
            120.0,
            "2024-03-31",
            "2024-05-01",
            period_type="instant",
        ),
    ]
    values = instant_statement_values(select_filing_facts(_classified(rows))).iloc[0]

    assert values["total_assets"] == 120.0
    assert values["total_assets_prior"] == 100.0
    assert values["total_assets_prior_period_end"] == pd.Timestamp("2023-12-31")


def test_monthly_statement_values_enforces_strict_timing_and_owns_cache(tmp_path) -> None:
    facts = _classified(
        [
            _fact(
                "us-gaap:Assets",
                120.0,
                "2024-03-31",
                "2024-05-15",
                period_type="instant",
            )
        ]
    )
    universe = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-05-15", "2024-06-28"]),
            "cik": [1, 1],
        }
    )
    cache = tmp_path / "monthly.parquet"

    monthly = monthly_statement_values(facts, universe, cache=cache)
    cached = monthly_statement_values(facts.iloc[:0], universe, cache=cache)

    assert monthly["decision_date"].tolist() == [pd.Timestamp("2024-06-28")]
    assert monthly.loc[0, "filed_date"] < monthly.loc[0, "decision_date"]
    assert monthly.loc[0, "total_assets"] == 120.0
    assert cached.loc[0, "total_assets"] == 120.0
    assert cached.loc[0, "decision_date"] == pd.Timestamp("2024-06-28")


def test_statement_reconstruction_checks_returns_auditable_tables() -> None:
    candidates = pd.DataFrame(
        {
            "cik": [1, 1],
            "field": ["revenue", "revenue"],
            "period_end": pd.to_datetime(["2024-03-31", "2024-03-31"]),
            "source_class": ["direct", "reconstructed_six_month_ytd"],
            "value": [30.0, 29.0],
        }
    )
    monthly = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-06-28"]),
            "filed_date": pd.to_datetime(["2024-05-01"]),
            "cik": [1],
            "revenue_ttm": [110.0],
            "revenue_ttm_method": ["four standalone quarters"],
        }
    )

    checks = statement_reconstruction_checks(candidates, monthly)

    assert set(checks) == {
        "summary",
        "reconciliation",
        "quarter_sources",
        "coverage",
        "ttm_sources",
    }
    assert checks["summary"].loc["point-in-time violations", "value"] == 0
    assert checks["reconciliation"].loc["direct vs six_month_ytd", "pairs"] == 1
    assert checks["coverage"].loc[2024, "revenue_ttm"] == 1.0
    assert np.isclose(checks["ttm_sources"].loc["revenue", "four standalone quarters"], 1.0)
