from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantfinlab.dataio import read_sec_facts, read_sec_metadata


def _write_sec_fixture(path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    mapping = {
        "ticker": "AAA",
        "cik": 1,
        "entity_name": "Alpha Inc.",
        "mapping_source": "SEC",
        "mapping_confidence": "high",
        "mapping_valid_from": pd.Timestamp("2010-01-01"),
        "mapping_valid_to": pd.Timestamp("2030-12-31"),
    }

    def fact(
        concept: str,
        *,
        period_start,
        period_end: str,
        filed_date: str,
        form_type: str = "10-K",
        value: float = 100.0,
        cik: int = 1,
        accession: str = "0001",
    ) -> dict[str, object]:
        row = {
            **mapping,
            "concept": concept,
            "label": concept.rsplit(":", 1)[-1],
            "value": value,
            "unit": "USD",
            "period_type": "instant" if period_start is None else "duration",
            "period_start": pd.NaT if period_start is None else pd.Timestamp(period_start),
            "period_end": pd.Timestamp(period_end),
            "fiscal_year": 2012,
            "fiscal_period": "FY",
            "filed_date": pd.Timestamp(filed_date),
            "form_type": form_type,
            "accession": accession,
            "statement_type": "BS",
            "taxonomy": "US-GAAP",
            "is_annual_filing": form_type == "10-K",
            "is_amendment": False,
            "filing_version": 1,
        }
        row["cik"] = cik
        row["ticker"] = "BBB" if cik == 2 else "AAA"
        row["entity_name"] = "Beta Inc." if cik == 2 else "Alpha Inc."
        return row

    rows = [
        fact(
            "us-gaap:Assets",
            period_start=None,
            period_end="2012-01-02",
            filed_date="2012-02-01",
            accession="assets-current",
        ),
        fact(
            "us-gaap:Revenues",
            period_start="2011-10-01",
            period_end="2012-01-02",
            filed_date="2012-02-01",
            accession="revenue-current",
        ),
        fact(
            "us-gaap:Revenues",
            period_start="2011-10-01",
            period_end="2012-01-02",
            filed_date="2012-02-01",
            accession="revenue-current",
        ),
        fact(
            "us-gaap:Assets",
            period_start=None,
            period_end="2011-12-31",
            filed_date="2012-02-01",
            accession="assets-old",
        ),
        fact(
            "us-gaap:GrossProfit",
            period_start="2011-10-01",
            period_end="2012-01-02",
            filed_date="2012-02-01",
            accession="other-concept",
        ),
        fact(
            "us-gaap:Assets",
            period_start=None,
            period_end="2012-01-02",
            filed_date="2012-02-01",
            cik=2,
            accession="other-cik",
        ),
        fact(
            "us-gaap:Assets",
            period_start=None,
            period_end="2012-01-02",
            filed_date="2012-02-01",
            form_type="8-K",
            accession="other-form",
        ),
        fact(
            "us-gaap:Assets",
            period_start=None,
            period_end="2012-01-02",
            filed_date="2012-02-01",
            value=np.inf,
            accession="invalid-value",
        ),
        fact(
            "us-gaap:Assets",
            period_start=None,
            period_end="2012-03-01",
            filed_date="2012-02-01",
            accession="early-filing",
        ),
    ]
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    table = table.replace_schema_metadata(
        {
            b"dataset": b"SEC fixture",
            b"validation": json.dumps({"status": "pass", "failures": []}).encode(),
        }
    )
    pq.write_table(table, path)


def test_read_sec_metadata_builds_compact_concept_and_mapping_catalogs(tmp_path) -> None:
    path = tmp_path / "fundamentals.parquet"
    _write_sec_fixture(path)

    result = read_sec_metadata(path, batch_size=2)

    assert result["metadata"]["validation"]["status"] == "pass"
    concepts = result["concepts"]
    assert set(concepts.columns) == {"concept", "label", "unit", "period_type", "rows"}
    assert concepts["rows"].sum() == 9
    mappings = result["mappings"]
    assert len(mappings) == 2
    assert pd.api.types.is_datetime64_any_dtype(mappings["mapping_valid_from"])


def test_read_sec_facts_filters_on_period_end_and_keeps_instant_facts(tmp_path) -> None:
    path = tmp_path / "fundamentals.parquet"
    _write_sec_fixture(path)

    facts = read_sec_facts(
        path,
        concepts=["us-gaap:Assets", "us-gaap:Revenues"],
        ciks=[1],
        forms=["10-k"],
        period_start="2012-01-01",
    )

    assert len(facts) == 2
    assert set(facts["concept"]) == {"us-gaap:Assets", "us-gaap:Revenues"}
    instant = facts.loc[facts["concept"].eq("us-gaap:Assets")].iloc[0]
    assert pd.isna(instant["period_start"])
    assert instant["period_end"] == pd.Timestamp("2012-01-02")
    assert set(facts["form_type"]) == {"10-K"}
    assert set(facts["taxonomy"]) == {"us-gaap"}
    assert np.isfinite(facts["value"]).all()
    assert (facts["filed_date"] >= facts["period_end"]).all()
