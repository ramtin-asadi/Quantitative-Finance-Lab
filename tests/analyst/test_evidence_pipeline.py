from datetime import timedelta

import pandas as pd
import pytest

from quantfinlab.analyst.documents import (
    DocumentStore,
    chunk_document,
    document_date,
    parse_html,
    text_hash,
)
from quantfinlab.analyst.events import candidate_events, event_candidate
from quantfinlab.analyst.macro import release_packet, select_macro_release
from quantfinlab.analyst.news import cluster_headlines
from quantfinlab.analyst.retrieval import DocumentIndex, pack_evidence
from quantfinlab.analyst.routing import query_plan
from quantfinlab.analyst.schemas import AnalysisTarget, Claim, DocumentRecord
from quantfinlab.analyst.sec import (
    accepted_utc,
    comparable_filing,
    compare_sections,
    select_filings,
)
from quantfinlab.analyst.sources import release_time
from quantfinlab.analyst.validation import validate_target


@pytest.fixture
def filing():
    text = "ITEM 1A. RISK FACTORS\n\n" + "Customer concentration creates exposure to changes in purchasing decisions. " * 35
    text += "\n\nITEM 7. MANAGEMENT'S DISCUSSION\n\nOperating margin was 25 percent, compared with 20 percent in the prior year."
    return DocumentRecord(document_id="sec-fixture", source="sec", source_type="filing", title="Example NVDA 10-K",
        entities=["Example Company"], tickers=["NVDA"], cik=42, form="10-K", accession="0001-26-00001",
        available_at="2026-02-01T21:00:00Z", accepted_at="2026-02-01T21:00:00Z",
        retrieved_at="2026-03-01T00:00:00Z", source_url="https://www.sec.gov/Archives/example.htm",
        raw_path="data/sec_documents/raw/fixture.raw", text=text, text_hash=text_hash(text), duplicate_group="accession-fixture")


def test_document_availability_requires_timezone(filing):
    with pytest.raises(ValueError, match="timezone"):
        DocumentRecord.model_validate({**filing.model_dump(), "available_at": "2026-02-01T21:00:00"})
    with pytest.raises(ValueError, match="precedes"):
        DocumentRecord.model_validate({**filing.model_dump(), "available_at": "2026-02-01T20:00:00Z"})


def test_html_removes_hidden_facts_but_preserves_table():
    text = parse_html('<html><body><nav>navigation</nav><ix:hidden>hidden xbrl</ix:hidden><h2>Risk Factors</h2><p>Useful filing evidence.</p><table><tr><th>Revenue</th><td>120</td></tr></table></body></html>')
    assert "hidden xbrl" not in text and "navigation" not in text
    assert "Revenue | 120" in text and "Useful filing evidence." in text


def test_chunks_preserve_sections_and_budget(filing):
    chunks = chunk_document(filing, max_tokens=500)
    assert len(chunks) >= 3
    assert all(chunk.token_count <= 500 for chunk in chunks)
    assert all(chunk.text.startswith(filing.title) for chunk in chunks)
    assert chunks[-1].section.startswith("ITEM 7")
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_partitioned_store_is_idempotent_and_immutable(tmp_path, filing):
    store = DocumentStore(tmp_path)
    assert store.put([filing]) == 1
    assert store.put([filing]) == 0
    assert store.get(filing.document_id) == filing
    assert not list(store.records(as_of="2026-01-31T23:59:59Z"))
    assert len(list(tmp_path.glob("source=sec/year=2026/*.parquet"))) == 1
    changed = filing.model_copy(update={"text_hash": text_hash("changed")})
    with pytest.raises(ValueError, match="immutable"):
        store.put([changed])
    moved = filing.model_copy(update={"available_at": filing.available_at + timedelta(days=366)})
    with pytest.raises(ValueError, match="immutable"):
        store.put([moved])


def test_fts_filters_before_limit_and_deduplicates(tmp_path, filing):
    index = DocumentIndex(tmp_path / "documents.sqlite")
    chunks = chunk_document(filing)
    assert index.add(chunks) == len(chunks)
    assert index.add(chunks) == 0
    assert not index.search("margin", as_of="2026-01-31T23:59:59Z")
    assert not index.search("margin", as_of="2026-02-02T00:00:00Z", ticker="AAPL")
    results = index.search('margin OR "', as_of="2026-02-02T00:00:00Z", ticker="NVDA", sources=["sec"])
    assert results
    packet = pack_evidence(results + results, as_of="2026-02-02T00:00:00Z", budget=900)
    assert packet["tokens"] <= 900
    assert len({e["evidence_id"] for e in packet["evidence"]}) == len(packet["evidence"])
    index.close()


def test_sec_comparable_period_and_utc_contract():
    rows = pd.DataFrame([
        {"form_type": "10-Q", "report_date": "2024-03-31", "accession": "a"},
        {"form_type": "10-Q", "report_date": "2024-12-31", "accession": "b"},
        {"form_type": "10-Q", "report_date": "2025-03-31", "accession": "c"}])
    rows["report_date"] = pd.to_datetime(rows.report_date)
    assert comparable_filing(rows.iloc[-1], rows).accession == "a"
    assert accepted_utc("2026-08-26 20:36:00").hour == 20


def test_sec_selection_is_bounded_and_point_in_time():
    rows = pd.DataFrame([{"cik": 42, "accession": str(i), "form_type": "8-K", "accepted_at": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                          "report_date": pd.Timestamp("2026-01-01"), "form_items": "2.02" if i % 2 else "9.01"} for i in range(100)])
    selected = select_filings(rows, as_of="2026-03-01T23:59:59Z")
    assert len(selected) <= 6 and selected.form_items.eq("2.02").all()
    assert selected.available_at.max() <= pd.Timestamp("2026-03-01T23:59:59Z")


def test_diff_deleted_sections_are_retained(filing):
    current = filing.model_copy(update={"document_id": "current", "available_at": filing.available_at + timedelta(days=366),
                                      "text": "ITEM 1A. RISK FACTORS\n\nNew supplier concentration risk was disclosed."})
    changes = compare_sections(filing, current)
    assert any(row["kind"] == "delete" for row in changes)
    assert any("supplier" in row["after"] for row in changes)


def test_heading_punctuation_is_not_a_disclosure_change(filing):
    prior = filing.model_copy(update={"text": "ITEM 2. MANAGEMENT'S DISCUSSION\n\nCash increased from operations."})
    current = filing.model_copy(update={"document_id": "current", "available_at": filing.available_at + timedelta(days=90),
        "text": "Item 2. Management’s Discussion\n\nCash increased from operations."})
    assert compare_sections(prior, current) == []


def test_dst_release_banner_uses_actual_date():
    value, basis = release_time("Embargoed until 8:30 a.m. Friday, March 4, 2016", source="bls", url="")
    assert value.hour == 13 and basis == "explicit_release_banner"
    value, _ = release_time("Embargoed until 8:30 a.m. Friday, July 8, 2016", source="bls", url="")
    assert value.hour == 12


def test_archived_release_download_is_not_a_new_event(tmp_path):
    old = DocumentRecord(document_id="old", source="bls", source_type="official_release", title="Producer prices",
        available_at="2026-09-12T20:00:00Z", retrieved_at="2026-09-12T20:00:00Z",
        source_url="https://www.bls.gov/news.release/archives/ppi_01132017.htm", raw_path="old.raw",
        text="Producer prices increased by 0.3 percent in December 2016.", text_hash="old", duplicate_group="old",
        metadata={"family": "ppi"})
    assert document_date(old) == "2017-01-13"
    assert not candidate_events([old], as_of="2026-09-14T00:00:00Z", days=7)
    index = DocumentIndex(tmp_path / "index.sqlite")
    index.add(chunk_document(old))
    assert not index.search("producer prices", as_of="2026-09-14T00:00:00Z", since="2026-09-01T00:00:00Z")
    assert index.search("producer prices", as_of="2026-09-14T00:00:00Z")
    index.close()
    newer = old.model_copy(update={"document_id": "new", "source_url": "https://www.bls.gov/news.release/archives/ppi_09102026.htm"})
    older = old.model_copy(update={"document_id": "previous", "source_url": "https://www.bls.gov/news.release/archives/ppi_08142026.htm"})

    class Store:
        def records(self, **kwargs):
            return [old, newer, older]

    current, previous = select_macro_release(Store(), "ppi", as_of="2026-09-14T00:00:00Z")
    assert current.document_id == "new" and previous.document_id == "previous"


def test_release_banner_accepts_comma_after_month():
    value, basis = release_time("Transmission of material is embargoed until USDL 17-0035 8:30 a.m. (EST), Friday, January, 13, 2017",
                               source="bls", url="")
    assert value.isoformat() == "2017-01-13T13:30:00+00:00" and basis == "explicit_release_banner"


def test_news_clusters_and_routing():
    groups = cluster_headlines([{"url":"https://example.com/a?utm_source=x","title":"Company announces earnings"},
                                {"url":"https://example.com/a","title":"Company announces earnings"}])
    assert len(groups) == 1
    plan = query_plan("NVDA 10-Q guidance and credit", tickers=["NVDA"])
    assert "fundamentals" in plan["contexts"] and plan["entities"] == ["NVDA"]


def test_event_triage_does_not_claim_model_analysis(filing):
    candidate = event_candidate(filing)
    assert candidate["status"] == "candidate" and candidate["requires_analysis"]


def test_numbers_must_be_supported_by_cited_evidence(filing):
    evidence = [{"evidence_id":"e1", "text":"Operating margin was 25 percent.", "text_hash":text_hash("Operating margin was 25 percent."),
                 "available_at":filing.available_at.isoformat()}]
    target = AnalysisTarget(conclusion="Margins matter.",materiality="medium",
        claims=[Claim(statement="Margin was 30 percent.",evidence_ids=["e1"],kind="fact")],
        what_changed="Margin changed.",why_it_matters="Profitability affects cash generation.",uncertainty=["No peer data."])
    errors = validate_target(target,{"as_of":"2026-02-02T00:00:00Z","evidence":evidence})
    assert any("Untraceable" in error for error in errors)


def test_macro_packet_rejects_future_expectations(filing):
    with pytest.raises(ValueError,match="Expectation"):
        release_packet(filing,as_of="2026-03-01T00:00:00Z",expectations={"available_at":"2026-02-02T00:00:00Z"})
