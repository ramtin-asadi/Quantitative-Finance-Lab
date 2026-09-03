from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import build
import companyfacts
import download
import pandas as pd
from bs4 import BeautifulSoup
from event_reviews import DYNAMIC_REVIEWS, MANUAL_REVIEWS

HERE = Path(__file__).resolve().parent
DOCUMENT_CACHE = HERE / "cache" / "bankruptcy_documents"
DOCUMENT_MANIFEST = HERE / "cache" / "bankruptcy_documents_manifest.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def read_manifest() -> dict[str, Any]:
    if not DOCUMENT_MANIFEST.exists():
        return {"files": {}}
    return json.loads(DOCUMENT_MANIFEST.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any]) -> None:
    DOCUMENT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = DOCUMENT_MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(DOCUMENT_MANIFEST)


def document_text(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    text = BeautifulSoup(decoded, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def event_excerpt(text: str, width: int = 1200) -> str:
    lowered = text.lower()
    positions = [
        lowered.find(term)
        for term in ["voluntary petition", "chapter 11", "bankruptcy", "receiver"]
        if lowered.find(term) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - width // 3)
    return text[start : start + width]


def referenced_date(
    text: str, match: re.Match[str] | None, filed_date: pd.Timestamp | None
) -> pd.Timestamp | None:
    if match is None or filed_date is None or pd.isna(filed_date):
        return None
    window = text[max(0, match.start() - 300) : match.end()]
    dates = re.findall(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}\b",
        window,
    )
    if not dates:
        return None
    parsed = pd.to_datetime(dates[-1], errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def classify(
    text: str,
    entity_name: str = "",
    filed_date: pd.Timestamp | None = None,
) -> tuple[str, bool, str, str]:
    lowered = re.sub(r"\s+", " ", text.lower())
    bankruptcy_terms = re.search(
        r"bankrupt|chapter\s+(?:7|11|15)|voluntary petition|involuntary petition|receivership|receiver",
        lowered,
    )
    if not bankruptcy_terms:
        return (
            "metadata_miscoded",
            False,
            "Automatic document review found no bankruptcy, Chapter 7/11/15, petition, or receivership language.",
            "no_bankruptcy_language",
        )

    filed_petition = re.search(
        r"(?:filed|commenced).{0,180}(?:voluntary|involuntary)?\s*(?:petition|case).{0,100}(?:chapter\s+(?:7|11|15)|bankruptcy)",
        lowered,
    ) or re.search(
        r"(?:voluntary|involuntary)\s+petitions?.{0,100}(?:were|was|have been|has been)?\s*filed",
        lowered,
    )
    intent = re.search(
        r"(?:intend|intends|expect|expects|plan|plans|anticipate|anticipates|will|may)\s+to\s+(?:seek|file|commence).{0,120}(?:chapter\s+(?:7|11|15)|bankrupt|petition|case)",
        lowered,
    )
    confirmation = re.search(
        r"(?:bankruptcy\s+)?court\s+(?:has\s+)?entered\s+an?\s+order.{0,120}(?:confirmation|confirm(?:ed|ing))|confirmation order.{0,40}was entered|plan.{0,80}(?:became|becomes) effective|emerg(?:e|ed|ing) from chapter|substantial consummation",
        lowered,
    )
    subsidiary_subject = re.search(
        r"(?:subsidiar(?:y|ies)|affiliate|portfolio company|joint venture).{0,180}(?:filed|commenced).{0,150}(?:petition|chapter\s+(?:7|11|15)|bankrupt)",
        lowered,
    ) or re.search(
        r"(?:filed|commenced).{0,120}(?:by|on behalf of).{0,100}(?:subsidiar(?:y|ies)|affiliate|portfolio company|joint venture)",
        lowered,
    )
    registrant_subject = re.search(
        r"(?:the company|the registrant|we)(?:\s+and\s+(?:certain\s+of\s+)?(?:its|our)\s+subsidiaries)?\s*(?:,|\(|\w|\s){0,100}(?:filed|commenced).{0,150}(?:petition|chapter\s+(?:7|11|15)|bankrupt)",
        lowered,
    )
    company_and_subsidiaries = re.search(
        r"(?:the company|the registrant).{0,80}(?:together with|along with|and).{0,80}subsidiar(?:y|ies).{0,100}(?:filed|commenced)",
        lowered,
    )
    registrant_receiver = re.search(
        r"(?:receiver|receivership).{0,160}(?:the company|the registrant)|(?:the company|the registrant).{0,160}(?:receiver|receivership)",
        lowered,
    )
    normalized_name = re.sub(r"[^a-z0-9]+", " ", entity_name.lower()).strip()
    normalized_name = re.sub(
        r"\b(?:incorporated|corporation|company|holdings?|limited|inc|corp|llc|plc)\b",
        "",
        normalized_name,
    )
    normalized_name = re.sub(r"\s+", " ", normalized_name).strip()
    debtor_filing = re.search(
        r"(?:the\s+)?debtors?.{0,80}(?:filed|commenced).{0,150}(?:petition|chapter\s+(?:7|11|15)|bankrupt)",
        lowered,
    )
    named_debtor_filing = False
    if debtor_filing and normalized_name:
        context = lowered[max(0, debtor_filing.start() - 350) : debtor_filing.end()]
        normalized_context = re.sub(r"[^a-z0-9]+", " ", context)
        named_debtor_filing = normalized_name in normalized_context

    if confirmation:
        return (
            "plan_confirmation_or_emergence",
            False,
            "Automatic document review found a court confirmation, effective plan, or emergence event rather than a new petition.",
            "confirmation_or_emergence",
        )
    if intent and not filed_petition:
        return (
            "announced_intent_to_file",
            False,
            "Automatic document review found prospective bankruptcy-filing language but no completed petition filing.",
            "intent_without_filing",
        )
    registrant_filing = company_and_subsidiaries or registrant_subject
    if registrant_filing is None and named_debtor_filing:
        registrant_filing = debtor_filing
    petition_date = referenced_date(lowered, registrant_filing, filed_date)
    if (
        petition_date is not None
        and filed_date is not None
        and abs((pd.Timestamp(filed_date).normalize() - petition_date).days) > 60
    ):
        return (
            "historical_bankruptcy_reference",
            False,
            f"Automatic document review found a registrant petition dated {petition_date.date()}, more than 60 days from this 8-K filing.",
            "historical_petition_date",
        )
    if registrant_filing is not None:
        return (
            "registrant_bankruptcy_petition",
            True,
            "Automatic document review found an explicit petition/case filing by the Company or registrant.",
            "explicit_registrant_filing",
        )
    if subsidiary_subject:
        return (
            "subsidiary_bankruptcy_petition",
            False,
            "Automatic document review identified a subsidiary, affiliate, portfolio company, or joint venture as the filing party.",
            "explicit_nonregistrant_filing",
        )
    receiver_date = referenced_date(lowered, registrant_receiver, filed_date)
    if (
        receiver_date is not None
        and filed_date is not None
        and abs((pd.Timestamp(filed_date).normalize() - receiver_date).days) > 60
    ):
        return (
            "historical_bankruptcy_reference",
            False,
            f"Automatic document review found a registrant receivership dated {receiver_date.date()}, more than 60 days from this 8-K filing.",
            "historical_receivership_date",
        )
    if registrant_receiver:
        return (
            "registrant_receivership",
            True,
            "Automatic document review found explicit receivership language referring to the Company or registrant.",
            "explicit_registrant_receivership",
        )
    return (
        "unreviewed",
        False,
        "Automatic document review found bankruptcy language but could not identify the registrant as the filing or receivership subject with high confidence.",
        "ambiguous_bankruptcy_language",
    )


def fetch_document(
    client: download.SecClient,
    cik: int,
    accession: str,
    primary_document: str,
    manifest: dict[str, Any],
    refresh: bool,
) -> tuple[str, str, int]:
    compact_accession = accession.replace("-", "")
    url = ARCHIVE_URL.format(
        cik=cik,
        accession=compact_accession,
        document=primary_document,
    )
    path = DOCUMENT_CACHE / f"{accession}.html"
    record = manifest.setdefault("files", {}).get(accession, {})
    if path.exists() and not refresh:
        return document_text(path.read_bytes()), url, 0
    headers = {}
    if path.exists() and record.get("etag"):
        headers["If-None-Match"] = record["etag"]
    if path.exists() and record.get("last_modified"):
        headers["If-Modified-Since"] = record["last_modified"]
    response = client.get(url, headers)
    if response.status_code == 304:
        return document_text(path.read_bytes()), url, 0
    DOCUMENT_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".html.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    manifest["files"][accession] = {
        "url": url,
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "etag": response.headers.get("ETag", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
    }
    transferred = int(response.headers.get("Content-Length", len(response.content)))
    return document_text(response.content), url, transferred


def review(
    identity: str,
    ciks: list[int] | None = None,
    refresh_documents: bool = False,
) -> dict[str, int]:
    selected = sorted(set(ciks if ciks is not None else companyfacts.eligible_ciks()))
    client = download.SecClient(identity)
    manifest = read_manifest()
    rows: list[dict[str, Any]] = []
    stats = {
        "ciks": len(selected),
        "item_1_03_filings": 0,
        "documents_requested": 0,
        "document_failures": 0,
        "transferred_bytes": 0,
    }
    start = pd.Timestamp(companyfacts.DEFAULT_START)
    for number, cik in enumerate(selected, start=1):
        if not (build.SUBMISSIONS / f"CIK{cik:010d}.json").exists():
            continue
        current, entity = build.load_entity(cik)
        if (
            pd.isna(entity["sec_sic"])
            or 6000 <= int(entity["sec_sic"]) <= 6999
            or entity["entity_type"] not in {"", "operating"}
        ):
            continue
        filings, entity = build.load_submission(cik, start, current=current)
        events = filings.loc[filings["is_bankruptcy_or_receivership"]]
        stats["item_1_03_filings"] += len(events)
        for event in events.itertuples(index=False):
            if event.accession in MANUAL_REVIEWS:
                scope, is_registrant, note = MANUAL_REVIEWS[event.accession]
                rule = "manual_document_review"
                url = ARCHIVE_URL.format(
                    cik=cik,
                    accession=event.accession.replace("-", ""),
                    document=event.primary_document,
                )
                excerpt = ""
            elif not event.primary_document:
                scope, is_registrant, note, rule = (
                    "unreviewed",
                    False,
                    "The SEC submission metadata did not identify a primary filing document.",
                    "missing_primary_document",
                )
                url = ""
                excerpt = ""
            else:
                try:
                    text, url, transferred = fetch_document(
                        client,
                        cik,
                        event.accession,
                        event.primary_document,
                        manifest,
                        refresh_documents,
                    )
                    stats["documents_requested"] += 1
                    stats["transferred_bytes"] += transferred
                    scope, is_registrant, note, rule = classify(
                        text, entity["entity_name"], event.filed_date
                    )
                    excerpt = event_excerpt(text)
                except (OSError, RuntimeError, ValueError) as exc:
                    stats["document_failures"] += 1
                    scope, is_registrant, note, rule = (
                        "unreviewed",
                        False,
                        f"Primary-document review failed with {type(exc).__name__}; retry review_events.py.",
                        "document_fetch_error",
                    )
                    url = ARCHIVE_URL.format(
                        cik=cik,
                        accession=event.accession.replace("-", ""),
                        document=event.primary_document,
                    )
                    excerpt = ""
            rows.append(
                {
                    "accession": event.accession,
                    "cik": cik,
                    "entity_name": entity["entity_name"],
                    "filed_date": event.filed_date,
                    "primary_document": event.primary_document,
                    "document_url": url,
                    "bankruptcy_scope": scope,
                    "is_registrant_bankruptcy_event": bool(is_registrant),
                    "bankruptcy_review_note": note,
                    "review_rule": rule,
                    "document_excerpt": excerpt,
                }
            )
        if number % 100 == 0 or number == len(selected):
            print(
                f"event review {number:,}/{len(selected):,} "
                f"events={stats['item_1_03_filings']:,}",
                flush=True,
            )
            write_manifest(manifest)

    fresh = pd.DataFrame.from_records(rows)
    old = pd.read_parquet(DYNAMIC_REVIEWS) if DYNAMIC_REVIEWS.exists() else pd.DataFrame()
    combined = pd.concat([old, fresh], ignore_index=True)
    if not combined.empty:
        combined = combined.sort_values(["filed_date", "accession"]).drop_duplicates(
            "accession", keep="last"
        )
        temporary = DYNAMIC_REVIEWS.with_suffix(".parquet.tmp")
        combined.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(DYNAMIC_REVIEWS)
    write_manifest(manifest)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and conservatively classify SEC Item 1.03 primary documents."
    )
    parser.add_argument("--identity", default=os.getenv("EDGAR_IDENTITY", ""))
    parser.add_argument("--cik", type=int, action="append")
    parser.add_argument("--refresh-documents", action="store_true")
    args = parser.parse_args()
    review(
        identity=download.validate_identity(args.identity),
        ciks=args.cik,
        refresh_documents=args.refresh_documents,
    )


if __name__ == "__main__":
    main()
