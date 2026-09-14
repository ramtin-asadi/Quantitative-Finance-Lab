import re
import sqlite3
from datetime import timedelta
from pathlib import Path

from .caching import load_response
from .documents import chunk_document, chunk_passage, document_date, text_hash
from .evidence import evidence_row
from .reports import unique_passages
from .schemas import DocumentRecord, EventRecord, utc


def event_candidate(document: DocumentRecord) -> dict | None:
    families = {"cpi": "inflation", "ppi": "inflation", "empsit": "labor", "jolts": "labor",
                "gdp": "growth", "pce": "inflation", "fomc": "monetary_policy", "minutes": "monetary_policy"}
    item_family = {"1.03": "bankruptcy_distress", "2.02": "corporate_results", "2.03": "financing",
        "2.04": "bankruptcy_distress", "2.05": "restructuring", "2.06": "impairment", "5.02": "management",
        "1.01": "material_agreement", "7.01": "guidance", "8.01": "other_material"}
    family = families.get(document.metadata.get("family"))
    triggers = [item for item in document.items if item in item_family]
    if triggers:
        family = item_family[triggers[0]]
    if family is None and document.form in {"10-K", "10-Q"}:
        family = "corporate_results"
    if family is None and document.source == "fed":
        family = "monetary_policy"
    if family is None and document.source == "bea":
        family = "growth" if "gross domestic" in document.title.lower() else "income_and_inflation"
    if family is None and document.source in {"eia", "cftc"}:
        family = "energy" if document.source == "eia" else "positioning"
    if family is None and document.source == "gdelt":
        terms = re.findall(r"\b(?:war|sanctions|tariff|bankruptcy|earthquake|oil|strike)\b", document.title, re.I)
        if terms:
            family, triggers = "global_market_discovery", terms
    if family is None:
        return None
    return {"document_id": document.document_id, "available_at": document.available_at.isoformat(),
            "content_date": document_date(document),
            "family": family, "triggers": triggers, "status": "candidate", "requires_analysis": True}


class EventStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, available_at TEXT, payload TEXT)")

    def put(self, event: EventRecord):
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO events VALUES(?,?,?)",
                (event.event_id, event.available_at.isoformat(), event.model_dump_json()))

    def list(self, *, as_of, since, model=None, prompt=None):
        rows = self.connection.execute("SELECT payload FROM events WHERE available_at<=? AND available_at>=? ORDER BY available_at DESC, rowid ASC",
                                       (utc(as_of).isoformat(), utc(since).isoformat()))
        events = [EventRecord.model_validate_json(row[0]) for row in rows]
        events = [event for event in events if (model is None or event.model_version == model)
                  and (prompt is None or event.prompt_version == prompt)]
        unique = {}
        for event in sorted(events, key=lambda item: item.metadata.get("analysis_as_of", "")):
            unique[event.metadata.get("origin_document_id", event.event_id)] = event
        events = list(unique.values())
        priority = {"high": 3, "medium": 2, "low": 1}
        return sorted(events, key=lambda event: (priority[event.importance], event.available_at), reverse=True)

    def close(self):
        self.connection.close()


def candidate_events(documents, *, as_of, days=7):
    candidates = []
    cutoff = utc(as_of)
    for document in documents:
        recent = (cutoff - timedelta(days=days)).date().isoformat() <= document_date(document) <= cutoff.date().isoformat()
        if recent and document.available_at <= cutoff:
            candidate = event_candidate(document)
            if candidate:
                candidates.append({**candidate, "source": document.source, "title": document.title,
                                   "duplicate_group": document.duplicate_group, "form": document.form})
    unique = {}
    for row in sorted(candidates, key=lambda row: row["form"] == "EX-99.1", reverse=True):
        unique.setdefault(row["duplicate_group"], row)
    return sorted(unique.values(), key=lambda row: (row["source"] != "gdelt", row["content_date"], row["available_at"]), reverse=True)


def event_evidence(document, *, max_chunks=6):
    if document.source == "cftc":
        records = positioning_records(document)
        return [evidence_row(
            f"{row['contract']}; position date {document.report_period}. Open interest {row['open_interest']:,} contracts. "
            f"Non-commercial long positions {row['long']:,}, short positions {row['short']:,}, "
            f"and spreading positions {row['spreading']:,} contracts. "
            f"Calculated non-commercial net position (long minus short): {row['net']:,} contracts. "
            f"Weekly change in that calculated net position: {row['net_change']:+,} contracts. "
            "Positions are measured on the report date; the availability timestamp records when this payload was observed.",
            key=f"{document.document_id}:position:{row['code']}", document_id=document.document_id,
            available_at=document.available_at, source="cftc", title=document.title, url=document.source_url)
            for row in records[:max_chunks]]
    if document.source == "eia":
        narrative = re.split(r"\nHighlights\s*\n|\nRefinery Activity", document.text)[0]
        narrative = re.sub(r"^.*?(?=For the week ending|U\.S\. crude oil refinery inputs)", "", narrative, flags=re.S)
        pieces = re.split(r"\n(?=Crude oil imports|Commercial crude oil|Over the past four|The price for|The national|U\.S\. crude oil imports|U\.S\. commercial|Total products supplied)", narrative)
        return [evidence_row(re.sub(r"\s+", " ", text).strip(), key=f"{document.document_id}:narrative:{i}",
            document_id=document.document_id, available_at=document.available_at, source="eia", title=document.title,
            url=document.source_url) for i, text in enumerate(pieces[:max_chunks]) if len(text.strip()) > 50]
    if document.source in {"bls", "bea", "fed"}:
        from .corpus import macro_evidence

        paragraphs = macro_evidence(document)
        if paragraphs:
            return [evidence_row(text, key=f"{document.document_id}:event:{i}:{text_hash(text)[:10]}",
                document_id=document.document_id, available_at=document.available_at, source=document.source,
                title=document.title, tickers=document.tickers, entities=document.entities, url=document.source_url)
                for i, text in enumerate(paragraphs[:max_chunks])]
    return [evidence_row(chunk_passage(chunk), key=chunk.chunk_id, document_id=document.document_id,
                available_at=document.available_at, source=document.source, title=document.title,
                tickers=document.tickers, entities=document.entities, url=document.source_url)
            for chunk in chunk_document(document)[:max_chunks]]


def positioning_records(document):
    """Label the legacy COT futures rows and calculate directional net positions."""
    pattern = r"(?m)^([^\n]+?)\s+Code-([\w]+)\s*\nFUTURES ONLY POSITIONS AS OF"
    matches = list(re.finditer(pattern, document.text))
    selected, codes = [], {"067651", "023651", "022651", "111659"}
    for i, match in enumerate(matches):
        if match[2] not in codes:
            continue
        block = document.text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(document.text)]
        interest = re.search(r"OPEN INTEREST:\s*([\d,]+)", block)
        commitments = re.search(r"\bCOMMITMENTS\s*\n([^\n]+)", block)
        changes = re.search(r"CHANGES FROM[^\n]*\n([^\n]+)", block)
        if not interest or not commitments or not changes:
            continue
        values = [int(value.replace(",", "")) for value in re.findall(r"-?\d[\d,]*", commitments[1])]
        deltas = [int(value.replace(",", "")) for value in re.findall(r"-?\d[\d,]*", changes[1])]
        if len(values) != 9 or len(deltas) != 9:
            continue
        selected.append({"contract": match[1].strip(), "code": match[2], "open_interest": int(interest[1].replace(",", "")),
            "long": values[0], "short": values[1], "spreading": values[2], "net": values[0] - values[1],
            "net_change": deltas[0] - deltas[1]})
    return selected


def event_from_report(report, document, *, model_sha, prompt_version):
    candidate = event_candidate(document)
    if not report.validated or candidate is None:
        return None
    answer = report.analysis
    return EventRecord(event_id="event-" + report.cache_key[:24],
        available_at=max(utc(row["available_at"]) for row in report.sources),
        family=candidate["family"], event_type=candidate["family"], entities=document.entities,
        tickers=document.tickers, asset_tags=document.asset_tags, what_happened=answer.conclusion,
        what_changed=answer.what_changed, importance=answer.materiality if answer.materiality != "uncertain" else "low",
        document_ids=sorted({row["document_id"] for row in report.sources}),
        evidence_ids=sorted({row["evidence_id"] for row in report.sources}),
        model_version=model_sha, prompt_version=prompt_version,
        metadata={"response_key": report.cache_key, "why_it_matters": answer.why_it_matters,
                  "content_date": document_date(document),
                  "uncertainty": answer.uncertainty, "discovery_only": document.source == "gdelt",
                  "origin_document_id": document.document_id, "analysis_as_of": report.as_of})


def events_to_evidence(events, response_folder):
    evidence = []
    for event in events:
        path = Path(response_folder) / (event.metadata["response_key"] + ".json")
        saved = load_response(path)
        if saved is None or not saved.validated:
            raise ValueError("An event is missing its validated analysis record.")
        facts = [claim.statement for claim in saved.analysis.claims if claim.kind == "fact"]
        if not facts:
            continue
        text = " ".join(unique_passages(facts))
        evidence.append(evidence_row(text, key=event.event_id, available_at=event.available_at,
            source="gdelt" if event.metadata.get("discovery_only") else "validated_event",
            title=event.family.replace("_", " "), tickers=event.tickers, entities=event.entities,
            url=saved.sources[0].get("source_url", "") if saved.sources else ""))
    return evidence
