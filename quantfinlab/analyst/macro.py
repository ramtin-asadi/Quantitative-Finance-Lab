import json
import re

from .documents import chunk_document, chunk_passage, document_date, text_hash
from .evidence import evidence_row
from .schemas import utc


def release_reaction(prices, *, published_at, as_of, assets=("SPY", "QQQ", "HYG", "TLT", "UUP", "GLD")):
    """Daily close window around publication, without attributing the move to the release."""
    from datetime import timezone
    from zoneinfo import ZoneInfo

    import pandas as pd

    from .context import close_time

    if published_at is None:
        return None
    publication, cutoff = utc(published_at), utc(as_of)
    eligible = prices.loc[[close_time(date) <= cutoff for date in prices.index]].sort_index()
    closes = [pd.Timestamp(date).to_pydatetime().replace(hour=16, minute=0, second=0,
              tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc) for date in eligible.index]
    before = eligible.loc[[stamp < publication for stamp in closes]]
    after = eligible.loc[[stamp >= publication for stamp in closes]]
    if before.empty or after.empty:
        return None
    left, right = before.iloc[-1], after.iloc[0]
    changes = {asset: round(float((right[asset] / left[asset] - 1) * 100), 2)
               for asset in assets if asset in eligible and pd.notna(left[asset]) and pd.notna(right[asset]) and left[asset] != 0}
    return {"published_at": publication.isoformat(), "before_date": str(before.index[-1].date()),
            "after_date": str(after.index[0].date()), "available_at": close_time(after.index[0]).isoformat(),
            "returns_percent": changes,
            "scope": "Daily close-to-close window using regular US session times; other news overlaps this window. These returns do not identify a causal release effect."}


def reaction_evidence(reaction, document):
    if reaction is None or not reaction["returns_percent"]:
        return []
    text = (f"Close-to-close changes around {document.title}, from {reaction['before_date']} to {reaction['after_date']}. "
            + "; ".join(f"{asset}: {value:+.2f} percent" for asset, value in reaction["returns_percent"].items())
            + ". " + reaction["scope"])
    return [evidence_row(text, key="release-window-" + text_hash(text)[:20],
        available_at=reaction["available_at"], source="structured_context", title="Calculated daily release window",
        url=document.source_url)]


def release_packet(current, previous=None, *, as_of, expectations=None, contexts=()):
    cutoff = utc(as_of)
    documents = [record for record in [current, previous] if record is not None]
    if any(record.available_at > cutoff for record in documents):
        raise ValueError("Macro packet includes a future release.")
    if previous is not None and (previous.source != current.source or previous.available_at >= current.available_at):
        raise ValueError("Previous release must precede current release from the same source.")
    if expectations is not None and utc(expectations["available_at"]) > current.available_at:
        raise ValueError("Expectation was not available before the release.")
    for context in contexts:
        if context.as_of > cutoff:
            raise ValueError("Macro packet contains future context.")
    return {"as_of": cutoff.isoformat(), "current": current.model_dump(mode="json"),
            "previous": previous.model_dump(mode="json") if previous else None,
            "expectations": expectations, "contexts": [context.model_dump(mode="json") for context in contexts],
            "surprise_available": expectations is not None}


def labor_release_changes(current, previous):
    """Compare explicitly reported payroll gains and unemployment rates, with source excerpts."""
    if previous is None:
        return []
    if current.source != previous.source or current.available_at <= previous.available_at:
        raise ValueError("Release comparisons require ordered reports from the same source.")
    patterns = {
        "payroll_gain": (r"total nonfarm payroll employment\s+(?:increased|rose|grew)\s+by\s+([\d,]+)", "persons"),
        "unemployment_rate": (r"unemployment rate[^.]{0,75}?\b(?:at|to)\s+(\d+\.\d+)\s+percent", "percent"),
    }
    result = []
    for name, (pattern, unit) in patterns.items():
        matches = [re.search(pattern, re.sub(r"\s+", " ", document.text[:20000]), re.I)
                   for document in [previous, current]]
        if all(matches):
            before, after = [float(match[1].replace(",", "")) for match in matches]
            result.append({"measure": name, "prior": before, "current": after, "unit": unit,
                "difference": round(after - before, 2), "difference_unit": "percentage points" if unit == "percent" else unit,
                "prior_excerpt": matches[0][0], "current_excerpt": matches[1][0],
                "comparison": "Reported headlines in successive releases; not a market-consensus surprise.",
                "previous_document_id": previous.document_id, "current_document_id": current.document_id})
    return result


def select_macro_release(store, family, *, as_of):
    aliases = {"payroll": "empsit", "employment": "empsit", "inflation": "cpi"}
    family = aliases.get(family.lower(), family.lower())
    terms = {"gdp": "gross domestic product", "pce": "personal income and outlays",
             "empsit": "employment situation", "fomc": "fomc", "minutes": "minutes", "cpi": "consumer price"}
    documents = [row for row in store.records(as_of=utc(as_of)) if row.source in {"bls", "bea", "fed"}
                 and (row.metadata.get("family") == family or family in row.title.casefold()
                      or terms.get(family, family) in row.title.casefold())]
    if not documents:
        raise ValueError(f"No cached {family} release is available by this cutoff. Update its official source.")
    current = max(documents, key=lambda row: (document_date(row), row.available_at))
    prior = [row for row in documents if row.source == current.source and document_date(row) < document_date(current)
             and (not row.report_period or row.report_period != current.report_period)]
    previous = max(prior, key=lambda row: (document_date(row), row.available_at)) if prior else None
    return current, previous


def release_evidence(current, previous=None, *, family=None):
    from .corpus import macro_evidence

    family = family or current.metadata.get("family", "")
    evidence = []
    for document in [current, previous]:
        if document is None:
            continue
        selected = macro_evidence(document)
        selected = selected or [chunk_passage(chunk) for chunk in chunk_document(document)[:4]]
        for i, text in enumerate(selected):
            evidence.append(evidence_row(text, key=f"{document.document_id}:release:{i}:{text_hash(text)[:10]}", document_id=document.document_id,
                available_at=document.available_at, source=document.source, title=document.title, url=document.source_url))
    comparisons = labor_release_changes(current, previous) if family == "empsit" else []
    if comparisons:
        text = json.dumps(comparisons, ensure_ascii=False)
        evidence.insert(0, evidence_row(text, key="release-change-" + text_hash(text)[:20],
            available_at=current.available_at, source="structured_context", title="Calculated changes between labor releases",
            url=current.source_url))
    return evidence
