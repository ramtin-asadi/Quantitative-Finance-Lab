import difflib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from .documents import DocumentStore, bounded_excerpt, parse_html, sections, text_hash
from .evidence import evidence_row
from .retrieval import index_new_documents
from .schemas import DocumentRecord, utc
from .sources import SourceClient

important_items = {"1.01", "1.03", "2.02", "2.03", "2.04", "2.05", "2.06", "5.02", "7.01", "8.01"}


def resolve_ticker(root: Path, ticker: str, *, as_of=None) -> tuple[int, str]:
    from quantfinlab.dataio import read_sec_metadata

    cache = Path(root) / "data/sp500_fundamentals/cache/ticker_cik_mapping.parquet"
    mapping = (pd.read_parquet(cache) if cache.exists() else
               read_sec_metadata(Path(root) / "data/sp500_fundamentals.parquet", include_concepts=False)["mappings"])
    rows = mapping[mapping.ticker.eq(ticker.upper())].copy()
    if as_of is not None:
        date = pd.Timestamp(utc(as_of)).tz_localize(None).normalize()
        mapping_end = pd.to_datetime(mapping.mapping_valid_to).max()
        current_mapping = rows.mapping_valid_to.eq(mapping_end) & rows.mapping_source.str.contains("sec_current_ticker_file", regex=False)
        rows = rows[(rows.mapping_valid_from.isna() | rows.mapping_valid_from.le(date)) &
                    (rows.mapping_valid_to.isna() | rows.mapping_valid_to.ge(date) | (current_mapping & (date >= mapping_end)))]
    if rows.cik.nunique() != 1:
        raise ValueError(f"No unambiguous existing SEC mapping for {ticker} at this cutoff.")
    return int(rows.iloc[-1].cik), str(rows.iloc[-1].entity_name)


def local_filings(root: Path, cik: int) -> pd.DataFrame:
    from quantfinlab.dataio import read_sec_filings

    path = Path(root) / "data/sec_credit.parquet"
    filings = read_sec_filings(path, ciks=[cik])
    primary = pd.read_parquet(path, columns=["accession", "primary_document"],
                              filters=[("record_type", "==", "filing"), ("cik", "==", cik)])
    return filings.merge(primary, on="accession", validate="one_to_one")


def accepted_utc(value) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def select_filings(filings: pd.DataFrame, *, as_of, historical=False, limit=20) -> pd.DataFrame:
    cutoff = utc(as_of)
    rows = filings.copy()
    rows["available_at"] = rows.accepted_at.map(accepted_utc)
    rows = rows[rows.available_at.le(cutoff)].sort_values(["available_at", "accession"])
    periodic = rows[rows.form_type.isin(["10-K", "10-Q"])]
    if historical:
        if len(periodic) > limit:
            positions = [round(i * (len(periodic) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [-1]
            periodic = periodic.iloc[positions]
        return periodic.reset_index(drop=True)
    selected = []
    for form in ["10-K", "10-Q"]:
        choices = periodic[periodic.form_type.eq(form)]
        if choices.empty:
            continue
        latest = choices.iloc[-1]
        selected.append(latest)
        previous = comparable_filing(latest, choices)
        if previous is not None:
            selected.append(previous)
    recent = rows[rows.form_type.eq("8-K") & rows.available_at.ge(cutoff - timedelta(days=90))]
    relevant = recent.form_items.fillna("").map(lambda x: bool(set(re.findall(r"\d\.\d{2}", x)) & important_items))
    selected.extend(row for _, row in recent[relevant].tail(6).iterrows())
    return pd.DataFrame(selected).drop_duplicates("accession").reset_index(drop=True)


def comparable_filing(current: pd.Series, filings: pd.DataFrame):
    prior = filings[filings.form_type.eq(current.form_type) & filings.report_date.lt(current.report_date)].copy()
    gap = (pd.Timestamp(current.report_date) - pd.to_datetime(prior.report_date)).dt.days
    prior = prior[gap.between(300, 430)]
    if prior.empty:
        return None
    return prior.sort_values("report_date").iloc[-1]


class SecDocuments:
    def __init__(self, root: Path, *, identity: str):
        self.root = Path(root)
        self.client = SourceClient(root, "sec", identity=identity)
        self.store = DocumentStore(self.root / "workspace/financial_analyst/documents")

    def refresh_metadata(self, cik: int, filings: pd.DataFrame):
        url = f"https://data.sec.gov/submissions/CIK{cik:010}.json"
        content, _, _ = self.client.fetch(url, refresh=True)
        payload = json.loads(content)
        recent = pd.DataFrame(payload["filings"]["recent"])
        recent = recent.rename(columns={"accessionNumber": "accession", "form": "form_type",
            "reportDate": "report_date", "acceptanceDateTime": "accepted_at", "primaryDocument": "primary_document", "items": "form_items"})
        recent["cik"] = cik
        recent["entity_name"] = payload["name"]
        recent["report_date"] = pd.to_datetime(recent.report_date, errors="coerce")
        recent["accepted_at"] = pd.to_datetime(recent.accepted_at, utc=True).dt.tz_localize(None)
        return pd.concat([filings, recent], ignore_index=True).drop_duplicates("accession", keep="last")

    def download(self, filing: pd.Series, *, ticker: str, entity: str, exhibits=True):
        accession = str(filing.accession)
        folder = f"https://www.sec.gov/Archives/edgar/data/{int(filing.cik)}/{accession.replace('-', '')}/"
        primary = str(filing.primary_document)
        if not primary or primary == "nan":
            raise ValueError(f"Missing primary document for {accession}.")
        urls = [(folder + primary, str(filing.form_type))]
        items = re.findall(r"\d\.\d{2}", str(filing.form_items))
        mapping = pd.read_parquet(self.root / "data/sp500_fundamentals/cache/ticker_cik_mapping.parquet")
        filing_day = pd.Timestamp(accepted_utc(filing.accepted_at)).tz_localize(None).normalize()
        aliases = mapping[mapping.cik.eq(int(filing.cik)) & mapping.mapping_valid_from.le(filing_day) & mapping.mapping_valid_to.ge(filing_day)]
        filing_tickers = sorted(aliases.ticker.dropna().unique().tolist())
        if exhibits and filing.form_type == "8-K" and set(items) & {"2.02", "7.01", "8.01"}:
            content, _, _ = self.client.fetch(folder + accession + "-index.html")
            soup = BeautifulSoup(content, "html.parser")
            for row in soup.select("table.tableFile tr"):
                cells = row.find_all("td")
                if len(cells) >= 4 and cells[3].get_text(strip=True).upper() == "EX-99.1":
                    link = cells[2].find("a", href=True)
                    if link:
                        urls.append((urljoin(folder, link["href"]), "EX-99.1"))
                        break
        documents = []
        for url, form in urls:
            content, provenance, path = self.client.fetch(url)
            text = parse_html(content.decode("utf-8", errors="replace"))
            digest = text_hash(text)
            accepted = accepted_utc(filing.accepted_at)
            document = DocumentRecord(document_id=f"sec-{text_hash(accession + url + digest)[:24]}",
                source="sec", source_type="filing", title=f"{entity} {form} {filing.report_date}",
                entities=[entity], tickers=filing_tickers, cik=int(filing.cik), form=form, items=items,
                accession=accession, report_period=str(pd.Timestamp(filing.report_date).date()) if pd.notna(filing.report_date) else None,
                accepted_at=accepted, available_at=accepted, retrieved_at=utc(provenance["retrieved_at"]),
                source_url=url, raw_path=str(path.relative_to(self.root)), text=text, text_hash=digest,
                duplicate_group=f"sec-{accession}", metadata={"availability_basis": "SEC acceptance timestamp",
                "historical_eligible": True, "raw_hash": provenance["hash"],
                "requested_ticker": ticker, "entity_name_basis": "canonical issuer alias; CIK establishes identity",
                "ticker_mapping_status": "mapped_at_filing" if filing_tickers else "outside_existing_mapping_coverage"})
            self.store.put([document])
            documents.append(document)
        return documents

    def collect(self, ticker: str, *, as_of=None, historical=False, limit=20, refresh=False):
        cutoff = utc(as_of) if as_of else datetime.now(timezone.utc)
        cik, entity = resolve_ticker(self.root, ticker, as_of=cutoff)
        filings = local_filings(self.root, cik)
        if refresh:
            filings = self.refresh_metadata(cik, filings)
        selected = select_filings(filings, as_of=cutoff, historical=historical, limit=limit)
        documents = []
        for _, filing in selected.iterrows():
            documents.extend(self.download(filing, ticker=ticker.upper(), entity=entity))
        return documents


def compare_sections(previous: DocumentRecord, current: DocumentRecord):
    if previous.cik != current.cik or previous.form != current.form:
        raise ValueError("Disclosure comparisons require the same issuer and filing form.")
    if previous.available_at >= current.available_at:
        raise ValueError("Prior filing must precede current filing.")
    old_sections = {re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip(): body for heading, body in sections(previous.text)}
    changes = []
    for heading, body in sections(current.text):
        before = old_sections.pop(re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip(), "")
        old, new = before.split("\n\n"), body.split("\n\n")
        matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for kind, a, b, c, d in matcher.get_opcodes():
            if kind == "equal":
                continue
            left, right = "\n\n".join(old[a:b]), "\n\n".join(new[c:d])
            changes.append({"section": heading, "kind": kind, "before": left, "after": right,
                "before_numbers": re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", left),
                "after_numbers": re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", right),
                "previous_document_id": previous.document_id, "current_document_id": current.document_id,
                "similarity": difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()})
    for heading, body in old_sections.items():
        changes.append({"section": heading, "kind": "delete", "before": body, "after": "",
                        "previous_document_id": previous.document_id, "current_document_id": current.document_id})
    return changes


def compensation_change(previous: DocumentRecord, current: DocumentRecord):
    """Compare an explicitly named unrecognized compensation balance in matched filings."""
    if previous.cik != current.cik or previous.form != current.form or previous.available_at >= current.available_at:
        raise ValueError("Compensation changes require ordered, comparable issuer filings.")
    pattern = re.compile(r"unrecognized\s+(?:stock[- ]based\s+)?compensation\s+(?:expense|cost)[^$]{0,100}\$\s*([\d,.]+)\s*(billion|million)", re.I)
    values = []
    for document in [previous, current]:
        matches = list(pattern.finditer(re.sub(r"\s+", " ", document.text)))
        amounts = {float(match[1].replace(",", "")) * (1000 if match[2].lower() == "billion" else 1) for match in matches}
        if len(amounts) != 1:
            return None
        values.append(amounts.pop())
    before, after = values
    return {"measure": "unrecognized stock-based compensation expense", "unit": "million USD",
            "prior": round(before, 2), "current": round(after, 2), "change": round(after - before, 2),
            "multiple": round(after / before, 2) if before else None,
            "change_percent": round((after / before - 1) * 100, 2) if before else None,
            "previous_document_id": previous.document_id, "current_document_id": current.document_id,
            "interpretation_scope": "Future compensation expense balance; not a realized loss or an executed repurchase."}


def company_filings(root, store, ticker, *, as_of, identity=None, refresh=False, index=None):
    if refresh:
        if not identity:
            raise ValueError("Set an identifying name and email with identity= before requesting SEC downloads.")
        documents = SecDocuments(root, identity=identity).collect(ticker, as_of=utc(as_of), refresh=True)
        if index is not None:
            index_new_documents(index, documents)
    cik, _ = resolve_ticker(root, ticker, as_of=utc(as_of))
    documents = [row for row in store.records(source="sec", as_of=utc(as_of))
                 if row.cik == cik and row.form in {"10-K", "10-Q"} and row.report_period]
    if not documents:
        raise ValueError(f"No filings cached for {ticker}. Run company_filings('{ticker}', refresh=True).")
    current = max(documents, key=lambda row: row.available_at)
    candidates = [row for row in documents if row.form == current.form and row.available_at < current.available_at
                  and 300 <= (pd.Timestamp(current.report_period) - pd.Timestamp(row.report_period)).days <= 430]
    if not candidates:
        raise ValueError(f"No comparable prior-year {current.form} cached for {ticker}. Refresh its selected filings.")
    return max(candidates, key=lambda row: row.available_at), current



def disclosure_relevance(text, question):
    topics = {"cash": r"cash flows?|net cash|operating activities|cash generation",
              "earnings": r"net income|earnings|operating income", "profit": r"margin|profit",
              "margin": r"margin|gross profit", "liquid": r"liquidity|cash flows?|debt",
              "customer": r"customer|concentration", "debt": r"debt|covenant|maturit|borrow",
              "compensation": r"stock.based|share.based|unrecognized compensation",
              "risk": r"risk|export control|contingen|litigation"}
    return sum(bool(re.search(pattern, text, re.I)) for key, pattern in topics.items() if key in question.lower())


def select_change_evidence(previous, current, changes, *, ticker, count, question="", budget=2600):
    terms = re.compile(r"liquidity|risk|margin|cash|debt|customer|concentration|dilut|compensation|capital|guidance|impair", re.I)
    candidates = [row for row in changes if row["section"].casefold() != "document"] or changes
    ranked = sorted(candidates, key=lambda row: (disclosure_relevance(row.get("before", "") + row.get("after", ""), question),
                                             len(set(terms.findall((row.get("before", "") + row.get("after", "")).lower()))),
                                             -row.get("similarity", 1)), reverse=True)
    evidence, selected = [], []
    compensation = compensation_change(previous, current)
    if compensation:
        text = (f"Unrecognized stock-based compensation expense: prior ${compensation['prior']:.2f} million; "
            f"current ${compensation['current']:.2f} million; increase ${compensation['change']:.2f} million. "
            + (f"Current/prior multiple {compensation['multiple']:.2f}; change {compensation['change_percent']:.2f} percent. "
               if compensation["multiple"] is not None else "") + compensation["interpretation_scope"])
        evidence.append(evidence_row(text, key="comparison-" + text_hash(text)[:20],
            available_at=current.available_at, source="structured_context", title="Calculated compensation balance comparison",
            tickers=[ticker], entities=current.entities, url=current.source_url))
    for i, change in enumerate(ranked):
        pair = []
        for side, document in [("before", previous), ("after", current)]:
            text = change.get(side, "")
            if text:
                paragraphs = [paragraph for paragraph in text.split("\n\n") if not re.search(
                    r"purchasing or owning.*securities|risks described in|forward-looking statements|table of contents|safe harbor", paragraph, re.I)]
                if not paragraphs:
                    continue
                paragraphs.sort(key=lambda value: (disclosure_relevance(value, question), len(set(terms.findall(value.lower())))), reverse=True)
                excerpt = "\n\n".join(paragraphs[:3])
                excerpt = bounded_excerpt(excerpt, count, budget=400)
                pair.append(evidence_row(excerpt, key=f"{document.document_id}:change:{i}:{side}",
                    document_id=document.document_id, available_at=document.available_at, source="sec",
                    title=f"{document.title} · {change['section']} · {side}", tickers=[ticker.upper()],
                    entities=document.entities, url=document.source_url))
        if not pair:
            continue
        if count(json.dumps([*evidence, *pair])) > budget:
            break
        evidence.extend(pair)
        selected.append({k: v for k, v in change.items() if k not in {"before", "after"}})
        if len(selected) >= 2:
            break
    return evidence, selected
