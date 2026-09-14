import difflib
import gzip
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .documents import DocumentStore, text_hash
from .schemas import DocumentRecord, utc
from .sources import SourceClient


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_") and k not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(sorted(query)), ""))


def cluster_headlines(articles: list[dict], threshold=0.9):
    groups = []
    for article in sorted(articles, key=lambda x: (x.get("seendate", ""), x["url"])):
        url = canonical_url(article["url"])
        match = next((group for group in groups if url == group[0]["canonical_url"] or
                      difflib.SequenceMatcher(None, article["title"].lower(), group[0]["title"].lower()).ratio() >= threshold), None)
        record = {**article, "canonical_url": url}
        if match is None:
            groups.append([record])
        else:
            match.append(record)
    return groups


def headline_query(query: str):
    tokens = re.findall(r'"[^"]+"|\(|\)|[^\s()]+', query)
    position = 0

    def term():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Incomplete news query.")
        token = tokens[position]
        position += 1
        if token == "(":
            expression = either()
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError("Unclosed news query group.")
            position += 1
            return expression
        if token in {"OR", "AND", ")"}:
            raise ValueError("Expected a search term.")
        if token.lower().startswith("sourcelang:"):
            language = token.split(":", 1)[1].lower()
            if language not in {"english", "en"}:
                raise ValueError("The GAL fallback currently supports sourcelang:english only.")
            return lambda article: article.get("lang", "").lower() in {"en", "eng", "english"}
        if ":" in token or token.startswith("-") or "*" in token:
            raise ValueError("GAL supports words, quoted phrases, AND, OR, parentheses and sourcelang:english; use --transport doc for other operators.")
        pattern = re.compile(r"(?<!\w)" + re.escape(token.strip('"')) + r"(?!\w)", re.I)
        return lambda article: bool(pattern.search(article.get("title", "") + " " + article.get("desc", "")))

    def both():
        nonlocal position
        expressions = [term()]
        while position < len(tokens) and tokens[position] not in {"OR", ")"}:
            if tokens[position] == "AND":
                position += 1
            expressions.append(term())
        return lambda article: all(expression(article) for expression in expressions)

    def either():
        nonlocal position
        expressions = [both()]
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            expressions.append(both())
        return lambda article: any(expression(article) for expression in expressions)

    predicate = either()
    if position != len(tokens):
        raise ValueError("Unexpected news query token.")
    return predicate


def discover_gal(client: SourceClient, *, query: str, limit: int, minutes: int, diagnostics: dict):
    if not 1 <= minutes <= 120:
        raise ValueError("GAL collection is bounded to 1–120 recent one-minute files.")
    predicate = headline_query(query)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=2)
    matches, seen, found = [], set(), 0
    diagnostics.update(transport="gal", search_scope="headline_and_publisher_summary", requested_minutes=minutes,
                       window_end=end.isoformat(), window_start=(end-timedelta(minutes=minutes-1)).isoformat())
    for offset in range(minutes):
        stamp = (end-timedelta(minutes=offset)).strftime("%Y%m%d%H%M00")
        url = f"https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/gal/{stamp}.gal.json.gz"
        try:
            content, provenance, path = client.fetch(url, timeout=(8, 30))
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                continue
            raise
        found += 1
        for line in gzip.decompress(content).decode("utf-8").splitlines():
            article = json.loads(line)
            if not article.get("title") or not article.get("url") or not predicate(article):
                continue
            canonical = canonical_url(article["url"])
            if canonical in seen:
                continue
            seen.add(canonical)
            matches.append(({**article, "batch_time": stamp, "seendate": article.get("date")}, provenance, path))
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break
    diagnostics.update(files_scanned=found, minutes_checked=offset+1, matched_articles=len(matches), limit_reached=len(matches) >= limit)
    if not found:
        raise ValueError("No GAL files were available in the bounded window; retry later or increase --minutes up to 120.")
    return matches


def discover_news(client: SourceClient, *, query: str, limit=100, timespan="1d", transport="auto", minutes=30, diagnostics=None):
    if not 1 <= limit <= 250:
        raise ValueError("GDELT discovery is bounded to 250 records per query.")
    if transport not in {"auto", "doc", "gal"}:
        raise ValueError("Unknown GDELT transport.")
    diagnostics = diagnostics if diagnostics is not None else {}
    matches = None
    if transport != "gal":
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode({"query": query, "mode": "artlist",
            "format": "json", "maxrecords": limit, "timespan": timespan, "sort": "datedesc"})
        try:
            content, provenance, path = client.fetch(url, refresh=True, timeout=(5, 20))
            payload = json.loads(content)
            if "articles" not in payload:
                raise ValueError("GDELT DOC response has no articles field.")
            matches = [(article, provenance, path) for article in payload["articles"]]
            diagnostics.update(transport="doc", search_scope="gdelt_doc_search", timespan=timespan)
        except (requests.RequestException, ValueError) as error:
            if transport == "doc":
                raise
            diagnostics["doc_error"] = str(error)
    if matches is None:
        matches = discover_gal(client, query=query, limit=limit, minutes=minutes, diagnostics=diagnostics)
    articles = [article for article, _, _ in matches]
    origins = {article["url"]: (provenance, path) for article, provenance, path in matches}
    store = DocumentStore(client.root / "workspace/financial_analyst/documents")
    existing = {path.stem for path in store.path.glob("source=gdelt/year=*/*.parquet")}
    records = []
    for group in cluster_headlines(articles):
        article = group[0]
        provenance, path = origins[article["url"]]
        available = utc(provenance["retrieved_at"])
        digest = text_hash(article["title"] + article["canonical_url"])
        text = "News discovery headline: " + article["title"] + "\n\nPublisher URL: " + article["canonical_url"]
        document_id = "gdelt-" + digest[:24]
        if document_id in existing:
            records.append(store.get(document_id))
            continue
        records.append(DocumentRecord(document_id=document_id, source="gdelt", source_type="news_discovery",
            title=article["title"], available_at=available, retrieved_at=available, source_url=article["canonical_url"],
            raw_path=str(path.relative_to(client.root)), text=text, text_hash=text_hash(text), duplicate_group="news-" + digest[:24],
            metadata={"seendate": article.get("seendate"), "cluster_urls": [a["url"] for a in group],
                      "transport": diagnostics["transport"], "search_scope": diagnostics["search_scope"],
                      "gal_batch_time": article.get("batch_time"), "query": query,
                      "historical_eligible": False, "availability_basis": "retrieved discovery metadata",
                      "factual_status": "secondary discovery; verify against primary source"}))
    return records
