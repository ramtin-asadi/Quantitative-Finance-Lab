import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from .config import AnalystConfig
from .documents import DocumentStore, chunk_document
from .news import discover_news
from .retrieval import DocumentIndex
from .sec import SecDocuments
from .sources import (
    SourceClient,
    collect_manifest,
    discover_bls,
    discover_cftc,
    discover_eia,
    discover_fed,
    discover_rss,
)


def source_main(source: str, *, update=False):
    parser = argparse.ArgumentParser(description=f"Collect selected {source} evidence for Project 24.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--as-of", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--query", default="(economy OR earnings OR tariff OR oil) sourcelang:english")
    parser.add_argument("--transport", choices=["auto", "doc", "gal"], default="auto")
    parser.add_argument("--minutes", type=int, default=30)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive.")
    identity = os.environ.get("EDGAR_IDENTITY", "")
    config = AnalystConfig.from_repo(args.root)
    existing = {path.stem for path in (config.workspace / "documents").glob("source=*/year=*/*.parquet")}
    client = SourceClient(config.root, source, identity=identity)
    errors = []
    diagnostics = {}
    if source == "sec":
        if not args.ticker:
            parser.error("SEC collection requires explicit --ticker entries; no all-issuer text crawl.")
        sec = SecDocuments(config.root, identity=identity)
        documents = []
        for ticker in args.ticker:
            documents.extend(sec.collect(ticker, as_of=args.as_of, historical=args.historical,
                                         limit=args.limit, refresh=update))
    elif source == "gdelt":
        try:
            documents = discover_news(client, query=args.query, limit=min(args.limit, 250),
                                      transport=args.transport, minutes=args.minutes, diagnostics=diagnostics)
        except (requests.RequestException, ValueError) as error:
            documents = []
            errors.append({"source": source, "error": str(error)})
    else:
        if args.manifest:
            records = json.loads(args.manifest.read_text(encoding="utf-8"))
        elif source == "bls":
            records = discover_bls(client, start=args.start_year, end=args.end_year)
        elif source == "fed" and args.historical:
            records = discover_fed(client, start=args.start_year, end=args.end_year)
        elif source == "fed":
            records = discover_rss(client, "https://www.federalreserve.gov/feeds/press_monetary.xml")
            records += discover_rss(client, "https://www.federalreserve.gov/feeds/speeches.xml")
            records.sort(key=lambda record: record["published_at"])
        elif source == "bea":
            records = discover_rss(client, "https://apps.bea.gov/rss/rss.xml")
            records = [r for r in records if any(word in r["title"].lower() for word in ["gross domestic", "personal income", "outlays"])]
        elif source == "eia":
            records = discover_eia(client, limit=args.limit)
        elif source == "cftc":
            records = discover_cftc(client)
        else:
            parser.error(f"{source} requires --manifest with official URLs and verified availability/position dates.")
        records = records[-args.limit:]
        documents, errors = collect_manifest(config.root, source, records, identity=identity)
    store = DocumentStore(config.workspace / "documents")
    store.put(documents)
    count = len({document.document_id for document in documents} - existing)
    index = DocumentIndex(config.workspace / "index/documents.sqlite")
    indexed = sum(index.add(chunk_document(record)) for record in documents)
    index.close()
    receipt = {"source": source, "new_documents": count, "processed": len(documents), "new_chunks": indexed, "errors": errors}
    if diagnostics:
        receipt["discovery"] = diagnostics
    destination = config.workspace / "updates"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{source}-latest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if errors:
        raise SystemExit(1)
