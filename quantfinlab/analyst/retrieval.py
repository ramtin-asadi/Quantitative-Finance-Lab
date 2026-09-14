import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .documents import chunk_document, chunk_passage, document_date, estimate_tokens, text_hash
from .schemas import DocumentChunk, utc


def verify_fts5() -> None:
    with sqlite3.connect(":memory:") as connection:
        try:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        except sqlite3.OperationalError as error:
            raise RuntimeError("This Python SQLite build needs FTS5 support.") from error


class DocumentIndex:
    def __init__(self, path: str | Path):
        verify_fts5()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, source TEXT NOT NULL,
                available_at TEXT NOT NULL, duplicate_group TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS chunks_time ON chunks(available_at, source);
            CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
                chunk_id UNINDEXED, title, entity, section, body, source, available_month, content_month, tokenize='unicode61');
        """)
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(chunks)")}
        if "content_date" not in columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN content_date TEXT")
        search_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(search)")}
        if not {"source", "available_month", "content_month"}.issubset(search_columns):
            with self.connection:
                self.connection.execute("CREATE VIRTUAL TABLE search_scoped USING fts5(chunk_id UNINDEXED, title, entity, section, body, source, available_month, content_month, tokenize='unicode61')")
                self.connection.execute("INSERT INTO search_scoped SELECT s.chunk_id,s.title,s.entity,s.section,s.body,c.source,replace(substr(c.available_at,1,7),'-',''),replace(substr(coalesce(c.content_date,c.available_at),1,7),'-','') FROM search s JOIN chunks c ON c.chunk_id=s.chunk_id")
                self.connection.execute("DROP TABLE search")
                self.connection.execute("ALTER TABLE search_scoped RENAME TO search")
        self.connection.execute("CREATE INDEX IF NOT EXISTS chunks_content_date ON chunks(content_date)")

    def close(self):
        self.connection.close()

    def add(self, chunks: Iterable[DocumentChunk]) -> int:
        inserted = 0
        with self.connection:
            for chunk in chunks:
                exists = self.connection.execute("SELECT 1 FROM chunks WHERE chunk_id=?", (chunk.chunk_id,)).fetchone()
                if exists:
                    continue
                self.connection.execute("INSERT INTO chunks(chunk_id,document_id,source,available_at,duplicate_group,payload,content_date) VALUES(?,?,?,?,?,?,?)", (
                    chunk.chunk_id, chunk.document_id, chunk.source, chunk.available_at.isoformat(),
                    chunk.duplicate_group, chunk.model_dump_json(), chunk.content_date))
                self.connection.execute("INSERT INTO search VALUES(?,?,?,?,?,?,?,?)", (
                    chunk.chunk_id, chunk.title, " ".join(chunk.entities + chunk.tickers), chunk.section, chunk.text, chunk.source,
                    chunk.available_at.strftime("%Y%m"), (chunk.content_date or chunk.available_at.isoformat())[:7].replace("-", "")))
                inserted += 1
        return inserted

    def search(self, query: str, *, as_of, sources=None, ticker=None, since=None, sections=None, limit=30):
        terms = re.findall(r"[\w]+", query)
        stopwords = {"what", "which", "the", "a", "an", "and", "or", "is", "are", "in", "of", "to", "for",
                     "with", "this", "that", "how", "does", "do", "it", "its", "from", "by", "as", "be", "on",
                     "changed", "changes", "latest", "including", "between", "tensions", "these", "those",
                     "happened", "matter", "matters", "simple", "interpretation", "news", "release", "report",
                     "document", "material", "financial", "results", "filing", "deserve", "attention"}
        terms = list(dict.fromkeys(word.lower() for word in terms if len(word) >= 3 and word.lower() not in stopwords))[:8]
        if not terms:
            return []
        match = " OR ".join('"' + word + '"' for word in terms)
        if ticker:
            match = 'entity:"' + ticker.upper().replace('"', '""') + '" AND (' + match + ')'
        if sources:
            source_match = " OR ".join('source:"' + source.replace('"', '""') + '"' for source in sources)
            match = '(' + source_match + ') AND (' + match + ')'
        if since is not None:
            first, last = utc(since), utc(as_of)
            months = [f'{month // 12:04d}{month % 12 + 1:02d}' for month in range(first.year * 12 + first.month - 1, last.year * 12 + last.month)]
            if not months:
                return []
            period_match = " OR ".join('"' + month + '"' for month in months)
            match = '(' + match + ') AND available_month:(' + period_match + ') AND content_month:(' + period_match + ')'
        sql = """SELECT c.payload, bm25(search, 0, 4, 5, 3, 1, 0, 0, 0) AS lexical_score
                 FROM search CROSS JOIN chunks c ON c.chunk_id=search.chunk_id
                 WHERE search MATCH ? AND c.available_at<=?"""
        params = [match, utc(as_of).isoformat()]
        if sources:
            sql += " AND c.source IN (" + ",".join("?" for _ in sources) + ")"
            params += list(sources)
        if since is not None:
            sql += " AND c.available_at>=?"
            params.append(utc(since).isoformat())
            sql += " AND (c.content_date IS NULL OR c.content_date>=?)"
            params.append(utc(since).date().isoformat())
        if ticker:
            sql += " AND EXISTS (SELECT 1 FROM json_each(c.payload,'$.tickers') WHERE value=?)"
            params.append(ticker.upper())
        if sections:
            sql += " AND (" + " OR ".join("lower(json_extract(c.payload,'$.section')) LIKE ?" for _ in sections) + ")"
            params.extend("%" + section.lower().replace("%", "").replace("_", " ") + "%" for section in sections)
        sql += " ORDER BY lexical_score, c.chunk_id LIMIT ?"
        params.append(limit)
        deadline = time.monotonic() + 60
        self.connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        try:
            rows = self.connection.execute(sql, params).fetchall()
        finally:
            self.connection.set_progress_handler(None, 0)
        return [(DocumentChunk.model_validate_json(row[0]), row[1]) for row in rows]


def pack_evidence(results, *, as_of, budget=15000, count: Callable[[str], int] = estimate_tokens,
                  max_per_document=3, max_items=16):
    if budget <= 0:
        raise ValueError("Evidence budget must be positive.")
    chosen, seen, used, documents = [], set(), 0, {}
    cutoff = utc(as_of)
    for chunk, _rank in sorted(results, key=lambda x: (x[1], x[0].chunk_id)):
        if chunk.available_at > cutoff or chunk.text_hash in seen:
            continue
        if documents.get(chunk.document_id, 0) >= max_per_document:
            continue
        passage = chunk_passage(chunk)
        evidence = {"evidence_id": chunk.chunk_id, "document_id": chunk.document_id,
                    "available_at": chunk.available_at.isoformat(), "source": chunk.source,
                    "entities": chunk.entities, "tickers": chunk.tickers, "text": passage,
                    "text_hash": text_hash(passage)}
        tokens = count(json.dumps({"as_of": cutoff.isoformat(), "evidence": [*chosen, evidence]}, ensure_ascii=False))
        if tokens > budget:
            continue
        chosen.append(evidence)
        used = tokens
        seen.add(chunk.text_hash)
        documents[chunk.document_id] = documents.get(chunk.document_id, 0) + 1
        if len(chosen) >= max_items:
            break
    return {"as_of": cutoff.isoformat(), "evidence": chosen, "tokens": used, "budget": budget}


def fuse_results(searches, *, rank_constant=60):
    """Reciprocal-rank fusion; negative scores retain the lower-is-better ordering."""
    scores, chunks = {}, {}
    for results in searches:
        seen = set()
        for rank, (chunk, _) in enumerate(results, 1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) - 1 / (rank_constant + rank)
    return sorted([(chunks[key], score) for key, score in scores.items()], key=lambda row: (row[1], row[0].chunk_id))


def retrieve_queries(index, queries, *, as_of, sources=(), ticker=None, since=None, sections=(), limit=30):
    searches = [index.search(query, as_of=as_of, sources=sources, ticker=ticker, since=since,
                             sections=sections, limit=limit) for query in queries]
    return fuse_results(searches)


def index_new_documents(index, documents):
    known = {row[0] for row in index.connection.execute("SELECT DISTINCT document_id FROM chunks")}
    missing_dates = {row[0] for row in index.connection.execute("SELECT DISTINCT document_id FROM chunks WHERE content_date IS NULL")}
    inserted, date_updates = 0, []
    for record in documents:
        if record.document_id in missing_dates:
            date_updates.append((document_date(record), record.document_id))
        if record.document_id not in known:
            inserted += index.add(chunk_document(record))
            known.add(record.document_id)
    if date_updates:
        with index.connection:
            index.connection.executemany("UPDATE chunks SET content_date=? WHERE document_id=?", date_updates)
    return inserted
