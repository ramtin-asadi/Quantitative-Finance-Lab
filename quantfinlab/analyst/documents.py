import hashlib
import json
import re
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from .schemas import DocumentChunk, DocumentRecord, utc


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return "\n\n".join(re.sub(r"[ \t\r\f\v]+", " ", p).strip()
                       for p in re.split(r"\n\s*\n", text) if p.strip())


def parse_html(html: str, selector: str | None = None) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select('script, style, nav, footer, header, [hidden], ix\\:hidden'):
        tag.decompose()
    body = soup.select_one(selector) if selector else None
    body = body or soup.find("main") or soup.find("article") or soup.body or soup
    for table in list(body.find_all("table")):
        if table.find_parent("table") is not None:
            continue
        rows = [" | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"], recursive=False))
                for row in table.find_all("tr")]
        table.replace_with("\n\n" + "\n".join(r for r in rows if r.strip(" |")) + "\n\n")
    for tag in body.find_all(["p", "div", "h1", "h2", "h3", "h4", "li", "pre", "br"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    return normalize_text(body.get_text(" "))


def sections(text: str) -> list[tuple[str, str]]:
    heading = "Document"
    paragraphs, result = [], []
    pattern = re.compile(r"^(?:part\s+[ivx]+\b|item\s+\d{1,2}(?:\.\d{2}|[a-c])?\b|management.s discussion|risk factors$|liquidity and capital resources$|business$|notes to|note\s+\d+\b|household survey data$|establishment survey data$|food$|energy$|all items less food and energy$|final demand$|job openings$|hires$|separations$)", re.I)
    for paragraph in text.split("\n\n"):
        if len(paragraph) < 220 and pattern.match(paragraph):
            if paragraphs:
                result.append((heading, "\n\n".join(paragraphs)))
            heading, paragraphs = paragraph, []
        else:
            paragraphs.append(paragraph)
    if paragraphs:
        result.append((heading, "\n\n".join(paragraphs)))
    return result


def estimate_tokens(text: str) -> int:
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


def bounded_excerpt(text, count, *, budget=1000):
    if count(text) <= budget:
        return text
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if count(" ".join(words[:middle])) <= budget:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low])


def document_coverage(documents):
    import pandas as pd

    rows = [{"source": record.source, "document_id": record.document_id, "available_at": record.available_at,
             "title": record.title, "form": record.form, "tickers": ", ".join(record.tickers),
             "characters": len(record.text), "report_period": record.report_period}
            for record in documents]
    return pd.DataFrame(rows, columns=["source", "document_id", "available_at", "title", "form", "tickers", "characters", "report_period"])


def document_date(document):
    """Date for relevance and recency; never a replacement for availability."""
    if document.published_at or document.accepted_at:
        return (document.published_at or document.accepted_at).date().isoformat()
    if document.metadata.get("publication_date"):
        return document.metadata["publication_date"]
    match = re.search(r"/archives/\w+_(\d{2})(\d{2})(\d{4})\.htm", document.source_url)
    if document.source == "bls" and match:
        return f"{match[3]}-{match[1]}-{match[2]}"
    match = re.search(r"/archive/\d{4}/(\d{4})_(\d{2})_(\d{2})/", document.source_url)
    if document.source == "eia" and match:
        return "-".join(match.groups())
    if document.source == "cftc" and document.report_period:
        return document.report_period
    return document.available_at.date().isoformat()


def chunk_passage(chunk):
    """Separate the original passage from the search-only heading prefix."""
    heading, separator, passage = chunk.text.partition("\n\n")
    return passage if separator and heading.startswith(chunk.title) else chunk.text


def chunk_document(document: DocumentRecord, count: Callable[[str], int] = estimate_tokens,
                   *, max_tokens: int = 800, token_method: str = "utf8_estimate") -> list[DocumentChunk]:
    chunks = []
    for heading, body in sections(document.text):
        prefix = " | ".join(x for x in [document.title, ", ".join(document.tickers),
                                       document.form, document.report_period, heading] if x)
        if count(prefix) >= max_tokens - 80:
            raise ValueError("Chunk context is longer than the token budget.")
        pieces = []
        for paragraph in body.split("\n\n"):
            if count(prefix + "\n\n" + paragraph) <= max_tokens:
                pieces.append(paragraph)
            else:
                words, part = paragraph.split(), []
                for word in words:
                    if count(prefix + "\n\n" + " ".join([*part, word])) > max_tokens:
                        if not part:
                            raise ValueError("Unbroken source token exceeds chunk budget.")
                        pieces.append(" ".join(part))
                        part = []
                    part.append(word)
                if part:
                    pieces.append(" ".join(part))
        grouped, current = [], []
        for paragraph in pieces:
            if current and count(prefix + "\n\n" + "\n\n".join([*current, paragraph])) > max_tokens:
                grouped.append("\n\n".join(current))
                current = []
            current.append(paragraph)
        if current:
            grouped.append("\n\n".join(current))
        for content in grouped:
            text = prefix + "\n\n" + content
            digest = text_hash(text)
            order = len(chunks)
            chunks.append(DocumentChunk(chunk_id=f"{document.document_id}:{order}:{digest[:10]}",
                document_id=document.document_id, source=document.source, title=document.title,
                entities=document.entities, tickers=document.tickers, form=document.form,
                section=heading, available_at=document.available_at, order=order,
                token_count=count(text), token_method=token_method, text=text, text_hash=digest,
                duplicate_group=document.duplicate_group, content_date=document_date(document)))
    return chunks


class DocumentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records = {}

    def put(self, documents: Iterable[DocumentRecord]) -> int:
        import pyarrow as pa
        import pyarrow.parquet as pq

        written = 0
        for document in documents:
            partition = self.path / f"source={document.source}" / f"year={document.available_at.year}"
            path = partition / f"{document.document_id}.parquet"
            if path.exists() or any(self.path.glob(f"source=*/year=*/{document.document_id}.parquet")):
                old = self.get(document.document_id)
                if old.text_hash != document.text_hash or old.available_at != document.available_at:
                    raise ValueError("An immutable document ID was reused for changed content or availability.")
                continue
            partition.mkdir(parents=True, exist_ok=True)
            row = document.model_dump(mode="json")
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
            schema = pa.schema([(k, pa.list_(pa.string()) if k in {"entities", "tickers", "asset_tags", "items"}
                                 else pa.int64() if k == "cik" else pa.string()) for k in row])
            temporary = path.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist([row], schema=schema), temporary, compression="zstd")
            temporary.replace(path)
            written += 1
        return written

    def records(self, *, source: str | None = None, as_of: datetime | str | None = None):
        import pyarrow.parquet as pq

        cutoff = utc(as_of) if as_of is not None else None
        pattern = f"source={source or '*'}/year=*/*.parquet"
        for path in sorted(self.path.glob(pattern)):
            row = pq.ParquetFile(path).read().to_pylist()[0]
            row["metadata"] = json.loads(row["metadata"])
            record = DocumentRecord.model_validate(row)
            if cutoff is None or record.available_at <= cutoff:
                yield record

    def dataset(self):
        import pyarrow.dataset as ds

        return ds.dataset(self.path, format="parquet", partitioning="hive", exclude_invalid_files=True)

    def get(self, document_id: str) -> DocumentRecord:
        import pyarrow.parquet as pq

        if document_id in self._records:
            return self._records[document_id]
        paths = list(self.path.glob(f"source=*/year=*/{document_id}.parquet"))
        if len(paths) != 1:
            raise KeyError(document_id)
        row = pq.ParquetFile(paths[0]).read().to_pylist()[0]
        row["metadata"] = json.loads(row["metadata"])
        record = DocumentRecord.model_validate(row)
        self._records[document_id] = record
        return record
