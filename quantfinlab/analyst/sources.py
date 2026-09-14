import email.utils
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from .documents import DocumentStore, normalize_text, parse_html, text_hash
from .schemas import DocumentRecord, utc

source_folders = {"sec": "sec_documents", "fed": "fed_documents", "bls": "bls_releases",
                  "bea": "bea_releases", "eia": "eia_energy", "cftc": "cftc_cot", "gdelt": "gdelt_news"}
source_domains = {"sec": {"www.sec.gov", "data.sec.gov"}, "fed": {"www.federalreserve.gov"},
                  "bls": {"www.bls.gov"}, "bea": {"www.bea.gov", "apps.bea.gov"}, "eia": {"www.eia.gov"},
                  "cftc": {"www.cftc.gov"}, "gdelt": {"api.gdeltproject.org", "data.gdeltproject.org", "storage.googleapis.com"}}


class SourceClient:
    def __init__(self, root: Path, source: str, *, identity: str, session=None):
        if "@" not in identity or " " not in identity.strip():
            raise ValueError("Supply an identifying name and contact email.")
        self.root, self.source = Path(root), source
        self.cache = self.root / "data" / source_folders[source] / "raw"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": identity, "Accept-Encoding": "gzip, deflate"})
        self.last_request = 0.0

    def fetch(self, url: str, *, refresh=False, timeout=(15, 60)):
        if urlparse(url).scheme != "https" or urlparse(url).hostname not in source_domains[self.source]:
            raise ValueError("URL does not belong to the selected official source.")
        if urlparse(url).hostname == "storage.googleapis.com" and not urlparse(url).path.startswith("/data.gdeltproject.org/"):
            raise ValueError("Only the official GDELT public bucket is allowed.")
        key = text_hash(url)
        self.cache.mkdir(parents=True, exist_ok=True)
        raw = self.cache / (key + ".raw")
        meta_path = self.cache / (key + ".json")
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if raw.exists() and not refresh:
            return raw.read_bytes(), metadata, raw
        headers = {}
        if raw.exists():
            if metadata.get("etag"):
                headers["If-None-Match"] = metadata["etag"]
            if metadata.get("last_modified"):
                headers["If-Modified-Since"] = metadata["last_modified"]
        for attempt in range(3):
            time.sleep(max(0, 0.25 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            response = self.session.get(url, headers=headers, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            response.raise_for_status()
            break
        if response.status_code == 304:
            return raw.read_bytes(), metadata, raw
        digest = text_hash(response.content.hex())
        now = datetime.now(timezone.utc).isoformat()
        if raw.exists() and metadata.get("hash") == digest:
            return raw.read_bytes(), metadata, raw
        if metadata.get("hash") and metadata["hash"] != digest:
            archive = self.cache / f"{key}.{metadata['hash'][:16]}.raw"
            if raw.exists() and not archive.exists():
                archive.write_bytes(raw.read_bytes())
        temporary = raw.with_suffix(".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(raw)
        metadata = {"url": url, "hash": digest, "retrieved_at": now,
                    "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type", "")}
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return response.content, metadata, raw


def release_time(text: str, *, source: str, url: str) -> tuple[datetime | None, str]:
    if source == "fed" and "fomcminutes" in url:
        return None, "minutes_publication_requires_calendar_or_announcement"
    banner = re.search(r"(?:for release at|embargoed until|transmission of material|embargoed until release at)(.{0,350})", text[:8000], re.I | re.S)
    time_match = re.search(r"(\d{1,2})[:.](\d{2})\s*(a\.?m\.?|p\.?m\.?)", banner[0] if banner else "", re.I)
    dates = re.findall(r"(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{1,2},?\s+20\d{2}", text[:8000], re.I)
    if source == "fed":
        match = re.search(r"/pressreleases/monetary(20\d{6})[a-z]?\.htm", url)
        if match:
            dates = [datetime.strptime(match[1], "%Y%m%d").strftime("%B %d, %Y")]
    if dates and time_match:
        date = datetime.strptime(dates[0].replace(",", ""), "%B %d %Y")
        hour = int(time_match[1]) % 12 + (12 if time_match[3].lower().startswith("p") else 0)
        return date.replace(hour=hour, minute=int(time_match[2]), tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc), "explicit_release_banner"
    return None, "retrieval_time_only"


def ingest_release(client: SourceClient, url: str, *, title=None, published_at=None,
                   report_period=None, source_type="official_release", refresh=False,
                   metadata=None) -> DocumentRecord:
    content, provenance, path = client.fetch(url, refresh=refresh)
    html = content.decode("utf-8", errors="replace")
    if content.startswith(b"%PDF"):
        from io import BytesIO

        from pypdf import PdfReader

        text = normalize_text("\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages))
    else:
        selectors = {"fed": "#content", "bls": "#bodytext", "bea": "article", "cftc": "main", "eia": "main"}
        text = parse_html(html, selectors.get(client.source))
    detected, basis = release_time(text, source=client.source, url=url)
    published = utc(published_at) if published_at else detected
    retrieved = utc(provenance["retrieved_at"])
    available = published or retrieved
    if client.source in {"eia", "cftc"} and published_at is None:
        available, basis = retrieved, "first_observed_public_payload"
    digest = text_hash(text)
    heading = BeautifulSoup(html, "html.parser").find("h1")
    title = title or (heading.get_text(" ", strip=True) if heading else url.rsplit("/", 1)[-1])
    info = dict(metadata or {})
    info.update({"availability_basis": "verified_input" if published_at else basis,
                 "raw_hash": provenance["hash"], "historical_eligible": published is not None and available == published})
    if client.source == "fed" and "fomcminutes" in url and info.get("publication_date"):
        date = datetime.strptime(info["publication_date"], "%Y-%m-%d")
        available = date.replace(hour=23, minute=59, second=59, tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)
        published = None
        info.update(availability_basis="official_calendar_release_date_conservative_end_of_day", historical_eligible=True)
    if client.source == "cftc" and not report_period:
        raise ValueError("CFTC position date is required separately from its public release time.")
    return DocumentRecord(document_id=f"{client.source}-{text_hash(url + digest)[:24]}",
        source=client.source, source_type=source_type, title=title, report_period=report_period,
        published_at=published, available_at=available, retrieved_at=retrieved,
        source_url=url, raw_path=str(path.relative_to(client.root)), text=text, text_hash=digest,
        duplicate_group=f"text-{digest[:24]}", metadata=info)


def discover_bls(client: SourceClient, *, families=("cpi", "empsit", "ppi", "jolts"), start=2016, end=2026):
    records = {}
    for family in families:
        url = f"https://www.bls.gov/bls/news-release/{family}.htm"
        content, _, _ = client.fetch(url, refresh=True)
        soup = BeautifulSoup(content, "html.parser")
        for anchor in soup.find_all("a", href=True):
            link = urljoin(url, anchor["href"])
            match = re.search(rf"/news\.release/archives/{family}_(\d{{2}})(\d{{2}})(\d{{4}})\.htm$", link)
            if match and start <= int(match[3]) <= end:
                records[link] = {"url": link, "family": family, "discovery_date": f"{match[3]}-{match[1]}-{match[2]}"}
    return sorted(records.values(), key=lambda x: (x["discovery_date"], x["url"]))


def discover_fed(client: SourceClient, *, start=2016, end=2026):
    records = {}
    calendars = set()
    for year in range(start, end + 1):
        url = (f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
               if year < datetime.now().year - 5 else "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
        if url in calendars:
            continue
        calendars.add(url)
        try:
            content, _, _ = client.fetch(url, refresh=True)
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                continue
            raise
        for anchor in BeautifulSoup(content, "html.parser").find_all("a", href=True):
            link = urljoin(url, anchor["href"])
            match = re.search(r"(?:monetary|fomcminutes)(20\d{6})[a-z]?\.htm$", link)
            if match and start <= int(match[1][:4]) <= end:
                record = {"url": link, "family": "minutes" if "minutes" in link else "fomc",
                          "discovery_date": datetime.strptime(match[1], "%Y%m%d").date().isoformat()}
                if "fomcminutes" in link:
                    parent = anchor.parent
                    for _ in range(4):
                        text = parent.get_text(" ", strip=True)
                        released = re.search(r"Released\s+(\w+\s+\d{1,2},\s*20\d{2}|\d{2}/\d{2}/20\d{2})", text, re.I)
                        if released:
                            raw_date = released[1]
                            date = datetime.strptime(raw_date, "%B %d, %Y" if "," in raw_date else "%m/%d/%Y")
                            record["publication_date"] = date.date().isoformat()
                            break
                        parent = parent.parent
                        if parent is None:
                            break
                records[link] = record
    return sorted(records.values(), key=lambda x: (x.get("publication_date", x["discovery_date"]), x["url"]))


def discover_rss(client: SourceClient, url: str):
    content, _, _ = client.fetch(url, refresh=True)
    soup = BeautifulSoup(content, "xml")
    rows = []
    for item in soup.find_all("item"):
        link, date = item.find("link"), item.find("pubDate")
        if link is None or date is None:
            continue
        rows.append({"url": link.get_text(strip=True), "title": item.title.get_text(strip=True),
                     "published_at": email.utils.parsedate_to_datetime(date.get_text()).isoformat()})
    return sorted(rows, key=lambda row: utc(row["published_at"]))


def discover_eia(client: SourceClient, *, limit=8):
    url = "https://www.eia.gov/petroleum/supply/weekly/archive/"
    content, _, _ = client.fetch(url, refresh=True)
    links = sorted({urljoin(url, a["href"]) for a in BeautifulSoup(content, "html.parser").find_all("a", href=True)
                    if re.search(r"/weekly/archive/20\d{2}/", a["href"])})
    records = []
    for link in links[-limit:]:
        content, _, _ = client.fetch(link)
        soup = BeautifulSoup(content, "html.parser")
        highlights = next((urljoin(link, a["href"]) for a in soup.find_all("a", href=True)
                           if "highlights.pdf" in a["href"]), None)
        if highlights:
            records.append({"url": highlights, "title": "Weekly Petroleum Status Report highlights", "family": "petroleum"})
    return records


def discover_cftc(client: SourceClient):
    url = "https://www.cftc.gov/dea/futures/deanymesf.htm"
    content, _, _ = client.fetch(url, refresh=True)
    text = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
    match = re.search(r"AS OF\s+(\d{2}/\d{2}/\d{2})", text, re.I)
    if match is None:
        raise ValueError("CFTC report has no verifiable position date.")
    return [{"url": url, "title": "CFTC NYMEX futures positioning", "family": "positioning",
             "report_period": datetime.strptime(match[1], "%m/%d/%y").date().isoformat(),
             "source_type": "position_report"}]


def collect_manifest(root: Path, source: str, records: list[dict], *, identity: str):
    client = SourceClient(root, source, identity=identity)
    store = DocumentStore(Path(root) / "workspace/financial_analyst/documents")
    existing = {path.stem for path in store.path.glob(f"source={source}/year=*/*.parquet")}
    errors, documents = [], []
    for record in records:
        options = {k: v for k, v in record.items() if k in {"title", "published_at", "report_period", "source_type"}}
        try:
            metadata = {"family": record.get("family", source)}
            if record.get("publication_date"):
                metadata["publication_date"] = record["publication_date"]
            doc = ingest_release(client, record["url"], metadata=metadata, **options)
            if doc.document_id in existing:
                doc = store.get(doc.document_id)
            store.put([doc])
            documents.append(doc)
        except (requests.RequestException, ValueError) as error:
            errors.append({"url": record["url"], "error": str(error)})
    return documents, errors
