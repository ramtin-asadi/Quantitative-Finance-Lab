from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import BOS_EXCLUDED_SERIES, VALET_TARGETS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "boc"
MARKET_OUTPUT = DATA / "canada_boc_market.parquet"
BOS_OUTPUT = DATA / "canada_boc_bos.csv"
MPS_OUTPUT = DATA / "canada_boc_mps.csv"
VALET = "https://www.bankofcanada.ca/valet"
ZERO_PAGE = "https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/"
ZERO_DOWNLOAD = "https://www.bankofcanada.ca/stats/results/csv"
BOS_PAGE = "https://www.bankofcanada.ca/publications/bos/business-outlook-survey-data/"
BOS_GROUP = "Business_Outlook_Survey"
MPS_SEARCH = "https://www.bankofcanada.ca/search/"

MARKET_SCHEMA = pa.schema(
    [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("date", pa.timestamp("ns"), nullable=False),
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("series_name", pa.string(), nullable=False),
        pa.field("maturity_years", pa.float32()),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("frequency", pa.string(), nullable=False),
        pa.field("source_url", pa.string(), nullable=False),
    ]
)


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    client = requests.Session()
    client.headers["User-Agent"] = "Quantitative-Finance-Lab Canada nowcast builder"
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def fetch_json(url: str, path: Path, *, refresh: bool = False) -> dict:
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    response = make_session().get(url, timeout=180)
    response.raise_for_status()
    data = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def resolve_valet_series(*, refresh: bool = False) -> dict[str, dict[str, str]]:
    listing = fetch_json(
        f"{VALET}/lists/series/json",
        CACHE / "series_list.json",
        refresh=refresh,
    )["series"]
    resolved = {}
    for name, target in VALET_TARGETS.items():
        known_id = target.get("known_id")
        if known_id and known_id in listing:
            candidates = [(known_id, listing[known_id])]
        else:
            terms = tuple(term.casefold() for term in target["label_terms"])
            candidates = []
            for series_id, details in listing.items():
                text = f"{details.get('label', '')} {details.get('description', '')}".casefold()
                if all(term in text for term in terms):
                    candidates.append((series_id, details))
        if not candidates:
            raise KeyError(f"Bank of Canada Valet series could not be resolved: {name}")

        def preferred(item: tuple[str, dict]) -> tuple:
            series_id = item[0]
            market_prefix = (
                series_id.startswith("TB.CDN.")
                if name.startswith("t_bill_")
                else series_id.startswith("BD.CDN.")
                if name.startswith("goc_")
                else series_id == "AVG.INTWO"
                if name == "corra"
                else True
            )
            return (
                not market_prefix,
                series_id.startswith(("SAN_", "MPR_", "BOS_", "FSR_")),
                len(series_id),
                series_id,
            )

        candidates.sort(key=preferred)
        series_id, details = candidates[0]
        resolved[name] = {
            "series_id": series_id,
            "label": details.get("label", series_id),
            "description": details.get("description", ""),
        }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "resolved_series.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    )
    return resolved


def valet_unit(name: str) -> str:
    return "CAD per USD" if name == "usd_cad" else "percent"


def fetch_valet(
    resolved: dict[str, dict[str, str]],
    *,
    start_date: pd.Timestamp | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    ids = [details["series_id"] for details in resolved.values()]
    query = f"?start_date={start_date.date()}" if start_date is not None else ""
    url = f"{VALET}/observations/{','.join(ids)}/json{query}"
    cache_name = "valet_full.json" if start_date is None else "valet_increment.json"
    payload = fetch_json(url, CACHE / cache_name, refresh=refresh)
    id_to_name = {details["series_id"]: name for name, details in resolved.items()}
    rows = []
    for observation in payload.get("observations", []):
        date = pd.to_datetime(observation.get("d"), errors="coerce")
        for series_id, value in observation.items():
            if series_id == "d" or series_id not in id_to_name:
                continue
            numeric = pd.to_numeric(value.get("v"), errors="coerce")
            if pd.isna(date) or pd.isna(numeric):
                continue
            name = id_to_name[series_id]
            rows.append(
                {
                    "dataset": "valet_market_series",
                    "date": date,
                    "series_id": name,
                    "series_name": resolved[name]["label"],
                    "maturity_years": pd.NA,
                    "value": float(numeric),
                    "unit": valet_unit(name),
                    "frequency": "business daily",
                    "source_url": url,
                }
            )
    return pd.DataFrame(rows)


def zero_url(start_date: pd.Timestamp | None = None) -> str:
    params = {
        "lookupPage": "lookup_yield_curve.php",
        "startRange": "1986-01-01",
        "searchRange": "all" if start_date is None else "",
        "dFrom": "1986-01-01" if start_date is None else str(start_date.date()),
        "dTo": str(pd.Timestamp.today().normalize().date()),
        "submit": "Submit",
    }
    return f"{ZERO_DOWNLOAD}?{urlencode(params)}"


def fetch_zero_curve(
    *,
    start_date: pd.Timestamp | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_name = "zero_curve_full.csv" if start_date is None else "zero_curve_increment.csv"
    path = CACHE / cache_name
    url = zero_url(start_date)
    if not path.exists() or refresh:
        response = make_session().get(url, timeout=300)
        response.raise_for_status()
        if "text/csv" not in response.headers.get("Content-Type", ""):
            raise ValueError("Bank of Canada zero-coupon response is not CSV.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    wide = pd.read_csv(path, skipinitialspace=True, na_values=["na", "NA"])
    wide.columns = [str(column).strip() for column in wide.columns]
    date_column = wide.columns[0]
    wide[date_column] = pd.to_datetime(wide[date_column], errors="coerce")
    long = wide.melt(id_vars=date_column, var_name="source_series", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=[date_column, "value"])
    maturity = long["source_series"].str.extract(r"ZC(\d{3,4})YR", expand=False)
    long["maturity_years"] = pd.to_numeric(maturity, errors="coerce") / 100
    result = pd.DataFrame(
        {
            "dataset": "government_zero_coupon_curve",
            "date": long[date_column],
            "series_id": long["source_series"].str.strip(),
            "series_name": "Government of Canada zero-coupon yield",
            "maturity_years": long["maturity_years"],
            "value": long["value"],
            "unit": "decimal yield",
            "frequency": "business daily",
            "source_url": url,
        }
    )
    return result


def validate_market(frame: pd.DataFrame) -> pd.DataFrame:
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["maturity_years"] = pd.to_numeric(frame["maturity_years"], errors="coerce")
    frame = frame.dropna(subset=["date", "value"])
    keys = ["dataset", "date", "series_id"]
    frame = frame.sort_values(keys).drop_duplicates(keys, keep="last").reset_index(drop=True)
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Bank of Canada market output is empty or duplicated.")
    curve = frame[frame["dataset"] == "government_zero_coupon_curve"]
    if curve["maturity_years"].nunique() != 120:
        raise ValueError("The Bank of Canada zero-coupon curve does not contain 120 maturities.")
    return frame


def write_market(frame: pd.DataFrame) -> None:
    frame = validate_market(frame)
    table = pa.Table.from_pandas(frame, schema=MARKET_SCHEMA, preserve_index=False, safe=False)
    metadata = {
        b"dataset": b"Bank of Canada policy, FX, money-market and yield-curve data",
        b"source_url": ZERO_PAGE.encode(),
        b"generated_at_utc": datetime.now(timezone.utc).isoformat().encode(),
        b"valet_resolution": b"series identifiers resolved from the live Valet Lists endpoint",
    }
    temporary = MARKET_OUTPUT.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(metadata), temporary, compression="zstd")
    temporary.replace(MARKET_OUTPUT)
    print(
        f"wrote {MARKET_OUTPUT} rows={len(frame):,} "
        f"size_mb={MARKET_OUTPUT.stat().st_size / 1e6:.1f}"
    )


def build_market(*, update: bool = False) -> None:
    resolved = resolve_valet_series(refresh=update)
    if update and MARKET_OUTPUT.exists():
        old = pd.read_parquet(MARKET_OUTPUT)
        starts = {
            dataset: group["date"].max() - pd.Timedelta(days=14)
            for dataset, group in old.groupby("dataset")
        }
        valet = fetch_valet(
            resolved,
            start_date=starts.get("valet_market_series"),
            refresh=True,
        )
        curve = fetch_zero_curve(
            start_date=starts.get("government_zero_coupon_curve"),
            refresh=True,
        )
        fresh = pd.concat([valet, curve], ignore_index=True)
        keep = pd.Series(True, index=old.index)
        for dataset, start in starts.items():
            keep &= ~((old["dataset"] == dataset) & (old["date"] >= start))
        frame = pd.concat([old.loc[keep], fresh], ignore_index=True)
    else:
        frame = pd.concat(
            [
                fetch_valet(resolved, refresh=True),
                fetch_zero_curve(refresh=True),
            ],
            ignore_index=True,
        )
    write_market(frame)


def bos_release_dates(*, refresh: bool = False) -> dict[pd.Timestamp, pd.Timestamp]:
    path = CACHE / "bos_release_dates.json"
    if path.exists() and not refresh:
        stored = json.loads(path.read_text())
        if stored:
            return {pd.Timestamp(key): pd.Timestamp(value) for key, value in stored.items()}
    dates = {}
    client = make_session()
    for page in range(1, 20):
        url = (
            "https://www.bankofcanada.ca/feed/?content_type=bos&"
            f"post_type%5B0%5D=post&post_type%5B1%5D=page&paged={page}"
        )
        response = client.get(url, timeout=60)
        if response.status_code >= 400:
            break
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        if not items:
            break
        for item in items:
            title = item.title.get_text(" ", strip=True) if item.title else ""
            if "business outlook survey" not in title.casefold():
                continue
            date_node = item.find("date")
            published = pd.to_datetime(
                date_node.get_text(strip=True) if date_node else None,
                errors="coerce",
                utc=True,
            )
            if pd.isna(published):
                continue
            release = published.tz_convert(None).normalize()
            quarter = (release - pd.DateOffset(months=1)).to_period("Q").start_time
            dates[quarter] = release
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {str(key.date()): str(value.date()) for key, value in sorted(dates.items())}, indent=2
        )
        + "\n"
    )
    return dates


def build_bos(*, update: bool = False) -> None:
    payload = fetch_json(
        f"{VALET}/observations/group/{BOS_GROUP}/json",
        CACHE / "bos.json",
        refresh=True,
    )
    releases = bos_release_dates(refresh=update)
    details = payload.get("seriesDetail", {})
    rows = []
    for observation in payload.get("observations", []):
        quarter = pd.to_datetime(observation.get("d"), errors="coerce")
        if pd.isna(quarter):
            continue
        for series_id, item in observation.items():
            if series_id == "d" or series_id in BOS_EXCLUDED_SERIES:
                continue
            numeric = pd.to_numeric(item.get("v"), errors="coerce")
            if pd.isna(numeric):
                continue
            metadata = details.get(series_id, {})
            label = metadata.get("label", series_id)
            unit = (
                "percent of firms" if "%" in label or "per cent" in label else "balance of opinion"
            )
            rows.append(
                {
                    "observation_quarter": quarter,
                    "release_date": releases.get(quarter),
                    "series_id": series_id,
                    "question": metadata.get("description", ""),
                    "response": label,
                    "value": float(numeric),
                    "unit": unit,
                    "source_url": BOS_PAGE,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["observation_quarter", "series_id"])
    if frame.empty or frame.duplicated(["observation_quarter", "series_id"]).any():
        raise ValueError("Business Outlook Survey output is empty or duplicated.")
    temporary = BOS_OUTPUT.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(BOS_OUTPUT)
    print(f"wrote {BOS_OUTPUT} rows={len(frame):,} series={frame['series_id'].nunique():,}")


def discover_mps_pages() -> list[str]:
    client = make_session()
    links = set()
    for page in (1, 2, 3):
        params = {"content_type[]": "market-participants-survey"}
        if page > 1:
            params["espage"] = str(page)
        response = client.get(MPS_SEARCH, params=params, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        before = len(links)
        for anchor in soup.find_all("a", href=True):
            url = urljoin(response.url, anchor["href"])
            path = urlparse(url).path.casefold()
            if re.search(r"/20\d{2}/\d{2}/markets?-participants-survey-.*quarter", path):
                links.add(url.split("?")[0])
        if page > 1 and len(links) == before:
            break
    if len(links) < 10:
        raise RuntimeError(f"Only {len(links)} official MPS releases were discovered.")
    return sorted(links)


def cached_html(url: str, *, refresh: bool = False) -> str:
    name = hashlib.sha256(url.encode()).hexdigest()[:16] + ".html"
    path = CACHE / "mps" / name
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    response = make_session().get(url, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return response.text


def flatten_label(value) -> str:
    if isinstance(value, tuple):
        parts = [str(item).strip() for item in value if "unnamed" not in str(item).casefold()]
        return " | ".join(dict.fromkeys(parts))
    return str(value).strip()


def numeric_value(value: str) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text or text.casefold() in {"nan", "na", "n/a", "-"}:
        return None
    match = re.fullmatch(r"(?:US\$|C\$|\$)?\s*(-?\d+(?:\.\d+)?)\s*%?", text)
    return float(match.group(1)) if match else None


def mps_quarter(title: str) -> pd.Timestamp:
    match = re.search(r"(First|Second|Third|Fourth) Quarter of (20\d{2})", title, re.IGNORECASE)
    if not match:
        raise ValueError(f"MPS quarter could not be parsed from {title!r}")
    number = {"first": 1, "second": 2, "third": 3, "fourth": 4}[match.group(1).casefold()]
    return pd.Period(f"{match.group(2)}Q{number}", freq="Q").start_time


def parse_mps_page(url: str, html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else soup.title.get_text(" ", strip=True)
    quarter = mps_quarter(title)
    post_date = soup.select_one(".post-date")
    release = pd.to_datetime(
        post_date.get_text(" ", strip=True) if post_date else None,
        errors="coerce",
    )
    if pd.isna(release):
        match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}",
            soup.get_text(" "),
        )
        release = pd.to_datetime(match.group(0) if match else None, errors="coerce")
    rows = []
    for table_number, table in enumerate(soup.find_all("table"), start=1):
        prior = table.find_previous(["h2", "h3", "h4"])
        question = prior.get_text(" ", strip=True) if prior else ""
        section_heading = prior.find_previous("h2") if prior else None
        section = section_heading.get_text(" ", strip=True) if section_heading else ""
        try:
            frame = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            continue
        frame.columns = [flatten_label(column) for column in frame.columns]
        if frame.empty:
            continue
        stub = frame.columns[0]
        for row_number, (_, record) in enumerate(frame.iterrows(), start=1):
            row_label = flatten_label(record[stub])
            for column in frame.columns[1:]:
                value_text = flatten_label(record[column])
                if value_text.casefold() == "nan":
                    continue
                unit = (
                    "percent"
                    if "%" in value_text
                    or "percent" in question.casefold()
                    or "rate" in question.casefold()
                    else "source text"
                )
                rows.append(
                    {
                        "survey_quarter": quarter,
                        "release_date": release,
                        "section": section,
                        "question": question,
                        "table_number": table_number,
                        "row_number": row_number,
                        "row_label": row_label,
                        "column_label": column,
                        "value_text": value_text,
                        "value_numeric": numeric_value(value_text),
                        "unit": unit,
                        "source_url": url,
                    }
                )
    return rows


def build_mps(*, update: bool = False) -> None:
    rows = []
    pages = discover_mps_pages()
    for url in pages:
        rows.extend(parse_mps_page(url, cached_html(url, refresh=False)))
    frame = pd.DataFrame(rows).sort_values(
        ["survey_quarter", "table_number", "row_number", "column_label"]
    )
    keys = [
        "survey_quarter",
        "table_number",
        "row_number",
        "column_label",
        "source_url",
    ]
    frame = frame.drop_duplicates(keys, keep="last")
    if frame.empty or frame["survey_quarter"].nunique() != len(pages):
        raise ValueError("MPS extraction did not produce one survey quarter per page.")
    temporary = MPS_OUTPUT.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
    temporary.replace(MPS_OUTPUT)
    print(f"wrote {MPS_OUTPUT} rows={len(frame):,} surveys={len(pages):,}")
