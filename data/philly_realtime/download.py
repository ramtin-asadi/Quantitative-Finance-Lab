from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
STATE = CACHE / "sources.json"
VINTAGE_OUTPUT = DATA / "philly_realtime_vintages.parquet"
RELEASE_OUTPUT = DATA / "philly_first_second_third.parquet"
BASE = "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/"
VARIABLES = [
    "routput", "rcon", "rinvbf", "rinvresid", "rex", "rimp", "rconm",
    "pcpi", "pcpix", "pconx", "employ", "h", "ipt", "ipm", "hstarts",
]


def discover() -> list[dict[str, str]]:
    sources = []
    session = requests.Session()
    for variable in VARIABLES:
        page = BASE + variable
        response = session.get(page, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        vintage_candidates = []
        release_candidates = []
        for anchor in soup.find_all("a", href=True):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            url = urljoin(response.url, anchor["href"])
            if Path(urlparse(url).path).suffix.lower() not in {".xlsx", ".xls"}:
                continue
            if text.lower().startswith("vintages"):
                vintage_candidates.append((text, url))
            if text.lower().startswith("all available observations"):
                release_candidates.append((text, url))
        if not vintage_candidates:
            raise RuntimeError(f"No complete-vintage workbook found for {variable}.")
        monthly = [item for item in vintage_candidates if "mv" in Path(urlparse(item[1]).path).name.lower()]
        text, url = (monthly or vintage_candidates)[0]
        sources.append({"variable": variable.upper(), "kind": "vintage", "text": text, "url": url})
        for text, url in release_candidates:
            sources.append({"variable": variable.upper(), "kind": "release", "text": text, "url": url})
    unique = {source["url"]: source for source in sources}
    return sorted(unique.values(), key=lambda source: source["url"])


def fetch(sources: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool]:
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = json.loads(STATE.read_text()) if STATE.exists() else {}
    changed = False
    session = requests.Session()
    session.headers["User-Agent"] = "Quantitative-Finance-Lab Philadelphia real-time builder"
    for source in sources:
        previous = saved.get(source["url"], {})
        headers = {}
        if previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
        response = session.get(source["url"], headers=headers, timeout=240)
        filename = f"{source['variable'].lower()}_{Path(urlparse(source['url']).path).name}"
        path = CACHE / filename
        if response.status_code == 304 and path.exists():
            source.update(previous)
            source["path"] = filename
            saved[source["url"]] = source
            continue
        response.raise_for_status()
        if response.content[:2] not in {b"PK", b"\xd0\xcf"}:
            raise ValueError(f"Philadelphia Fed response is not Excel: {source['url']}")
        sha256 = hashlib.sha256(response.content).hexdigest()
        if path.exists() and previous.get("sha256") == sha256:
            source.update(previous)
            source["path"] = filename
            saved[source["url"]] = source
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        source.update(
            {
                "path": filename,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "sha256": sha256,
                "bytes": len(response.content),
            }
        )
        saved[source["url"]] = source
        changed = True
        print(f"downloaded {filename} ({len(response.content) / 1e6:.2f} MB)")
    STATE.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    return list(saved.values()), changed


def period_date(value: object) -> tuple[pd.Timestamp | None, str | None]:
    text = str(value).strip()
    quarterly = re.fullmatch(r"(\d{4}):Q([1-4])", text, re.IGNORECASE)
    if quarterly:
        return pd.Period(f"{quarterly.group(1)}Q{quarterly.group(2)}", freq="Q").start_time, "quarterly"
    monthly = re.fullmatch(r"(\d{4}):(\d{1,2})", text)
    if monthly:
        return pd.Timestamp(int(monthly.group(1)), int(monthly.group(2)), 1), "monthly"
    date = pd.to_datetime(value, errors="coerce")
    return (pd.Timestamp(date), "dated") if pd.notna(date) else (None, None)


def vintage_date(column: object) -> tuple[pd.Timestamp, str]:
    text = str(column)
    monthly = re.search(r"(\d{2})M(\d{1,2})$", text, re.IGNORECASE)
    quarterly = re.search(r"(\d{2})Q([1-4])$", text, re.IGNORECASE)
    match = monthly or quarterly
    if match is None:
        raise ValueError(f"Could not parse RTDSM vintage column {column}")
    year = int(match.group(1))
    year += 1900 if year >= 50 else 2000
    period = int(match.group(2))
    if monthly:
        return pd.Timestamp(year, period, 1), "monthly"
    return pd.Period(f"{year}Q{period}", freq="Q").end_time.normalize(), "quarterly"


def read_excel(path: Path, sheet_name=0, header=0) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    return pd.read_excel(path, sheet_name=sheet_name, header=header, engine=engine)


def parse_vintage(source: dict[str, str]) -> pd.DataFrame:
    path = CACHE / source["path"]
    frame = read_excel(path)
    date_column = frame.columns[0]
    long = frame.melt(id_vars=date_column, var_name="vintage_column", value_name="value")
    parsed = long[date_column].map(period_date)
    long["observation_date"] = [item[0] for item in parsed]
    long["observation_frequency"] = [item[1] for item in parsed]
    vintages = long["vintage_column"].map(vintage_date)
    long["vintage_date"] = [item[0] for item in vintages]
    long["vintage_frequency"] = [item[1] for item in vintages]
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["observation_date", "value"])
    long.insert(0, "variable", source["variable"])
    long["source_file"] = source["path"]
    return long[
        [
            "variable",
            "observation_date",
            "observation_frequency",
            "vintage_date",
            "vintage_frequency",
            "value",
            "source_file",
        ]
    ]


def parse_release(source: dict[str, str]) -> pd.DataFrame:
    path = CACHE / source["path"]
    book = pd.ExcelFile(path, engine="xlrd" if path.suffix.lower() == ".xls" else "openpyxl")
    sheet = "DATA" if "DATA" in book.sheet_names else book.sheet_names[-1]
    raw = book.parse(sheet_name=sheet, header=None)
    header = None
    for row in range(raw.shape[0]):
        values = [str(value).strip().lower() for value in raw.iloc[row].tolist()]
        if values and values[0] == "date" and "first" in values:
            header = row
            break
    if header is None:
        raise ValueError(f"No first/second/third header found in {path.name}")
    columns = [str(value).strip() for value in raw.iloc[header].tolist()]
    frame = raw.iloc[header + 1 :].copy()
    frame.columns = columns
    date_column = columns[0]
    release_columns = [column for column in columns[1:] if column]
    long = frame.melt(id_vars=date_column, value_vars=release_columns, var_name="release", value_name="value")
    parsed = long[date_column].map(period_date)
    long["observation_date"] = [item[0] for item in parsed]
    long["observation_frequency"] = [item[1] for item in parsed]
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["observation_date", "value"])
    long.insert(0, "variable", source["variable"])
    filename = source["path"].lower()
    measure = "level_change" if "employ_level" in filename else "percent_change" if "pct_chg" in filename else "published"
    long.insert(1, "measure", measure)
    long["release"] = long["release"].str.strip().str.lower()
    long["source_file"] = source["path"]
    return long[
        ["variable", "measure", "observation_date", "observation_frequency", "release", "value", "source_file"]
    ]


def write(frame: pd.DataFrame, path: Path, dataset: str, key: list[str]) -> None:
    frame = frame.sort_values(key, ignore_index=True).drop_duplicates(key, keep="last")
    if frame.duplicated(key).any() or not np.isfinite(frame["value"]).all():
        raise ValueError(f"{path.name} has duplicate keys or non-finite values.")
    metadata = {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Federal Reserve Bank of Philadelphia RTDSM",
        "source_url": BASE + "real-time-data-set-for-macroeconomists",
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {name.encode(): value.encode() for name, value in metadata.items()}
    )
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    print(f"wrote {path} rows={len(frame):,}")


def build(catalog: list[dict[str, str]]) -> None:
    vintages = [parse_vintage(source) for source in catalog if source.get("kind") == "vintage"]
    releases = [parse_release(source) for source in catalog if source.get("kind") == "release"]
    write(
        pd.concat(vintages, ignore_index=True),
        VINTAGE_OUTPUT,
        "Philadelphia Fed RTDSM complete vintage histories",
        ["variable", "observation_date", "vintage_date"],
    )
    write(
        pd.concat(releases, ignore_index=True),
        RELEASE_OUTPUT,
        "Philadelphia Fed RTDSM first, second, third, and latest releases",
        ["variable", "measure", "observation_date", "release"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.build_only:
        if not STATE.exists():
            raise FileNotFoundError("No cached source catalog exists. Run download.py first.")
        build(list(json.loads(STATE.read_text()).values()))
        return
    catalog, changed = fetch(discover())
    if changed or not VINTAGE_OUTPUT.exists() or not RELEASE_OUTPUT.exists():
        build(catalog)
    else:
        print("Philadelphia Fed real-time workbooks are unchanged")


if __name__ == "__main__":
    main()
