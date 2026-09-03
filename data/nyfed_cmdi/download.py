from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
WORKBOOK = CACHE / "market_cmdi.xlsx"
STATE = CACHE / "source.json"
OUTPUT = DATA / "nyfed_cmdi.parquet"
SOURCE_URL = (
    "https://www.newyorkfed.org/medialibrary/research/interactives/"
    "cmdi/downloads/Market%20CMDI.xlsx"
)
PAGE_URL = "https://www.newyorkfed.org/research/policy/cmdi"


def session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    client = requests.Session()
    client.headers["User-Agent"] = "Quantitative-Finance-Lab CMDI builder"
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def fetch(*, force: bool = False) -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    headers = {}
    if not force and WORKBOOK.exists():
        if previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
    response = session().get(SOURCE_URL, headers=headers, timeout=180)
    if response.status_code == 304:
        return False
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"PK"):
        raise ValueError("The NY Fed CMDI download is not an Excel workbook.")
    digest = hashlib.sha256(content).hexdigest()
    if WORKBOOK.exists() and digest == previous.get("sha256"):
        return False
    temporary = WORKBOOK.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(WORKBOOK)
    STATE.write_text(
        json.dumps(
            {
                "url": SOURCE_URL,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "sha256": digest,
                "bytes": len(content),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return True


def build() -> pd.DataFrame:
    raw = pd.read_excel(WORKBOOK, sheet_name="Index Data", header=5)
    raw = raw.iloc[:, :4].copy()
    raw.columns = ["date", "market_cmdi", "ig_cmdi", "hy_cmdi"]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for column in raw.columns[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    result = raw.dropna(subset=["date"]).sort_values("date")
    result = result.drop_duplicates("date", keep="last").reset_index(drop=True)
    if result.empty or result["date"].duplicated().any():
        raise ValueError("CMDI workbook produced an empty or duplicate-date table.")
    if result[["market_cmdi", "ig_cmdi", "hy_cmdi"]].isna().all(axis=1).any():
        raise ValueError("CMDI workbook contains a date with no index values.")
    table = pa.Table.from_pandas(result, preserve_index=False)
    metadata = {
        **(table.schema.metadata or {}),
        b"dataset": b"New York Fed Corporate Bond Market Distress Index",
        b"source_url": SOURCE_URL.encode(),
        b"page_url": PAGE_URL.encode(),
        b"frequency": b"weekly, end-of-week Friday",
        b"generated_at_utc": datetime.now(timezone.utc).isoformat().encode(),
    }
    temporary = OUTPUT.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(metadata), temporary, compression="zstd")
    temporary.replace(OUTPUT)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    changed = fetch(force=args.force)
    if changed or not OUTPUT.exists():
        frame = build()
        print(
            f"wrote {OUTPUT} ({len(frame):,} rows, "
            f"{frame['date'].min().date()} through {frame['date'].max().date()})"
        )
    else:
        print("NY Fed CMDI source is unchanged; existing Parquet was preserved.")


if __name__ == "__main__":
    main()
