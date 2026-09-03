from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
STATE = CACHE / "sources.json"
RAW = CACHE / "mpt_histdata.xlsx"
SOURCE_CODE = CACHE / "mpt_source.zip"
OUTPUT = DATA / "atlanta_mpt.parquet"
URLS = {
    "mpt_histdata.xlsx": "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cenfis/market-probability-tracker/mpt_histdata.xlsx",
    "mpt_source.zip": "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cenfis/market-probability-tracker/mpt_source.zip",
}


def fetch() -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    changed = False
    for filename, url in URLS.items():
        previous = state.get(url, {})
        headers = {}
        if previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
        response = requests.get(url, headers=headers, timeout=240)
        path = CACHE / filename
        if response.status_code == 304 and path.exists():
            continue
        response.raise_for_status()
        if not response.content.startswith(b"PK"):
            raise ValueError(f"Atlanta MPT response is not ZIP/XLSX: {url}")
        sha256 = hashlib.sha256(response.content).hexdigest()
        if path.exists() and previous.get("sha256") == sha256:
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
        state[url] = {
            "path": filename,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": sha256,
            "bytes": len(response.content),
        }
        changed = changed or filename == "mpt_histdata.xlsx"
        print(f"downloaded {filename} ({len(response.content) / 1e6:.2f} MB)")
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return changed


def build() -> None:
    frame = pd.read_excel(RAW, sheet_name="DATA", engine="openpyxl")
    expected = ["date", "reference_start", "target_range", "field", "value"]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected MPT columns: {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["reference_start"] = pd.to_datetime(frame["reference_start"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    key = ["date", "reference_start", "field"]
    frame = frame.sort_values(key, ignore_index=True).drop_duplicates(key, keep="last")
    if frame.duplicated(key).any() or not np.isfinite(frame["value"]).all():
        raise ValueError("MPT data contain duplicate keys or non-finite values.")
    metadata = {
        "dataset": "Atlanta Fed Market Probability Tracker historical distributions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": URLS["mpt_histdata.xlsx"],
        "method": "Options on three-month compounded SOFR",
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {name.encode(): value.encode() for name, value in metadata.items()}
    )
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT} rows={len(frame):,} date={frame['date'].min().date()}..{frame['date'].max().date()}")


def main() -> None:
    changed = fetch()
    if changed or not OUTPUT.exists():
        build()
    else:
        print("Atlanta MPT historical workbook is unchanged")


if __name__ == "__main__":
    main()
