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
RAW = CACHE / "ebp_csv.csv"
STATE = CACHE / "state.json"
OUTPUT = DATA / "fed_credit.parquet"
URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"


def fetch() -> bool:
    CACHE.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    headers = {}
    if state.get("etag"):
        headers["If-None-Match"] = state["etag"]
    if state.get("last_modified"):
        headers["If-Modified-Since"] = state["last_modified"]
    response = requests.get(URL, headers=headers, timeout=120)
    if response.status_code == 304 and RAW.exists():
        return False
    response.raise_for_status()
    if not response.content.startswith(b"date,"):
        raise ValueError("The Federal Reserve EBP response is not the expected CSV.")
    sha256 = hashlib.sha256(response.content).hexdigest()
    if RAW.exists() and state.get("sha256") == sha256:
        return False
    temporary = RAW.with_suffix(".csv.tmp")
    temporary.write_bytes(response.content)
    temporary.replace(RAW)
    STATE.write_text(
        json.dumps(
            {
                "url": URL,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "sha256": sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return True


def build() -> None:
    frame = pd.read_csv(RAW)
    expected = ["date", "gz_spread", "ebp", "est_prob"]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected EBP columns: {list(frame.columns)}")
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    for column in expected[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("date", ignore_index=True)
    if frame["date"].duplicated().any() or not np.isfinite(frame[expected[1:]]).all().all():
        raise ValueError("EBP source contains duplicate dates or non-finite values.")

    metadata = {
        "dataset": "Federal Reserve excess bond premium and recession probability",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": URL,
        "frequency": "monthly",
        "revision_note": "The Federal Reserve may revise the full history each month.",
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()}
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
        print("Federal Reserve EBP source is unchanged")


if __name__ == "__main__":
    main()
