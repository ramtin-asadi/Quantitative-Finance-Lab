from __future__ import annotations

import argparse
import io
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
OUTPUT = DATA / "macro_high_frequency.parquet"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = {
    "DCOILBRENTEU": ("inflation_input", "Brent crude spot price"),
    "DCOILWTICO": ("inflation_input", "WTI crude spot price challenger"),
    "GASREGW": ("inflation_input", "U.S. regular retail gasoline price"),
    "SOFR": ("policy_rate", "Secured Overnight Financing Rate"),
    "DFF": ("policy_rate", "Effective federal funds rate"),
    "DFEDTARU": ("policy_rate", "Federal funds target range upper limit"),
    "DFEDTARL": ("policy_rate", "Federal funds target range lower limit"),
    "DGS2": ("nominal_rate", "2-year Treasury constant maturity"),
    "DGS5": ("nominal_rate", "5-year Treasury constant maturity"),
    "DGS10": ("nominal_rate", "10-year Treasury constant maturity"),
    "DFII5": ("real_rate", "5-year TIPS constant maturity"),
    "DFII10": ("real_rate", "10-year TIPS constant maturity"),
    "T5YIE": ("breakeven", "5-year breakeven inflation"),
    "T10YIE": ("breakeven", "10-year breakeven inflation"),
    "DTWEXBGS": ("financial_control", "Broad nominal U.S. dollar index"),
    "VIXCLS": ("financial_control", "CBOE VIX"),
}


def request_series(series_id: str, start: pd.Timestamp | None) -> pd.DataFrame:
    params = {"id": series_id}
    if start is not None:
        params["cosd"] = str(start.date())
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()
    if not response.text.startswith("observation_date"):
        raise ValueError(f"Unexpected FRED graph response for {series_id}.")
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def update_cache(series_id: str, update: bool, overlap_days: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{series_id}.csv"
    old = pd.read_csv(path, parse_dates=["date"]) if update and path.exists() else pd.DataFrame()
    start = old["date"].max() - pd.Timedelta(days=overlap_days) if not old.empty else None
    fresh = request_series(series_id, start)
    combined = pd.concat([old, fresh], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates("date", keep="last")
    temporary = path.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    print(f"{series_id}: rows={len(combined):,} new_window={len(fresh):,}")


def build() -> None:
    pieces = []
    for series_id, (category, label) in SERIES.items():
        frame = pd.read_csv(CACHE / f"{series_id}.csv", parse_dates=["date"])
        frame.insert(0, "series_id", series_id)
        frame.insert(1, "category", category)
        frame.insert(2, "label", label)
        pieces.append(frame)
    data = pd.concat(pieces, ignore_index=True).sort_values(["series_id", "date"], ignore_index=True)
    key = ["series_id", "date"]
    if data.duplicated(key).any() or not np.isfinite(data["value"]).all():
        raise ValueError("High-frequency macro output has duplicate keys or non-finite values.")
    metadata = {
        "dataset": "Small high-frequency macro and policy control panel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "FRED graph CSV; underlying EIA, Federal Reserve, Treasury, and CBOE sources",
        "source_url": URL,
        "units": "Source-native; see series label/FRED documentation",
    }
    table = pa.Table.from_pandas(data, preserve_index=False).replace_schema_metadata(
        {name.encode(): value.encode() for name, value in metadata.items()}
    )
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT} rows={len(data):,} series={data['series_id'].nunique():,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--overlap-days", type=int, default=90)
    args = parser.parse_args()
    for series_id in SERIES:
        update_cache(series_id, args.update, args.overlap_days)
    build()


if __name__ == "__main__":
    main()
