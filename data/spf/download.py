from __future__ import annotations

import hashlib
import json
import re
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
FORECAST_OUTPUT = DATA / "spf_forecasts.parquet"
DATES_OUTPUT = DATA / "spf_release_dates.parquet"
BASE = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/"
URLS = {
    "meanLevel.xlsx": BASE + "historical-data/meanLevel.xlsx",
    "meanGrowth.xlsx": BASE + "historical-data/meanGrowth.xlsx",
    "medianLevel.xlsx": BASE + "historical-data/medianLevel.xlsx",
    "medianGrowth.xlsx": BASE + "historical-data/medianGrowth.xlsx",
    "spf-release-dates.txt": BASE + "spf-release-dates.txt",
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
        if path.suffix == ".xlsx" and not response.content.startswith(b"PK"):
            raise ValueError(f"SPF response is not XLSX: {url}")
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
        changed = True
        print(f"downloaded {filename} ({len(response.content) / 1e6:.2f} MB)")
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return changed


def release_dates() -> pd.DataFrame:
    pattern = re.compile(
        r"^\s*(?:(\d{4})\s+)?Q([1-4])\s+(\d{1,2}/\d{1,2}/\d{2})\**\s+(\d{1,2}/\d{1,2}/\d{2})\**"
    )
    year = None
    rows = []
    for line in (CACHE / "spf-release-dates.txt").read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        if match.group(1):
            year = int(match.group(1))
        if year is None:
            continue
        rows.append(
            {
                "survey_year": year,
                "survey_quarter": int(match.group(2)),
                "deadline_date": pd.to_datetime(match.group(3), format="%m/%d/%y"),
                "release_date": pd.to_datetime(match.group(4), format="%m/%d/%y"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Could not parse SPF deadline/release dates.")
    frame["survey_period"] = frame["survey_year"].astype(str) + "Q" + frame["survey_quarter"].astype(str)
    return frame.sort_values(["survey_year", "survey_quarter"], ignore_index=True)


def parse_workbook(path: Path) -> pd.DataFrame:
    name = path.stem.lower()
    statistic = "median" if "median" in name else "mean"
    measure = "growth" if "growth" in name else "level"
    book = pd.ExcelFile(path, engine="openpyxl")
    pieces = []
    for variable in book.sheet_names:
        frame = book.parse(sheet_name=variable)
        if frame.shape[1] < 3:
            continue
        year_column, quarter_column = frame.columns[:2]
        frame[year_column] = pd.to_numeric(frame[year_column], errors="coerce")
        frame[quarter_column] = pd.to_numeric(frame[quarter_column], errors="coerce")
        frame = frame.dropna(subset=[year_column, quarter_column])
        long = frame.melt(
            id_vars=[year_column, quarter_column],
            var_name="forecast_horizon",
            value_name="forecast_value",
        )
        long["forecast_value"] = pd.to_numeric(long["forecast_value"], errors="coerce")
        long = long.dropna(subset=["forecast_value"])
        long = long.rename(columns={year_column: "survey_year", quarter_column: "survey_quarter"})
        long["survey_year"] = long["survey_year"].astype("int16")
        long["survey_quarter"] = long["survey_quarter"].astype("int8")
        long.insert(0, "variable", variable)
        long.insert(1, "statistic", statistic)
        long.insert(2, "measure", measure)
        long["source_file"] = path.name
        pieces.append(long)
    return pd.concat(pieces, ignore_index=True)


def write(frame: pd.DataFrame, path: Path, dataset: str, key: list[str]) -> None:
    frame = frame.sort_values(key, ignore_index=True).drop_duplicates(key, keep="last")
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty and not np.isfinite(numeric).all().all():
        raise ValueError(f"{path.name} contains non-finite numeric values.")
    metadata = {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Federal Reserve Bank of Philadelphia Survey of Professional Forecasters",
        "source_url": "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters",
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {name.encode(): value.encode() for name, value in metadata.items()}
    )
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    print(f"wrote {path} rows={len(frame):,}")


def build() -> None:
    dates = release_dates()
    forecasts = pd.concat(
        [parse_workbook(CACHE / filename) for filename in URLS if filename.endswith(".xlsx")],
        ignore_index=True,
    )
    forecasts["survey_period"] = (
        forecasts["survey_year"].astype(str) + "Q" + forecasts["survey_quarter"].astype(str)
    )
    forecasts = forecasts.merge(
        dates[["survey_year", "survey_quarter", "deadline_date", "release_date"]],
        on=["survey_year", "survey_quarter"],
        how="left",
        validate="many_to_one",
    )
    write(
        forecasts,
        FORECAST_OUTPUT,
        "SPF mean and median level/growth forecast histories",
        ["variable", "statistic", "measure", "survey_year", "survey_quarter", "forecast_horizon"],
    )
    write(
        dates,
        DATES_OUTPUT,
        "SPF historical survey deadlines and publication dates",
        ["survey_year", "survey_quarter"],
    )


def main() -> None:
    changed = fetch()
    if changed or not FORECAST_OUTPUT.exists() or not DATES_OUTPUT.exists():
        build()
    else:
        print("SPF source files are unchanged")


if __name__ == "__main__":
    main()
