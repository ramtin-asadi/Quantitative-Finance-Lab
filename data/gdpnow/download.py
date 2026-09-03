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
WORKBOOK = CACHE / "GDPTrackingModelDataAndForecasts.xlsx"
RELEASE_WORKBOOK = CACHE / "GDPNowcastDataReleaseDates.xlsx"
FORECAST_OUTPUT = DATA / "gdpnow_forecasts.parquet"
CONTRIBUTION_OUTPUT = DATA / "gdpnow_contributions.parquet"
TRACK_OUTPUT = DATA / "gdpnow_track_record.parquet"
RELEASE_OUTPUT = DATA / "gdpnow_release_dates.parquet"
URLS = {
    "GDPTrackingModelDataAndForecasts.xlsx": "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx",
    "GDPNowcastDataReleaseDates.xlsx": "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cqer/researchcq/gdpnow/GDPNowcastDataReleaseDates.xlsx",
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
        response = requests.get(url, headers=headers, timeout=300)
        path = CACHE / filename
        if response.status_code == 304 and path.exists():
            continue
        response.raise_for_status()
        if not response.content.startswith(b"PK"):
            raise ValueError(f"Atlanta Fed response is not XLSX: {url}")
        sha256 = hashlib.sha256(response.content).hexdigest()
        if path.exists() and previous.get("sha256") == sha256:
            continue
        temporary = path.with_suffix(".xlsx.tmp")
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


def clean_component(value: object) -> str:
    text = " ".join(str(value).replace("*", "").split())
    return re.sub(r"^\d+\s*-\s*", "", text).strip()


def archive_long(sheet: str, value_name: str) -> pd.DataFrame:
    frame = pd.read_excel(WORKBOOK, sheet_name=sheet, engine="openpyxl")
    frame = frame.rename(
        columns={
            frame.columns[0]: "forecast_date",
            frame.columns[1]: "target_quarter",
        }
    )
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"], errors="coerce")
    frame["target_quarter"] = pd.to_datetime(frame["target_quarter"], errors="coerce")
    frame = frame.dropna(subset=["forecast_date", "target_quarter"])
    id_columns = ["forecast_date", "target_quarter"]
    value_columns = []
    for column in frame.columns[2:]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            frame[column] = numeric
            value_columns.append(column)
    long = frame.melt(id_vars=id_columns, value_vars=value_columns, var_name="component", value_name=value_name)
    long = long.dropna(subset=[value_name])
    long["component"] = long["component"].map(clean_component)
    long["source_sheet"] = sheet
    return long


def current_long(sheet: str, value_name: str) -> pd.DataFrame:
    raw = pd.read_excel(WORKBOOK, sheet_name=sheet, header=None, engine="openpyxl")
    title = " ".join(raw.iloc[1].dropna().astype(str).tolist())
    target = re.search(r"(20\d{2})q([1-4])", title, re.IGNORECASE)
    if not target:
        raise ValueError(f"Could not identify current GDPNow target quarter in {sheet}.")
    target_quarter = pd.Period(f"{target.group(1)}Q{target.group(2)}", freq="Q").end_time.normalize()
    dates = pd.to_datetime(raw.iloc[0, 2:], errors="coerce")
    rows = []
    for row in range(2, raw.shape[0]):
        component = clean_component(raw.iat[row, 1])
        if not component or component.lower() == "nan":
            component = clean_component(raw.iat[row, 0])
        if not component or component.lower() == "nan":
            continue
        for column, forecast_date in zip(range(2, raw.shape[1]), dates):
            value = pd.to_numeric(raw.iat[row, column], errors="coerce")
            if pd.notna(forecast_date) and pd.notna(value):
                rows.append(
                    {
                        "forecast_date": forecast_date,
                        "target_quarter": target_quarter,
                        "component": component,
                        value_name: float(value),
                        "source_sheet": sheet,
                    }
                )
    return pd.DataFrame(rows)


def track_record() -> pd.DataFrame:
    frame = pd.read_excel(WORKBOOK, sheet_name="TrackRecord", engine="openpyxl")
    frame = frame.iloc[:, :8].copy()
    frame.columns = [
        "target_quarter", "final_model_forecast", "bea_advance_estimate",
        "bea_release_date", "blank", "error", "absolute_error", "squared_error",
    ]
    frame = frame.drop(columns="blank")
    frame["target_quarter"] = pd.to_datetime(frame["target_quarter"], errors="coerce")
    frame["bea_release_date"] = pd.to_datetime(frame["bea_release_date"], errors="coerce")
    for column in frame.columns.difference(["target_quarter", "bea_release_date"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["target_quarter", "final_model_forecast"]).sort_values("target_quarter")


def release_dates() -> pd.DataFrame:
    rows = []
    for sheet in ("PostedUpdates", "InternalUpdates"):
        raw = pd.read_excel(RELEASE_WORKBOOK, sheet_name=sheet, header=None, engine="openpyxl")
        frame = raw.iloc[:, :3].copy()
        frame.columns = ["release_inputs", "release_date", "release_time"]
        frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
        frame = frame.dropna(subset=["release_date"])
        frame.insert(0, "schedule_type", "posted" if sheet == "PostedUpdates" else "internal")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["release_date", "schedule_type"])


def write(frame: pd.DataFrame, path: Path, dataset: str, key: list[str]) -> None:
    frame = frame.sort_values(key, ignore_index=True).drop_duplicates(key, keep="last")
    numeric = frame.select_dtypes(include="number")
    if not numeric.empty and not np.isfinite(numeric).all().all():
        raise ValueError(f"{path.name} contains non-finite numeric values.")
    metadata = {
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": URLS["GDPTrackingModelDataAndForecasts.xlsx"],
        "source": "Federal Reserve Bank of Atlanta GDPNow official workbook",
    }
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {name.encode(): value.encode() for name, value in metadata.items()}
    )
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
    print(f"wrote {path} rows={len(frame):,}")


def build() -> None:
    forecasts = pd.concat(
        [
            archive_long("TrackingDeepArchives", "forecast_value"),
            archive_long("TrackingArchives", "forecast_value"),
            current_long("TrackingHistory", "forecast_value"),
        ],
        ignore_index=True,
    )
    contributions = pd.concat(
        [
            archive_long("ContribArchives", "contribution_pp"),
            current_long("ContribHistory", "contribution_pp"),
        ],
        ignore_index=True,
    )
    write(
        forecasts, FORECAST_OUTPUT, "GDPNow headline and component forecast history",
        ["forecast_date", "target_quarter", "component"],
    )
    write(
        contributions, CONTRIBUTION_OUTPUT, "GDPNow component contribution history",
        ["forecast_date", "target_quarter", "component"],
    )
    write(track_record(), TRACK_OUTPUT, "GDPNow track record against BEA advance GDP", ["target_quarter"])
    write(release_dates(), RELEASE_OUTPUT, "GDPNow scheduled source-data release dates", ["schedule_type", "release_date", "release_inputs"])


def main() -> None:
    changed = fetch()
    outputs = [FORECAST_OUTPUT, CONTRIBUTION_OUTPUT, TRACK_OUTPUT, RELEASE_OUTPUT]
    if changed or not all(path.exists() for path in outputs):
        build()
    else:
        print("GDPNow official workbooks are unchanged")


if __name__ == "__main__":
    main()
