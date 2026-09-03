from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

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
STATE = CACHE / "state.json"
OUTPUT = DATA / "alfred_realtime.parquet"
CATALOG_OUTPUT = DATA / "alfred_series_catalog.parquet"
DOWNLOAD_PAGE = "https://alfred.stlouisfed.org/series/downloaddata?seid={}"
CURL = shutil.which("curl")

SERIES = {
    "PCEC96": "consumption",
    "PCE": "consumption",
    "RSAFS": "consumption",
    "RSFSXMV": "consumption",
    "PI": "income",
    "DSPI": "income",
    "DSPIC96": "income",
    "TOTALSA": "consumption",
    "INDPRO": "production",
    "IPMAN": "production",
    "TCU": "production",
    "AWHI": "production",
    "HOUST": "housing",
    "PERMIT": "housing",
    "TLRESCONS": "housing",
    "HSN1F": "housing",
    "DGORDER": "business_investment",
    "NEWORDER": "business_investment",
    "ANXAVS": "business_investment",
    "TLNRESCONS": "business_investment",
    "BUSINV": "inventories",
    "MNFCTRIMSA": "inventories",
    "WHLSLRIMSA": "inventories",
    "RETAILIMSA": "inventories",
    "EXPGS": "trade",
    "IMPGS": "trade",
    "BOPGSTB": "trade",
    "PAYEMS": "labor",
    "UNRATE": "labor",
    "AWHAETP": "labor",
    "TEMPHELPS": "labor",
    "ICSA": "labor",
    "CCSA": "labor",
    "IURSA": "labor",
    "JTSJOL": "labor",
    "JTSHIL": "labor",
    "CPIAUCSL": "inflation",
    "CPILFESL": "inflation",
    "CPIUFDSL": "inflation",
    "CUSR0000SAF11": "inflation",
    "CUSR0000SETB01": "inflation",
    "PCEPI": "inflation",
    "PCEPILFE": "inflation",
    "DFXARG3M086SBEA": "inflation",
    "WPSFD49207": "inflation",
    "IR": "inflation",
    "CUSR0000SAH1": "inflation",
    "FEDFUNDS": "policy_financial",
    "GS2": "policy_financial",
    "GS5": "policy_financial",
    "GS10": "policy_financial",
    "TWEXBGSMTH": "policy_financial",
}


def curl_request(url: str, fields: list[tuple[str, str]] | None = None) -> bytes:
    command = [
        CURL,
        "-fsSL",
        "--retry",
        "5",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "600",
    ]
    body = None
    if fields is not None:
        command.extend(
            [
                "--header",
                "Content-Type: application/x-www-form-urlencoded",
                "--data-binary",
                "@-",
            ]
        )
        body = urlencode(fields).encode()
    command.append(url)
    return subprocess.run(command, input=body, check=True, stdout=subprocess.PIPE).stdout


def series_info(session: requests.Session, series_id: str) -> dict[str, object]:
    url = DOWNLOAD_PAGE.format(series_id)
    if CURL:
        page = curl_request(url).decode("utf-8")
    else:
        response = session.get(url, timeout=120)
        response.raise_for_status()
        page = response.text
    soup = BeautifulSoup(page, "html.parser")
    vintage_select = soup.select_one('select[name="form[selected_vintage_dates][]"]')
    start_input = soup.select_one('[name="form[obs_start_date]"]')
    end_input = soup.select_one('[name="form[obs_end_date]"]')
    if vintage_select is None or start_input is None or end_input is None:
        raise ValueError(f"ALFRED download form is unavailable for {series_id}.")
    vintages = [option["value"] for option in vintage_select.find_all("option")]
    title = soup.title.get_text(" ", strip=True)
    title = re.sub(r"^Download Data for\s+", "", title).split(" | ALFRED", 1)[0]
    return {
        "series_id": series_id,
        "category": SERIES[series_id],
        "title": title,
        "observation_start": start_input["value"],
        "observation_end": end_input["value"],
        "vintages": vintages,
        "url": url,
    }


def request_archive(
    session: requests.Session, info: dict[str, object], vintages: list[str], tag: str
) -> bytes:
    fields = [
        ("form[units]", "lin"),
        ("form[obs_start_date]", str(info["observation_start"])),
        ("form[obs_end_date]", str(info["observation_end"])),
        ("form[entered_vintage_dates]", ""),
        ("form[file_type]", "1"),
        ("form[file_format]", "csv"),
        ("form[download_data]", ""),
    ]
    fields.extend(("form[selected_vintage_dates][]", vintage) for vintage in vintages)
    if CURL:
        content = curl_request(str(info["url"]), fields)
    else:
        response = session.post(str(info["url"]), data=fields, timeout=300)
        response.raise_for_status()
        content = response.content
    if not content.startswith(b"PK"):
        raise ValueError(f"ALFRED did not return a ZIP for {info['series_id']}.")
    path = CACHE / f"{info['series_id']}_{tag}.zip"
    temporary = path.with_suffix(".zip.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return content


def parse_archive(content: bytes, series_id: str, category: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        frame = pd.read_csv(archive.open(csv_name))
    expected = ["period_start_date", series_id, "realtime_start_date", "realtime_end_date"]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected ALFRED columns for {series_id}: {list(frame.columns)}")
    frame = frame.rename(
        columns={
            "period_start_date": "observation_date",
            series_id: "value",
            "realtime_start_date": "realtime_start",
            "realtime_end_date": "realtime_end",
        }
    )
    frame.insert(0, "series_id", series_id)
    frame.insert(1, "category", category)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="raise")
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"], errors="raise")
    frame["realtime_end"] = pd.to_datetime(frame["realtime_end"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def write_outputs(data: pd.DataFrame, catalog: pd.DataFrame) -> None:
    key = ["series_id", "observation_date", "realtime_start"]
    data = data.sort_values(key, ignore_index=True).drop_duplicates(key, keep="last")
    if data.duplicated(key).any() or not np.isfinite(data["value"]).all():
        raise ValueError("ALFRED output has duplicate revision keys or non-finite values.")
    metadata = {
        "dataset": "Targeted ALFRED real-time macroeconomic revision panel",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "series_count": str(data["series_id"].nunique()),
        "point_in_time_rule": "A value is valid from realtime_start through realtime_end; a null end is current.",
        "source": "Official ALFRED Observations by Real-Time Period downloads",
    }
    table = pa.Table.from_pandas(data, preserve_index=False).replace_schema_metadata(
        {name.encode(): value.encode() for name, value in metadata.items()}
    )
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(OUTPUT)
    catalog.to_parquet(CATALOG_OUTPUT.with_suffix(".parquet.tmp"), index=False, compression="zstd")
    CATALOG_OUTPUT.with_suffix(".parquet.tmp").replace(CATALOG_OUTPUT)
    print(
        f"wrote {OUTPUT} rows={len(data):,} series={data['series_id'].nunique():,} "
        f"size_mb={OUTPUT.stat().st_size / 1e6:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--series", action="append", choices=tuple(SERIES))
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state = {series_id: value for series_id, value in state.items() if series_id in SERIES}
    old = pd.read_parquet(OUTPUT) if args.update and OUTPUT.exists() else pd.DataFrame()
    old_catalog = pd.read_parquet(CATALOG_OUTPUT) if args.update and CATALOG_OUTPUT.exists() else pd.DataFrame()
    if not old.empty:
        old = old.loc[old["series_id"].isin(SERIES)]
    if not old_catalog.empty:
        old_catalog = old_catalog.loc[old_catalog["series_id"].isin(SERIES)]
    selected = args.series or list(SERIES)
    session = requests.Session()
    session.headers["User-Agent"] = "Quantitative-Finance-Lab ALFRED real-time builder"
    fresh_frames = []
    catalog_rows = []
    new_state = dict(state)
    for series_id in selected:
        info = series_info(session, series_id)
        vintages = list(info.pop("vintages"))
        last = state.get(series_id, {}).get("last_vintage") if args.update else None
        requested = [vintage for vintage in vintages if last is None or vintage > last]
        catalog_rows.append(
            {
                **info,
                "first_vintage": pd.Timestamp(vintages[0]),
                "last_vintage": pd.Timestamp(vintages[-1]),
                "vintage_count": len(vintages),
            }
        )
        if requested:
            tag = "full" if last is None else f"update_{requested[0]}_{requested[-1]}"
            content = request_archive(session, info, requested, tag)
            fresh = parse_archive(content, series_id, SERIES[series_id])
            fresh_frames.append(fresh)
            print(f"{series_id}: vintages={len(requested):,} rows={len(fresh):,}")
        else:
            print(f"{series_id}: no new vintage dates")
        new_state[series_id] = {"last_vintage": vintages[-1], "vintage_count": len(vintages)}

    if fresh_frames or not OUTPUT.exists():
        combined = pd.concat([old, *fresh_frames], ignore_index=True)
        catalog = pd.concat([old_catalog, pd.DataFrame(catalog_rows)], ignore_index=True)
        catalog = catalog.drop_duplicates("series_id", keep="last").sort_values("series_id")
        write_outputs(combined, catalog)
        STATE.write_text(json.dumps(new_state, indent=2, sort_keys=True) + "\n")
    else:
        print("ALFRED targeted panel is current")


if __name__ == "__main__":
    main()
