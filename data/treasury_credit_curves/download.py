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
OUTPUT = DATA / "treasury_credit_curves.parquet"
PAGES = {
    "HQM": "https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curve/corporate-bond-yield-curve",
    "TNC": "https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves/treasury-coupon-issues",
}


def source_kind(family: str, text: str) -> tuple[str, str] | None:
    lower = " ".join(text.lower().split())
    if family == "HQM":
        if "hqm corporate bond yield curve spot rates" in lower:
            rate_type = "spot"
        elif "hqm corporate bond yield curve par yields" in lower:
            rate_type = "par"
        else:
            return None
    else:
        if "tnc treasury yield curve spot rates, monthly average" in lower:
            rate_type = "spot"
        elif "tnc treasury yield curve spot rates, end of month" in lower:
            rate_type = "spot"
        elif "tnc treasury yield curve par yields, monthly average" in lower:
            rate_type = "par"
        elif "tnc treasury yield curve par yields, end of month" in lower:
            rate_type = "par"
        else:
            return None
    observation_type = "end_of_month" if "end of month" in lower else "monthly_average"
    return rate_type, observation_type


def source_end_year(text: str) -> int:
    if "present" in text.lower():
        return 9999
    years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", text)]
    return max(years, default=0)


def discover_sources(update_only: bool) -> list[dict[str, str]]:
    session = requests.Session()
    session.headers["User-Agent"] = "Quantitative-Finance-Lab Treasury data builder"
    sources = []
    for family, page in PAGES.items():
        response = session.get(page, timeout=120)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for anchor in soup.find_all("a", href=True):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            url = urljoin(response.url, anchor["href"])
            if Path(urlparse(url).path).suffix.lower() not in {".xls", ".xlsx"}:
                continue
            kind = source_kind(family, text)
            if kind is None:
                continue
            rate_type, observation_type = kind
            sources.append(
                {
                    "family": family,
                    "rate_type": rate_type,
                    "observation_type": observation_type,
                    "text": text,
                    "url": url,
                }
            )
    unique = {source["url"]: source for source in sources}
    if not unique:
        raise RuntimeError("No Treasury HQM/TNC workbooks were discovered.")
    sources = list(unique.values())
    if update_only:
        latest: dict[tuple[str, str, str], dict[str, str]] = {}
        for source in sources:
            key = (source["family"], source["rate_type"], source["observation_type"])
            if key not in latest or source_end_year(source["text"]) > source_end_year(latest[key]["text"]):
                latest[key] = source
        sources = list(latest.values())
    return sorted(sources, key=lambda source: source["url"])


def fetch_sources(sources: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool]:
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = json.loads(STATE.read_text()) if STATE.exists() else {}
    session = requests.Session()
    session.headers["User-Agent"] = "Quantitative-Finance-Lab Treasury data builder"
    changed = False
    for source in sources:
        previous = saved.get(source["url"], {})
        headers = {}
        if previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
        response = session.get(source["url"], headers=headers, timeout=180)
        filename = f"{source['family'].lower()}_{Path(urlparse(source['url']).path).name}"
        path = CACHE / filename
        if response.status_code == 304 and path.exists():
            source.update(previous)
            source["path"] = filename
            saved[source["url"]] = source
            continue
        response.raise_for_status()
        if response.content[:2] not in {b"PK", b"\xd0\xcf"}:
            raise ValueError(f"Treasury response is not an Excel workbook: {source['url']}")
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


def read_workbook(path: Path) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    return pd.read_excel(path, sheet_name=0, header=None, engine=engine)


def maturity_number(value: object) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def parse_spot(frame: pd.DataFrame, source: dict[str, str]) -> pd.DataFrame:
    matches = frame.index[frame.iloc[:, 0].astype(str).str.strip().eq("Maturity")]
    if matches.empty:
        raise ValueError(f"Could not locate spot-curve headers in {source['path']}")
    header = int(matches[0])
    current_year: int | None = None
    columns: list[tuple[int, pd.Timestamp]] = []
    for column in range(2, frame.shape[1]):
        year = pd.to_numeric(frame.iat[header - 1, column], errors="coerce")
        if pd.notna(year):
            current_year = int(year)
        month = str(frame.iat[header, column]).strip()[:3]
        if current_year is None or month.lower() == "nan":
            continue
        try:
            date = pd.Timestamp(f"{current_year}-{month}-01") + pd.offsets.MonthEnd(0)
        except ValueError:
            continue
        columns.append((column, date))

    rows = []
    for row in range(header + 1, frame.shape[0]):
        maturity = pd.to_numeric(frame.iat[row, 0], errors="coerce")
        if pd.isna(maturity):
            continue
        for column, date in columns:
            value = pd.to_numeric(frame.iat[row, column], errors="coerce")
            if pd.notna(value):
                rows.append((date, float(maturity), float(value)))
    return pd.DataFrame(rows, columns=["date", "maturity_years", "yield_percent"])


def parse_par(frame: pd.DataFrame, source: dict[str, str]) -> pd.DataFrame:
    matches = frame.index[frame.iloc[:, 0].astype(str).str.strip().eq("Date")]
    if matches.empty:
        raise ValueError(f"Could not locate par-curve headers in {source['path']}")
    header = int(matches[0])
    maturities = {
        column: maturity_number(frame.iat[header + 1, column])
        for column in range(2, frame.shape[1])
    }
    rows = []
    for row in range(header + 2, frame.shape[0]):
        date = pd.to_datetime(frame.iat[row, 0], errors="coerce")
        if pd.isna(date):
            continue
        date = pd.Timestamp(date) + pd.offsets.MonthEnd(0)
        for column, maturity in maturities.items():
            value = pd.to_numeric(frame.iat[row, column], errors="coerce")
            if maturity is not None and pd.notna(value):
                rows.append((date, maturity, float(value)))
    return pd.DataFrame(rows, columns=["date", "maturity_years", "yield_percent"])


def build(catalog: list[dict[str, str]]) -> None:
    pieces = []
    for source in catalog:
        path = CACHE / source["path"]
        if not path.exists() or source.get("rate_type") not in {"spot", "par"}:
            continue
        frame = read_workbook(path)
        parsed = parse_spot(frame, source) if source["rate_type"] == "spot" else parse_par(frame, source)
        parsed.insert(0, "curve_family", source["family"])
        parsed.insert(1, "rate_type", source["rate_type"])
        parsed.insert(2, "observation_type", source["observation_type"])
        parsed["source_file"] = source["path"]
        pieces.append(parsed)
    if not pieces:
        raise RuntimeError("No cached Treasury curve workbooks could be parsed.")
    curves = pd.concat(pieces, ignore_index=True)
    key = ["curve_family", "rate_type", "observation_type", "date", "maturity_years"]
    curves = curves.sort_values(key + ["source_file"]).drop_duplicates(key, keep="last")
    curves = curves.sort_values(key, ignore_index=True)
    if curves.duplicated(key).any() or not np.isfinite(curves["yield_percent"]).all():
        raise ValueError("Treasury curve output has duplicate keys or non-finite yields.")
    if curves.loc[curves["curve_family"].eq("HQM"), "date"].min().year != 1984:
        raise ValueError("HQM history does not begin in 1984.")
    if curves.loc[curves["curve_family"].eq("TNC"), "date"].min().year != 1976:
        raise ValueError("TNC history does not begin in 1976.")

    metadata = {
        "dataset": "U.S. Treasury HQM corporate and TNC nominal Treasury yield curves",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pages": json.dumps(PAGES, sort_keys=True),
        "units": "percent",
        "date_min": str(curves["date"].min().date()),
        "date_max": str(curves["date"].max().date()),
    }
    table = pa.Table.from_pandas(curves, preserve_index=False).replace_schema_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()}
    )
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(OUTPUT)
    print(
        f"wrote {OUTPUT} rows={len(curves):,} "
        f"date={curves['date'].min().date()}..{curves['date'].max().date()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    update_only = args.update and OUTPUT.exists() and STATE.exists()
    sources = discover_sources(update_only)
    catalog, changed = fetch_sources(sources)
    if changed or not OUTPUT.exists():
        build(catalog)
    else:
        print("Treasury curve sources are unchanged")


if __name__ == "__main__":
    main()
