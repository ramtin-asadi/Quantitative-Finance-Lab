from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
STATE = CACHE / "sources.json"
OUTPUT = DATA / "fred_md_qd_vintages.parquet"
PAGE = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"
VINTAGE_PATTERN = re.compile(r"((?:19|20)\d{2})[-_m](1[0-2]|0?[1-9])(?!\d)", re.IGNORECASE)
SCHEMA = pa.schema(
    [
        pa.field("panel", pa.string(), nullable=False),
        pa.field("vintage_date", pa.timestamp("ns"), nullable=False),
        pa.field("observation_date", pa.timestamp("ns"), nullable=False),
        pa.field("series_id", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("transformation", pa.int8()),
        pa.field("factor_group", pa.int8()),
        pa.field("source_file", pa.string(), nullable=False),
    ]
)
CURL = shutil.which("curl")


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = "Quantitative-Finance-Lab FRED-MD-QD builder"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def read_page() -> tuple[str, str]:
    if CURL:
        result = subprocess.run(
            [CURL, "-fsSL", "--retry", "5", "--retry-all-errors", PAGE],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout, PAGE
    response = make_session().get(PAGE, timeout=120)
    response.raise_for_status()
    return response.text, response.url


def curl_download(url: str, path: Path) -> None:
    subprocess.run(
        [
            CURL,
            "-fL",
            "--retry",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "1200",
            "--output",
            str(path),
            url,
        ],
        check=True,
    )


def discover(update_only: bool) -> list[dict[str, str]]:
    page, final_url = read_page()
    soup = BeautifulSoup(page, "html.parser")
    sources = []
    for anchor in soup.find_all("a", href=True):
        text = " ".join(anchor.get_text(" ", strip=True).split())
        url = urljoin(final_url, anchor["href"])
        lower = f"{text} {url}".lower()
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in {".zip", ".csv"}:
            continue
        if "fred-md" not in lower and "fred_qd" not in lower and "fred-qd" not in lower:
            continue
        panel = "MD" if "monthly" in lower or "fred-md" in text.lower() else "QD"
        historical = "historical" in text.lower()
        if update_only and historical:
            continue
        if suffix == ".zip" and not historical:
            continue
        if suffix == ".csv" and not VINTAGE_PATTERN.search(Path(urlparse(url).path).name):
            continue
        sources.append({"panel": panel, "text": text, "url": url, "historical": historical})
    unique = {source["url"]: source for source in sources}
    if not unique:
        raise RuntimeError("No FRED-MD/QD archives or snapshots were discovered.")
    return sorted(unique.values(), key=lambda source: source["url"])


def fetch(sources: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = json.loads(STATE.read_text()) if STATE.exists() else {}
    changed = []
    session = make_session()
    for source in sources:
        previous = saved.get(source["url"], {})
        filename = f"{source['panel'].lower()}_{Path(urlparse(source['url']).path).name}"
        path = CACHE / filename
        if path.exists() and previous.get("sha256"):
            source.update(previous)
            source["path"] = filename
            saved[source["url"]] = source
            continue
        headers = {}
        if previous.get("etag"):
            headers["If-None-Match"] = previous["etag"]
        if previous.get("last_modified"):
            headers["If-Modified-Since"] = previous["last_modified"]
        temporary = path.with_suffix(path.suffix + ".tmp")
        if CURL:
            curl_download(source["url"], temporary)
            content = temporary.read_bytes()
            etag = None
            last_modified = None
        else:
            response = session.get(source["url"], headers=headers, timeout=300)
            if response.status_code == 304 and path.exists():
                source.update(previous)
                source["path"] = filename
                saved[source["url"]] = source
                continue
            response.raise_for_status()
            content = response.content
            temporary.write_bytes(content)
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
        expected = b"PK" if path.suffix.lower() == ".zip" else b"sasdate,"
        if not content.startswith(expected):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Unexpected FRED-MD/QD response: {source['url']}")
        sha256 = hashlib.sha256(content).hexdigest()
        if path.exists() and previous.get("sha256") == sha256:
            temporary.unlink(missing_ok=True)
            source.update(previous)
            source["path"] = filename
            saved[source["url"]] = source
            continue
        temporary.replace(path)
        source.update(
            {
                "path": filename,
                "etag": etag,
                "last_modified": last_modified,
                "sha256": sha256,
                "bytes": len(content),
            }
        )
        saved[source["url"]] = source
        changed.append(source)
        print(f"downloaded {filename} ({len(content) / 1e6:.2f} MB)")
    STATE.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    return list(saved.values()), changed


def vintage_from_name(name: str) -> pd.Timestamp:
    match = VINTAGE_PATTERN.search(name)
    if not match:
        raise ValueError(f"No monthly vintage in {name}")
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def parse_snapshot(content: bytes, panel: str, vintage: pd.Timestamp, source_file: str) -> pa.Table:
    frame = pd.read_csv(io.BytesIO(content), low_memory=False)
    date_column = frame.columns[0]
    if panel == "MD":
        transformation = pd.to_numeric(frame.iloc[0, 1:], errors="coerce")
        factor = pd.Series(pd.NA, index=frame.columns[1:], dtype="Int8")
        values = frame.iloc[1:].copy()
    else:
        factor = pd.to_numeric(frame.iloc[0, 1:], errors="coerce").astype("Int8")
        transformation = pd.to_numeric(frame.iloc[1, 1:], errors="coerce")
        values = frame.iloc[2:].copy()
    values[date_column] = pd.to_datetime(values[date_column], errors="coerce")
    values = values.dropna(subset=[date_column])
    long = values.melt(id_vars=date_column, var_name="series_id", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long.insert(0, "panel", panel)
    long.insert(1, "vintage_date", vintage)
    long = long.rename(columns={date_column: "observation_date"})
    long["transformation"] = long["series_id"].map(transformation).astype("Int8")
    long["factor_group"] = long["series_id"].map(factor).astype("Int8")
    long["source_file"] = source_file
    return pa.Table.from_pandas(long, schema=SCHEMA, preserve_index=False, safe=False)


def snapshots(source: dict[str, str]):
    path = CACHE / source["path"]
    if path.suffix.lower() == ".csv":
        yield path.read_bytes(), source["panel"], vintage_from_name(path.name), path.name
        return
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.lower().endswith(".csv") or not VINTAGE_PATTERN.search(name):
                continue
            yield archive.read(name), source["panel"], vintage_from_name(name), f"{path.name}:{name}"


def output_metadata() -> dict[bytes, bytes]:
    values = {
        "dataset": "FRED-MD and FRED-QD real-time vintage archive",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": PAGE,
        "timing": "vintage_date is the St. Louis Fed snapshot month, not an observation release timestamp",
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def write_full(catalog: list[dict[str, str]]) -> None:
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, SCHEMA.with_metadata(output_metadata()), compression="zstd")
    rows = 0
    try:
        for source in sorted(catalog, key=lambda item: item["path"]):
            if not (CACHE / source["path"]).exists():
                continue
            for content, panel, vintage, source_file in snapshots(source):
                table = parse_snapshot(content, panel, vintage, source_file)
                writer.write_table(table, row_group_size=250_000)
                rows += table.num_rows
                print(f"parsed {panel} {vintage:%Y-%m}: {table.num_rows:,} rows")
    finally:
        writer.close()
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("FRED-MD/QD parsing produced no rows.")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT} rows={rows:,} size_mb={OUTPUT.stat().st_size / 1e6:.1f}")


def write_increment(changed: list[dict[str, str]]) -> None:
    changed_snapshots = []
    for source in changed:
        changed_snapshots.extend(list(snapshots(source)))
    replace = {(panel, vintage.value) for _, panel, vintage, _ in changed_snapshots}
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, SCHEMA.with_metadata(output_metadata()), compression="zstd")
    old = pq.ParquetFile(OUTPUT)
    try:
        for batch in old.iter_batches(batch_size=250_000):
            table = pa.Table.from_batches([batch])
            keep = pa.array(
                [
                    (panel, pd.Timestamp(vintage).value) not in replace
                    for panel, vintage in zip(
                        table["panel"].to_pylist(), table["vintage_date"].to_pylist()
                    )
                ]
            )
            filtered = table.filter(keep)
            if filtered.num_rows:
                writer.write_table(filtered)
        for content, panel, vintage, source_file in changed_snapshots:
            writer.write_table(parse_snapshot(content, panel, vintage, source_file))
    finally:
        writer.close()
        old.close()
    temporary.replace(OUTPUT)
    print(f"incrementally updated {OUTPUT} snapshots={len(replace):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.build_only:
        if not STATE.exists():
            raise FileNotFoundError("No cached source catalog exists. Run download.py first.")
        write_full(list(json.loads(STATE.read_text()).values()))
        return
    update_only = args.update and OUTPUT.exists() and STATE.exists()
    catalog, changed = fetch(discover(update_only))
    if not OUTPUT.exists():
        write_full(catalog)
    elif changed:
        if any(Path(source["path"]).suffix.lower() == ".zip" for source in changed):
            write_full(catalog)
        else:
            write_increment(changed)
    else:
        print("FRED-MD/QD sources are unchanged")


if __name__ == "__main__":
    main()
