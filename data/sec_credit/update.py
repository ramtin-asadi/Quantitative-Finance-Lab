from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import companyfacts
import download as source
import orjson
import pandas as pd
import pyarrow.parquet as pq
import review_events

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STATE = HERE / "cache" / "update_state.json"
COMPANYFACTS_HTTP_MANIFEST = HERE / "cache" / "companyfacts_http_manifest.json"
BUILDER = HERE / "build.py"
OUTPUT = ROOT / "data" / "sec_credit.parquet"
FUNDAMENTALS_UPDATER = ROOT / "data" / "sp500_fundamentals" / "update.py"
MASTER_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
RELEVANT_FORMS = {
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "8-K",
    "8-K/A",
    "NT 10-K",
    "NT 10-Q",
    "25",
    "25-NSE",
    "15-12B",
    "15-12G",
    "15-15D",
}


def quarter_sequence(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    year, quarter = start.year, (start.month - 1) // 3 + 1
    final = (end.year, (end.month - 1) // 3 + 1)
    result = []
    while (year, quarter) <= final:
        result.append((year, quarter))
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    return result


def parse_master(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith("master.idx")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one master.idx, found {len(names)}")
        text = archive.read(names[0]).decode("latin-1")
    lines = text.splitlines()
    separator = next((index for index, line in enumerate(lines) if line.startswith("---")), None)
    if separator is None:
        raise RuntimeError("Unexpected SEC master index format")
    rows = [line.split("|") for line in lines[separator + 1 :] if line.count("|") == 4]
    frame = pd.DataFrame(
        rows, columns=["cik", "company_name", "form_type", "filed_date", "filename"]
    )
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")
    frame["filed_date"] = pd.to_datetime(frame["filed_date"], errors="coerce")
    return frame.dropna(subset=["cik", "filed_date"])


def fetch_index(
    identity: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, list[str]]:
    client = source.SecClient(identity)
    frames = []
    urls = []
    for year, quarter in quarter_sequence(start, end):
        url = MASTER_URL.format(year=year, quarter=quarter)
        response = client.get(url)
        frames.append(parse_master(response.content))
        urls.append(url)
    index = pd.concat(frames, ignore_index=True).drop_duplicates()
    return index.loc[index["filed_date"].between(start, end, inclusive="both")], urls


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh_companyfacts(
    identity: str, ciks: list[int], workers: int = 6
) -> dict[str, int]:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    client = source.SecClient(identity)
    manifest = read_json(COMPANYFACTS_HTTP_MANIFEST)
    files = manifest.setdefault("files", {})
    prepared = companyfacts.read_manifest()
    known_hashes = (
        prepared.set_index("cik")["source_sha256"].astype(str).to_dict()
        if not prepared.empty
        else {}
    )
    stats = {"ciks": len(ciks), "changed": 0, "unchanged": 0, "transferred_bytes": 0}

    def fetch(cik: int) -> tuple[int, bool, int, dict[str, str] | None]:
        path = companyfacts.COMPANYFACTS / f"CIK{cik:010d}.json"
        record = files.get(str(cik), {})
        headers = {}
        if path.exists() and record.get("etag"):
            headers["If-None-Match"] = record["etag"]
        if path.exists() and record.get("last_modified"):
            headers["If-Modified-Since"] = record["last_modified"]
        response = client.get(COMPANYFACTS_URL.format(cik=cik), headers)
        if response.status_code == 304:
            return cik, False, 0, None
        payload = orjson.loads(response.content)
        if int(payload.get("cik", cik)) != cik:
            raise ValueError(f"Company Facts CIK mismatch: expected {cik}")
        digest = hashlib.sha256(response.content).hexdigest()
        baseline = record.get("sha256") or known_hashes.get(cik, "")
        changed = digest != baseline or not path.exists()
        if changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
        transferred = int(response.headers.get("Content-Length", len(response.content)))
        new_record = {
            "sha256": digest,
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return cik, changed, transferred, new_record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, cik) for cik in ciks]
        for number, future in enumerate(as_completed(futures), start=1):
            cik, changed, transferred, record = future.result()
            stats["changed" if changed else "unchanged"] += 1
            stats["transferred_bytes"] += transferred
            if record is not None:
                files[str(cik)] = record
            if number % 100 == 0 or number == len(ciks):
                print(
                    f"companyfacts update {number:,}/{len(ciks):,} "
                    f"changed={stats['changed']:,} "
                    f"network_mb={stats['transferred_bytes'] / 1e6:.1f}",
                    flush=True,
                )
    manifest["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(COMPANYFACTS_HTTP_MANIFEST, manifest)
    return stats


def accounting_backlog(
    relevant: pd.DataFrame,
    state: dict[str, object],
    model_ciks: set[int],
) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
    periodic = relevant.loc[relevant["form_type"].isin(companyfacts.PERIODIC_FORMS)].copy()
    periodic["cik_int"] = periodic["cik"].astype(int)
    periodic["accession"] = periodic["filename"].str.extract(
        r"(\d{10}-\d{2}-\d{6})", expand=False
    )
    periodic = periodic.dropna(subset=["accession"])
    periodic = periodic.loc[periodic["cik_int"].isin(model_ciks)]
    checked = {
        str(cik): list(dict.fromkeys(accessions))
        for cik, accessions in dict(state.get("checked_accounting_accessions", {})).items()
    }
    pending = {
        int(cik): list(dict.fromkeys(accessions))
        for cik, accessions in dict(state.get("pending_accounting_accessions", {})).items()
        if int(cik) in model_ciks
    }
    for cik, group in periodic.groupby("cik_int", sort=False):
        path = companyfacts.fact_cache_path(int(cik))
        cached = (
            set(pd.read_parquet(path, columns=["accession"])["accession"].unique())
            if path.exists()
            else set()
        )
        known = set(checked.get(str(int(cik)), []))
        missing = sorted(set(group["accession"]) - cached - known)
        if missing:
            pending[int(cik)] = sorted(set(pending.get(int(cik), [])) | set(missing))
    return pending, checked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh the broad SEC credit universe without bulk ZIPs."
    )
    parser.add_argument("--identity", default=os.getenv("EDGAR_IDENTITY", ""))
    parser.add_argument("--overlap-days", type=int, default=14)
    parser.add_argument("--skip-sp500-fundamentals", action="store_true")
    parser.add_argument(
        "--accounting-batch-size",
        type=int,
        default=100,
        help=(
            "Maximum affected CIKs whose full per-CIK Company Facts JSON may be "
            "refreshed this run; zero updates filing events only."
        ),
    )
    args = parser.parse_args()
    identity = source.validate_identity(args.identity)
    if args.overlap_days < 0:
        raise ValueError("--overlap-days cannot be negative")
    if args.accounting_batch_size < 0:
        raise ValueError("--accounting-batch-size cannot be negative")

    if not args.skip_sp500_fundamentals:
        subprocess.run(
            [sys.executable, str(FUNDAMENTALS_UPDATER), "--identity", identity],
            cwd=ROOT,
            check=True,
        )

    state = read_json(STATE)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    checkpoint = pd.to_datetime(state.get("last_index_date"), errors="coerce")
    if pd.isna(checkpoint):
        checkpoint = today
    scan_start = pd.Timestamp(checkpoint) - pd.Timedelta(days=args.overlap_days)
    index, urls = fetch_index(identity, scan_start, today)
    relevant = index.loc[index["form_type"].isin(RELEVANT_FORMS)].copy()

    model_ciks = {
        int(value.as_py())
        for value in pq.read_table(OUTPUT, columns=["cik"]).column("cik").unique()
    }
    pending_accounting, checked_accounting = accounting_backlog(
        relevant, state, model_ciks
    )
    fact_targets = sorted(pending_accounting)[: args.accounting_batch_size]
    fact_stats = refresh_companyfacts(identity, fact_targets) if fact_targets else {}
    if fact_targets:
        prepare_stats = companyfacts.prepare_companyfacts(ciks=fact_targets)
        for cik in fact_targets:
            checked = set(checked_accounting.get(str(cik), []))
            checked.update(pending_accounting.pop(cik, []))
            checked_accounting[str(cik)] = sorted(checked)
    else:
        prepare_stats = {}

    submission_targets = set(
        relevant.loc[relevant["cik"].astype(int).isin(model_ciks), "cik"].astype(int)
    )
    submission_targets.update(
        cik
        for cik in model_ciks
        if not (source.SUBMISSIONS / f"CIK{cik:010d}.json").exists()
    )
    submission_stats: dict[str, int] = {}
    if submission_targets:
        submission_stats = source.download(
            identity=identity,
            start=source.DEFAULT_START,
            ciks=sorted(submission_targets),
            refresh_current=True,
        )
        review_events.review(identity=identity, ciks=sorted(submission_targets))
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)

    observed = index["filed_date"].max()
    write_json(
        STATE,
        {
            "last_successful_run_utc": datetime.now(timezone.utc).isoformat(),
            "last_index_date": str((observed if pd.notna(observed) else today).date()),
            "scan_start": str(scan_start.date()),
            "index_urls": urls,
            "relevant_index_filings": len(relevant),
            "companyfacts_targets": fact_targets,
            "pending_accounting_accessions": {
                str(cik): accessions for cik, accessions in sorted(pending_accounting.items())
            },
            "checked_accounting_accessions": checked_accounting,
            "submission_targets": sorted(submission_targets),
            "companyfacts_network_stats": fact_stats,
            "companyfacts_prepare_stats": prepare_stats,
            "submissions_network_stats": submission_stats,
            "submissions_zip_downloaded": False,
            "companyfacts_zip_downloaded": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "scan_start": str(scan_start.date()),
                "scan_end": str(today.date()),
                "relevant_index_filings": len(relevant),
                "companyfacts_ciks_refreshed": len(fact_targets),
                "pending_accounting_ciks": len(pending_accounting),
                "submission_ciks_refreshed": len(submission_targets),
                "companyfacts_transferred_mb": round(
                    fact_stats.get("transferred_bytes", 0) / 1e6, 2
                ),
                "submissions_transferred_mb": round(
                    submission_stats.get("transferred_bytes", 0) / 1e6, 2
                ),
                "bulk_archives_downloaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
