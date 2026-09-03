from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import companyfacts
import orjson
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SUBMISSIONS = CACHE / "submissions"
MANIFEST = CACHE / "submissions_manifest.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
DEFAULT_START = "2012-01-01"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_identity(identity: str) -> str:
    cleaned = " ".join(identity.strip().split())
    if "@" not in cleaned or " " not in cleaned:
        raise ValueError(
            "Set EDGAR_IDENTITY to a descriptive name and email, for example "
            "'Jane Doe jane@example.com'."
        )
    return cleaned


class SecClient:
    def __init__(self, identity: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": validate_identity(identity), "Accept-Encoding": "gzip, deflate"}
        )
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.last_request = 0.0
        self.rate_lock = threading.Lock()

    def get(self, url: str, headers: dict[str, str] | None = None) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(5):
            with self.rate_lock:
                wait = 0.12 - (time.monotonic() - self.last_request)
                if wait > 0:
                    time.sleep(wait)
                self.last_request = time.monotonic()
            try:
                response = self.session.get(url, headers=headers, timeout=(15, 180))
                if response.status_code == 304:
                    return response
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"SEC returned HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"SEC request failed for {url}: {last_error}") from last_error


def read_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"files": {}}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(MANIFEST)


def cached_payload(name: str) -> dict[str, Any]:
    path = SUBMISSIONS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return orjson.loads(path.read_bytes())


def fetch_file(
    client: SecClient,
    name: str,
    manifest: dict[str, Any],
    refresh: bool,
) -> tuple[dict[str, Any], bool, int]:
    path = SUBMISSIONS / name
    record = manifest.setdefault("files", {}).get(name, {})
    if path.exists() and not refresh:
        return cached_payload(name), False, 0

    conditional = {}
    if path.exists() and record.get("etag"):
        conditional["If-None-Match"] = record["etag"]
    if path.exists() and record.get("last_modified"):
        conditional["If-Modified-Since"] = record["last_modified"]
    url = SUBMISSIONS_URL.format(name=name)
    response = client.get(url, conditional)
    if response.status_code == 304:
        record["checked_at_utc"] = utc_now()
        manifest["files"][name] = record
        return cached_payload(name), False, 0

    payload = orjson.loads(response.content)
    digest = hashlib.sha256(response.content).hexdigest()
    changed = record.get("sha256") != digest or not path.exists()
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    manifest["files"][name] = {
        "url": url,
        "sha256": digest,
        "decoded_bytes": len(response.content),
        "etag": response.headers.get("ETag", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
        "checked_at_utc": utc_now(),
    }
    transferred = int(response.headers.get("Content-Length", len(response.content)))
    return payload, changed, transferred


def history_names(payload: dict[str, Any], start: str) -> list[str]:
    files = payload.get("filings", {}).get("files", [])
    return sorted(
        item["name"]
        for item in files
        if item.get("name") and item.get("filingTo", "") >= start
    )


def nonfinancial_operating(payload: dict[str, Any]) -> tuple[bool, str]:
    sic = pd.to_numeric(payload.get("sic"), errors="coerce")
    entity_type = str(payload.get("entityType", "")).lower()
    if pd.isna(sic):
        return False, "missing_sic"
    if 6000 <= int(sic) <= 6999:
        return False, "financial"
    if entity_type not in {"", "operating"}:
        return False, "nonoperating"
    return True, "kept"


def download(
    identity: str,
    start: str = DEFAULT_START,
    ciks: list[int] | None = None,
    refresh_current: bool = False,
    workers: int = 6,
) -> dict[str, int]:
    candidates = sorted(set(ciks if ciks is not None else companyfacts.eligible_ciks()))
    if not candidates:
        raise ValueError("No eligible SEC CIKs were selected")
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    client = SecClient(identity)
    stats = {
        "ciks": len(candidates),
        "nonfinancial_operating_ciks": 0,
        "excluded_financial_ciks": 0,
        "excluded_nonoperating_ciks": 0,
        "excluded_missing_sic_ciks": 0,
        "requests_changed": 0,
        "requests_unchanged": 0,
        "transferred_bytes": 0,
        "history_files": 0,
    }
    history_candidates: list[int] = []

    def fetch_current(cik: int) -> tuple[int, dict[str, Any], bool, int]:
        name = f"CIK{cik:010d}.json"
        payload, changed, transferred = fetch_file(
            client, name, manifest, refresh=refresh_current
        )
        payload_cik = int(payload.get("cik", cik))
        if payload_cik != cik:
            raise ValueError(f"Submissions CIK mismatch: expected {cik}, got {payload_cik}")
        return cik, payload, changed, transferred

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_current, cik) for cik in candidates]
        for number, future in enumerate(as_completed(futures), start=1):
            cik, payload, changed, transferred = future.result()
            stats["requests_changed" if changed else "requests_unchanged"] += 1
            stats["transferred_bytes"] += transferred
            keep, reason = nonfinancial_operating(payload)
            if keep:
                history_candidates.append(cik)
                stats["nonfinancial_operating_ciks"] += 1
            else:
                stats[f"excluded_{reason}_ciks"] += 1
            if number % 100 == 0 or number == len(candidates):
                print(
                    f"submissions current {number:,}/{len(candidates):,} "
                    f"kept={len(history_candidates):,} "
                    f"network_mb={stats['transferred_bytes'] / 1e6:.1f}",
                    flush=True,
                )

    history_files = sorted(
        {
            name
            for cik in history_candidates
            for name in history_names(cached_payload(f"CIK{cik:010d}.json"), start)
        }
    )

    def fetch_history(name: str) -> tuple[bool, int]:
        _payload, changed, transferred = fetch_file(
            client, name, manifest, refresh=False
        )
        return changed, transferred

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_history, name) for name in history_files]
        for number, future in enumerate(as_completed(futures), start=1):
            history_changed, history_bytes = future.result()
            stats["history_files"] += 1
            stats["requests_changed" if history_changed else "requests_unchanged"] += 1
            stats["transferred_bytes"] += history_bytes
            if number % 100 == 0 or number == len(history_files):
                print(
                    f"submissions history {number:,}/{len(history_files):,} "
                    f"network_mb={stats['transferred_bytes'] / 1e6:.1f}",
                    flush=True,
                )
    write_manifest(manifest)
    manifest["last_run_utc"] = utc_now()
    manifest["start"] = start
    manifest["eligible_companyfacts_ciks"] = len(companyfacts.eligible_ciks())
    manifest["submissions_zip_downloaded"] = False
    write_manifest(manifest)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Screen the existing all-filer Company Facts cache, then download SEC "
            "submissions only for qualifying U.S.-GAAP operating-company candidates."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--identity", default=os.getenv("EDGAR_IDENTITY", ""))
    parser.add_argument("--cik", type=int, action="append")
    parser.add_argument("--refresh-current", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--skip-event-review", action="store_true")
    parser.add_argument("--min-periodic-filings", type=int, default=8)
    parser.add_argument("--min-history-years", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    pd.Timestamp(args.start)
    if args.prepare_only and args.skip_prepare:
        raise ValueError("--prepare-only and --skip-prepare cannot be combined")
    if not args.skip_prepare:
        stats = companyfacts.prepare_companyfacts(
            start=args.start,
            ciks=args.cik,
            min_periodic_filings=args.min_periodic_filings,
            min_history_years=args.min_history_years,
            force=args.force_prepare,
        )
        print(json.dumps({"companyfacts": stats}, indent=2, sort_keys=True))
    if args.prepare_only:
        return
    download(
        identity=args.identity,
        start=args.start,
        ciks=args.cik,
        refresh_current=args.refresh_current,
        workers=args.workers,
    )
    if not args.skip_event_review:
        command = [
            sys.executable,
            str(HERE / "review_events.py"),
            "--identity",
            validate_identity(args.identity),
        ]
        for cik in args.cik or []:
            command.extend(["--cik", str(cik)])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
