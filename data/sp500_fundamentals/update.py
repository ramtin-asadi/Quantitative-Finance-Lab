from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import download as base
import orjson
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests

STATE_PATH = base.CACHE_DIR / "incremental_update_state.json"
MAPPING_CACHE_PATH = base.CACHE_DIR / "ticker_cik_mapping.parquet"
SEC_XBRL_INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/xbrl.zip"
SEC_COMPANYFACTS_CIK_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010}.json"
DEFAULT_INDEX_OVERLAP_DAYS = 10


def log(message: str, indent: int = 0) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {'  ' * indent}{message}", flush=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SecClient:
    def __init__(self, identity: str, min_interval_seconds: float = 0.12) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": base.validate_identity(identity),
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self.min_interval_seconds = min_interval_seconds
        self.last_request_at = 0.0

    def get(self, url: str, timeout: tuple[int, int] = (15, 180)) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(5):
            wait = self.min_interval_seconds - (time.monotonic() - self.last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, timeout=timeout)
                self.last_request_at = time.monotonic()
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


def read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATE_PATH)


def parquet_metadata(path: Path) -> dict[str, str]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    parquet.close()
    return {key.decode(): value.decode() for key, value in metadata.items()}


def existing_period_start(path: Path) -> pd.Timestamp:
    metadata = parquet_metadata(path)
    value = metadata.get("period_end_start", base.DEFAULT_PERIOD_START)
    return pd.Timestamp(value)


def existing_max_filed_date(path: Path) -> pd.Timestamp:
    metadata = parquet_metadata(path)
    validation = json.loads(metadata.get("validation", "{}"))
    value = validation.get("filed_date_max")
    if value:
        return pd.Timestamp(value)
    dataset = ds.dataset(path, format="parquet")
    maximum: pd.Timestamp | None = None
    for batch in dataset.scanner(columns=["filed_date"], batch_size=262_144).to_batches():
        candidate = batch.column(0).to_pandas().max()
        if pd.notna(candidate):
            maximum = candidate if maximum is None else max(maximum, candidate)
    if maximum is None:
        raise RuntimeError("Existing fundamentals parquet has no filing dates")
    return pd.Timestamp(maximum)


def load_existing_mapping(path: Path) -> pd.DataFrame:
    columns = [
        "ticker",
        "cik",
        "entity_name",
        "mapping_source",
        "mapping_confidence",
        "member_first_date",
        "member_last_date",
        "mapping_valid_from",
        "mapping_valid_to",
        "ticker_evidence_first_filed",
        "ticker_evidence_last_filed",
    ]
    pieces: list[pd.DataFrame] = []
    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.scanner(columns=columns, batch_size=262_144).to_batches():
        pieces.append(batch.to_pandas().drop_duplicates())
    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates()
        .sort_values(["ticker", "cik"], ignore_index=True)
    )


def quarter_sequence(start: date, end: date) -> list[tuple[int, int]]:
    year, quarter = start.year, (start.month - 1) // 3 + 1
    final = (end.year, (end.month - 1) // 3 + 1)
    result: list[tuple[int, int]] = []
    while (year, quarter) <= final:
        result.append((year, quarter))
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    return result


def parse_xbrl_index(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith("xbrl.idx")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one xbrl.idx, found {len(names)}")
        text = archive.read(names[0]).decode("latin-1")
    lines = text.splitlines()
    separator = next(
        (index for index, line in enumerate(lines) if line.startswith("---")),
        None,
    )
    if separator is None:
        raise RuntimeError("Unexpected SEC XBRL index format")
    rows: list[list[str]] = []
    for line in lines[separator + 1 :]:
        fields = line.split("|")
        if len(fields) == 5:
            rows.append(fields)
    frame = pd.DataFrame(
        rows,
        columns=["cik", "company_name", "form_type", "filed_date", "filename"],
    )
    frame["cik"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64")
    frame["filed_date"] = pd.to_datetime(frame["filed_date"], errors="coerce")
    frame["form_type"] = frame["form_type"].astype("string").str.upper()
    frame["accession"] = (
        frame["filename"].str.rsplit("/", n=1).str[-1].str.replace(".txt", "", regex=False)
    )
    return frame.dropna(subset=["cik", "filed_date"]).reset_index(drop=True)


def fetch_xbrl_indexes(
    client: SecClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    urls: list[str] = []
    for year, quarter in quarter_sequence(start.date(), end.date()):
        url = SEC_XBRL_INDEX_URL.format(year=year, quarter=quarter)
        try:
            response = client.get(url)
        except RuntimeError as exc:
            log(f"WARNING: skipped unavailable SEC index {year} Q{quarter}: {exc}", 1)
            continue
        frame = parse_xbrl_index(response.content)
        frames.append(frame)
        urls.append(url)
        log(f"SEC XBRL index {year} Q{quarter}: {len(frame):,} filings", 1)
    if not frames:
        raise RuntimeError("No SEC XBRL indexes were available")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["cik", "form_type", "filed_date", "accession"]
    )
    combined = combined.loc[
        combined["filed_date"].between(start, end, inclusive="both")
    ].reset_index(drop=True)
    return combined, urls


def cached_source_accessions(cik: int) -> set[str]:
    path = base.COMPANYFACTS_DIR / f"CIK{int(cik):010}.json"
    if not path.exists():
        return set()
    try:
        payload = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return set()
    return payload_accessions(payload)


def payload_accessions(payload: dict[str, Any]) -> set[str]:
    accessions: set[str] = set()
    for taxonomy in payload.get("facts", {}).values():
        for concept in taxonomy.values():
            for unit_facts in concept.get("units", {}).values():
                for fact in unit_facts:
                    accession = fact.get("accn")
                    if accession:
                        accessions.add(str(accession))
    return accessions


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def fetch_companyfacts(
    client: SecClient,
    cik: int,
) -> tuple[bool, dict[str, Any], str]:
    url = SEC_COMPANYFACTS_CIK_URL.format(cik=int(cik))
    response = client.get(url)
    payload = orjson.loads(response.content)
    payload_cik = int(payload.get("cik", 0))
    if payload_cik != int(cik):
        raise ValueError(f"Company Facts CIK mismatch: expected {cik}, got {payload_cik}")
    path = base.COMPANYFACTS_DIR / f"CIK{int(cik):010}.json"
    old_hash = source_hash(path)
    new_hash = hashlib.sha256(response.content).hexdigest()
    changed = old_hash != new_hash
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return changed, payload, url


def current_candidate_ciks(
    current_rows: list[list[Any]],
    target_tickers: set[str],
    verified_aliases: dict[str, str],
) -> set[int]:
    current_targets = set(target_tickers)
    current_targets.update(verified_aliases.values())
    result: set[int] = set()
    for cik, _name, raw_ticker, _exchange in current_rows:
        if base.normalize_ticker(raw_ticker) in current_targets:
            result.add(int(cik))
    for path in [base.VERIFIED_OVERRIDES_PATH, base.OVERRIDES_PATH]:
        if path.exists():
            overrides = pd.read_csv(path)
            for row in overrides.itertuples(index=False):
                if base.normalize_ticker(row.ticker) in target_tickers:
                    result.add(int(row.cik))
    return result


def resolve_incremental_mapping(
    client: SecClient,
    membership: pd.DataFrame,
    verified_aliases: dict[str, str],
    existing_mapping: pd.DataFrame,
    identity: str,
    current_rows: list[list[Any]] | None = None,
) -> tuple[pd.DataFrame, list[int], list[list[Any]]]:
    target_tickers = set(membership["ticker"])
    if current_rows is None:
        current_rows = base.fetch_current_sec_tickers(identity)
    candidate_ciks = set(existing_mapping["cik"].astype(int))
    candidate_ciks |= current_candidate_ciks(current_rows, target_tickers, verified_aliases)

    downloaded_new: list[int] = []
    for cik in sorted(candidate_ciks):
        path = base.COMPANYFACTS_DIR / f"CIK{cik:010}.json"
        if path.exists():
            continue
        try:
            changed, _payload, _url = fetch_companyfacts(client, cik)
            if changed:
                downloaded_new.append(cik)
        except RuntimeError as exc:
            log(f"WARNING: could not bootstrap Company Facts for CIK {cik}: {exc}", 1)

    company_files = [
        base.COMPANYFACTS_DIR / f"CIK{cik:010}.json"
        for cik in sorted(candidate_ciks)
        if (base.COMPANYFACTS_DIR / f"CIK{cik:010}.json").exists()
    ]
    candidates = base.scan_companyfacts_ticker_evidence(company_files, target_tickers)
    base.add_current_ticker_candidates(candidates, current_rows, verified_aliases)
    forced = base.add_manual_overrides(
        candidates,
        base.VERIFIED_OVERRIDES_PATH,
        source="repository_verified_override",
    )
    forced |= base.add_manual_overrides(candidates, base.OVERRIDES_PATH)
    mapping = base.select_mappings(candidates, membership, forced)
    return mapping, downloaded_new, current_rows


def comparable_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "cik",
        "entity_name",
        "mapping_source",
        "mapping_confidence",
        "member_first_date",
        "member_last_date",
        "mapping_valid_from",
        "mapping_valid_to",
        "ticker_evidence_first_filed",
        "ticker_evidence_last_filed",
    ]
    result = frame[columns].copy()
    for column in [
        "member_first_date",
        "member_last_date",
        "mapping_valid_from",
        "mapping_valid_to",
        "ticker_evidence_first_filed",
        "ticker_evidence_last_filed",
    ]:
        result[column] = (
            pd.to_datetime(result[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        )
    return result.sort_values(["ticker", "cik"], ignore_index=True)


def mappings_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left = comparable_mapping(left)
    right = comparable_mapping(right)
    if left.shape != right.shape:
        return False
    return left.fillna("").equals(right.fillna(""))


def ciks_requiring_parse(mapping: pd.DataFrame) -> set[int]:
    manifest = base.read_parse_manifest()
    stale: set[int] = set()
    for raw_cik in mapping["cik"].astype(int).unique():
        cik = int(raw_cik)
        source_path = base.COMPANYFACTS_DIR / f"CIK{cik:010}.json"
        record = manifest.get(cik, {})
        status = str(record.get("status", ""))
        parsed_cache_missing = status == "ok" and not base.parsed_path(cik).exists()
        if (
            not source_path.exists()
            or record.get("source_sha256", "") != source_hash(source_path)
            or status not in {"ok", "no_eligible_facts"}
            or parsed_cache_missing
        ):
            stale.add(cik)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update S&P fundamentals using SEC XBRL indexes and "
            "per-CIK Company Facts; never downloads companyfacts.zip."
        )
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("EDGAR_IDENTITY", ""),
        help="SEC User-Agent identity; prefer EDGAR_IDENTITY.",
    )
    parser.add_argument(
        "--index-overlap-days",
        type=int,
        default=DEFAULT_INDEX_OVERLAP_DAYS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    identity = base.validate_identity(args.identity)
    if args.index_overlap_days < 0:
        raise ValueError("--index-overlap-days cannot be negative")
    if not base.OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{base.OUTPUT_PATH} does not exist; run the full fundamentals builder once."
        )
    client = SecClient(identity)
    state = read_state()
    period_start = existing_period_start(base.OUTPUT_PATH)
    existing_parquet = pq.ParquetFile(base.OUTPUT_PATH)
    old_rows = existing_parquet.metadata.num_rows
    existing_parquet.close()
    old_max_filed = existing_max_filed_date(base.OUTPUT_PATH)

    log("1/7 Read updated market universe and existing SEC mapping")
    market, membership, verified_aliases = base.require_market_data(base.MARKET_PATH)
    output_mapping = load_existing_mapping(base.OUTPUT_PATH)
    existing_mapping = (
        pd.read_parquet(MAPPING_CACHE_PATH) if MAPPING_CACHE_PATH.exists() else output_mapping
    )
    log(
        f"{membership['ticker'].nunique():,} market tickers; "
        f"{output_mapping['cik'].nunique():,} existing output CIKs",
        1,
    )

    log("2/7 Refresh mappings without scanning or downloading the bulk archive")
    mapping, new_source_ciks, current_rows = resolve_incremental_mapping(
        client,
        membership,
        verified_aliases,
        existing_mapping,
        identity,
    )
    mapping_changed = not mappings_equal(existing_mapping, mapping)
    stale_parse_ciks = ciks_requiring_parse(mapping)
    log(
        f"{mapping['ticker'].nunique():,} tickers mapped to "
        f"{mapping['cik'].nunique():,} CIKs; mapping changed={mapping_changed}; "
        f"{len(stale_parse_ciks):,} local issuer caches require parsing",
        1,
    )

    state_checked = pd.to_datetime(state.get("last_index_date"), errors="coerce")
    checkpoint = (
        pd.Timestamp(state_checked) if pd.notna(state_checked) else pd.Timestamp(old_max_filed)
    )
    scan_start = checkpoint - pd.Timedelta(days=args.index_overlap_days)
    scan_end = pd.Timestamp(utc_now().date())

    log("3/7 Read small SEC XBRL indexes for the unchecked window")
    index, index_urls = fetch_xbrl_indexes(client, scan_start, scan_end)
    mapped_ciks = set(mapping["cik"].astype(int))
    relevant = index.loc[
        index["cik"].astype(int).isin(mapped_ciks) & index["form_type"].isin(base.ALLOWED_FORMS)
    ].copy()

    # Accessions absent from Company Facts after a successful API response are
    # often XBRL-only cover-page filings, not delayed financial facts. Remember
    # them for the short index-overlap window so they do not trigger repeatedly.
    checked_state = state.get(
        "checked_index_accessions",
        state.get("pending_accessions", {}),  # migrate the first updater version
    )
    checked = {int(cik): set(values) for cik, values in checked_state.items()}
    retry = {int(cik): set(values) for cik, values in state.get("retry_accessions", {}).items()}
    expected_by_cik: dict[int, set[str]] = {}
    for cik, group in relevant.groupby("cik", observed=True):
        cik_int = int(cik)
        known = cached_source_accessions(cik_int)
        already_checked = checked.get(cik_int, set())
        missing = set(group["accession"].astype(str)) - known - already_checked
        new_date_rows = group.loc[group["filed_date"] > checkpoint]
        accessions = missing | (set(new_date_rows["accession"].astype(str)) - already_checked)
        if accessions:
            expected_by_cik[cik_int] = accessions
    for cik, accessions in retry.items():
        if cik in mapped_ciks:
            expected_by_cik.setdefault(cik, set()).update(accessions)
    for cik in new_source_ciks:
        if cik in mapped_ciks:
            expected_by_cik.setdefault(cik, set())
    log(
        f"{len(relevant):,} relevant indexed filings; "
        f"{len(expected_by_cik):,} CIKs require an API check",
        1,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "checkpoint": str(checkpoint.date()),
                    "scan_start": str(scan_start.date()),
                    "scan_end": str(scan_end.date()),
                    "mapped_ciks": len(mapped_ciks),
                    "relevant_index_filings": len(relevant),
                    "ciks_to_fetch": sorted(expected_by_cik),
                    "ciks_to_reparse_from_local_cache": sorted(stale_parse_ciks),
                    "mapping_changed": mapping_changed,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    log("4/7 Fetch full Company Facts only for triggered CIKs")
    changed_ciks: set[int] = set()
    failed_accessions: dict[int, list[str]] = {}
    checked_this_run: dict[int, set[str]] = {}
    api_urls: list[str] = []
    for number, (cik, expected_accessions) in enumerate(sorted(expected_by_cik.items()), start=1):
        try:
            changed, payload, url = fetch_companyfacts(client, cik)
            api_urls.append(url)
            if changed:
                changed_ciks.add(cik)
            missing_from_api = expected_accessions - payload_accessions(payload)
            checked_this_run[cik] = set(expected_accessions)
            log(
                f"{number}/{len(expected_by_cik)} CIK {cik}: "
                f"changed={changed}; index-only accessions={len(missing_from_api)}",
                2,
            )
        except (RuntimeError, ValueError, orjson.JSONDecodeError) as exc:
            failed_accessions[cik] = sorted(expected_accessions)
            log(f"WARNING: CIK {cik} update failed: {exc}", 1)

    if changed_ciks:
        log("Refresh ticker evidence from the changed local Company Facts caches", 1)
        mapping, _additional_sources, _ = resolve_incremental_mapping(
            client,
            membership,
            verified_aliases,
            mapping,
            identity,
            current_rows=current_rows,
        )
        mapped_ciks = set(mapping["cik"].astype(int))
        mapping_changed = not mappings_equal(existing_mapping, mapping)
        stale_parse_ciks |= ciks_requiring_parse(mapping)

    log("5/7 Reparse only changed issuer caches with EdgarTools")
    parse_ciks = (changed_ciks | stale_parse_ciks) & mapped_ciks
    if parse_ciks:
        parse_manifest = base.parse_selected_companies(
            mapping.loc[mapping["cik"].isin(parse_ciks)],
            period_start,
            reparse=False,
        )
    else:
        parse_manifest = base.read_parse_manifest()
    log(f"{len(parse_ciks):,} changed CIK caches reparsed", 1)

    rewrite_needed = bool(parse_ciks) or mapping_changed
    if rewrite_needed:
        log("6/7 Validate all cached facts and atomically rewrite the single parquet")
        coverage = base.mapping_coverage(market, mapping, parse_manifest)
        validation = base.validate_parsed_facts(mapping, parse_manifest, coverage, period_start)
        printable = {key: value for key, value in validation.items() if key != "rows_by_cik"}
        log(json.dumps(printable, indent=2, sort_keys=True), 1)
        if validation["status"] != "pass":
            raise RuntimeError(
                "Incremental fundamentals validation failed; parquet was not replaced. "
                + "; ".join(validation["failures"])
            )
        base.write_final_parquet(mapping, validation, period_start)
    else:
        log("6/7 No source or mapping changes; fundamentals parquet left byte-for-byte intact")

    MAPPING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(MAPPING_CACHE_PATH, index=False, compression="zstd")

    new_rows = pq.ParquetFile(base.OUTPUT_PATH).metadata.num_rows
    observed_index_date = index["filed_date"].max()
    last_index_date = (
        max(checkpoint, pd.Timestamp(observed_index_date))
        if pd.notna(observed_index_date)
        else checkpoint
    )
    new_metadata = parquet_metadata(base.OUTPUT_PATH)
    new_validation = json.loads(new_metadata.get("validation", "{}"))
    relevant_accessions = set(relevant["accession"].astype(str))
    checked_out: dict[str, list[str]] = {}
    for cik in mapped_ciks:
        values = (checked.get(cik, set()) | checked_this_run.get(cik, set())) & relevant_accessions
        if values:
            checked_out[str(cik)] = sorted(values)
    state_out = {
        "last_successful_run_utc": utc_now().isoformat(),
        "last_index_date": str(pd.Timestamp(last_index_date).date()),
        "last_output_filed_date": new_validation.get("filed_date_max", str(old_max_filed.date())),
        "checked_index_accessions": checked_out,
        "retry_accessions": {
            str(cik): accessions for cik, accessions in sorted(failed_accessions.items())
        },
        "last_index_urls": index_urls,
        "last_api_ciks": sorted(expected_by_cik),
        "last_changed_ciks": sorted(changed_ciks),
        "bulk_archive_downloaded": False,
    }
    write_state(state_out)

    log("7/7 Incremental SEC update complete")
    print(
        json.dumps(
            {
                "status": "pass",
                "old_rows": int(old_rows),
                "new_rows": int(new_rows),
                "rows_added_net": int(new_rows - old_rows),
                "ciks_checked": len(expected_by_cik),
                "ciks_changed": len(changed_ciks),
                "ciks_reparsed": len(parse_ciks),
                "failed_ciks": len(failed_accessions),
                "mapping_changed": mapping_changed,
                "companyfacts_zip_downloaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
