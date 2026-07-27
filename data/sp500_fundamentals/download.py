from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from edgar import set_identity
from edgar.entity.parser import EntityFactsParser
from edgar.storage import download_edgar_data, use_local_storage

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
CACHE_DIR = HERE / "cache"
EDGAR_DIR = CACHE_DIR / "edgar"
COMPANYFACTS_DIR = EDGAR_DIR / "companyfacts"
PARSED_DIR = CACHE_DIR / "parsed"
PARSE_MANIFEST_PATH = CACHE_DIR / "parse_manifest.csv"
MARKET_PATH = DATA_DIR / "sp500_market_data.parquet"
OUTPUT_PATH = DATA_DIR / "sp500_fundamentals.parquet"
OVERRIDES_PATH = RAW_DIR / "ticker_cik_overrides.csv"
VERIFIED_OVERRIDES_PATH = HERE / "verified_ticker_cik_overrides.csv"

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json"
SEC_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
DEFAULT_PERIOD_START = "2012-01-01"
RECOMMENDED_ANALYSIS_START = pd.Timestamp("2012-01-01")

ALLOWED_FORMS = {
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
    "6-K",
    "6-K/A",
    "8-K",
    "8-K/A",
}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
FINANCIAL_TAXONOMIES = {"us-gaap", "ifrs-full"}
DEI_NUMERIC_CONCEPTS = {
    "EntityCommonStockSharesOutstanding",
    "EntityPublicFloat",
}

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("cik", pa.int64(), nullable=False),
        pa.field("entity_name", pa.string(), nullable=False),
        pa.field("mapping_source", pa.string(), nullable=False),
        pa.field("mapping_confidence", pa.string(), nullable=False),
        pa.field("member_first_date", pa.timestamp("ns")),
        pa.field("member_last_date", pa.timestamp("ns")),
        pa.field("mapping_valid_from", pa.timestamp("ns")),
        pa.field("mapping_valid_to", pa.timestamp("ns")),
        pa.field("ticker_evidence_first_filed", pa.timestamp("ns")),
        pa.field("ticker_evidence_last_filed", pa.timestamp("ns")),
        pa.field("concept", pa.string(), nullable=False),
        pa.field("label", pa.string()),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("period_type", pa.string(), nullable=False),
        pa.field("period_start", pa.timestamp("ns")),
        pa.field("period_end", pa.timestamp("ns"), nullable=False),
        pa.field("fiscal_year", pa.int32()),
        pa.field("fiscal_period", pa.string()),
        pa.field("filed_date", pa.timestamp("ns"), nullable=False),
        pa.field("form_type", pa.string(), nullable=False),
        pa.field("accession", pa.string(), nullable=False),
        pa.field("statement_type", pa.string()),
        pa.field("taxonomy", pa.string(), nullable=False),
        pa.field("data_quality", pa.string()),
        pa.field("confidence_score", pa.float32()),
        pa.field("is_annual_filing", pa.bool_(), nullable=False),
        pa.field("is_amendment", pa.bool_(), nullable=False),
        pa.field("filing_version", pa.int32(), nullable=False),
    ]
)


def log(message: str, indent: int = 0) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {'  ' * indent}{message}", flush=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_identity(identity: str) -> str:
    cleaned = " ".join(identity.strip().split())
    if "@" not in cleaned or " " not in cleaned:
        raise ValueError(
            "SEC identity must identify a person/project and contact email, for example "
            "'Jane Doe jane@example.com'. Set EDGAR_IDENTITY or pass --identity."
        )
    return cleaned


def normalize_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    ticker = re.sub(r"-\d{6}$", "", ticker)
    return ticker.replace("-", ".")


def trading_symbol_tokens(value: object) -> list[str]:
    if value is None:
        return []
    raw = str(value).upper().strip()
    parts = re.split(r"[,;/|\s]+", raw)
    return [normalize_ticker(part) for part in parts if part.strip()]


def require_market_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Build the market dataset first with "
            "data/sp500_market/download.py."
        )
    market = pd.read_parquet(path, columns=["date", "ticker", "yf_ticker", "is_sp500_member"])
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market["ticker"] = market["ticker"].map(normalize_ticker)
    market["yf_ticker"] = market["yf_ticker"].map(normalize_ticker)
    market = market.dropna(subset=["date", "ticker"]).drop_duplicates(
        ["date", "ticker"], keep="last"
    )
    members = market.loc[market["is_sp500_member"]].copy()
    membership = (
        members.groupby("ticker", observed=True)["date"]
        .agg(member_first_date="min", member_last_date="max", member_days="count")
        .reset_index()
    )
    if membership.empty:
        raise ValueError("Market parquet contains no point-in-time member rows.")
    aliases = (
        market[["ticker", "yf_ticker"]]
        .dropna()
        .drop_duplicates()
        .loc[lambda frame: frame["ticker"] != frame["yf_ticker"]]
        .set_index("ticker")["yf_ticker"]
        .to_dict()
    )
    return market, membership, aliases


def ensure_companyfacts(identity: str, refresh: bool) -> list[Path]:
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    set_identity(identity)
    use_local_storage(EDGAR_DIR, True, allow_network_fallback=False)
    existing = sorted(COMPANYFACTS_DIR.glob("CIK*.json"))
    if refresh or len(existing) < 10_000:
        log(
            "Downloading and extracting the official SEC companyfacts bulk archive; "
            "this is a multi-gigabyte first-run cache.",
            1,
        )
        download_edgar_data(
            submissions=False,
            facts=True,
            reference=False,
            disable_progress=False,
        )
        existing = sorted(COMPANYFACTS_DIR.glob("CIK*.json"))
    if len(existing) < 10_000:
        raise RuntimeError(
            f"SEC companyfacts cache looks incomplete ({len(existing):,} JSON files)."
        )
    return existing


def read_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    return orjson.loads(content), hashlib.sha256(content).hexdigest()


def add_candidate(
    candidates: dict[str, dict[int, dict[str, Any]]],
    ticker: str,
    cik: int,
    entity_name: str,
    ticker_filed_dates: list[pd.Timestamp],
    source: str,
    entity_filed_dates: list[pd.Timestamp] | None = None,
) -> None:
    if ticker not in candidates:
        return
    record = candidates[ticker].setdefault(
        int(cik),
        {
            "ticker": ticker,
            "cik": int(cik),
            "entity_name": entity_name,
            "ticker_filed_dates": [],
            "entity_filed_dates": [],
            "sources": set(),
        },
    )
    if entity_name and not record["entity_name"]:
        record["entity_name"] = entity_name
    record["ticker_filed_dates"].extend(ticker_filed_dates)
    record["entity_filed_dates"].extend(entity_filed_dates or [])
    record["sources"].add(source)


def scan_companyfacts_ticker_evidence(
    company_files: list[Path], target_tickers: set[str]
) -> dict[str, dict[int, dict[str, Any]]]:
    candidates: dict[str, dict[int, dict[str, Any]]] = {ticker: {} for ticker in target_tickers}
    for index, path in enumerate(company_files, start=1):
        try:
            payload = orjson.loads(path.read_bytes())
            cik = int(payload.get("cik", 0))
            entity_name = str(payload.get("entityName", ""))
            trading_symbol = (
                payload.get("facts", {}).get("dei", {}).get("TradingSymbol", {}).get("units", {})
            )
            for unit_facts in trading_symbol.values():
                for fact in unit_facts:
                    dates = pd.to_datetime([fact.get("filed")], errors="coerce").dropna().tolist()
                    for ticker in trading_symbol_tokens(fact.get("val")):
                        add_candidate(
                            candidates,
                            ticker,
                            cik,
                            entity_name,
                            dates,
                            "companyfacts_dei_trading_symbol",
                        )
        except (OSError, orjson.JSONDecodeError, TypeError, ValueError):
            continue
        if index % 2_500 == 0:
            log(f"scanned {index:,}/{len(company_files):,} companyfacts files", 2)
    return candidates


def fetch_current_sec_tickers(identity: str) -> list[list[Any]]:
    response = requests.get(
        SEC_TICKERS_URL,
        timeout=(15, 120),
        headers={
            "User-Agent": identity,
            "Accept-Encoding": "gzip, deflate",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("fields") != ["cik", "name", "ticker", "exchange"]:
        raise ValueError("Unexpected SEC company_tickers_exchange.json schema.")
    return payload["data"]


def company_filing_date_bounds(cik: int) -> list[pd.Timestamp]:
    path = COMPANYFACTS_DIR / f"CIK{int(cik):010}.json"
    if not path.exists():
        return []
    try:
        payload = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return []

    first: str | None = None
    last: str | None = None
    for taxonomy in payload.get("facts", {}).values():
        for concept in taxonomy.values():
            for unit_facts in concept.get("units", {}).values():
                for fact in unit_facts:
                    filed = fact.get("filed")
                    if not isinstance(filed, str) or len(filed) != 10:
                        continue
                    first = filed if first is None else min(first, filed)
                    last = filed if last is None else max(last, filed)
    return (
        [pd.Timestamp(first), pd.Timestamp(last)] if first is not None and last is not None else []
    )


def add_current_ticker_candidates(
    candidates: dict[str, dict[int, dict[str, Any]]],
    current_rows: list[list[Any]],
    verified_aliases: dict[str, str],
) -> None:
    alias_targets: dict[str, set[str]] = {}
    for historical_ticker, current_ticker in verified_aliases.items():
        alias_targets.setdefault(current_ticker, set()).add(historical_ticker)
    bounds_cache: dict[int, list[pd.Timestamp]] = {}
    for cik, name, raw_ticker, _exchange in current_rows:
        current_ticker = normalize_ticker(raw_ticker)
        targets = set(alias_targets.get(current_ticker, set()))
        if current_ticker in candidates:
            targets.add(current_ticker)
        if not targets:
            continue
        cik = int(cik)
        if cik not in bounds_cache:
            bounds_cache[cik] = company_filing_date_bounds(cik)
        entity_filed_dates = bounds_cache[cik]
        for ticker in targets:
            add_candidate(
                candidates,
                ticker,
                cik,
                str(name),
                [],
                "sec_current_ticker_file",
                entity_filed_dates=entity_filed_dates,
            )
            if ticker != current_ticker:
                candidates[ticker][cik]["sources"].add("market_verified_ticker_rename")


def add_manual_overrides(
    candidates: dict[str, dict[int, dict[str, Any]]],
    path: Path,
    source: str = "manual_override",
) -> set[tuple[str, int]]:
    forced: set[tuple[str, int]] = set()
    if not path.exists():
        return forced
    overrides = pd.read_csv(path)
    required = {"ticker", "cik"}
    if not required.issubset(overrides.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}.")
    for row in overrides.itertuples(index=False):
        ticker = normalize_ticker(row.ticker)
        cik = int(row.cik)
        if ticker not in candidates:
            continue
        entity_name = ""
        source_path = COMPANYFACTS_DIR / f"CIK{cik:010}.json"
        if source_path.exists():
            try:
                entity_name = str(orjson.loads(source_path.read_bytes()).get("entityName", ""))
            except (OSError, orjson.JSONDecodeError):
                entity_name = ""
        add_candidate(candidates, ticker, cik, entity_name, [], source)
        forced.add((ticker, cik))
    return forced


def select_mappings(
    candidates: dict[str, dict[int, dict[str, Any]]],
    membership: pd.DataFrame,
    forced: set[tuple[str, int]],
) -> pd.DataFrame:
    member_lookup = membership.set_index("ticker")
    rows: list[dict[str, Any]] = []
    today = pd.Timestamp(utc_now().date())
    for ticker, cik_records in candidates.items():
        member = member_lookup.loc[ticker]
        member_start = pd.Timestamp(member["member_first_date"])
        member_end = pd.Timestamp(member["member_last_date"])
        for cik, record in cik_records.items():
            ticker_dates = pd.DatetimeIndex(record["ticker_filed_dates"]).dropna().sort_values()
            entity_dates = pd.DatetimeIndex(record["entity_filed_dates"]).dropna().sort_values()
            first = ticker_dates.min() if len(ticker_dates) else pd.NaT
            last = ticker_dates.max() if len(ticker_dates) else pd.NaT
            manual = (ticker, cik) in forced
            if manual:
                selected = True
                confidence = "manual_verified"
                overlap_days = int((member_end - member_start).days + 1)
                mapping_valid_from = member_start
                mapping_valid_to = member_end
            elif len(ticker_dates):
                evidence_start = first - pd.Timedelta(days=550)
                evidence_end = last + pd.Timedelta(days=550)
                overlap_start = max(member_start, evidence_start)
                overlap_end = min(member_end, evidence_end)
                overlap_days = max(0, int((overlap_end - overlap_start).days + 1))
                selected = overlap_days > 0
                confidence = "high" if overlap_days >= 90 else "medium"
                mapping_valid_from = overlap_start
                mapping_valid_to = overlap_end
            else:
                entity_first = entity_dates.min() if len(entity_dates) else pd.NaT
                entity_last = entity_dates.max() if len(entity_dates) else pd.NaT
                if (
                    "sec_current_ticker_file" in record["sources"]
                    and pd.notna(entity_first)
                    and pd.notna(entity_last)
                ):
                    overlap_start = max(member_start, entity_first)
                    overlap_end = min(member_end, max(entity_last, today))
                    overlap_days = max(0, int((overlap_end - overlap_start).days + 1))
                    selected = overlap_days > 0
                    confidence = "high" if overlap_days >= 90 else "medium"
                    mapping_valid_from = overlap_start
                    mapping_valid_to = overlap_end
                    if selected:
                        record["sources"].add("companyfacts_entity_filing_overlap")
                else:
                    selected = False
                    confidence = "low"
                    overlap_days = 0
                    mapping_valid_from = pd.NaT
                    mapping_valid_to = pd.NaT
            if not selected:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "cik": int(cik),
                    "entity_name": str(record["entity_name"]),
                    "mapping_source": "+".join(sorted(record["sources"])),
                    "mapping_confidence": confidence,
                    "member_first_date": member_start,
                    "member_last_date": member_end,
                    "member_days": int(member["member_days"]),
                    "mapping_valid_from": mapping_valid_from,
                    "mapping_valid_to": mapping_valid_to,
                    "ticker_evidence_first_filed": first,
                    "ticker_evidence_last_filed": last,
                    "overlap_days": overlap_days,
                }
            )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise RuntimeError("No SEC CIK mappings could be resolved.")

    # If overlapping evidence produces multiple CIKs for one ticker, keep disjoint
    # historical candidates, but discard a weak candidate dominated by a stronger one.
    mapping = mapping.sort_values(
        ["ticker", "overlap_days", "mapping_confidence"],
        ascending=[True, False, True],
    )
    keep_indices: list[int] = []
    for _ticker, group in mapping.groupby("ticker", sort=False):
        best_overlap = int(group["overlap_days"].max())
        for index, row in group.iterrows():
            if (
                len(group) == 1
                or row["mapping_confidence"] == "manual_verified"
                or int(row["overlap_days"]) >= max(30, int(best_overlap * 0.20))
            ):
                keep_indices.append(index)
    return mapping.loc[keep_indices].sort_values(["ticker", "cik"]).reset_index(drop=True)


def read_parse_manifest() -> dict[int, dict[str, Any]]:
    if not PARSE_MANIFEST_PATH.exists():
        return {}
    frame = pd.read_csv(PARSE_MANIFEST_PATH).fillna("")
    return {int(row["cik"]): row.to_dict() for _, row in frame.iterrows()}


def write_parse_manifest(manifest: dict[int, dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(manifest.values()))
    if not frame.empty:
        frame = frame.sort_values("cik")
    frame.to_csv(PARSE_MANIFEST_PATH, index=False)


def parsed_path(cik: int) -> Path:
    return PARSED_DIR / f"CIK{int(cik):010}.parquet"


def clean_entity_facts(payload: dict[str, Any], period_start: pd.Timestamp) -> pd.DataFrame:
    entity = EntityFactsParser.parse_company_facts(payload)
    if entity is None:
        return pd.DataFrame()
    frame = entity.to_dataframe(include_metadata=True, pit_mode=True)
    if frame.empty:
        return frame
    frame["taxonomy"] = frame["taxonomy"].astype("string")
    frame["base_concept"] = frame["concept"].astype("string").str.split(":", n=1).str[-1]
    frame["form_type"] = frame["form_type"].astype("string").str.upper()
    frame["numeric_value"] = pd.to_numeric(frame["numeric_value"], errors="coerce")
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="coerce")
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")

    taxonomy_ok = frame["taxonomy"].isin(FINANCIAL_TAXONOMIES) | (
        frame["taxonomy"].eq("dei") & frame["base_concept"].isin(DEI_NUMERIC_CONCEPTS)
    )
    frame = frame.loc[
        taxonomy_ok
        & frame["form_type"].isin(ALLOWED_FORMS)
        & frame["numeric_value"].notna()
        & np.isfinite(frame["numeric_value"])
        & frame["period_end"].notna()
        & frame["filing_date"].notna()
        & (frame["period_end"] >= period_start)
        & (frame["filing_date"] >= frame["period_end"])
    ].copy()
    if frame.empty:
        return frame

    frame["concept"] = frame["concept"].astype("string")
    frame["label"] = frame["label"].astype("string")
    frame["unit"] = frame["unit"].fillna("").astype("string")
    frame["period_type"] = frame["period_type"].fillna("").astype("string")
    frame["fiscal_period"] = frame["fiscal_period"].fillna("").astype("string")
    frame["accession"] = frame["accession"].fillna("").astype("string")
    frame["statement_type"] = frame["statement_type"].astype("string")
    frame["data_quality"] = frame["data_quality"].astype("string")
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int32")
    frame["confidence_score"] = pd.to_numeric(frame["confidence_score"], errors="coerce").astype(
        "Float32"
    )
    frame["is_annual_filing"] = frame["form_type"].isin(ANNUAL_FORMS)
    frame["is_amendment"] = frame["form_type"].str.endswith("/A")

    key = [
        "concept",
        "unit",
        "period_start",
        "period_end",
        "filing_date",
        "form_type",
        "accession",
        "numeric_value",
    ]
    frame = frame.drop_duplicates(key, keep="last")
    version_key = ["concept", "unit", "period_start", "period_end"]
    frame = frame.sort_values(version_key + ["filing_date", "accession"])
    frame["filing_version"] = (
        frame.groupby(version_key, dropna=False, observed=True).cumcount() + 1
    ).astype("int32")
    frame = frame.drop(columns=["value"])
    frame = frame.rename(columns={"numeric_value": "value"})
    return frame[
        [
            "concept",
            "label",
            "value",
            "unit",
            "period_type",
            "period_start",
            "period_end",
            "fiscal_year",
            "fiscal_period",
            "filing_date",
            "form_type",
            "accession",
            "statement_type",
            "taxonomy",
            "data_quality",
            "confidence_score",
            "is_annual_filing",
            "is_amendment",
            "filing_version",
        ]
    ].rename(columns={"filing_date": "filed_date"})


def eligible_cached_facts(frame: pd.DataFrame, period_start: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        (frame["period_end"] >= period_start) & (frame["filed_date"] >= frame["period_end"])
    ].copy()


def parse_selected_companies(
    mapping: pd.DataFrame,
    period_start: pd.Timestamp,
    reparse: bool,
) -> dict[int, dict[str, Any]]:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_parse_manifest()
    ciks = sorted(mapping["cik"].unique())
    for index, cik in enumerate(ciks, start=1):
        source_path = COMPANYFACTS_DIR / f"CIK{int(cik):010}.json"
        if not source_path.exists():
            manifest[int(cik)] = {
                "cik": int(cik),
                "status": "source_missing",
                "source_sha256": "",
                "fact_rows": 0,
                "entity_name": "",
                "parsed_at": utc_now().isoformat(),
            }
            continue
        payload, source_hash = read_json_bytes(source_path)
        prior = manifest.get(int(cik), {})
        prior_start = pd.to_datetime(prior.get("period_start"), errors="coerce")
        cache_valid = (
            not reparse
            and parsed_path(int(cik)).exists()
            and str(prior.get("source_sha256", "")) == source_hash
            and pd.notna(prior_start)
            and prior_start <= period_start
            and str(prior.get("status", "")) == "ok"
        )
        if cache_valid:
            continue
        frame = clean_entity_facts(payload, period_start)
        if frame.empty:
            status = "no_eligible_facts"
            fact_rows = 0
            if parsed_path(int(cik)).exists():
                parsed_path(int(cik)).unlink()
        else:
            frame.to_parquet(
                parsed_path(int(cik)),
                index=False,
                compression="zstd",
            )
            status = "ok"
            fact_rows = len(frame)
        manifest[int(cik)] = {
            "cik": int(cik),
            "status": status,
            "source_sha256": source_hash,
            "fact_rows": fact_rows,
            "entity_name": str(payload.get("entityName", "")),
            "period_start": str(period_start.date()),
            "parsed_at": utc_now().isoformat(),
        }
        if index % 25 == 0 or index == len(ciks):
            write_parse_manifest(manifest)
            log(f"parsed {index:,}/{len(ciks):,} mapped SEC entities", 2)
    write_parse_manifest(manifest)
    return manifest


def mapping_coverage(
    market: pd.DataFrame,
    mapping: pd.DataFrame,
    parse_manifest: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    mapped_tickers = set(mapping["ticker"])
    facts_ciks = {
        cik
        for cik, record in parse_manifest.items()
        if str(record.get("status")) == "ok" and int(record.get("fact_rows", 0)) > 0
    }
    members = market.loc[market["is_sp500_member"]].copy()
    members["mapped"] = False
    members["has_facts"] = False
    intervals_by_ticker = {
        ticker: group for ticker, group in mapping.groupby("ticker", observed=True, sort=False)
    }
    for ticker, index in members.groupby("ticker", observed=True).groups.items():
        intervals = intervals_by_ticker.get(ticker)
        if intervals is None:
            continue
        dates = members.loc[index, "date"]
        mapped = pd.Series(False, index=index)
        has_facts = pd.Series(False, index=index)
        for row in intervals.itertuples(index=False):
            in_interval = dates.between(
                row.mapping_valid_from, row.mapping_valid_to, inclusive="both"
            )
            mapped |= in_interval
            if int(row.cik) in facts_ciks:
                has_facts |= in_interval
        members.loc[index, "mapped"] = mapped
        members.loc[index, "has_facts"] = has_facts
    after_2012 = members.loc[members["date"] >= RECOMMENDED_ANALYSIS_START]
    recent_cutoff = members["date"].max() - pd.DateOffset(years=2)
    recent = members.loc[members["date"] >= recent_cutoff]

    def rate(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].mean()) if len(frame) else float("nan")

    member_tickers = set(members["ticker"])
    return {
        "market_tickers": len(member_tickers),
        "mapped_tickers": len(mapped_tickers),
        "facts_tickers": int(mapping.loc[mapping["cik"].isin(facts_ciks), "ticker"].nunique()),
        "unmapped_tickers": sorted(member_tickers - mapped_tickers),
        "mapped_member_day_rate_all": round(rate(members, "mapped"), 6),
        "mapped_member_day_rate_2012_plus": round(rate(after_2012, "mapped"), 6),
        "facts_member_day_rate_2012_plus": round(rate(after_2012, "has_facts"), 6),
        "facts_member_day_rate_recent": round(rate(recent, "has_facts"), 6),
    }


def validate_parsed_facts(
    mapping: pd.DataFrame,
    parse_manifest: dict[int, dict[str, Any]],
    coverage: dict[str, Any],
    period_start: pd.Timestamp,
) -> dict[str, Any]:
    duplicate_keys = 0
    nonfinite_values = 0
    filed_before_period_end = 0
    total_rows = 0
    min_period: pd.Timestamp | None = None
    max_period: pd.Timestamp | None = None
    min_filed: pd.Timestamp | None = None
    max_filed: pd.Timestamp | None = None
    repeated_versions = 0
    output_rows = 0
    rows_by_cik: dict[int, int] = {}

    for cik in sorted(mapping["cik"].unique()):
        record = parse_manifest.get(int(cik), {})
        if str(record.get("status", "")) != "ok":
            rows_by_cik[int(cik)] = 0
            continue
        frame = eligible_cached_facts(pd.read_parquet(parsed_path(int(cik))), period_start)
        rows_by_cik[int(cik)] = len(frame)
        total_rows += len(frame)
        aliases = int((mapping["cik"] == cik).sum())
        output_rows += len(frame) * aliases
        key = [
            "concept",
            "unit",
            "period_start",
            "period_end",
            "filed_date",
            "form_type",
            "accession",
            "value",
        ]
        duplicate_keys += int(frame.duplicated(key).sum())
        nonfinite_values += int((~np.isfinite(frame["value"].astype(float))).sum())
        filed_before_period_end += int((frame["filed_date"] < frame["period_end"]).sum())
        repeated_versions += int((frame["filing_version"] > 1).sum())
        pmin, pmax = frame["period_end"].min(), frame["period_end"].max()
        fmin, fmax = frame["filed_date"].min(), frame["filed_date"].max()
        min_period = pmin if min_period is None else min(min_period, pmin)
        max_period = pmax if max_period is None else max(max_period, pmax)
        min_filed = fmin if min_filed is None else min(min_filed, fmin)
        max_filed = fmax if max_filed is None else max(max_filed, fmax)

    failures: list[str] = []
    if duplicate_keys:
        failures.append(f"{duplicate_keys} duplicate fact-version keys")
    if nonfinite_values:
        failures.append(f"{nonfinite_values} non-finite values")
    if total_rows == 0:
        failures.append("no eligible SEC facts")
    if coverage["mapped_member_day_rate_2012_plus"] < 0.95:
        failures.append(
            "CIK mapping covers less than 95% of constituent-days from 2012 onward "
            f"({coverage['mapped_member_day_rate_2012_plus']:.2%})"
        )
    if coverage["facts_member_day_rate_recent"] < 0.95:
        failures.append(
            "SEC facts cover less than 95% of recent constituent-days "
            f"({coverage['facts_member_day_rate_recent']:.2%})"
        )
    temporal_rate = filed_before_period_end / max(total_rows, 1)
    if temporal_rate > 0.001:
        failures.append(
            "more than 0.1% of facts were filed before their period ended " f"({temporal_rate:.3%})"
        )

    source_hashes = [
        f"{cik}:{parse_manifest[int(cik)].get('source_sha256', '')}"
        for cik in sorted(mapping["cik"].unique())
        if int(cik) in parse_manifest
    ]
    source_digest = hashlib.sha256("|".join(source_hashes).encode()).hexdigest()
    return {
        "status": "pass" if not failures else "fail",
        "unique_entity_fact_rows": int(total_rows),
        "expected_output_rows_after_ticker_aliases": int(output_rows),
        "mapped_ciks": int(mapping["cik"].nunique()),
        "duplicate_fact_version_keys": duplicate_keys,
        "nonfinite_values": nonfinite_values,
        "filed_before_period_end": filed_before_period_end,
        "filed_before_period_end_rate": round(temporal_rate, 8),
        "repeated_point_in_time_versions": repeated_versions,
        "period_end_min": str(min_period.date()) if min_period is not None else None,
        "period_end_max": str(max_period.date()) if max_period is not None else None,
        "filed_date_min": str(min_filed.date()) if min_filed is not None else None,
        "filed_date_max": str(max_filed.date()) if max_filed is not None else None,
        "selected_companyfacts_source_digest": source_digest,
        "rows_by_cik": rows_by_cik,
        **coverage,
        "failures": failures,
    }


def output_chunk(mapping_row: pd.Series, facts: pd.DataFrame) -> pd.DataFrame:
    out = facts.copy()
    out.insert(0, "ticker", str(mapping_row["ticker"]))
    out.insert(1, "cik", int(mapping_row["cik"]))
    out.insert(2, "entity_name", str(mapping_row["entity_name"]))
    out.insert(3, "mapping_source", str(mapping_row["mapping_source"]))
    out.insert(4, "mapping_confidence", str(mapping_row["mapping_confidence"]))
    out.insert(5, "member_first_date", mapping_row["member_first_date"])
    out.insert(6, "member_last_date", mapping_row["member_last_date"])
    out.insert(7, "mapping_valid_from", mapping_row["mapping_valid_from"])
    out.insert(8, "mapping_valid_to", mapping_row["mapping_valid_to"])
    out.insert(
        9,
        "ticker_evidence_first_filed",
        mapping_row["ticker_evidence_first_filed"],
    )
    out.insert(
        10,
        "ticker_evidence_last_filed",
        mapping_row["ticker_evidence_last_filed"],
    )
    return out[[field.name for field in OUTPUT_SCHEMA]]


def write_final_parquet(
    mapping: pd.DataFrame,
    validation: dict[str, Any],
    period_start: pd.Timestamp,
) -> None:
    output_tmp = OUTPUT_PATH.with_suffix(".parquet.tmp")
    metadata = {
        "dataset": "Point-in-time SEC fundamentals for the S&P 500 historical universe",
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now().isoformat(),
        "period_end_start": str(period_start.date()),
        "recommended_modeling_start": str(RECOMMENDED_ANALYSIS_START.date()),
        "source": (
            "SEC EDGAR Company Facts; bulk archive bootstrap with optional "
            "selective per-CIK API updates"
        ),
        "source_url": SEC_COMPANYFACTS_URL,
        "bulk_source_url": SEC_BULK_URL,
        "edgartools_role": (
            "EdgarTools EntityFactsParser and point-in-time metadata export; "
            "SEC bulk archive cached locally and selectively refreshable by CIK"
        ),
        "validation": json.dumps(validation, sort_keys=True),
        "point_in_time_rule": (
            "Filter filed_date <= decision date, then take the greatest filing_version "
            "within (cik, concept, unit, period_start, period_end). Amendments and "
            "later comparative restatements are preserved."
        ),
        "identity": "SEC User-Agent configured at runtime; identity value not persisted",
    }
    schema = OUTPUT_SCHEMA.with_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()}
    )
    writer = pq.ParquetWriter(
        output_tmp,
        schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
    )
    rows_written = 0
    try:
        facts_cache: dict[int, pd.DataFrame] = {}
        for index, mapping_row in mapping.iterrows():
            cik = int(mapping_row["cik"])
            if cik not in facts_cache:
                path = parsed_path(cik)
                if not path.exists():
                    continue
                facts_cache = {cik: eligible_cached_facts(pd.read_parquet(path), period_start)}
            chunk = output_chunk(mapping_row, facts_cache[cik])
            table = pa.Table.from_pandas(
                chunk,
                schema=OUTPUT_SCHEMA,
                preserve_index=False,
                safe=False,
            ).replace_schema_metadata(schema.metadata)
            writer.write_table(table, row_group_size=250_000)
            rows_written += len(chunk)
            if (index + 1) % 50 == 0:
                log(f"wrote {index + 1:,}/{len(mapping):,} ticker-CIK mappings", 2)
    finally:
        writer.close()

    expected = int(validation["expected_output_rows_after_ticker_aliases"])
    parquet = pq.ParquetFile(output_tmp)
    parquet_rows = parquet.metadata.num_rows
    parquet.close()
    if rows_written != expected or parquet_rows != expected:
        output_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Final parquet row-count mismatch: expected {expected:,}, "
            f"wrote {rows_written:,}, metadata says {parquet_rows:,}."
        )
    output_tmp.replace(OUTPUT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time SEC fundamentals for the market-data universe."
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("EDGAR_IDENTITY", ""),
        help="SEC User-Agent identity. Prefer the EDGAR_IDENTITY environment variable.",
    )
    parser.add_argument("--market-path", type=Path, default=MARKET_PATH)
    parser.add_argument("--period-start", default=DEFAULT_PERIOD_START)
    parser.add_argument("--refresh-sec", action="store_true")
    parser.add_argument("--reparse", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity = validate_identity(args.identity)
    period_start = pd.Timestamp(args.period_start)

    log("1/6 Read the exact historical market-data universe")
    market, membership, verified_aliases = require_market_data(args.market_path)
    target_tickers = set(membership["ticker"])
    log(
        f"{len(target_tickers):,} priced historical tickers; "
        f"{market['date'].min().date()} to {market['date'].max().date()}",
        1,
    )

    log("2/6 Ensure official SEC bulk Company Facts cache")
    company_files = ensure_companyfacts(identity, refresh=args.refresh_sec)
    log(f"{len(company_files):,} SEC companyfacts files available", 1)

    log("3/6 Resolve historical ticker-to-CIK evidence")
    candidates = scan_companyfacts_ticker_evidence(company_files, target_tickers)
    add_current_ticker_candidates(candidates, fetch_current_sec_tickers(identity), verified_aliases)
    forced = add_manual_overrides(
        candidates,
        VERIFIED_OVERRIDES_PATH,
        source="repository_verified_override",
    )
    forced |= add_manual_overrides(candidates, OVERRIDES_PATH)
    mapping = select_mappings(candidates, membership, forced)
    log(
        f"{mapping['ticker'].nunique():,}/{len(target_tickers):,} tickers mapped "
        f"to {mapping['cik'].nunique():,} SEC entities",
        1,
    )

    log("4/6 Parse all eligible point-in-time facts with EdgarTools")
    parse_manifest = parse_selected_companies(mapping, period_start, reparse=args.reparse)

    log("5/6 Validate source, temporal grain, and market-universe coverage")
    coverage = mapping_coverage(market, mapping, parse_manifest)
    validation = validate_parsed_facts(mapping, parse_manifest, coverage, period_start)
    printable = {key: value for key, value in validation.items() if key != "rows_by_cik"}
    log(json.dumps(printable, indent=2, sort_keys=True), 1)
    if validation["status"] != "pass":
        raise RuntimeError(
            "Fundamentals validation failed; parquet was not replaced. "
            + "; ".join(validation["failures"])
        )

    log("6/6 Write the single validated fundamentals parquet")
    write_final_parquet(mapping, validation, period_start)
    log(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted; SEC and parsed-company caches are safe to resume.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
