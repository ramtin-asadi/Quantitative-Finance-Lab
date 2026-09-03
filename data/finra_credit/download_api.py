from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
API_CACHE = HERE / "cache" / "api"
TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
API_ROOT = "https://api.finra.org/data/group/fixedIncomeMarket/name"
DATASETS = {
    "corporateMarketBreadth": "finra_corporate_market_breadth.parquet",
    "corporateMarketSentiment": "finra_corporate_market_sentiment.parquet",
    "corporate144AMarketBreadth": "finra_corporate_144a_market_breadth.parquet",
    "corporate144AMarketSentiment": "finra_corporate_144a_market_sentiment.parquet",
    "corporatesAndAgenciesCappedVolume": "finra_corporate_agency_capped_volume.parquet",
    "collateralizedObligationPricing": "finra_cbo_cdo_clo_pricing.parquet",
    "securitizedProductTradingActivity": "finra_securitized_product_activity.parquet",
    "securitizedProductsCappedVolume": "finra_securitized_product_capped_volume.parquet",
    "securitizedProductErrata": "finra_securitized_product_errata.parquet",
    "agencyMarketBreadth": "finra_agency_market_breadth.parquet",
    "agencyMarketSentiment": "finra_agency_market_sentiment.parquet",
    "treasuryDailyAggregates": "finra_treasury_daily_aggregates.parquet",
    "treasuryMonthlyAggregates": "finra_treasury_monthly_aggregates.parquet",
}
DATE_FIELDS = {
    "corporateMarketBreadth": "tradeReportDate",
    "corporateMarketSentiment": "tradeReportDate",
    "corporate144AMarketBreadth": "tradeReportDate",
    "corporate144AMarketSentiment": "tradeReportDate",
    "corporatesAndAgenciesCappedVolume": "tradeReportDate",
    "collateralizedObligationPricing": "reportDate",
    "securitizedProductTradingActivity": "reportDate",
    "securitizedProductsCappedVolume": "tradeReportDate",
    "securitizedProductErrata": "reportDate",
    "agencyMarketBreadth": "tradeReportDate",
    "agencyMarketSentiment": "tradeReportDate",
    "treasuryDailyAggregates": "tradeDate",
    "treasuryMonthlyAggregates": "beginningOfTheMonthDate",
}


def token(client_id: str, client_secret: str) -> str:
    response = requests.post(TOKEN_URL, auth=(client_id, client_secret), timeout=120)
    response.raise_for_status()
    return response.json()["access_token"]


def post_page(
    session: requests.Session, dataset: str, payload: dict[str, object]
) -> requests.Response:
    url = f"{API_ROOT}/{dataset}"
    for attempt in range(6):
        response = session.post(url, json=payload, timeout=180)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == 5:
            response.raise_for_status()
        wait = min(2**attempt, 30)
        print(f"{dataset}: HTTP {response.status_code}; retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"FINRA request failed for {dataset}")


def page(
    session: requests.Session,
    dataset: str,
    offset: int,
    limit: int,
    start_date: str | None,
) -> list[dict]:
    payload: dict[str, object] = {"limit": limit, "offset": offset}
    if start_date:
        payload["compareFilters"] = [
            {
                "compareType": "GREATER",
                "fieldName": DATE_FIELDS[dataset],
                "fieldValue": start_date,
            }
        ]
    response = post_page(session, dataset, payload)
    records = response.json()
    if not isinstance(records, list):
        raise ValueError(f"FINRA returned a non-list response for {dataset}: {records}")
    return records


def normalize(records: list[dict]) -> pd.DataFrame:
    frame = pd.json_normalize(records, sep="_")
    for column in frame.columns:
        if re.search(r"date|timestamp", column, re.IGNORECASE):
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().any():
                frame[column] = converted
                continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().sum() and numeric.notna().sum() == frame[column].notna().sum():
            frame[column] = numeric
    return frame


def fetch_dataset(
    session: requests.Session, dataset: str, output: Path, update: bool, limit: int
) -> None:
    old = pd.read_parquet(output) if update and output.exists() else pd.DataFrame()
    start_date = None
    if not old.empty:
        date_field = DATE_FIELDS[dataset]
        if date_field in old:
            maximum = pd.to_datetime(old[date_field], errors="coerce").max()
            if pd.notna(maximum):
                start_date = str((maximum - pd.Timedelta(days=14)).date())
    records = []
    offset = 0
    while True:
        batch = page(session, dataset, offset, limit, start_date)
        records.extend(batch)
        print(f"{dataset}: offset={offset:,} rows={len(batch):,}")
        if len(batch) < limit:
            break
        offset += limit
    fresh = normalize(records)
    if fresh.empty and old.empty:
        raise RuntimeError(f"FINRA returned no rows for {dataset}")
    combined = pd.concat([old, fresh], ignore_index=True).drop_duplicates()
    date_field = DATE_FIELDS[dataset]
    if date_field in combined:
        combined = combined.sort_values(
            [date_field] + [column for column in combined.columns[:1] if column != date_field],
            ignore_index=True,
        )
    if combined.duplicated().any():
        raise ValueError(f"Duplicate rows remain in {dataset}")
    temporary = output.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)
    date_range = ""
    if date_field in combined:
        dates = pd.to_datetime(combined[date_field], errors="coerce")
        date_range = f" dates={dates.min().date()}..{dates.max().date()}"
    print(f"wrote {output} rows={len(combined):,}{date_range}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--dataset", choices=tuple(DATASETS), action="append")
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 5000:
        raise ValueError("--limit must be between 1 and FINRA's 5,000-row page limit")
    client_id = os.getenv("FINRA_CLIENT_ID", "").strip()
    client_secret = os.getenv("FINRA_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError("Set FINRA_CLIENT_ID and FINRA_CLIENT_SECRET from a FINRA Public API credential.")
    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {token(client_id, client_secret)}", "Accept": "application/json"}
    )
    API_CACHE.mkdir(parents=True, exist_ok=True)
    selected = args.dataset or list(DATASETS)
    for dataset in selected:
        fetch_dataset(session, dataset, API_CACHE / DATASETS[dataset], args.update, args.limit)


if __name__ == "__main__":
    main()
