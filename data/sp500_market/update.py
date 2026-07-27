from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, time as clock_time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import download as base
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OVERLAP_DAYS = 14
ACTIVE_LOOKBACK_DAYS = 30
NEW_YORK = ZoneInfo("America/New_York")


def log(message: str, indent: int = 0) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {'  ' * indent}{message}", flush=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_exclusive_end() -> pd.Timestamp:
    """Use only completed U.S. sessions; include today after an evening buffer."""
    now = datetime.now(NEW_YORK)
    end_date = now.date()
    if now.time() >= clock_time(20, 0):
        end_date += timedelta(days=1)
    return pd.Timestamp(end_date)


def recent_snapshot_union(
    snapshots: pd.DataFrame,
    replace_start: pd.Timestamp,
) -> list[str]:
    before = snapshots.loc[snapshots["snapshot_date"] <= replace_start].tail(1)
    after = snapshots.loc[snapshots["snapshot_date"] > replace_start]
    selected = pd.concat([before, after], ignore_index=True)
    return base.universe_union(selected)


def fetch_incremental_group(
    tickers: list[str],
    fetch_start: pd.Timestamp,
    end: pd.Timestamp,
    manifest: dict[str, dict[str, Any]],
    chunk_size: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    if not tickers:
        return manifest, 0
    symbol_to_tickers: dict[str, list[str]] = {}
    for ticker in tickers:
        symbol_to_tickers.setdefault(base.yf_symbol(ticker), []).append(ticker)
    symbols = sorted(symbol_to_tickers)
    updated = 0
    for batch_number, symbol_batch in enumerate(base.chunks(symbols, chunk_size), start=1):
        fetched = base.yf_download(
            symbol_batch,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            threads=True,
        )
        missed: list[str] = []
        for symbol in symbol_batch:
            frame = fetched.get(symbol)
            if frame is None or frame.empty:
                missed.append(symbol)
                continue
            for ticker in symbol_to_tickers[symbol]:
                combined = base.merge_cache(base.existing_cache(ticker), frame)
                combined = combined.loc[
                    (combined.index >= pd.Timestamp(base.DEFAULT_START)) & (combined.index < end)
                ]
                if combined.empty:
                    missed.append(symbol)
                    continue
                note = (
                    f"historical ticker mapped to current symbol {symbol}"
                    if ticker != symbol
                    else "incremental update"
                )
                base.store_ticker(ticker, symbol, combined, manifest, note)
                manifest[ticker]["last_requested_end_exclusive"] = str(end.date())
                updated += 1

        for symbol in dict.fromkeys(missed):
            retry = base.yf_download(
                [symbol],
                start=fetch_start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                threads=False,
            ).get(symbol)
            for ticker in symbol_to_tickers[symbol]:
                if retry is not None and not retry.empty:
                    combined = base.merge_cache(base.existing_cache(ticker), retry)
                    combined = combined.loc[
                        (combined.index >= pd.Timestamp(base.DEFAULT_START))
                        & (combined.index < end)
                    ]
                    base.store_ticker(
                        ticker,
                        symbol,
                        combined,
                        manifest,
                        "incremental individual retry",
                    )
                    manifest[ticker]["last_requested_end_exclusive"] = str(end.date())
                    updated += 1
                elif base.existing_cache(ticker) is None:
                    base.mark_missing(
                        ticker,
                        symbol,
                        manifest,
                        "no data returned during incremental update",
                    )
                    manifest[ticker]["last_requested_end_exclusive"] = str(end.date())
                elif ticker in manifest:
                    manifest[ticker]["last_requested_end_exclusive"] = str(end.date())
        base.write_manifest(manifest)
        log(
            f"batch {batch_number}/{math.ceil(len(symbols) / chunk_size)} complete",
            2,
        )
        time.sleep(0.4)
    return manifest, updated


def add_existing_features(
    refreshed: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    feature_columns = [
        column for column in ["industry", "market_cap"] if column in existing.columns
    ]
    if not feature_columns:
        return refreshed
    old_features = existing[["date", "ticker", *feature_columns]].drop_duplicates(
        ["date", "ticker"]
    )
    return refreshed.merge(
        old_features,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )


def write_market_atomic(
    frame: pd.DataFrame,
    original_schema: pa.Schema,
    source_metadata: dict[str, str],
    validation: dict[str, Any],
    update_metadata: dict[str, Any],
) -> None:
    output_tmp = base.OUTPUT_PATH.with_name(f".{base.OUTPUT_PATH.name}.update.tmp")
    if output_tmp.exists():
        output_tmp.unlink()
    schema_without_metadata = original_schema.remove_metadata()
    table = pa.Table.from_pandas(
        frame,
        schema=schema_without_metadata,
        preserve_index=False,
        safe=False,
    )
    metadata = {
        key: value for key, value in (original_schema.metadata or {}).items() if key != b"pandas"
    }
    metadata[b"generated_at_utc"] = utc_now().isoformat().encode()
    metadata[b"window_end_exclusive"] = str(update_metadata["end_exclusive"]).encode()
    metadata[b"constituent_source_metadata"] = json.dumps(source_metadata, sort_keys=True).encode()
    metadata[b"validation"] = json.dumps(validation, sort_keys=True).encode()
    metadata[b"incremental_market_update"] = json.dumps(update_metadata, sort_keys=True).encode()
    table = table.replace_schema_metadata(metadata)
    try:
        pq.write_table(
            table,
            output_tmp,
            compression="zstd",
            compression_level=9,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=250_000,
        )
        check = pq.ParquetFile(output_tmp)
        if check.metadata.num_rows != len(frame):
            raise RuntimeError("Incremental market parquet row-count mismatch")
        check.close()
        output_tmp.replace(base.OUTPUT_PATH)
    finally:
        if output_tmp.exists():
            output_tmp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update S&P market prices and membership from a short "
            "overlap window; does not rebuild or redownload full histories."
        )
    )
    parser.add_argument("--end", help="Exclusive end date; defaults to completed U.S. sessions.")
    parser.add_argument("--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS)
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not base.OUTPUT_PATH.exists():
        raise FileNotFoundError(f"{base.OUTPUT_PATH} does not exist; run download.py once first.")
    if args.overlap_days < 7:
        raise ValueError("--overlap-days must be at least 7")
    end = pd.Timestamp(args.end) if args.end else default_exclusive_end()

    log("1/6 Read the existing validated market parquet")
    original_parquet = pq.ParquetFile(base.OUTPUT_PATH)
    original_schema = original_parquet.schema_arrow
    original_parquet.close()
    existing = pd.read_parquet(base.OUTPUT_PATH)
    existing["date"] = pd.to_datetime(existing["date"])
    old_rows = len(existing)
    old_max = existing["date"].max()
    replace_start = old_max - pd.Timedelta(days=args.overlap_days)
    if end <= replace_start:
        raise ValueError("Exclusive end must be later than the overlap start")
    log(
        f"{old_rows:,} rows through {old_max.date()}; "
        f"refresh window starts {replace_start.date()}",
        1,
    )

    log("2/6 Refresh constituent snapshots and identify recent securities")
    constituent_path, source_metadata = base.download_constituents(
        refresh=not args.offline,
        offline=args.offline,
    )
    snapshots = base.load_snapshots(
        constituent_path,
        pd.Timestamp(base.DEFAULT_START),
        end,
    )
    recent_universe = recent_snapshot_union(snapshots, replace_start)
    recent_existing_members = set(
        existing.loc[
            existing["date"].ge(replace_start) & existing["is_sp500_member"],
            "ticker",
        ].astype(str)
    )
    active_cutoff = old_max - pd.Timedelta(days=ACTIVE_LOOKBACK_DAYS)
    recently_active_existing = set(
        existing.loc[existing["date"].ge(active_cutoff), "ticker"].astype(str)
    )
    update_universe = sorted(
        set(recent_universe) | recent_existing_members | recently_active_existing
    )
    existing_tickers = set(existing["ticker"].astype(str))
    new_tickers = sorted(set(update_universe) - existing_tickers)
    log(
        f"{len(update_universe):,} recent/active securities; "
        f"{len(new_tickers):,} newly encountered tickers",
        1,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "old_max_date": str(old_max.date()),
                    "replace_start": str(replace_start.date()),
                    "end_exclusive": str(end.date()),
                    "tickers_to_update": len(update_universe),
                    "recently_active_existing_tickers": len(recently_active_existing),
                    "new_tickers": new_tickers,
                    "offline": args.offline,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    log("3/6 Download only the overlap window (full history only for new tickers)")
    manifest = base.read_manifest()
    updated_tickers = 0
    if not args.offline:
        old_tickers = sorted(set(update_universe) - set(new_tickers))
        expected_latest_business_day = (end - pd.offsets.BDay(1)).normalize()
        if expected_latest_business_day <= old_max:
            stale_tickers: list[str] = []
            for ticker in old_tickers:
                cached = base.existing_cache(ticker)
                requested_end = pd.to_datetime(
                    manifest.get(ticker, {}).get("last_requested_end_exclusive"),
                    errors="coerce",
                )
                already_attempted = pd.notna(requested_end) and requested_end >= end
                cache_is_stale = cached is None or cached.empty or cached.index.max() < old_max
                if cache_is_stale and not already_attempted:
                    stale_tickers.append(ticker)
            old_tickers = stale_tickers
            log(
                f"Completed-session caches are already current; "
                f"only {len(old_tickers):,} stale tickers need Yahoo",
                1,
            )
        manifest, count = fetch_incremental_group(
            old_tickers,
            replace_start - pd.Timedelta(days=3),
            end,
            manifest,
            args.chunk_size,
        )
        updated_tickers += count
        manifest, count = fetch_incremental_group(
            new_tickers,
            pd.Timestamp(base.DEFAULT_START),
            end,
            manifest,
            args.chunk_size,
        )
        updated_tickers += count
    else:
        log("Offline mode: using existing per-ticker caches", 1)

    log("4/6 Assemble only refreshed rows and point-in-time membership")
    refreshed, cleaning = base.assemble_long_panel(
        update_universe,
        manifest,
        replace_start,
        end,
    )
    refreshed = base.add_membership(refreshed, snapshots)
    if new_tickers:
        new_history, new_cleaning = base.assemble_long_panel(
            new_tickers,
            manifest,
            pd.Timestamp(base.DEFAULT_START),
            end,
        )
        new_history = base.add_membership(new_history, snapshots)
        refreshed = pd.concat(
            [
                refreshed.loc[~refreshed["ticker"].isin(new_tickers)],
                new_history,
            ],
            ignore_index=True,
        ).drop_duplicates(["date", "ticker"], keep="last")
        cleaning["glitch_observations_removed"] += new_cleaning["glitch_observations_removed"]

    refreshed = add_existing_features(refreshed, existing)
    for column in original_schema.names:
        if column not in refreshed.columns:
            refreshed[column] = pd.NA if column == "industry" else np.nan
    refreshed = refreshed[original_schema.names]

    replace_mask = existing["date"].ge(replace_start) & existing["ticker"].isin(update_universe)
    if new_tickers:
        replace_mask |= existing["ticker"].isin(new_tickers)
    combined = pd.concat(
        [existing.loc[~replace_mask], refreshed],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(["date", "ticker"], keep="last")
    combined = combined.sort_values(["date", "ticker"], ignore_index=True)

    log("5/6 Validate unchanged history plus the incremental window")
    duplicate_keys = int(combined.duplicated(["date", "ticker"]).sum())
    invalid_prices = int((combined["adj_close"].isna() | combined["adj_close"].le(0)).sum())
    negative_volume = int((combined["volume"].dropna() < 0).sum())
    dates_at_or_after_end = int(combined["date"].ge(end).sum())
    failures: list[str] = []
    if duplicate_keys:
        failures.append(f"{duplicate_keys} duplicate date/ticker keys")
    if invalid_prices:
        failures.append(f"{invalid_prices} invalid adjusted closes")
    if negative_volume:
        failures.append(f"{negative_volume} negative volume rows")
    if dates_at_or_after_end:
        failures.append(f"{dates_at_or_after_end} rows at/after exclusive end")
    if combined["date"].max() < old_max:
        failures.append("output maximum date moved backward")
    incremental_validation = {
        "status": "pass" if not failures else "fail",
        "old_rows": int(old_rows),
        "new_rows": int(len(combined)),
        "rows_added_net": int(len(combined) - old_rows),
        "old_max_date": str(old_max.date()),
        "new_max_date": str(combined["date"].max().date()),
        "replace_start": str(replace_start.date()),
        "end_exclusive": str(end.date()),
        "tickers_requested": len(update_universe),
        "recently_active_existing_tickers": len(recently_active_existing),
        "tickers_with_downloaded_updates": int(updated_tickers),
        "new_tickers": new_tickers,
        "duplicate_date_ticker_keys": duplicate_keys,
        "invalid_adjusted_closes": invalid_prices,
        "negative_volumes": negative_volume,
        "dates_at_or_after_end": dates_at_or_after_end,
        **cleaning,
        "failures": failures,
    }
    log(json.dumps(incremental_validation, indent=2, sort_keys=True), 1)
    if failures:
        raise RuntimeError(
            "Incremental market validation failed; parquet was not replaced. " + "; ".join(failures)
        )

    log("6/6 Atomically replace the single parquet")
    old_metadata = {
        key.decode(): value.decode() for key, value in (original_schema.metadata or {}).items()
    }
    base_validation = json.loads(old_metadata.get("validation", "{}"))
    base_validation.update(
        {
            "status": "pass",
            "rows": int(len(combined)),
            "tickers_with_prices": int(combined["ticker"].nunique()),
            "date_min": str(combined["date"].min().date()),
            "date_max": str(combined["date"].max().date()),
            "duplicate_date_ticker_keys": duplicate_keys,
            "invalid_adjusted_closes": invalid_prices,
            "negative_volumes": negative_volume,
        }
    )
    write_market_atomic(
        combined,
        original_schema,
        source_metadata,
        base_validation,
        {
            **incremental_validation,
            "updated_at_utc": utc_now().isoformat(),
        },
    )
    log(f"Wrote {base.OUTPUT_PATH} " f"({base.OUTPUT_PATH.stat().st_size / 1_000_000:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
