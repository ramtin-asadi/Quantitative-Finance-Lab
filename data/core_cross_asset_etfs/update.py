"""Incrementally update the core cross-asset ETF panel.

The updater downloads only a recent overlap, retries missing symbols one at a
time, merges successful observations into the existing wide CSV, and replaces
the output atomically only after validation. Older history is never rewritten.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, time as clock_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import download as base


DEFAULT_OVERLAP_DAYS = 14
NEW_YORK = ZoneInfo("America/New_York")
FIELDS = ("Close", "Volume", "Dividends", "Stock Splits")
OUTPUT_NAMES = ("close", "volume", "dividends", "stock_splits")


def default_exclusive_end() -> pd.Timestamp:
    """Include today only after a conservative U.S. evening buffer."""

    now = datetime.now(NEW_YORK)
    end_date = now.date()
    if now.time() >= clock_time(20, 0):
        end_date += timedelta(days=1)
    return pd.Timestamp(end_date)


def extract(raw: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        for key in ((ticker, field), (field, ticker)):
            if key in raw.columns:
                return pd.to_numeric(raw[key], errors="coerce")
    elif field in raw.columns:
        return pd.to_numeric(raw[field], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype=float)


def normalize_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index
    return frame.loc[frame.index.notna()].sort_index()


def fetch_batch(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return normalize_download(
        yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            actions=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    )


def ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    columns = {}
    for field, output_name in zip(FIELDS, OUTPUT_NAMES, strict=True):
        columns[f"{ticker}__{output_name}"] = extract(raw, ticker, field)
    return pd.DataFrame(columns, index=raw.index)


def fetch_recent_panel(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    batch = fetch_batch(base.TICKERS, start, end)
    pieces: list[pd.DataFrame] = []
    missing: list[str] = []
    for ticker in base.TICKERS:
        piece = ticker_frame(batch, ticker)
        if piece[f"{ticker}__close"].notna().any():
            pieces.append(piece)
        else:
            missing.append(ticker)

    unresolved: list[str] = []
    for position, ticker in enumerate(missing, start=1):
        retry = fetch_batch([ticker], start, end)
        piece = ticker_frame(retry, ticker)
        if piece[f"{ticker}__close"].notna().any():
            pieces.append(piece)
        else:
            unresolved.append(ticker)
        if position < len(missing):
            time.sleep(0.2)

    if not pieces:
        raise RuntimeError("Yahoo returned no usable ETF observations")
    return pd.concat(pieces, axis=1).sort_index(), unresolved


def read_existing(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("existing ETF panel has invalid dates")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    for ticker in base.TICKERS:
        column = f"{ticker}__close"
        if column not in frame or frame[column].notna().sum() < 252:
            raise ValueError(
                f"existing ETF panel has insufficient history for {ticker}; "
                "run download.py once to rebuild it"
            )
    return frame


def merge_recent(
    existing: pd.DataFrame,
    recent: pd.DataFrame,
    replace_start: pd.Timestamp,
) -> pd.DataFrame:
    expected_columns = [
        f"{ticker}__{field}"
        for ticker in base.TICKERS
        for field in OUTPUT_NAMES
    ]
    if list(existing.columns) != expected_columns:
        raise ValueError("existing ETF schema does not match the configured universe")

    index = existing.index.union(recent.index).sort_values()
    combined = existing.reindex(index=index, columns=expected_columns)
    recent = recent.loc[recent.index >= replace_start]
    for column in recent.columns.intersection(combined.columns):
        incoming = pd.to_numeric(recent[column], errors="coerce").dropna()
        combined.loc[incoming.index, column] = incoming
    return combined


def validate(
    existing: pd.DataFrame,
    updated: pd.DataFrame,
    replace_start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    if updated.index.has_duplicates or not updated.index.is_monotonic_increasing:
        raise ValueError("updated ETF dates are invalid")
    if bool((updated.index >= end).any()):
        raise ValueError("updated ETF panel contains a date at or after the exclusive end")

    frozen = existing.index[existing.index < replace_start]
    pd.testing.assert_frame_equal(
        updated.reindex(index=frozen, columns=existing.columns),
        existing.reindex(index=frozen, columns=existing.columns),
        check_dtype=False,
        check_freq=False,
    )

    latest_dates: dict[str, str] = {}
    for ticker in base.TICKERS:
        close = updated[f"{ticker}__close"]
        if not close.notna().any():
            raise ValueError(f"{ticker} has no closing-price history")
        if bool(close.dropna().le(0.0).any()):
            raise ValueError(f"{ticker} has a non-positive closing price")
        volume = updated[f"{ticker}__volume"].dropna()
        if bool(volume.lt(0.0).any()):
            raise ValueError(f"{ticker} has negative volume")
        latest_dates[ticker] = str(updated.index[close.notna()].max().date())

    max_date = max(pd.Timestamp(value) for value in latest_dates.values())
    latest_coverage = sum(
        pd.Timestamp(value) == max_date for value in latest_dates.values()
    ) / len(latest_dates)
    if latest_coverage < 0.90:
        raise ValueError(
            f"only {latest_coverage:.1%} of ETFs reach the panel's latest date"
        )
    return {
        "status": "pass",
        "old_end": str(existing.index.max().date()),
        "new_end": str(updated.index.max().date()),
        "old_rows": len(existing),
        "new_rows": len(updated),
        "latest_coverage": round(latest_coverage, 6),
        "latest_dates": latest_dates,
    }


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index_label="date")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally update the core cross-asset ETF CSV."
    )
    parser.add_argument("--end", help="Exclusive Yahoo end date (YYYY-MM-DD).")
    parser.add_argument(
        "--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.overlap_days < 7:
        raise ValueError("--overlap-days must be at least 7")
    if not base.OUTPUT.exists():
        raise FileNotFoundError(
            f"{base.OUTPUT} is missing; run download.py once first"
        )

    end = pd.Timestamp(args.end) if args.end else default_exclusive_end()
    existing = read_existing(base.OUTPUT)
    replace_start = existing.index.max() - pd.Timedelta(days=args.overlap_days)
    recent, unresolved = fetch_recent_panel(replace_start, end)
    updated = merge_recent(existing, recent, replace_start)
    report = validate(existing, updated, replace_start, end)
    report["download_start"] = str(replace_start.date())
    report["download_end_exclusive"] = str(end.date())
    report["unresolved_tickers"] = unresolved

    if not args.dry_run:
        write_atomic(updated, base.OUTPUT)
    print(pd.Series(report).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
