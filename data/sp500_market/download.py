from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
CACHE_DIR = HERE / "cache"
PRICE_CACHE_DIR = CACHE_DIR / "prices"
MANIFEST_PATH = CACHE_DIR / "download_manifest.csv"
OUTPUT_PATH = DATA_DIR / "sp500_market_data.parquet"
FUNDAMENTALS_PATH = DATA_DIR / "sp500_fundamentals.parquet"
CONSTITUENTS_PATH = RAW_DIR / "sp500_historical_components.csv"

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)
SOURCE_REPOSITORY = "https://github.com/fja05680/sp500"
DEFAULT_START = "2005-01-01"
YF_FIELDS = [
    "adj_close",
    "close",
    "volume",
    "dividends",
    "stock_splits",
    "was_repaired",
]

# True symbol changes only. Acquisitions are deliberately not substituted.
RENAME_MAP = {
    "ABC": "COR",
    "ANTM": "ELV",
    "BLL": "BALL",
    "COG": "CTRA",
    "CTL": "LUMN",
    "DPS": "KDP",
    "FB": "META",
    "FBHS": "FBIN",
    "HFC": "DINO",
    "HRS": "LHX",
    "KORS": "CPRI",
    "LB": "BBWI",
    "MYL": "VTRS",
    "PKI": "RVTY",
    "RE": "EG",
    "UTX": "RTX",
    "WLTW": "WTW",
}


def log(message: str, indent: int = 0) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {'  ' * indent}{message}", flush=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    return re.sub(r"-\d{6}$", "", ticker)


def yf_symbol(ticker: str) -> str:
    return RENAME_MAP.get(ticker, ticker).replace(".", "-")


def safe_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")


def cache_path(ticker: str) -> Path:
    return PRICE_CACHE_DIR / f"{safe_name(ticker)}.parquet"


def download_constituents(refresh: bool, offline: bool) -> tuple[Path, dict[str, str]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CONSTITUENTS_PATH.exists() and not refresh:
        content = CONSTITUENTS_PATH.read_bytes()
        return CONSTITUENTS_PATH, {
            "url": CONSTITUENTS_URL,
            "sha256": sha256_bytes(content),
            "retrieved_at": datetime.fromtimestamp(
                CONSTITUENTS_PATH.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "mode": "cached",
        }
    if offline:
        raise FileNotFoundError(
            f"Offline mode requires {CONSTITUENTS_PATH}. See {HERE / 'README.md'}."
        )

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(
                CONSTITUENTS_URL,
                timeout=(15, 120),
                headers={"User-Agent": "Quantitative-Finance-Lab data reproducibility script"},
            )
            response.raise_for_status()
            content = response.content
            trial = pd.read_csv(pd.io.common.BytesIO(content), nrows=5)
            if not {"date", "tickers"}.issubset(trial.columns):
                raise ValueError(f"Unexpected constituent columns: {trial.columns.tolist()}")
            CONSTITUENTS_PATH.write_bytes(content)
            return CONSTITUENTS_PATH, {
                "url": CONSTITUENTS_URL,
                "sha256": sha256_bytes(content),
                "retrieved_at": utc_now().isoformat(),
                "etag": response.headers.get("ETag", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "mode": "downloaded",
            }
        except Exception as exc:  # network retry boundary
            last_error = exc
            time.sleep(2**attempt)

    if CONSTITUENTS_PATH.exists():
        log(f"WARNING: GitHub download failed; using cached file ({last_error})", 1)
        content = CONSTITUENTS_PATH.read_bytes()
        return CONSTITUENTS_PATH, {
            "url": CONSTITUENTS_URL,
            "sha256": sha256_bytes(content),
            "retrieved_at": datetime.fromtimestamp(
                CONSTITUENTS_PATH.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "mode": "cached_after_download_failure",
        }
    raise RuntimeError(
        "Could not download the historical constituents file and no cached copy exists. "
        f"Download it from {SOURCE_REPOSITORY} and save it as {CONSTITUENTS_PATH}."
    ) from last_error


def load_snapshots(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if not {"date", "tickers"}.issubset(raw.columns):
        raise ValueError(f"Bad constituent file columns: {raw.columns.tolist()}")
    raw["date"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    rows: list[dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        tokens = [
            normalize_ticker(token) for token in str(row.tickers).split(",") if str(token).strip()
        ]
        tokens = list(dict.fromkeys(tokens))
        rows.append({"snapshot_date": pd.Timestamp(row.date), "tickers": tokens})
    snapshots = pd.DataFrame(rows).sort_values("snapshot_date")
    before = snapshots.loc[snapshots["snapshot_date"] < start].tail(1)
    inside = snapshots.loc[
        (snapshots["snapshot_date"] >= start) & (snapshots["snapshot_date"] < end)
    ]
    selected = pd.concat([before, inside], ignore_index=True).drop_duplicates(
        "snapshot_date", keep="last"
    )
    selected["snapshot_date"] = selected["snapshot_date"].astype("datetime64[ns]")
    if selected.empty:
        raise ValueError(f"No constituent snapshots cover {start.date()} to {end.date()}.")
    counts = selected["tickers"].map(len)
    if counts.min() < 450 or counts.max() > 550:
        raise ValueError(
            "Constituent counts are implausible after parsing: "
            f"min={counts.min()}, max={counts.max()}."
        )
    return selected.reset_index(drop=True)


def universe_union(snapshots: pd.DataFrame) -> list[str]:
    universe: set[str] = set()
    for tickers in snapshots["tickers"]:
        universe.update(tickers)
    return sorted(universe)


def read_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    frame = pd.read_csv(MANIFEST_PATH).fillna("")
    return {str(row["ticker"]): row.to_dict() for _, row in frame.iterrows()}


def write_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(manifest.values()))
    if not frame.empty:
        frame = frame.sort_values("ticker")
    frame.to_csv(MANIFEST_PATH, index=False)


def normalize_yf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if out.columns.nlevels != 2:
            raise ValueError(f"Unexpected yfinance column levels: {out.columns}")
        out.columns = out.columns.get_level_values(-1)
    out.columns = [str(column).strip().lower().replace(" ", "_") for column in out.columns]
    aliases = {"repaired?": "was_repaired"}
    out = out.rename(columns=aliases)
    for column in YF_FIELDS:
        if column not in out.columns:
            if column in {"dividends", "stock_splits"}:
                out[column] = 0.0
            elif column == "was_repaired":
                out[column] = False
            else:
                out[column] = np.nan
    out.index = pd.to_datetime(out.index, errors="coerce").tz_localize(None)
    out.index.name = "date"
    out = out.loc[out.index.notna(), YF_FIELDS].sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    for column in ["adj_close", "close", "volume", "dividends", "stock_splits"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["was_repaired"] = out["was_repaired"].fillna(False).astype(bool)
    out.loc[out["adj_close"] <= 0, "adj_close"] = np.nan
    out.loc[out["close"] <= 0, "close"] = np.nan
    out.loc[out["volume"] < 0, "volume"] = np.nan
    out.loc[out["dividends"] < 0, "dividends"] = np.nan
    out.loc[out["stock_splits"] < 0, "stock_splits"] = np.nan
    return out.dropna(subset=["adj_close"])


def split_batch_result(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}
    output: dict[str, pd.DataFrame] = {}
    unique_symbols = list(dict.fromkeys(symbols))
    if len(unique_symbols) == 1:
        output[unique_symbols[0]] = normalize_yf_frame(raw)
        return output
    if not isinstance(raw.columns, pd.MultiIndex):
        return output
    top = set(map(str, raw.columns.get_level_values(0)))
    for symbol in unique_symbols:
        try:
            if symbol in top:
                sub = raw[symbol]
            else:
                sub = raw.xs(symbol, axis=1, level=-1)
            normalized = normalize_yf_frame(sub)
            if not normalized.empty:
                output[symbol] = normalized
        except (KeyError, ValueError):
            continue
    return output


def yf_download(symbols: list[str], start: str, end: str, threads: bool) -> dict[str, pd.DataFrame]:
    unique_symbols = list(dict.fromkeys(symbols))
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(
                unique_symbols,
                start=start,
                end=end,
                auto_adjust=False,
                back_adjust=False,
                actions=True,
                repair=True,
                keepna=False,
                progress=False,
                threads=threads,
                group_by="ticker",
                timeout=30,
            )
            return split_batch_result(raw, unique_symbols)
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** (attempt + 1))
    if last_error:
        log(f"yfinance batch failed: {last_error}", 2)
    return {}


def merge_cache(existing: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return fresh
    combined = pd.concat([existing, fresh]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    return combined


def store_ticker(
    ticker: str,
    symbol: str,
    frame: pd.DataFrame,
    manifest: dict[str, dict[str, Any]],
    note: str = "",
) -> None:
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path(ticker))
    manifest[ticker] = {
        "ticker": ticker,
        "yf_ticker": symbol,
        "source": "yfinance",
        "status": "ok",
        "n_rows": len(frame),
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "fetched_at": utc_now().isoformat(),
        "note": note,
    }


def mark_missing(
    ticker: str,
    symbol: str,
    manifest: dict[str, dict[str, Any]],
    note: str,
) -> None:
    manifest[ticker] = {
        "ticker": ticker,
        "yf_ticker": symbol,
        "source": "yfinance",
        "status": "unavailable",
        "n_rows": 0,
        "first_date": "",
        "last_date": "",
        "fetched_at": utc_now().isoformat(),
        "note": note,
    }


def existing_cache(ticker: str) -> pd.DataFrame | None:
    path = cache_path(ticker)
    if not path.exists():
        return None
    try:
        return normalize_yf_frame(pd.read_parquet(path))
    except Exception as exc:
        log(f"WARNING: ignoring corrupt cache for {ticker}: {exc}", 2)
        return None


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def download_prices(
    universe: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int,
    refresh: bool,
    retry_missing: bool,
) -> dict[str, dict[str, Any]]:
    manifest = read_manifest()
    end_string = end.strftime("%Y-%m-%d")
    todo: list[str] = []
    for ticker in universe:
        cached = existing_cache(ticker)
        status = str(manifest.get(ticker, {}).get("status", ""))
        if refresh or cached is None:
            if status == "unavailable" and not retry_missing and not refresh:
                continue
            todo.append(ticker)
            continue
        last = cached.index.max()
        if last < end - pd.Timedelta(days=5):
            todo.append(ticker)

    log(f"yfinance: {len(todo)} to fetch/update; {len(universe) - len(todo)} cached/marked")
    symbol_to_tickers: dict[str, list[str]] = {}
    for ticker in todo:
        symbol_to_tickers.setdefault(yf_symbol(ticker), []).append(ticker)
    symbols = sorted(symbol_to_tickers)

    for batch_number, symbol_batch in enumerate(chunks(symbols, chunk_size), start=1):
        fetch_start = start
        if not refresh:
            cached_starts: list[pd.Timestamp] = []
            needs_full_history = False
            for symbol in symbol_batch:
                for ticker in symbol_to_tickers[symbol]:
                    cached = existing_cache(ticker)
                    if cached is not None and not cached.empty:
                        cached_starts.append(cached.index.max() - pd.Timedelta(days=14))
                    else:
                        needs_full_history = True
            if cached_starts and not needs_full_history:
                fetch_start = min(cached_starts)
        fetched = yf_download(
            symbol_batch,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end_string,
            threads=True,
        )

        missed_symbols: list[str] = []
        for symbol in symbol_batch:
            frame = fetched.get(symbol)
            if frame is None or frame.empty:
                missed_symbols.append(symbol)
                continue
            for ticker in symbol_to_tickers[symbol]:
                combined = merge_cache(existing_cache(ticker), frame)
                combined = combined.loc[(combined.index >= start) & (combined.index < end)]
                if combined.empty:
                    missed_symbols.append(symbol)
                    continue
                note = (
                    f"historical ticker mapped to current symbol {symbol}"
                    if ticker != symbol
                    else ""
                )
                store_ticker(ticker, symbol, combined, manifest, note)

        # Individual retries catch symbols lost inside a large Yahoo batch.
        for symbol in dict.fromkeys(missed_symbols):
            retry = yf_download(
                [symbol],
                start=start.strftime("%Y-%m-%d"),
                end=end_string,
                threads=False,
            ).get(symbol)
            for ticker in symbol_to_tickers[symbol]:
                if retry is not None and not retry.empty:
                    combined = merge_cache(existing_cache(ticker), retry)
                    combined = combined.loc[(combined.index >= start) & (combined.index < end)]
                    store_ticker(ticker, symbol, combined, manifest, "individual retry")
                elif existing_cache(ticker) is None:
                    mark_missing(ticker, symbol, manifest, "no data returned by yfinance")

        write_manifest(manifest)
        log(
            f"batch {batch_number}/{math.ceil(len(symbols) / chunk_size) or 1} complete",
            1,
        )
        time.sleep(0.5)

    write_manifest(manifest)
    return manifest


def clean_adjusted_prices(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, bool]:
    """Remove isolated/reverting Yahoo artifacts without deleting real one-way moves."""
    out = frame.copy()
    values = out["adj_close"].to_numpy(dtype=float, copy=True)
    dates = out.index.to_numpy(dtype="datetime64[D]")
    nulled = 0

    for _ in range(4):
        valid = np.where(np.isfinite(values) & (values > 0))[0]
        if len(valid) < 3:
            break
        pass_nulled = 0
        for pos in range(1, len(valid) - 1):
            left_i, mid_i, right_i = valid[pos - 1], valid[pos], valid[pos + 1]
            if (dates[mid_i] - dates[left_i]).astype(int) > 10:
                continue
            if (dates[right_i] - dates[mid_i]).astype(int) > 10:
                continue
            left, middle, right = values[left_i], values[mid_i], values[right_i]
            move_in = middle / left - 1.0
            move_out = right / middle - 1.0
            round_trip = abs(right / left - 1.0) < 1.0
            if (
                abs(move_in) > 0.50
                and abs(move_out) > 0.50
                and move_in * move_out < 0
                and round_trip
            ):
                values[mid_i] = np.nan
                pass_nulled += 1
        nulled += pass_nulled
        if pass_nulled == 0:
            break

    # A >3x or <-75% one-session adjusted move is not credible for an S&P 500 member.
    valid = np.where(np.isfinite(values) & (values > 0))[0]
    if len(valid) >= 2:
        anchor_i = valid[0]
        for current_i in valid[1:]:
            if (dates[current_i] - dates[anchor_i]).astype(int) > 10:
                anchor_i = current_i
                continue
            ratio = values[current_i] / values[anchor_i]
            if ratio > 3.0 or ratio < 0.25:
                values[current_i] = np.nan
                nulled += 1
            else:
                anchor_i = current_i

    out["adj_close"] = values
    out = out.dropna(subset=["adj_close"])
    returns = out["adj_close"].pct_change(fill_method=None).abs()
    irrecoverable = int((returns > 0.50).sum()) > 15
    return out, nulled, irrecoverable


def assemble_long_panel(
    universe: list[str],
    manifest: dict[str, dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    total_nulled = 0
    dropped: list[str] = []
    for ticker in universe:
        if str(manifest.get(ticker, {}).get("status", "")) != "ok":
            continue
        cached = existing_cache(ticker)
        if cached is None or cached.empty:
            continue
        cached = cached.loc[(cached.index >= start) & (cached.index < end)]
        cleaned, nulled, irrecoverable = clean_adjusted_prices(cached)
        total_nulled += nulled
        if irrecoverable:
            dropped.append(ticker)
            continue
        cleaned = cleaned.reset_index()
        cleaned.insert(1, "ticker", ticker)
        cleaned.insert(2, "yf_ticker", str(manifest[ticker]["yf_ticker"]))
        frames.append(cleaned)
    if not frames:
        raise RuntimeError("No valid ticker histories were assembled.")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").astype("datetime64[ns]")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    panel["ticker"] = panel["ticker"].astype("string")
    panel["yf_ticker"] = panel["yf_ticker"].astype("string")
    panel["volume"] = panel["volume"].round().astype("Int64")
    panel["dividends"] = panel["dividends"].fillna(0.0)
    panel["stock_splits"] = panel["stock_splits"].fillna(0.0)
    panel["was_repaired"] = panel["was_repaired"].fillna(False).astype(bool)
    return panel, {
        "glitch_observations_removed": total_nulled,
        "irrecoverable_tickers_dropped": dropped,
    }


def add_membership(panel: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"date": panel["date"].drop_duplicates().sort_values()})
    snapshot_dates = snapshots[["snapshot_date"]].sort_values("snapshot_date")
    dates = pd.merge_asof(
        dates,
        snapshot_dates,
        left_on="date",
        right_on="snapshot_date",
        direction="backward",
    )
    if dates["snapshot_date"].isna().any():
        raise ValueError("Some price dates precede the first selected constituent snapshot.")

    membership_dates: list[np.ndarray] = []
    membership_tickers: list[np.ndarray] = []
    snap_lookup = snapshots.set_index("snapshot_date")["tickers"]
    for row in dates.itertuples(index=False):
        tickers = np.asarray(snap_lookup.loc[row.snapshot_date], dtype=object)
        membership_dates.append(np.repeat(np.datetime64(row.date, "ns"), len(tickers)))
        membership_tickers.append(tickers)
    membership = pd.DataFrame(
        {
            "date": np.concatenate(membership_dates),
            "ticker": np.concatenate(membership_tickers),
        }
    )
    membership["is_sp500_member"] = True
    out = panel.merge(membership, on=["date", "ticker"], how="left", validate="one_to_one")
    out["is_sp500_member"] = out["is_sp500_member"].fillna(False).astype(bool)
    out = out.merge(dates, on="date", how="left", validate="many_to_one")
    return out[
        [
            "date",
            "ticker",
            "yf_ticker",
            "adj_close",
            "close",
            "volume",
            "dividends",
            "stock_splits",
            "was_repaired",
            "is_sp500_member",
            "snapshot_date",
        ]
    ].sort_values(["date", "ticker"], ignore_index=True)


def equal_weight_member_returns(panel: pd.DataFrame) -> pd.Series:
    returns = panel[["date", "ticker", "adj_close", "is_sp500_member"]].copy()
    returns["return"] = returns.groupby("ticker", observed=True)["adj_close"].pct_change(
        fill_method=None
    )
    member = returns.loc[returns["is_sp500_member"]].copy()
    member.loc[(member["return"] <= -0.90) | (member["return"] >= 1.50), "return"] = np.nan
    counts = member.groupby("date", observed=True)["return"].count()
    ew = member.groupby("date", observed=True)["return"].mean()
    return ew.where(counts >= 20).dropna()


def benchmark_correlation(
    panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    ew = equal_weight_member_returns(panel)
    results: dict[str, Any] = {}
    for symbol in ["RSP", "SPY"]:
        fetched = yf_download(
            [symbol],
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            threads=False,
        ).get(symbol)
        if fetched is None or fetched.empty:
            results[symbol] = {"error": "benchmark unavailable"}
            continue
        benchmark = fetched["adj_close"].pct_change(fill_method=None).dropna()
        common = ew.index.intersection(benchmark.index)
        if len(common) < 100:
            results[symbol] = {"error": "insufficient overlapping days"}
            continue
        correlation = float(ew.reindex(common).corr(benchmark.reindex(common)))
        results[symbol] = {
            "daily_return_correlation": round(correlation, 6),
            "common_days": int(len(common)),
        }
    return results


def validate_panel(
    panel: pd.DataFrame,
    snapshots: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cleaning: dict[str, Any],
) -> dict[str, Any]:
    duplicate_keys = int(panel.duplicated(["date", "ticker"]).sum())
    invalid_price = int((panel["adj_close"].isna() | (panel["adj_close"] <= 0)).sum())
    negative_volume = int((panel["volume"].dropna() < 0).sum())
    future_dates = int((panel["date"] >= pd.Timestamp(end)).sum())

    daily_membership = snapshots.assign(n_constituents=snapshots["tickers"].map(len))[
        ["snapshot_date", "n_constituents"]
    ]
    member_prices = (
        panel.loc[panel["is_sp500_member"]]
        .groupby("date", observed=True)["ticker"]
        .nunique()
        .rename("member_prices")
        .reset_index()
    )
    calendar = pd.DataFrame({"date": panel["date"].drop_duplicates().sort_values()})
    calendar = pd.merge_asof(
        calendar,
        daily_membership.sort_values("snapshot_date"),
        left_on="date",
        right_on="snapshot_date",
        direction="backward",
    ).merge(member_prices, on="date", how="left")
    calendar["member_prices"] = calendar["member_prices"].fillna(0)
    calendar["coverage"] = calendar["member_prices"] / calendar["n_constituents"]
    recent_cutoff = max(start, end - pd.DateOffset(years=2))
    recent = calendar.loc[calendar["date"] >= recent_cutoff]

    split_rows = panel.loc[panel["stock_splits"] > 0].copy()
    if split_rows.empty:
        split_discontinuities = 0
    else:
        returns = panel[["date", "ticker", "adj_close"]].copy()
        returns["adj_return"] = returns.groupby("ticker", observed=True)["adj_close"].pct_change(
            fill_method=None
        )
        split_rows = split_rows.merge(
            returns[["date", "ticker", "adj_return"]],
            on=["date", "ticker"],
            how="left",
        )
        split_discontinuities = int((split_rows["adj_return"].abs() > 0.50).sum())

    benchmark = benchmark_correlation(panel, start, end)
    rsp_corr = benchmark.get("RSP", {}).get("daily_return_correlation")
    failures: list[str] = []
    if duplicate_keys:
        failures.append(f"{duplicate_keys} duplicate date/ticker keys")
    if invalid_price:
        failures.append(f"{invalid_price} invalid adjusted closes")
    if negative_volume:
        failures.append(f"{negative_volume} negative volumes")
    if future_dates:
        failures.append(f"{future_dates} rows at/after the exclusive end date")
    if rsp_corr is None or rsp_corr < 0.95:
        failures.append(f"EW constituent return correlation with RSP below 0.95 ({rsp_corr})")
    if float(recent["coverage"].median()) < 0.95:
        failures.append(
            "median member price coverage in the last two years is below 95% "
            f"({recent['coverage'].median():.2%})"
        )

    result = {
        "status": "pass" if not failures else "fail",
        "rows": int(len(panel)),
        "tickers_with_prices": int(panel["ticker"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "duplicate_date_ticker_keys": duplicate_keys,
        "invalid_adjusted_closes": invalid_price,
        "negative_volumes": negative_volume,
        "median_member_price_coverage": round(float(calendar["coverage"].median()), 6),
        "p05_member_price_coverage": round(float(calendar["coverage"].quantile(0.05)), 6),
        "recent_median_member_price_coverage": round(float(recent["coverage"].median()), 6),
        "split_dates_with_gt_50pct_adjusted_move": split_discontinuities,
        "benchmark_validation": benchmark,
        **cleaning,
        "failures": failures,
    }
    return result


def write_parquet(
    panel: pd.DataFrame,
    output: Path,
    source_metadata: dict[str, str],
    validation: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    additions = {
        "dataset": "S&P 500 historical-constituent market data",
        "schema_version": "1.0.0",
        "generated_at_utc": utc_now().isoformat(),
        "window_start": str(start.date()),
        "window_end_exclusive": str(end.date()),
        "price_source": "Yahoo Finance accessed with yfinance",
        "constituent_source": SOURCE_REPOSITORY,
        "constituent_source_metadata": json.dumps(source_metadata, sort_keys=True),
        "validation": json.dumps(validation, sort_keys=True),
        "notes": (
            "Long panel. Use is_sp500_member for historical-universe filtering. "
            "adj_close is split/dividend adjusted; dividends and stock_splits are raw actions."
        ),
    }
    metadata.update({key.encode(): value.encode() for key, value in additions.items()})
    table = table.replace_schema_metadata(metadata)
    pq.write_table(
        table,
        output,
        compression="zstd",
        compression_level=9,
        use_dictionary=["ticker", "yf_ticker"],
        write_statistics=True,
    )


def parse_args() -> argparse.Namespace:
    tomorrow = (utc_now() + timedelta(days=1)).date().isoformat()
    parser = argparse.ArgumentParser(
        description="Build survivorship-aware S&P 500 adjusted-close market data."
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=tomorrow, help="Exclusive end date.")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--refresh-constituents", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help=(
            "Skip constituents and Yahoo; add SEC industry and market cap to "
            "the existing market parquet."
        ),
    )
    parser.add_argument(
        "--no-sec-enrichment",
        action="store_true",
        help="Write only the 11-column market bootstrap even if fundamentals exist.",
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("EDGAR_IDENTITY", ""),
        help="SEC User-Agent identity; prefer EDGAR_IDENTITY.",
    )
    parser.add_argument(
        "--refresh-sic",
        action="store_true",
        help="Refresh cached SEC quarterly submission/SIC data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.enrich_only and args.no_sec_enrichment:
        raise ValueError("--enrich-only and --no-sec-enrichment cannot be combined")
    if args.enrich_only:
        from _sec_enrichment import DEFAULT_MAX_SHARE_AGE_DAYS, enrich_market_file

        log("SEC-only pass: no constituent or Yahoo requests will be made")
        enrich_market_file(
            market_path=OUTPUT_PATH,
            fundamentals_path=FUNDAMENTALS_PATH,
            output_path=OUTPUT_PATH,
            identity=args.identity,
            refresh_sic=args.refresh_sic,
            offline=args.offline,
            max_share_age_days=DEFAULT_MAX_SHARE_AGE_DAYS,
            validate_only=args.validate_only,
        )
        return 0
    enrich_after_build = (
        not args.no_sec_enrichment and FUNDAMENTALS_PATH.exists()
    )
    if enrich_after_build and not args.offline and not args.identity.strip():
        raise ValueError(
            "The final SEC enrichment requires EDGAR_IDENTITY with a name and "
            "contact email. Use --no-sec-enrichment only when intentionally "
            "building the 11-column bootstrap."
        )

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start >= end:
        raise ValueError("--start must be earlier than --end")

    log("1/5 Historical S&P 500 membership")
    constituent_path, source_metadata = download_constituents(
        refresh=args.refresh_constituents, offline=args.offline
    )
    snapshots = load_snapshots(constituent_path, start, end)
    universe = universe_union(snapshots)
    log(
        f"{len(snapshots):,} snapshots; {len(universe):,} unique historical tickers; "
        f"source SHA-256 {source_metadata['sha256'][:12]}",
        1,
    )

    if not args.validate_only:
        log("2/5 Yahoo Finance download/update")
        manifest = download_prices(
            universe,
            start,
            end,
            chunk_size=args.chunk_size,
            refresh=args.refresh_prices,
            retry_missing=args.retry_missing,
        )
    else:
        log("2/5 Using existing cache (--validate-only)")
        manifest = read_manifest()

    log("3/5 Assemble and clean long panel")
    panel, cleaning = assemble_long_panel(universe, manifest, start, end)
    panel = add_membership(panel, snapshots)
    has_member_observation = panel.groupby("ticker", observed=True)["is_sp500_member"].transform(
        "any"
    )
    no_overlap = sorted(panel.loc[~has_member_observation, "ticker"].unique())
    panel = panel.loc[has_member_observation].reset_index(drop=True)
    cleaning["tickers_without_member_date_overlap_dropped"] = no_overlap
    log(
        f"{len(panel):,} rows x {panel.shape[1]} columns; "
        f"{panel['ticker'].nunique():,} tickers",
        1,
    )

    log("4/5 Validate")
    validation = validate_panel(panel, snapshots, start, end, cleaning)
    log(json.dumps(validation, indent=2, sort_keys=True), 1)
    if validation["status"] != "pass":
        raise RuntimeError(
            "Market-data validation failed; parquet was not replaced. "
            + "; ".join(validation["failures"])
        )

    log("5/5 Write single validated parquet")
    build_path = (
        OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.building")
        if enrich_after_build
        else OUTPUT_PATH
    )
    if build_path != OUTPUT_PATH and build_path.exists():
        build_path.unlink()
    write_parquet(panel, build_path, source_metadata, validation, start, end)
    log(f"Wrote {build_path} ({build_path.stat().st_size / 1_000_000:.1f} MB)")

    if enrich_after_build:
        from _sec_enrichment import DEFAULT_MAX_SHARE_AGE_DAYS, enrich_market_file

        log("Fundamentals found: append point-in-time SEC industry and market cap")
        try:
            enrich_market_file(
                market_path=build_path,
                fundamentals_path=FUNDAMENTALS_PATH,
                output_path=build_path,
                identity=args.identity,
                refresh_sic=args.refresh_sic,
                offline=args.offline,
                max_share_age_days=DEFAULT_MAX_SHARE_AGE_DAYS,
                validate_only=False,
            )
            build_path.replace(OUTPUT_PATH)
            log(f"Published final enriched parquet at {OUTPUT_PATH}")
        finally:
            if build_path.exists():
                build_path.unlink()
    elif not FUNDAMENTALS_PATH.exists() and not args.no_sec_enrichment:
        log(
            "Wrote the 11-column bootstrap. Build sp500_fundamentals.parquet, "
            "then run this script with --enrich-only for the final 13 columns."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; cached ticker downloads are safe to resume.", file=sys.stderr)
        raise SystemExit(130) from None
