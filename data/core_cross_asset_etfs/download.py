from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "IEFA",
    "EWJ",
    "EWU",
    "EWG",
    "EWC",
    "EWA",
    "EWH",
    "EWS",
    "EEM",
    "EWZ",
    "EWW",
    "EWT",
    "EWY",
    "EZA",
    "FXI",
    "EPI",
    "SHY",
    "IEF",
    "TLT",
    "TIP",
    "AGG",
    "BND",
    "LQD",
    "HYG",
    "EMB",
    "BKLN",
    "GLD",
    "SLV",
    "GDX",
    "DBC",
    "DBA",
    "DBB",
    "USO",
    "VNQ",
    "IYR",
    "UUP",
    "FXY",
    "FXE",
]
OUTPUT = DATA / "core_cross_asset_etfs.csv"
FIELDS = ("Close", "Volume", "Dividends", "Stock Splits")
OUTPUT_NAMES = ("close", "volume", "dividends", "stock_splits")


def extract(raw: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        if (ticker, field) in raw.columns:
            return pd.to_numeric(raw[(ticker, field)], errors="coerce")
        if (field, ticker) in raw.columns:
            return pd.to_numeric(raw[(field, ticker)], errors="coerce")
    elif field in raw.columns and len(TICKERS) == 1:
        return pd.to_numeric(raw[field], errors="coerce")
    return pd.Series(np.nan, index=raw.index)


def fetch(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start="1999-01-01",
        auto_adjust=True,
        actions=True,
        group_by="ticker",
        threads=len(tickers) > 1,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    return raw.loc[raw.index.notna()].sort_index()


def ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{ticker}__{name}": extract(raw, ticker, field)
            for field, name in zip(FIELDS, OUTPUT_NAMES, strict=True)
        },
        index=raw.index,
    )


def main() -> int:
    batch = fetch(TICKERS)
    pieces: list[pd.DataFrame] = []
    missing: list[str] = []
    for ticker in TICKERS:
        piece = ticker_frame(batch, ticker)
        if piece[f"{ticker}__close"].notna().any():
            pieces.append(piece)
        else:
            missing.append(ticker)

    for position, ticker in enumerate(missing, start=1):
        piece = ticker_frame(fetch([ticker]), ticker)
        if not piece[f"{ticker}__close"].notna().any():
            raise RuntimeError(f"Yahoo returned no usable history for {ticker}")
        pieces.append(piece)
        if position < len(missing):
            time.sleep(0.2)

    out = pd.concat(pieces, axis=1).sort_index()
    expected = [
        f"{ticker}__{name}" for ticker in TICKERS for name in OUTPUT_NAMES
    ]
    out = out.reindex(columns=expected)
    for ticker in TICKERS:
        if out[f"{ticker}__close"].notna().sum() < 252:
            raise RuntimeError(f"insufficient closing-price history for {ticker}")
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    out.to_csv(temporary, index_label="date")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT} rows={len(out):,} cols={len(out.columns):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
