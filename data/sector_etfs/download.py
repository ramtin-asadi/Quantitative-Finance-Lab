from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TICKERS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'SMH', 'IYZ']
OUTPUT = DATA / "sector_etfs.csv"


def extract(raw: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        if (ticker, field) in raw.columns:
            return pd.to_numeric(raw[(ticker, field)], errors="coerce")
        if (field, ticker) in raw.columns:
            return pd.to_numeric(raw[(field, ticker)], errors="coerce")
    elif field in raw.columns and len(TICKERS) == 1:
        return pd.to_numeric(raw[field], errors="coerce")
    return pd.Series(np.nan, index=raw.index)


def main() -> int:
    raw = yf.download(
        TICKERS,
        start="1999-01-01",
        auto_adjust=True,
        actions=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no rows")
    raw.index = pd.to_datetime(raw.index, errors="coerce").tz_localize(None)
    raw = raw[raw.index.notna()].sort_index()
    out = pd.DataFrame({"date": raw.index})
    for ticker in TICKERS:
        out[f"{ticker}__close"] = extract(raw, ticker, "Close").to_numpy()
        out[f"{ticker}__volume"] = extract(raw, ticker, "Volume").to_numpy()
        out[f"{ticker}__dividends"] = extract(raw, ticker, "Dividends").to_numpy()
        out[f"{ticker}__stock_splits"] = extract(raw, ticker, "Stock Splits").to_numpy()
    out.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} rows={len(out):,} cols={len(out.columns):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
