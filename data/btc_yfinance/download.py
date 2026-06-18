from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        wanted = {"open", "high", "low", "close", "adj close", "volume", "dividends", "stock splits"}
        out.columns = [
            next((str(part) for part in col if str(part).strip().lower() in wanted), str(col[-1]))
            for col in out.columns
        ]
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    out = out.reset_index()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "datetime" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"datetime": "date"})
    if "index" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    preferred = ["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
    for col in preferred:
        if col not in out.columns and col != "date":
            out[col] = pd.NA
    return out[preferred].dropna(subset=["date"]).sort_values("date")


def download_one(ticker: str, output_name: str) -> None:
    raw = yf.download(
        ticker,
        start="1990-01-01",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    out = normalize_columns(raw)
    output = DATA / output_name
    out.to_csv(output, index=False)
    print(f"wrote {output} rows={len(out):,}")

def main() -> int:
    download_one("BTC-USD", "btc_usd_ohlcv.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
