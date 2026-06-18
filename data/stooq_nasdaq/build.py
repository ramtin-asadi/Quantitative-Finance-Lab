from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = Path(__file__).resolve().parent / "raw"
FIELD_ORDER = ("close", "volume")


def ticker_from_path(path: Path, market: str) -> str:
    name = path.name.lower()
    if market == "us":
        return name.replace(".us.txt", "").upper()
    code = name.replace(".hk.txt", "")
    if code.isdigit() and len(code) <= 4:
        code = code.zfill(4)
    return f"{code.upper()}.HK"


def discover(market: str) -> list[Path]:
    suffix = "*.us.txt" if market == "us" else "*.hk.txt"
    files = sorted(RAW.rglob(suffix))
    if not files:
        raise FileNotFoundError(f"No Stooq {market} raw files found under {RAW}")
    return files


def read_one(path: Path, ticker: str) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except (EmptyDataError, ParserError) as exc:
        print(f"skip {path.name}: {exc}")
        return None
    frame.columns = [str(c).strip().strip("<>") for c in frame.columns]
    required = {"DATE", "CLOSE", "VOL"}
    if not required.issubset(frame.columns):
        print(f"skip {path.name}: missing {sorted(required - set(frame.columns))}")
        return None
    out = frame[["DATE", "CLOSE", "VOL"]].copy()
    out["DATE"] = pd.to_datetime(out["DATE"].astype(str), format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["DATE"]).drop_duplicates("DATE", keep="last").sort_values("DATE")
    if out.empty:
        return None
    out = out.set_index("DATE")
    out[f"{ticker}__close"] = pd.to_numeric(out["CLOSE"], errors="coerce")
    out[f"{ticker}__volume"] = pd.to_numeric(out["VOL"], errors="coerce")
    return out[[f"{ticker}__close", f"{ticker}__volume"]]


def build(market: str, output_name: str) -> int:
    files = discover(market)
    frames = []
    seen = set()
    for i, path in enumerate(files, 1):
        ticker = ticker_from_path(path, market)
        if ticker in seen:
            continue
        seen.add(ticker)
        frame = read_one(path, ticker)
        if frame is not None:
            frames.append(frame)
        if i % 500 == 0:
            print(f"processed {i:,}/{len(files):,} files")
    if not frames:
        raise RuntimeError(f"No readable Stooq {market} files under {RAW}")
    combined = pd.concat(frames, axis=1, join="outer").sort_index()
    ordered = []
    tickers = sorted({c.rsplit("__", 1)[0] for c in combined.columns})
    for ticker in tickers:
        for field in FIELD_ORDER:
            col = f"{ticker}__{field}"
            if col in combined.columns:
                ordered.append(col)
    combined = combined[ordered].replace([np.inf, -np.inf], np.nan)
    out = combined.reset_index().rename(columns={"DATE": "date", "index": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    output = DATA / output_name
    out.to_parquet(output, index=False, compression="zstd")
    print(f"wrote {output} rows={len(out):,} tickers={len(tickers):,} cols={len(out.columns):,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(build("us", "nasdaq_close_volume.parquet"))
