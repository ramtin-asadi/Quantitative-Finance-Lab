from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = Path(__file__).resolve().parent / "raw"
DATE_COLS = {"quote_readtime", "quote_date", "expire_date", "expiry_date"}
TEXT_COLS = {
    "instrument_name",
    "base_currency",
    "underlying_index",
    "expiry_time",
    "option_right",
    "c_size",
    "p_size",
    "source_file",
    "source_month",
}


def clean_columns(columns) -> list[str]:
    return [str(c).strip().strip("[]").strip().lower().replace(" ", "_") for c in columns]


def source_month(path: Path) -> str:
    for token in path.stem.replace("-", "_").split("_"):
        if len(token) == 6 and token.isdigit():
            return token
    return path.stem.split("_")[-1]


def normalize(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = frame.copy()
    out.columns = clean_columns(out.columns)
    for col in DATE_COLS & set(out.columns):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in out.columns:
        if col in DATE_COLS or col in TEXT_COLS:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["source_file"] = path.name
    out["source_month"] = source_month(path)
    return out


def build(output_name: str) -> int:
    files = sorted([p for p in RAW.rglob("*") if p.suffix.lower() in {".txt", ".csv"}])
    if not files:
        raise FileNotFoundError(
            f"No OptionsDX .txt/.csv files found under {RAW}. Put purchased monthly files there first."
        )
    output = DATA / output_name
    if output.exists():
        output.unlink()
    writer = None
    total = 0
    try:
        for path in files:
            raw = pd.read_csv(path, skipinitialspace=True, low_memory=False)
            frame = normalize(raw, path)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(frame)
            print(f"{path.name}: {len(frame):,} rows")
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {output} files={len(files):,} rows={total:,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(build("spx_options_chain.parquet"))
