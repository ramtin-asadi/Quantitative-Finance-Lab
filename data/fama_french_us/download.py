from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def download_zip_csv(url: str) -> str:
    req = Request(url, headers={"User-Agent": "quantfinlab-data-builder/1.0"})
    with urlopen(req, timeout=120) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name).decode("utf-8-sig", errors="replace")


def is_date_token(value: object) -> bool:
    text = str(value).strip()
    return text.isdigit() and len(text) in {6, 8}


def parse_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 6:
        return pd.to_datetime(text + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    return pd.to_datetime(text, format="%Y%m%d")


def parse_first_table(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:-1]):
        next_token = lines[i + 1].split(",", 1)[0].strip()
        if is_date_token(next_token):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No Fama-French monthly table found")
    rows = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        token = line.split(",", 1)[0].strip()
        if not is_date_token(token):
            break
        rows.append(line)
    frame = pd.read_csv(StringIO("\n".join(rows)))
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = frame["date"].map(parse_date)
    for col in frame.columns:
        if col != "date":
            frame[col] = pd.to_numeric(frame[col], errors="coerce") / 100.0
    return frame.replace([-0.9999, -9.99, -99.99], np.nan).sort_values("date")


def load_dataset(url: str) -> pd.DataFrame:
    try:
        return parse_first_table(download_zip_csv(url))
    except Exception as exc:
        raise RuntimeError(f"Kenneth French download failed: {url}") from exc

def main() -> int:
    factors = load_dataset(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    )
    momentum = load_dataset(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
    )
    momentum = momentum.rename(columns={momentum.columns[1]: "MOM"})
    industries = load_dataset(
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_CSV.zip"
    )
    factors.to_csv(DATA / "fama_french_us_5_factors.csv", index=False)
    momentum.to_csv(DATA / "fama_french_us_momentum.csv", index=False)
    industries.to_csv(DATA / "fama_french_us_12_industries.csv", index=False)
    print(f"wrote {DATA / 'fama_french_us_5_factors.csv'} rows={len(factors):,}")
    print(f"wrote {DATA / 'fama_french_us_momentum.csv'} rows={len(momentum):,}")
    print(f"wrote {DATA / 'fama_french_us_12_industries.csv'} rows={len(industries):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
