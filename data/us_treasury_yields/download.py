from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def read_url_csv(url: str, **kwargs) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "quantfinlab-data-builder/1.0"})
    with urlopen(req, timeout=120) as response:
        return pd.read_csv(response, **kwargs)


def clean_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out

SERIES = {
    "DGS1MO": "1 mo",
    "DGS2MO": "2 mo",
    "DGS3MO": "3 mo",
    "DGS4MO": "4 mo",
    "DGS6MO": "6 mo",
    "DGS1": "1 yr",
    "DGS2": "2 yr",
    "DGS3": "3 yr",
    "DGS5": "5 yr",
    "DGS7": "7 yr",
    "DGS10": "10 yr",
    "DGS20": "20 yr",
    "DGS30": "30 yr",
}


def main() -> int:
    output = DATA / "us_treasury_yields.csv"
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(SERIES)
    try:
        frame = read_url_csv(url, na_values=[".", ""])
        frame = clean_date_frame(frame).rename(columns=SERIES)
    except Exception as exc:
        raise RuntimeError(f"FRED Treasury download failed: {url}") from exc
    keep = ["date"] + [v for v in SERIES.values() if v in frame.columns]
    frame = frame[keep]
    for col in keep[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.to_csv(output, index=False)
    print(f"wrote {output} rows={len(frame):,} cols={len(frame.columns):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
