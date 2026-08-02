from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def read_url_csv(url: str, **kwargs) -> pd.DataFrame:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        response = session.get(url, timeout=60)
        response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), **kwargs)


def clean_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI"


def main() -> int:
    output = DATA / "nfci.csv"
    try:
        frame = clean_date_frame(read_url_csv(URL, na_values=[".", ""]))
    except Exception as exc:
        raise RuntimeError(f"FRED NFCI download failed: {URL}") from exc
    keep = ["date", "NFCI"] if "NFCI" in frame.columns else list(frame.columns[:2])
    frame = frame[keep].rename(columns={keep[1]: "NFCI"})
    frame["NFCI"] = pd.to_numeric(frame["NFCI"], errors="coerce")
    frame.to_csv(output, index=False)
    print(f"wrote {output} rows={len(frame):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
