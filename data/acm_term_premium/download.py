from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

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

URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    date_candidates = [c for c in out.columns if str(c).lower() in {"date", "dates"}]
    date_col = date_candidates[0] if date_candidates else out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> int:
    output = DATA / "acm_term_premium.csv"
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0 quantfinlab-data-builder/1.0"})
    with urlopen(req, timeout=120) as response:
        frame = pd.read_excel(BytesIO(response.read()))
    out = normalize(frame)
    out.to_csv(output, index=False)
    print(f"wrote {output} rows={len(out):,} cols={len(out.columns):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
