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

URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv"


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    first_cell = str(out.iloc[0, 0]).strip().lower() if not out.empty else ""
    if first_cell.startswith("interest rate"):
        out = out.iloc[2:].copy()
        out.columns = frame.iloc[1].tolist()
    elif first_cell == "date":
        out = out.iloc[1:].copy()
        out.columns = frame.iloc[0].tolist()
    out.columns = [str(c).strip() for c in out.columns]
    date_col = "Date" if "Date" in out.columns else out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col].replace("-", pd.NA), errors="coerce")
    return out


def main() -> int:
    output = DATA / "japan_mof_yields.csv"
    try:
        raw = read_url_csv(URL, header=None, na_values=["", "-"])
        frame = normalize(raw)
    except Exception as exc:
        raise RuntimeError(f"MOF JGB historical CSV download failed: {URL}") from exc
    frame.to_csv(output, index=False)
    print(f"wrote {output} rows={len(frame):,} cols={len(frame.columns):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
