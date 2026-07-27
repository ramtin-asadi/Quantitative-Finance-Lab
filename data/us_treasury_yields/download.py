from __future__ import annotations

from http.client import RemoteDisconnected
from pathlib import Path
from datetime import date
from time import sleep
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data"


def read_url_csv(url: str, attempts: int = 4, **kwargs) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "quantfinlab-data-builder/1.0"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=120) as response:
                return pd.read_csv(response, **kwargs)
        except (RemoteDisconnected, TimeoutError, URLError):
            if attempt == attempts - 1:
                raise
            sleep(2 ** attempt)
    raise RuntimeError("download attempts exhausted")


def clean_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).strip() for c in out.columns]
    date_col = out.columns[0]
    out = out.rename(columns={date_col: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out

treasury_columns = {
    "Date": "date",
    "1 Mo": "1 mo",
    "2 Mo": "2 mo",
    "3 Mo": "3 mo",
    "4 Mo": "4 mo",
    "6 Mo": "6 mo",
    "1 Yr": "1 yr",
    "2 Yr": "2 yr",
    "3 Yr": "3 yr",
    "5 Yr": "5 yr",
    "7 Yr": "7 yr",
    "10 Yr": "10 yr",
    "20 Yr": "20 yr",
    "30 Yr": "30 yr",
}


def download_year(year: int) -> pd.DataFrame:
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all"
        f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
    )
    frame = read_url_csv(url, na_values=["N/A", ""])
    frame = frame.rename(columns=treasury_columns)
    frame = clean_date_frame(frame)
    columns = ["date"] + [column for column in treasury_columns.values() if column != "date"]
    return frame.reindex(columns=columns)


def main() -> int:
    output = data_dir / "us_treasury_yields.csv"
    current_year = date.today().year
    current = download_year(current_year)
    if output.exists():
        history = clean_date_frame(pd.read_csv(output))
        history = history[history["date"].dt.year < current_year]
    else:
        history = pd.concat(
            [download_year(year) for year in range(1990, current_year)],
            ignore_index=True,
        )
    frame = pd.concat([history, current], ignore_index=True)
    frame = clean_date_frame(frame)
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(output, index=False)
    latest = frame["3 mo"].last_valid_index()
    latest_date = frame.loc[latest, "date"].date()
    print(f"wrote {output} rows={len(frame):,} cols={len(frame.columns):,} latest_3m={latest_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
