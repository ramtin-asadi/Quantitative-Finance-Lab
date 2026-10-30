from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "sofr.csv"
URL = "https://markets.newyorkfed.org/api/rates/secured/sofrai/search.json"
START_DATE = "2020-03-02"


def main() -> None:
    response = requests.get(
        URL,
        params={"startDate": START_DATE, "endDate": date.today().isoformat()},
        timeout=120,
    )
    response.raise_for_status()
    records = response.json()["refRates"]
    if not records:
        raise ValueError("The NY Fed returned no SOFR index observations.")
    frame = pd.DataFrame(records)[["effectiveDate", "index"]].rename(
        columns={"effectiveDate": "date", "index": "sofr_index"}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["sofr_index"] = pd.to_numeric(frame["sofr_index"], errors="raise")
    frame = frame.sort_values("date", ignore_index=True)
    if frame["date"].isna().any() or frame["date"].duplicated().any() or not np.isfinite(frame["sofr_index"]).all():
        raise ValueError("The NY Fed SOFR index response is empty, duplicated, or incomplete.")
    if (frame["sofr_index"] <= 0).any():
        raise ValueError("The NY Fed SOFR index must be positive.")

    frame.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    print(f"wrote {OUTPUT} rows={len(frame):,} dates={frame['date'].min().date()} to {frame['date'].max().date()}")


if __name__ == "__main__":
    main()
