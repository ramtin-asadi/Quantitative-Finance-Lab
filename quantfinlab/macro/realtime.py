"""Release chronology and point-in-time macro information sets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vintage_asof(data: pd.DataFrame, as_of, *, series=None, wide=False) -> pd.DataFrame:
    """Latest valid values known at a timestamp, retaining observation-date identity.

    Required columns: series_id, observation_date, available_at, value. An optional
    valid_to column is inclusive (ALFRED convention). Acquisition snapshots must
    already have availability no earlier than their acquisition timestamp.
    """
    date = pd.Timestamp(as_of)
    known = data[data["available_at"].le(date)]
    if "valid_to" in known:
        known = known[known["valid_to"].isna() | known["valid_to"].ge(date)]
    if series is not None:
        known = known[known["series_id"].isin(series)]
    known = known.sort_values("available_at").drop_duplicates(["series_id", "observation_date"], keep="last")
    known = known.sort_values(["series_id", "observation_date"])
    if wide:
        return known.pivot(index="observation_date", columns="series_id", values="value").sort_index()
    return known.reset_index(drop=True)


def releases_between(data: pd.DataFrame, start, end, *, series=None) -> pd.DataFrame:
    """New observations and revisions published in (start, end]."""
    rows = data[data["available_at"].gt(pd.Timestamp(start)) & data["available_at"].le(pd.Timestamp(end))]
    if series is not None:
        rows = rows[rows["series_id"].isin(series)]
    return rows.sort_values(["available_at", "series_id", "observation_date"]).copy()


def first_releases(data: pd.DataFrame, *, max_delay=None) -> pd.DataFrame:
    """First observed source vintage, optionally excluding old archive backfill rows."""
    first = data.dropna(subset=["value", "available_at"]).sort_values("available_at").drop_duplicates(["series_id", "observation_date"])
    if max_delay is not None:
        first = first[(first["available_at"] - first["observation_date"]).dt.days.le(max_delay)]
    return first.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def release_growth(data: pd.DataFrame, *, periods=1, frequency="M", scale=1200,
                    method="log", max_delay=None, annualization=1) -> pd.DataFrame:
    """Growth using current and lagged levels from the same release vintage.

    This differs from differencing a string of first-release levels: earlier
    observations may have been revised by the time the new observation appears.
    """
    rows = []
    first = first_releases(data, max_delay=max_delay)
    by_series = dict(tuple(data.groupby("series_id", sort=False)))
    for row in first.itertuples():
        known = vintage_asof(by_series[row.series_id], row.available_at)
        period = row.observation_date.to_period(frequency)
        values = known.set_index(known["observation_date"].dt.to_period(frequency))["value"]
        previous = values.get(period - periods, np.nan)
        if method == "level":
            value = row.value
        elif method == "difference":
            value = scale * (row.value - previous)
        elif method == "percent":
            value = scale * ((row.value / previous) ** annualization - 1) if previous != 0 else np.nan
        elif method == "log":
            value = scale * np.log(row.value / previous) if row.value > 0 and previous > 0 else np.nan
        else:
            raise ValueError("Unknown growth method.")
        rows.append({"series_id": row.series_id, "observation_date": row.observation_date,
                     "release_date": row.available_at, "first": value})
    return pd.DataFrame(rows)


def release_history(data: pd.DataFrame) -> pd.DataFrame:
    """First, second, third and latest known values without discarding revision dates."""
    source = data.sort_values("available_at").copy()
    source["release_number"] = source.groupby(["series_id", "observation_date"]).cumcount() + 1
    rows = []
    for (series, date), group in source.groupby(["series_id", "observation_date"]):
        record = {"series_id": series, "observation_date": date,
                  "release_date": group["available_at"].iloc[0], "latest": group["value"].iloc[-1]}
        for number, name in enumerate(["first", "second", "third"]):
            record[name] = group["value"].iloc[number] if len(group) > number else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def monthly_snapshot(data: pd.DataFrame, as_of, *, frequencies=None, series=None,
                       start=None) -> pd.DataFrame:
    """Monthly information matrix from released monthly, weekly, daily and survey data.

    Daily/weekly observations are averaged within month after the availability
    filter. Quarterly survey balances are carried for two additional months;
    their publication dates still control whether the entire observation exists.
    """
    frequencies = frequencies or {}
    known = vintage_asof(data, as_of, series=series)
    values = {}
    for name, source in known.groupby("series_id", sort=False):
        x = source.set_index("observation_date")["value"].sort_index()
        x = x.groupby(x.index.to_period("M")).mean()
        x.index = x.index.to_timestamp()
        if frequencies.get(name) == "Q":
            end = min(pd.Timestamp(as_of).to_period("M").to_timestamp(), x.index.max() + pd.offsets.MonthBegin(2))
            x = x.reindex(pd.date_range(x.index.min(), end, freq="MS")).ffill(limit=2)
        values[name] = x
    result = pd.DataFrame(values).sort_index().asfreq("MS")
    if series is not None:
        result = result.reindex(columns=list(series))
    return result.loc[pd.Timestamp(start):] if start is not None else result
