"""Source adapters that preserve macro observation and publication dates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_alfred(path: str | Path, *, series=None, start=None) -> pd.DataFrame:
    """ALFRED observations with inclusive validity intervals and exact release days."""
    filters = []
    if series is not None:
        filters.append(("series_id", "in", list(series)))
    if start is not None:
        filters.append(("observation_date", ">=", pd.Timestamp(start)))
    data = pd.read_parquet(path, filters=filters or None).rename(
        columns={"realtime_start": "available_at", "realtime_end": "valid_to"})
    for name in ["observation_date", "available_at", "valid_to"]:
        data[name] = pd.to_datetime(data[name])
    data["timing"] = "source release day"
    return data.sort_values(["series_id", "observation_date", "available_at"]).reset_index(drop=True)


def read_fred_vintages(path: str | Path, *, frequency="MD", series=None,
                        start=None, as_of=None) -> pd.DataFrame:
    """FRED-MD/QD snapshots, conservatively available from the following month.

    Monthly vintage labels do not establish exact publication dates. When
    ``as_of`` is supplied, select only the latest conservatively usable snapshot.
    """
    import duckdb

    clauses, params = ["panel = ?"], [frequency]
    if as_of is not None:
        clauses.append("vintage_date = (SELECT max(vintage_date) FROM read_parquet(?) WHERE panel=? AND vintage_date < ?)")
        params.extend([str(path), frequency, pd.Timestamp(as_of).to_period("M").start_time])
    if series is not None:
        clauses.append("series_id IN (" + ",".join("?" for _ in series) + ")")
        params.extend(series)
    if start is not None:
        clauses.append("observation_date >= ?")
        params.append(pd.Timestamp(start))
    with duckdb.connect() as con:
        data = con.execute("SELECT * FROM read_parquet(?) WHERE " + " AND ".join(clauses),
                            [str(path), *params]).fetchdf()
    data["available_at"] = pd.to_datetime(data["vintage_date"]) + pd.offsets.MonthBegin(1)
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    data["timing"] = "following-month availability bound"
    return data


def read_philly_vintages(path: str | Path, *, variables=None) -> pd.DataFrame:
    """Philadelphia vintage levels; vintage labels remain distinct from release chronology."""
    filters = [("variable", "in", list(variables))] if variables is not None else None
    data = pd.read_parquet(path, filters=filters)
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    data["vintage_date"] = pd.to_datetime(data["vintage_date"])
    return data


def read_philly_releases(path: str | Path) -> pd.DataFrame:
    """Source first/second/third-release values; attach exact calendars separately."""
    return pd.read_parquet(path)


def statscan_catalog(path: str | Path) -> pd.DataFrame:
    """Distinct source series for selecting national totals and stable economic definitions."""
    import duckdb

    with duckdb.connect() as con:
        return con.execute("SELECT DISTINCT table_id, series_title, unit, scalar_factor FROM read_parquet(?) ORDER BY table_id, series_title", [str(path)]).fetchdf()


def read_statscan_vintages(path: str | Path, series: dict, *, start=None) -> pd.DataFrame:
    """Read explicitly selected StatsCan series into a common vintage schema.

    ``series`` maps an economic name to {table_id, series_title}. Vectors change
    between vintages and therefore do not identify a stable economic series.
    Forward snapshots are usable only from max(release_date, snapshot_date).
    Unrevised CPI rows with no publication date retain missing availability.
    """
    import duckdb

    parts = []
    with duckdb.connect() as con:
        for name, selection in series.items():
            clauses, params = [], [str(path)]
            for key in ["table_id", "series_title"]:
                clauses.append(f"{key} = ?")
                params.append(selection[key])
            if start is not None:
                clauses.append("reference_date >= ?")
                params.append(pd.Timestamp(start))
            data = con.execute("SELECT reference_date, release_date, snapshot_date, value, unit, scalar_factor, source_kind, table_id FROM read_parquet(?) WHERE "
                                + " AND ".join(clauses), params).fetchdf()
            if data.empty:
                raise ValueError(f"No source records match {name}: {selection}.")
            data = data.rename(columns={"reference_date": "observation_date", "release_date": "available_at"})
            data["series_id"] = name
            forward = data["source_kind"].eq("forward_snapshot")
            data.loc[forward, "available_at"] = data.loc[forward, ["available_at", "snapshot_date"]].max(axis=1)
            data["timing"] = np.where(forward, "forward snapshot", "source release day")
            parts.append(data)
    data = pd.concat(parts, ignore_index=True)
    keys = ["series_id", "observation_date", "available_at"]
    conflicts = data.groupby(keys, dropna=False)["value"].nunique().gt(1)
    if conflicts.any():
        raise ValueError("Selected StatsCan dimensions contain conflicting observations.")
    return data.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
