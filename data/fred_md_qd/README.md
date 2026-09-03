# FRED-MD and FRED-QD Vintages

Builds `data/fred_md_qd_vintages.parquet` from the official St. Louis Fed
archives and current monthly snapshots.

Included source history:

- FRED-MD 1999-08 through 2014-12 archive;
- FRED-MD 2015-01 through 2025-12 archive;
- FRED-QD 2018-05 through 2025-12 archive;
- every individually listed 2026 MD and QD snapshot, with later snapshots
  discovered automatically.

Official page:

https://www.stlouisfed.org/research/economists/mccracken/fred-databases

```powershell
.\.venv\Scripts\python.exe data\fred_md_qd\download.py
```

The long Parquet keeps panel, snapshot vintage, observation date, original
series ID, raw value, published transformation code, FRED-QD factor-group flag,
and exact source file. It does not apply transformations, fill ragged edges, or
extract factors; those belong in the notebook.

Update only newly listed snapshots:

```powershell
.\.venv\Scripts\python.exe data\fred_md_qd\update.py
```

Historical ZIPs remain cached and are not requested during an ordinary update.
When a new standalone snapshot appears, the updater copies existing Parquet
row groups and appends only that panel/vintage partition. Published snapshots
are treated as immutable vintages.
