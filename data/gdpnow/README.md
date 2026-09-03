# GDPNow

Downloads the Atlanta Fed's official model/history workbook and release-date
workbook.

```powershell
.\.venv\Scripts\python.exe data\gdpnow\download.py
```

Outputs:

- `data/gdpnow_forecasts.parquet`: headline and component forecast paths,
  including the 2011Q3–2014Q1 pre-live deep archive, live archive from 2014Q2,
  and the current quarter;
- `data/gdpnow_contributions.parquet`: component contributions to GDP growth;
- `data/gdpnow_track_record.parquet`: final pre-advance forecast versus the BEA
  advance estimate;
- `data/gdpnow_release_dates.parquet`: the current posted/internal source-data
  update schedule.

Official source:

https://www.atlantafed.org/research-and-data/data/gdpnow

The approximately 10–11 MB workbook contains many internal model and source
tabs. The builder keeps only the published forecast histories, component
contributions, track record, and release timing. It does not redistribute or
turn internal coefficients/source levels into notebook features.

```powershell
.\.venv\Scripts\python.exe data\gdpnow\update.py
```

Atlanta replaces the same workbooks in place. The updater sends HTTP validators
and compares returned content by SHA-256; only a changed workbook triggers an
atomic rebuild from cache.
