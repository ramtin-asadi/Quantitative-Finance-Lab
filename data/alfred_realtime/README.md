# Targeted ALFRED Real-Time Panel

Builds a 52-series release/revision panel rather than duplicating the hundreds
of FRED-MD series:

- consumption, nominal/real disposable income, retail, vehicles;
- production, capacity, aggregate hours;
- housing and residential construction;
- durable/core-capital-goods orders, shipments, and nonresidential construction;
- manufacturing, wholesale, retail, and total business inventories;
- exports, imports, and trade balance;
- payrolls, unemployment, hours, temporary help, claims, insured unemployment,
  and JOLTS openings/hires;
- headline/core/food/food-at-home/gasoline CPI, headline/core/food PCE prices,
  final-demand PPI, import prices, and shelter;
- monthly fed funds, 2/5/10-year Treasury yields, and the broad dollar index.

```powershell
.\.venv\Scripts\python.exe data\alfred_realtime\download.py
```

Outputs:

- `data/alfred_realtime.parquet` with observation date, value, and exact
  `realtime_start`/`realtime_end` validity interval;
- `data/alfred_series_catalog.parquet` with category, title, source date range,
  and vintage coverage.

The script uses ALFRED's official **Observations by Real-Time Period** bulk
download form, so no API key is required. This is the same real-time data model
documented by the FRED API endpoints:

- https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html
- https://fred.stlouisfed.org/docs/api/fred/series_observations.html

Incremental update:

```powershell
.\.venv\Scripts\python.exe data\alfred_realtime\update.py
```

The updater scrapes each official vintage-date list, submits only dates newer
than the saved checkpoint, and merges returned validity intervals by
`(series_id, observation_date, realtime_start)`. It never requests the older
vintage dates again. Daily market/rate controls are intentionally kept in
`macro_high_frequency`, where their unrevised source history is much smaller.
