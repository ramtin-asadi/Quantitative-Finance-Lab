# Small High-Frequency Macro Panel

Builds `data/macro_high_frequency.parquet` from keyless FRED graph CSVs.

The panel stays intentionally small:

- daily Brent, with WTI as a challenger;
- weekly U.S. regular retail gasoline;
- SOFR, effective fed funds, and both target-range bounds;
- 2/5/10-year nominal Treasury yields;
- 5/10-year real yields and breakevens;
- broad dollar index and VIX.

These are source-native levels. The script does not forward-fill holidays,
resample, calculate returns, construct a policy midpoint, or create inflation
features.

```powershell
.\.venv\Scripts\python.exe data\macro_high_frequency\download.py
```

Cleveland Fed methodology confirms that the high-frequency inputs needed for
inflation nowcasting are daily Brent and weekly retail gasoline; the other rate
series cover information arriving between monthly macro releases:

https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting

Weekly claims are already preserved with exact vintage chronology in
`alfred_realtime`, so they are not duplicated here.

```powershell
.\.venv\Scripts\python.exe data\macro_high_frequency\update.py
```

The updater requests only a 90-day overlap per series, merges new/revised
source dates into each cache, and rebuilds the compact Parquet locally.
