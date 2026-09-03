# Philadelphia Fed Real-Time Ground Truth

Downloads selected complete RTDSM vintage matrices and the matching
first/second/third/latest release files.

The selected variables cover real GDP, consumption, nonresidential and
residential investment, exports/imports, monthly real PCE, CPI/core CPI, core
PCE prices, payroll changes, aggregate hours, total/manufacturing industrial
production, and housing starts.

```powershell
.\.venv\Scripts\python.exe data\philly_realtime\download.py
```

Outputs:

- `data/philly_realtime_vintages.parquet`: the level of each observation in
  every monthly information vintage;
- `data/philly_first_second_third.parquet`: official first, second, third, and
  most-recent releases where Philadelphia publishes them.

Official sources:

- https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-for-macroeconomists
- https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/first-second-third

The scripts preserve published levels/growth values and timing. They do not
calculate revision errors, targets, surprises, transformations, or nowcast
features.

`vintage_date` is the date implied by Philadelphia's source column label, not
an exact release timestamp. The provider can publish a late-month revision in
a column named for the following monthly/quarterly vintage (for example, the
2026-08-27 BEA revision appears in `26M9`/`26Q3`). The builder preserves that
label unchanged, which is conservative for point-in-time use. Use ALFRED and
the relevant release calendar when exact day-level availability is required;
use the Philadelphia first/second/third table as the revision target.

```powershell
.\.venv\Scripts\python.exe data\philly_realtime\update.py
```

Philadelphia replaces each official workbook in place. The updater sends HTTP
validators and compares returned files by SHA-256 when the provider still
returns a full response. It rebuilds the two Parquets only when source content
changed.
