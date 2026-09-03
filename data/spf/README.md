# Survey of Professional Forecasters

Downloads the Philadelphia Fed's complete mean and median level/growth
workbooks and exact historical survey deadline/publication dates.

```powershell
.\.venv\Scripts\python.exe data\spf\download.py
```

Outputs:

- `data/spf_forecasts.parquet`: every published mean/median forecast sheet in
  tidy long form, including GDP/components, unemployment, payrolls, industrial
  production, CPI/core CPI, PCE/core PCE, and Treasury rates;
- `data/spf_release_dates.parquet`: true deadline and news-release dates from
  1990Q2 onward.

Mean/median histories begin in 1968 where the variable was part of the survey.
The forecast output keeps raw horizon names because their exact interpretation
depends on variable and survey-era conventions documented by the Philadelphia
Fed. It does not convert levels into growth, splice variables, or calculate
forecast errors.

Official sources:

- https://www.philadelphiafed.org/surveys-and-data/data-files
- https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/mean-forecasts
- https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/median-forecasts

```powershell
.\.venv\Scripts\python.exe data\spf\update.py
```

The provider replaces full-history workbooks. The updater sends HTTP validators
and also compares SHA-256 hashes when the server returns a full response;
unchanged content does not trigger a rebuild.
