# Atlanta Fed Market Probability Tracker

Downloads the official historical data and source-code archive for the Market
Probability Tracker, then writes `data/atlanta_mpt.parquet`.

```powershell
.\.venv\Scripts\python.exe data\atlanta_mpt\download.py
```

The data contain each observation date, three-month SOFR reference window,
then-current federal-funds target range, distribution statistic/probability
field, and published value. Field names and values are preserved exactly; the
builder does not map SOFR distributions into an FOMC path or rescale basis
points/probabilities.

Official page:

https://www.atlantafed.org/research-and-data/data/market-probability-tracker

```powershell
.\.venv\Scripts\python.exe data\atlanta_mpt\update.py
```

The Atlanta Fed replaces the full historical workbook. The updater sends HTTP
validators and compares returned content by SHA-256, rebuilding the Parquet
only when the workbook changes.
