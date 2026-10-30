# SOFR Index

Builds `data/sofr.csv` from the Federal Reserve Bank of New York's published
SOFR Index. The file contains `date` and `sofr_index`, the columns used by
project 9 to calculate monthly cash returns from ratios of index levels.

The download starts on March 2, 2020, the first date used in the notebook,
and includes all subsequently available observations. No API key or manual
raw files are needed.

## Source

- [New York Fed SOFR Averages and Index](https://www.newyorkfed.org/markets/reference-rates/sofr-averages-and-index)
- [Latest SOFR Index observation from the New York Fed API](https://markets.newyorkfed.org/api/rates/secured/sofrai/last/1.json)

The `SOFR` series in `data/macro_high_frequency.parquet` is the overnight rate
in percent. It is a different series from the compounded SOFR Index used here.

## Rebuild

From the repository root:

```bash
python data/sofr/download.py
```

If a system proxy blocks the request, a PowerShell session can bypass the
proxy for the New York Fed API:

```powershell
$env:NO_PROXY = "markets.newyorkfed.org"
.\.venv\Scripts\python.exe data\sofr\download.py
```

The generated CSV is ignored by Git, like the other final data files. Running
the downloader again refreshes it with the observations currently published by
the New York Fed.
