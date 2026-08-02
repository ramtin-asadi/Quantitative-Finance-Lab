# Chicago Fed NFCI

Builds `data/nfci.csv`, the weekly Chicago Fed National Financial Conditions Index from FRED.

## Source

- FRED NFCI page: https://fred.stlouisfed.org/series/NFCI
- No-key FRED CSV endpoint used by the script: https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI
- Underlying source: Federal Reserve Bank of Chicago.

## What To Put In `raw/`

Nothing. `download.py` downloads the FRED `NFCI` series directly and does not read manual raw files.

## Rebuild

```bash
python data/chicago_fed_nfci/download.py
```

The output has weekly Friday observations with columns `date` and `NFCI`.

The FRED graph endpoint returns the current/latest-vintage history, not the
value as originally published on each historical date. Lagging this series can
address publication timing but not later revisions. Do not use it as a
point-in-time predictive backtest feature unless a genuine vintage history is
supplied; it is still useful for current monitoring and labelled ex-post
interpretation.
