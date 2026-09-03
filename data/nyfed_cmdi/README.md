# New York Fed Corporate Bond Market Distress Index

Downloads the permanent official CMDI workbook and writes
`data/nyfed_cmdi.parquet`.

The output contains the weekly Friday date and the market-wide, investment-grade,
and high-yield CMDI series. It begins in 2005. The three source indexes are kept
as published; the builder does not smooth, standardize, lag, or transform them.

Official source:

https://www.newyorkfed.org/research/policy/cmdi

Build or update:

```powershell
.\.venv\Scripts\python.exe data\nyfed_cmdi\download.py
.\.venv\Scripts\python.exe data\nyfed_cmdi\update.py
```

The workbook is cached. Later runs use HTTP validators and a SHA-256 comparison,
then rebuild the Parquet only when the NY Fed file changes.
