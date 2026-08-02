# Macro Factors

Source: FRED, Statistics Canada, and Bank of Canada APIs.

Script: `download.py`

Final output files:

- `data/us_macro_factors.csv`
- `data/canada_macro_factors.csv`
- `data/macro_factor_summary.csv`
- `data/macro_download_issues.csv`

Source and download links:

- https://fred.stlouisfed.org/
- https://www150.statcan.gc.ca/n1/en/type/data
- https://www.bankofcanada.ca/valet/docs

Notes:

NFCI is intentionally handled by `chicago_fed_nfci` instead of this macro
bundle.

The downloaded histories are current/latest-vintage observations. A publication
lag can repair release timing, but it cannot undo later revisions. Therefore
these files must not be described as point-in-time vintage data. Historical
backtests should either use a genuine vintage source or keep these series out of
predictive features; they remain suitable for data monitoring and clearly
labelled ex-post interpretation.

Rebuild from the repository root:

```bash
python data/macro_factors/download.py
```

Incremental refresh (recommended for routine updates):

```bash
python data/macro_factors/update.py
```

The incremental updater requests only a six-month overlap plus new dates,
preserves the older history, incorporates recent revisions, rejects incomplete
new monthly rows, and atomically replaces an output only after validation. Use
`--country us` when only the U.S. Project 22 inputs need refreshing.

Historical holes are not redownloaded during a normal update. If a quality
audit finds a sparse FRED column, repair only that named series once:

```bash
python data/macro_factors/update.py --country us --backfill-series cpi_all_items
```

Raw/manual files, when required, belong in this folder's `raw/` directory and
should not be committed.
