# Core Cross Asset ETFs

Source: Yahoo Finance ETF data accessed through yfinance.

Scripts:

- `download.py` performs the initial full build.
- `update.py` performs subsequent incremental updates.

Final output file:

- `data/core_cross_asset_etfs.csv`

Source and download links:

- https://ranaroussi.github.io/yfinance/
- https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html

The output uses
`date,TICKER__close,TICKER__volume,TICKER__dividends,TICKER__stock_splits`
format.

Rebuild from the repository root:

```bash
python data/core_cross_asset_etfs/download.py
```

Incremental update after the initial build:

```bash
python data/core_cross_asset_etfs/update.py
```

The updater downloads a short recent overlap, retries failed tickers
individually, preserves older observations, validates coverage, and atomically
replaces the output only after all checks pass.

Raw/manual files, when required, belong in this folder's `raw/` directory and
should not be committed.
