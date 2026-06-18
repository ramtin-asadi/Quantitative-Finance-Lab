# Core Cross Asset Etfs

    Source: Yahoo Finance ETF data accessed through yfinance.

    Script: `download.py`

    Final output files:
    - `data/core_cross_asset_etfs.csv`

    Source and download links:
    - https://ranaroussi.github.io/yfinance/
- https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html

    Notes:
    Output uses date,TICKER__close,TICKER__volume,TICKER__dividends,TICKER__stock_splits format.

    Rebuild from the repository root:

    ```bash
    python data/core_cross_asset_etfs/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
