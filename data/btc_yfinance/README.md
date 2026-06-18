# Btc Yfinance

    Source: Yahoo Finance BTC-USD data accessed through yfinance.

    Script: `download.py`

    Final output files:
    - `data/btc_usd_ohlcv.csv`

    Source and download links:
    - https://ranaroussi.github.io/yfinance/
- https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html

    Notes:
    Output is normalized to snake_case OHLCV/action columns.

    Rebuild from the repository root:

    ```bash
    python data/btc_yfinance/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
