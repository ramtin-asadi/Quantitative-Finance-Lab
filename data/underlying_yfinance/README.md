# Underlying Yfinance

    Source: Yahoo Finance data accessed through yfinance.

    Script: `download.py`

    Final output files:
    - `data/spx_index_ohlcv.csv`
- `data/spy_ohlcv.csv`
- `data/qqq_ohlcv.csv`

    Source and download links:
    - https://ranaroussi.github.io/yfinance/
- https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html

    Notes:
    Yahoo/yfinance data is for personal, research, or educational use subject to Yahoo terms.

    Rebuild from the repository root:

    ```bash
    python data/underlying_yfinance/download.py
    ```

    Raw/manual files, when required, belong in this folder's `raw/` directory and should not be committed.
