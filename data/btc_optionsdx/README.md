# BTC OptionsDX

Builds `data/btc_options_chain.parquet` from purchased monthly OptionsDX Deribit BTC option files.

## Source

- OptionsDX BTC/Deribit product page: https://www.optionsdx.com/product/btc-option-chains-deribit/

## What To Put In `raw/`

1. Purchase or download the BTC/Deribit monthly option-chain files from OptionsDX (for the months that we use, it's free and can be bought for 0$, it just needs making an account).
2. Extract the files from their ZIP files and put the downloaded monthly `.txt` or `.csv` files under:
   `data/btc_optionsdx/raw/`
3. Keep the vendor filenames if possible. The builder reads every `.txt` and `.csv` recursively.

Expected raw contents:

- Monthly BTC files such as `btc_eod_202401.txt`, `btc_eod_202402.txt`, or equivalent OptionsDX CSV/TXT names.
- Columns should include Deribit/OptionsDX option fields such as `quote_date` or quote timestamp, expiry, strike, option right, bid/ask/mark/IV fields, and underlying/index fields.

Do not put SPX, SPY, or QQQ files in this folder.

## Rebuild

```bash
python data/btc_optionsdx/build.py
```

The script prints per-file row counts, writes one compressed Parquet output, and intentionally does not write any `*_month_counts.csv` file. Raw OptionsDX files are paid/manual data and are ignored by Git.
