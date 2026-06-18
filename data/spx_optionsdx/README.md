# SPX OptionsDX

Builds `data/spx_options_chain.parquet` from purchased monthly OptionsDX SPX files.

## Source

- OptionsDX SPX product page: https://www.optionsdx.com/product/spx-option-chain/

## What To Put In `raw/`

1. Purchase or download the SPX monthly option-chain files from OptionsDX. (for the months that we use, it's free and can be bought for 0$, it just needs making an account)
2. Extract all the ZIP files and put the downloaded monthly `.txt` or `.csv` files under:
   `data/spx_optionsdx/raw/`
3. Keep the vendor filenames if possible. The builder reads every `.txt` and `.csv` recursively.

Expected raw contents:

- Monthly SPX files such as `SPX_2022_01.txt`, `SPX_2022_02.txt`, or equivalent OptionsDX CSV/TXT names.
- Columns should include OptionsDX fields such as `quote_date`, `quote_readtime`, `expire_date`, `underlying_last`, `strike`, call bid/ask fields, and put bid/ask fields.

Do not put SPY, QQQ, or BTC files in this folder.

## Rebuild

```bash
python data/spx_optionsdx/build.py
```

The script prints per-file row counts, writes one compressed Parquet output, and intentionally does not write any `*_month_counts.csv` file. Raw OptionsDX files are paid/manual data and are ignored by Git.
