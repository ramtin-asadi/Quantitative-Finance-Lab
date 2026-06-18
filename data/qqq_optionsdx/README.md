# QQQ OptionsDX

Builds `data/qqq_options_chain.parquet` from purchased monthly OptionsDX QQQ files.

## Source

- OptionsDX QQQ product page: https://www.optionsdx.com/product/qqq-option-chain/

## What To Put In `raw/`

1. Purchase or download the QQQ monthly option-chain files from OptionsDX (for the months that we use, it's free and can be bought for 0$, it just needs making an account).
2. Extract the files from the ZIP files and put the downloaded monthly `.txt` or `.csv` files under:
   `data/qqq_optionsdx/raw/`
3. Keep the vendor filenames if possible. The builder reads every `.txt` and `.csv` recursively.

Expected raw contents:

- Monthly QQQ files such as `QQQ_2022_01.txt`, `QQQ_2022_02.txt`, or equivalent OptionsDX CSV/TXT names.
- Columns should include `quote_date`, `quote_readtime`, `expire_date`, `underlying_last`, `strike`, call bid/ask fields, and put bid/ask fields.

This workspace does not currently have QQQ raw files, so the script will fail clearly until the files are placed here. Do not put SPX, SPY, or BTC files in this folder.

## Rebuild

```bash
python data/qqq_optionsdx/build.py
```

The script prints per-file row counts, writes one compressed Parquet output, and intentionally does not write any `*_month_counts.csv` file. Raw OptionsDX files are paid/manual data and are ignored by Git.
