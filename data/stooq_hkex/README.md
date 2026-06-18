# Stooq HKEX

Builds `data/hkex_close_volume.parquet` from the user-downloaded Stooq Hong Kong stock database. The builder downloads the official HKEX List of Securities workbook during the run and uses it to rename Stooq numeric stock codes to security names.

## Source

- Stooq historical data download page: https://stooq.com/db/h/
- Use the Hong Kong daily stock archive and keep only the HKEX stocks subtree.
- HKEX List of Securities workbook, used by the script for security names: https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx

## What To Put In `raw/`

1. Open https://stooq.com/db/h/.
2. Download the Hong Kong daily historical stock database archive from Stooq.
3. Unzip it outside Git if possible.
4. Find the HKEX stocks folder. In the current Stooq layout it is normally:
   `daily/hk/hkex stocks/`
5. Move that folder, or all of its `*.hk.txt` stock files, under:
   `data/stooq_hkex/raw/`

Expected raw contents:

- Many text files named like `5.hk.txt`, `700.hk.txt`, `9988.hk.txt`.
- Each file should contain Stooq columns such as `<DATE>`, `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<VOL>`.

Do not include HKEX derivative warrant, ETF, or non-stock subfolders here. The script recursively reads `*.hk.txt`, downloads the HKEX security-name workbook, and writes columns with security names:

- `5.hk.txt` -> `HSBC HOLDINGS__close`
- `700.hk.txt` -> `TENCENT__close`
- `9988.hk.txt` -> `BABA-W__close`

Files whose numeric code is not present in the current HKEX List of Securities are skipped rather than falling back to numeric column names.

## Rebuild

```bash
python data/stooq_hkex/build.py
```

The output is a wide Parquet file with columns like `HSBC HOLDINGS__close` and `TENCENT__volume`. Raw Stooq files are ignored by Git and should not be redistributed.
