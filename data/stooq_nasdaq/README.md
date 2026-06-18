# Stooq NASDAQ

Builds `data/nasdaq_close_volume.parquet` from the user-downloaded Stooq historical stock database.

## Source

- Stooq historical data download page: https://stooq.com/db/h/
- Use the U.S. daily stock archive and keep only the NASDAQ stocks subtree.

## What To Put In `raw/`

1. Open https://stooq.com/db/h/.
2. Download the U.S. daily historical stock database archive from Stooq.
3. Unzip it outside Git if possible.
4. Find the NASDAQ stocks folder. In the current Stooq layout it is normally:
   `daily/us/nasdaq stocks/`
5. Move that folder, or all of its `*.us.txt` files, under:
   `data/stooq_nasdaq/raw/`

Expected raw contents:

- Many text files named like `aapl.us.txt`, `msft.us.txt`, `nvda.us.txt`.
- Each file should contain Stooq columns such as `<DATE>`, `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<VOL>`.

Do not include NYSE, AMEX, ETF, warrant, or other non-NASDAQ folders here. The script recursively reads `*.us.txt`, so either of these layouts is fine:

```text
data/stooq_nasdaq/raw/aapl.us.txt
data/stooq_nasdaq/raw/msft.us.txt
```

or:

```text
data/stooq_nasdaq/raw/nasdaq stocks/aapl.us.txt
data/stooq_nasdaq/raw/nasdaq stocks/msft.us.txt
```

## Rebuild

```bash
python data/stooq_nasdaq/build.py
```

The output is a wide Parquet file with columns like `AAPL__close` and `AAPL__volume`. Raw Stooq files are ignored by Git and should not be redistributed.
