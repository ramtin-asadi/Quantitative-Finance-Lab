# Data Reproducibility Layer

This folder is a reproducibility layer, not a redistributed data archive. Final files are produced directly under `data/` by one script per source folder. Raw/manual vendor files stay under the relevant `raw/` directory and are ignored by Git.

Script-downloadable sources include FRED U.S. Treasury yields, FRED NFCI, Japan MOF JGB yields, NY Fed ACM term premia, yfinance OHLCV/ETF data, macro factors from FRED/StatCan/Bank of Canada, and Kenneth French factor files. The HKEX stock-name workbook is downloaded inside the Stooq HKEX processor so the Hong Kong panel uses security names as its asset columns.

Manual or restricted sources include Stooq bulk stock downloads and OptionsDX option-chain files. Those files must be placed under the matching `raw/` folder before running the local processor. Raw folders are tracked with `.gitkeep` placeholders, but all real raw files inside them are ignored.

## Raw Folder Checklist

Most folders do not need manual raw files because their scripts download from public source endpoints. The table below lists every tracked raw folder and exactly what belongs there.

| folder | user should put inside `raw/` | source link |
|---|---|---|
| `stooq_nasdaq/raw/` | Stooq U.S. NASDAQ stock files only, usually `daily/us/nasdaq stocks/*.us.txt`; examples: `aapl.us.txt`, `msft.us.txt`. | https://stooq.com/db/h/ |
| `stooq_hkex/raw/` | Stooq Hong Kong HKEX stock files only, usually `daily/hk/hkex stocks/*.hk.txt`; examples: `5.hk.txt`, `700.hk.txt`. The builder also downloads the HKEX List of Securities workbook directly and renames numeric stock codes to security names. | https://stooq.com/db/h/ and https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx |
| `spx_optionsdx/raw/` | Purchased (Free) SPX OptionsDX monthly `.txt` or `.csv` files. | https://www.optionsdx.com/product/spx-option-chain/ |
| `spy_optionsdx/raw/` | Purchased (Free) SPY OptionsDX monthly `.txt` or `.csv` files. | https://www.optionsdx.com/product/spy-option-chain/ |
| `qqq_optionsdx/raw/` | Purchased (Free) QQQ OptionsDX monthly `.txt` or `.csv` files. | https://www.optionsdx.com/product/qqq-option-chain/ |
| `btc_optionsdx/raw/` | Purchased (Free) BTC/Deribit OptionsDX monthly `.txt` or `.csv` files. | https://www.optionsdx.com/product/btc-option-chains-deribit/ |
| `acm_term_premium/raw/` | Nothing. `download.py` downloads the official NY Fed workbook directly. | https://www.newyorkfed.org/research/data_indicators/term-premia-tabs |
| `chicago_fed_nfci/raw/` | Nothing. `download.py` downloads FRED `NFCI` directly. | https://fred.stlouisfed.org/series/NFCI |

Do not put archived files into these raw folders. The build/download scripts do not use archive folders as an input fallback.

Rebuild all generated data files from the repository root:

```bash
python data/us_treasury_yields/download.py
python data/japan_mof_yields/download.py
python data/chicago_fed_nfci/download.py
python data/acm_term_premium/download.py
python data/stooq_nasdaq/build.py
python data/stooq_hkex/build.py
python data/underlying_yfinance/download.py
python data/btc_yfinance/download.py
python data/spx_optionsdx/build.py
python data/spy_optionsdx/build.py
python data/qqq_optionsdx/build.py
python data/btc_optionsdx/build.py
python data/core_cross_asset_etfs/download.py
python data/sector_etfs/download.py
python data/factor_proxy_etfs/download.py
python data/international_country_etfs/download.py
python data/international_hedging_etfs/download.py
python data/canada_sector_etfs/download.py
python data/macro_factors/download.py
python data/fama_french_us/download.py
python data/fama_french_developed_ex_us/download.py
```

OptionsDX scripts require the corresponding monthly raw files to be present in their `raw/` folders. The files need purchasing but the files we use are 0$ and just need creating an account on website.

Source and terms cautions:

- Stooq bulk files come from https://stooq.com/db/h/ and should be used according to Stooq terms. Do not commit or redistribute the raw bulk archive.
- OptionsDX option-chain files come from product pages such as https://www.optionsdx.com/product/spx-option-chain/ and are paid/manual data. Do not redistribute raw files.
- yfinance accesses Yahoo Finance data. See https://ranaroussi.github.io/yfinance/ and Yahoo terms at https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html.
- FRED data should be cited to FRED and the underlying source, including the Federal Reserve Bank of Chicago for NFCI.
- NY Fed ACM term premia should be cited to the Federal Reserve Bank of New York.
- Japan MOF JGB data should be cited to the Ministry of Finance Japan.
- HKEX security names used by `data/stooq_hkex/build.py` come from HKEX securities lists: https://www.hkex.com.hk/Services/Trading/Securities/Securities-Lists?sc_lang=en.
