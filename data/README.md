# Data Reproducibility

We don't redistribute data used in every project, but we included guides and helpers for remaking all the data files that are used in notebooks. Final files are produced directly under `data/` by one script per source folder. Raw/manual vendor files stay under the relevant `raw/` directory.

Script-downloadable sources include FRED U.S. Treasury yields, FRED NFCI,
Japan MOF JGB yields, NY Fed ACM term premia, yfinance OHLCV/ETF data,
historical S&P 500 membership, SEC Company Facts, macro factors from
FRED/StatCan/Bank of Canada, and Kenneth French factor files. The HKEX
stock-name workbook is downloaded inside the Stooq HKEX processor so the Hong
Kong panel uses security names as its asset columns.

Manual or restricted sources include Stooq bulk stock downloads and OptionsDX option-chain files. Those files must be downloaded manually by user and placed under the matching `raw/` folder before running the local processor. Raw folders are tracked with `.gitkeep` placeholders, but all real raw files inside them are ignored.

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
| `sp500_market/raw/` | Normally nothing. The script downloads and caches the historical constituent CSV. If GitHub access fails, save the linked CSV here as `sp500_historical_components.csv`. | https://github.com/fja05680/sp500 |
| `sp500_fundamentals/raw/` | Normally nothing. Optionally add ignored `ticker_cik_overrides.csv` only for ticker-CIK links independently verified against an SEC filing. | https://www.sec.gov/search-filings/edgar-application-programming-interfaces |

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
python data/sp500_market/download.py --no-sec-enrichment
python data/sp500_fundamentals/download.py
python data/sp500_market/download.py --enrich-only
```

OptionsDX scripts require the corresponding monthly raw files to be present in their `raw/` folders. The files need purchasing but the files we use are 0$ and just need creating an account on website.

The S&P datasets have an explicit three-stage bootstrap. First, the market
builder creates the 11-column price/membership universe. Second, the
fundamentals builder uses that universe to create SEC facts and ticker-CIK
mappings. Third, the same market `download.py` is run with `--enrich-only` to
add the final `industry` and `market_cap` columns without requesting Yahoo
again. Later ordinary market rebuilds automatically enrich when the
fundamentals parquet already exists.

The SEC stages require `EDGAR_IDENTITY` to contain a name and contact email.
This is an SEC identifying User-Agent, not an API key or account credential.
See `data/sp500_market/README.md` and `data/sp500_fundamentals/README.md`.

After the three-stage bootstrap has succeeded once, maintain both files with:

```powershell
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\update.py
```

The updater refreshes a short Yahoo overlap and the latest constituent
snapshots, uses the SEC quarterly XBRL indexes to identify only mapped CIKs
with new filings, fetches those issuers through the per-CIK Company Facts API,
and reapplies point-in-time `industry` and `market_cap`. It never downloads
`companyfacts.zip`. Parquet does not support safe in-place row appends, so each
final single-file parquet is still validated and atomically rewritten; the
large source archive and unaffected issuer caches are neither downloaded nor
reparsed.

Source and terms cautions:

- Stooq bulk files come from https://stooq.com/db/h/ and should be used according to Stooq terms. Do not commit or redistribute the raw bulk archive.
- OptionsDX option-chain files come from product pages such as https://www.optionsdx.com/product/spx-option-chain/ and are paid/manual data. Do not redistribute raw files.
- yfinance accesses Yahoo Finance data. See https://ranaroussi.github.io/yfinance/ and Yahoo terms at https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html.
- FRED data should be cited to FRED and the underlying source, including the Federal Reserve Bank of Chicago for NFCI.
- NY Fed ACM term premia should be cited to the Federal Reserve Bank of New York.
- Japan MOF JGB data should be cited to the Ministry of Finance Japan.
- HKEX security names used by `data/stooq_hkex/build.py` come from HKEX securities lists: https://www.hkex.com.hk/Services/Trading/Securities/Securities-Lists?sc_lang=en.
- Historical S&P membership comes from the community-maintained https://github.com/fja05680/sp500 dataset. Its own README documents early-history and delisted-price limitations.
- S&P prices use Yahoo Finance through yfinance and therefore are suitable for reproducible research with the recorded validation metrics, but are not a licensed replacement for CRSP/Norgate.
- S&P fundamentals come from the official SEC Company Facts bulk archive and must be accessed under the SEC fair-access policy: https://www.sec.gov/about/developer-resources.
