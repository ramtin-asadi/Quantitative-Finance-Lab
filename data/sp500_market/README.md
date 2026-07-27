# S&P 500 Historical-Constituent Market Data

This folder builds one long-format parquet:

- `data/sp500_market_data.parquet`

The universe is the union of S&P 500 constituents from 2005 onward, not today's
members projected backward. Historical membership comes from
[`fja05680/sp500`](https://github.com/fja05680/sp500), and prices/actions come
from Yahoo Finance through `yfinance`. The final two optional columns use
official SEC filing data and the companion fundamentals parquet.

## Why this design

The output keeps each ticker's available price history and adds
`is_sp500_member` for every trading date. Filter on that column before forming a
historical portfolio or cross-section. This avoids the main survivorship error
of using only the current S&P 500 list. A Yahoo series is retained only if it
has at least one price observation during that ticker's membership dates; this
rejects current companies that merely reuse an old constituent's symbol.

The script downloads the repository's updated constituent CSV directly from its
raw GitHub URL; Git is not required and the repository is not cloned. The first
successful download is cached at
`data/sp500_market/raw/sp500_historical_components.csv`. Its SHA-256 checksum
and retrieval metadata are embedded in the final parquet. Normal reruns reuse
that exact cached file for reproducibility. Pass `--refresh-constituents` only
when you intentionally want the latest upstream revision.

If GitHub cannot be reached and there is no cached file, manually download
`S&P 500 Historical Components & Changes (Updated).csv` from the repository,
rename it to `sp500_historical_components.csv`, and place it in this folder's
`raw/` directory. Then rerun with `--offline`.

## Columns

| column | meaning |
|---|---|
| `date` | exchange trading date, timezone-naive |
| `ticker` | historical constituent symbol from the membership source |
| `yf_ticker` | Yahoo symbol actually requested; class-share dots become dashes and documented true renames use the successor symbol |
| `adj_close` | Yahoo split- and dividend-adjusted close |
| `close` | unadjusted close |
| `volume` | reported share volume |
| `dividends` | cash dividend per share on the action date |
| `stock_splits` | split ratio on the action date; zero otherwise |
| `was_repaired` | `yfinance` marked the row as repaired |
| `is_sp500_member` | ticker belongs to the point-in-time S&P 500 snapshot |
| `snapshot_date` | membership snapshot used as of `date` |
| `industry` | official SEC SIC industry title from the latest filing strictly before `date` |
| `market_cap` | point-in-time issuer-level estimate from `close` and validated SEC shares |

Open/high/low are intentionally omitted because they are not needed for the
planned fundamental-equity work. Yahoo's historical shares series and Yahoo's
current industry profile are intentionally not used: the former is incomplete
for this survivorship-aware universe, and the latter would apply today's
classification backward.

## SEC enrichment logic

There is one user-facing builder: `download.py`. It supports the unavoidable
bootstrap dependency explicitly:

1. the first market run builds the 11-column price/membership base;
2. the fundamentals builder reads that base to define its ticker universe and
   creates point-in-time SEC facts plus ticker-CIK evidence;
3. `download.py --enrich-only` reads those fundamentals and adds `industry`
   and `market_cap` without making any constituent or Yahoo request.

After the fundamentals parquet exists, an ordinary market rebuild automatically
performs the SEC enrichment at the end. `--no-sec-enrichment` is available when
an explicit 11-column bootstrap is wanted.

`industry` comes from the official SEC Financial Statement Data Sets `SUB`
table. SEC documentation defines its SIC field as the filer classification as
of the filing date. The helper reads only `sub.txt` from each quarterly ZIP by
HTTP Range request and caches those small extracts. The earliest SEC archive is
2009 Q1, so industry can begin in 2009 when a valid ticker-CIK mapping exists.
It is not forced to start in 2012.

`market_cap` does begin with the reliable fundamentals window in 2012. The
internal selection prefers
`dei:EntityCommonStockSharesOutstanding` and falls back to
`us-gaap:CommonStockSharesOutstanding` only when the preferred fact is absent.
It rejects implausible scale values and isolated filing spikes, applies a
maximum 400-day age, and leaves unavailable observations null.

Yahoo historical `close` is normalized for later stock splits, while an SEC
share fact is expressed in the basis represented by its filing. The calculation
therefore applies Yahoo split ratios strictly after the filing date before
multiplying by `close`. This prevents, for example, a roughly fourfold
understatement of Apple's pre-2020 market cap. The raw share facts and all
selection provenance remain in `sp500_fundamentals.parquet`; they are not
duplicated as extra market columns.

Both fields use the conservative availability rule
`source_filed_date < market_date`, so a same-day filing first affects the next
trading date. `market_cap` is an issuer-level estimate, not vendor float market
cap. Multiple share classes can share a CIK and aggregate share fact.

## Cleaning and validation

The downloader uses `auto_adjust=False`, corporate actions, and yfinance's
currency/unit repair. It then removes only isolated or reversing adjusted-price
artifacts and rejects irrecoverably noisy series. It does not winsorize genuine
returns or emit a separate "extreme returns" file.

Before replacing the final parquet it checks:

- unique `(date, ticker)` grain, positive adjusted prices, nonnegative volume,
  and date bounds;
- point-in-time constituent coverage, including a 95% recent median threshold;
- split dates for remaining adjusted-price discontinuities;
- daily equal-weight constituent-return correlation with RSP (must be at least
  0.95) and a secondary comparison with SPY.

The SEC pass additionally verifies strict filing availability, share staleness,
positive values, exact price-times-shares recomputation, recent coverage, and
share-event cleaning. Base and enrichment validation results are stored in
parquet metadata. The only final dataset is the parquet; per-ticker and SEC
caches remain ignored under `cache/` so interrupted runs can resume.

### Current validated build

The incremental build validated on 2026-07-27 contains 3,223,459 rows, 13 columns,
and 666 tickers from 2005-01-03 through 2026-07-24. It has no duplicate keys,
invalid adjusted closes, negative volumes, or unresolved split
discontinuities. Daily
equal-weight constituent returns correlate 0.99278 with RSP and 0.958636 with
SPY over 5,422 common days.

Filing-date industry covers 617 tickers and validated market cap covers 568.
Over the latest two years, S&P member-row coverage is 99.9148% for industry and
92.4675% for market cap. From 2012 onward, member-row market-cap coverage is
89.9530%. No enriched value uses a same-day or future filing. The embedded
parquet metadata is authoritative for later rebuilds.

## Rebuild

From the repository root, the complete first-time order is:

```powershell
# 1. Bootstrap prices, actions, and membership.
.\.venv\Scripts\python.exe data\sp500_market\download.py --no-sec-enrichment

# 2. Build point-in-time SEC facts and ticker-CIK mappings.
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\sp500_fundamentals\download.py

# 3. Add only industry and market cap; no Yahoo download occurs.
.\.venv\Scripts\python.exe data\sp500_market\download.py --enrich-only
```

The SEC identity is a declared User-Agent, not an account or API key. Its value
is never persisted. Once all three stages exist, an ordinary
`data/sp500_market/download.py` run updates prices and automatically reapplies
the SEC columns.

## Incremental maintenance

For normal maintenance after the first bootstrap, run the repository-level
orchestrator:

```powershell
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\update.py
```

The market step refreshes the upstream constituent CSV and requests a
14-calendar-day Yahoo overlap for the recent membership union plus every
historical constituent that has traded during the latest 30 days. This keeps
available post-membership histories current without repeatedly requesting
long-delisted securities. It downloads full history only if a genuinely new
constituent ticker appears. The overlap
captures late or corrected Yahoo bars without redownloading old histories.
It rebuilds the affected slice from per-ticker caches, retains all earlier
rows, validates the combined panel, and atomically replaces the parquet.

The orchestrator then updates SEC fundamentals and runs `download.py
--enrich-only`; therefore new market rows receive `industry` and `market_cap`
without any second Yahoo request. `data/sp500_market/update.py` can be run
alone, but its new rows intentionally keep the old enrichment values only
where the exact `(date, ticker)` already existed. Prefer `data/update.py` for a
fully maintained pair.

Useful options:

```powershell
# Update an existing price cache through today
.\.venv\Scripts\python.exe data\sp500_market\download.py

# Retry symbols Yahoo previously returned as unavailable
.\.venv\Scripts\python.exe data\sp500_market\download.py --retry-missing

# Intentionally update the historical-constituent source
.\.venv\Scripts\python.exe data\sp500_market\download.py --refresh-constituents

# Reassemble and revalidate without any price download
.\.venv\Scripts\python.exe data\sp500_market\download.py --validate-only --offline

# Reapply enrichment entirely from completed local SEC caches
.\.venv\Scripts\python.exe data\sp500_market\download.py --enrich-only --offline

# Intentionally refresh the quarterly SEC submission/SIC caches
.\.venv\Scripts\python.exe data\sp500_market\download.py --enrich-only --refresh-sic
```

The source is useful and transparent, but it is not an official S&P Dow Jones
Indices constituent history. Its own maintainer notes that early history is
partly reconstructed and that free Yahoo data is incomplete for some delisted
securities. For publication-grade claims or trading-capital decisions, compare
against a licensed point-in-time security master and delisted-price vendor.

Yahoo data is subject to Yahoo's terms:
<https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html>.

Official SEC enrichment sources:

- <https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets>
- <https://www.sec.gov/files/fsds.pdf>
- <https://www.sec.gov/search-filings/standard-industrial-classification-sic-code-list>
