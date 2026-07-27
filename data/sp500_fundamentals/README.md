# Point-in-Time S&P 500 Fundamentals

This folder builds one long-format parquet:

- `data/sp500_fundamentals.parquet`

It reads the ticker universe from `data/sp500_market_data.parquet`, resolves the
historical tickers to SEC Central Index Keys (CIKs), and extracts every eligible
numeric `us-gaap`, `ifrs-full`, and selected numeric DEI fact from the official
SEC Company Facts bulk archive. EdgarTools parses the facts and supplies filing,
statement-type, and quality metadata.

For a first build, create the market bootstrap first:

```powershell
.\.venv\Scripts\python.exe data\sp500_market\download.py --no-sec-enrichment
```

After this fundamentals builder finishes, run:

```powershell
.\.venv\Scripts\python.exe data\sp500_market\download.py --enrich-only
```

That final SEC-only market pass adds just `industry` and `market_cap`; it does
not download prices again. This three-stage order resolves the dependency
cleanly: market membership defines the SEC universe, then SEC facts enrich the
already-built market panel.

## SEC access: identity, not an API key

No SEC API key is required. SEC fair-access policy requires an identifying
`User-Agent` containing a name/project and contact email. Keep it outside the
repository:

```powershell
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\sp500_fundamentals\download.py
```

The identity is used only for SEC requests. Its value is not written to a
script, cache manifest, or parquet metadata.

The first run downloads and extracts the official SEC `companyfacts.zip` bulk
archive under this folder's ignored `cache/edgar/` directory. Expect a
multi-gigabyte download and substantially more extracted disk usage. Reruns
reuse both the SEC cache and per-CIK parsed parquet caches.

## Incremental updates without the bulk archive

After the first bootstrap, use the unified updater:

```powershell
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\update.py
```

`sp500_fundamentals/update.py` reads the SEC quarterly `xbrl.idx` files over a
10-day overlap around its saved checkpoint. These indexes contain filing CIK,
form, filing date, and accession, so they act as a change detector for the
CIKs mapped to the market universe. Only triggered CIKs are requested from:

```text
https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
```

That endpoint returns the complete current Company Facts JSON for one issuer;
it does not offer a `since` parameter. The updater hashes each response,
reparses only changed issuer caches with EdgarTools, and never requests
`companyfacts.zip`. Successful index-only filings are remembered during the
overlap window so cover-page XBRL filings that add no Company Facts do not
cause repeated downloads. Failed requests are retained for retry.

The output remains one parquet as requested. Parquet files are immutable, so
when facts or ticker-CIK mapping metadata change the updater streams all
already-parsed local issuer caches into a temporary parquet, validates the row
count and schema, and atomically replaces the old file. This rewrites the
compact final file but does not redownload or reparse unaffected issuers. If
neither sources nor mappings changed, the fundamentals parquet is left
byte-for-byte intact.

## What “point in time” means here

Every row retains `filed_date`, `form_type`, and `accession`. Amendments and
later filings that repeat or revise an earlier period are not collapsed.
`filing_version` orders observations within:

```text
(cik, concept, unit, period_start, period_end)
```

For a decision or training date `t`:

1. filter to `filed_date <= t`;
2. within the key above, retain the greatest `filing_version`;
3. join to the market panel using `ticker` or, preferably, the stable `cik`;
4. filter the market panel to `is_sp500_member` on `t`.

Never select facts by `period_end` alone. A fiscal quarter or year was not known
to the market until it was filed. The dataset deliberately preserves later
comparative restatements so an as-of query sees only the versions then
available.

## Coverage and history

The output begins on 2012-01-01. SEC structured Company Facts began with the
XBRL transition around 2009, but the first years are uneven across issuers and
concepts. Starting in 2012 is the more defensible default for a cross-sectional
ML panel. The market dataset keeps its longer 2005 price history.

Historical ticker-to-CIK resolution is the hard part of a survivorship-aware SEC
dataset. The script scans `dei:TradingSymbol` evidence across the entire SEC
bulk archive, compares its filing-date range to the ticker's S&P membership
range, and supplements current names from the SEC's official current ticker
file only when the entity's actual Company Facts filing range overlaps the
ticker's S&P membership dates. The market builder's small, documented set of
true ticker renames is also carried through to the corresponding current SEC
ticker. `mapping_valid_from` and `mapping_valid_to` record the interval supported
by that evidence. This prevents a current company from being silently assigned
to an old, reused ticker. Mapping and fact coverage rates, plus any unmapped
tickers, are stored in final parquet metadata and printed during the run.

Rare corporate-identity transitions that cannot be recovered from the SEC
current ticker file are kept in the versioned
`verified_ticker_cik_overrides.csv`, with an official SEC source URL and a
research note. Keep this file small and evidence-backed.

If a known historical mapping is still absent, create the ignored file
`data/sp500_fundamentals/raw/ticker_cik_overrides.csv` with:

```text
ticker,cik,note
OLD,123456,Verified against SEC filing accession ...
```

Only add an override after verifying the issuer in an SEC filing. The script
marks such rows `manual_verified`. The override file is local research input and
is intentionally not redistributed.

## Columns

| column | meaning |
|---|---|
| `ticker` | S&P historical ticker alias used by the market dataset |
| `cik` | stable SEC filer identifier |
| `entity_name` | SEC entity name |
| `mapping_source`, `mapping_confidence` | ticker-to-CIK provenance |
| `member_first_date`, `member_last_date` | observed membership range in the market parquet |
| `mapping_valid_from`, `mapping_valid_to` | dates for which the ticker-CIK evidence overlaps membership |
| `ticker_evidence_first_filed`, `ticker_evidence_last_filed` | first/last SEC filing carrying that trading-symbol evidence |
| `concept`, `label` | taxonomy-qualified XBRL concept and human label |
| `value`, `unit` | reported numeric value and unit |
| `period_type`, `period_start`, `period_end` | instant or duration reporting period |
| `fiscal_year`, `fiscal_period` | issuer-reported fiscal labels |
| `filed_date` | earliest date this fact version may be used |
| `form_type`, `accession` | filing provenance |
| `statement_type`, `taxonomy` | EdgarTools classification and SEC taxonomy |
| `data_quality`, `confidence_score` | EdgarTools parsing metadata |
| `is_annual_filing`, `is_amendment` | filing-type helpers |
| `filing_version` | chronological version within the point-in-time fact key |

The output includes 10-K, 10-Q, 20-F, 40-F, 6-K, and 8-K facts and their
amendments. Keeping 8-K/6-K facts allows an as-of model to see structured
earnings information when it was disclosed before the later periodic filing.
Users who want only periodic statements can filter `form_type`.

## Validation

Before replacing the final parquet the script verifies:

- exact fact-version uniqueness and finite numeric values;
- filing dates, period dates, allowed forms, and amendment preservation;
- ticker-to-CIK coverage of at least 95% of S&P constituent-days from 2012;
- fact coverage of at least 95% of recent constituent-days;
- source hashes for every selected Company Facts JSON, combined into a dataset
  digest in parquet metadata;
- final parquet row count and schema.

### Current validated build

The incremental build validated on 2026-07-27 contains 13,528,328 rows and 30
columns for
617 tickers with eligible facts. It maps 618 of the market dataset's 666
historical tickers; one mapped recent entity has no eligible Company Facts yet.
Fact coverage is 97.7167% of S&P constituent-days from 2012 onward and 99.9164%
over the latest two years. The remaining 49 tickers without output facts are
mostly older, delisted, acquired, or ambiguous identities. They are left
unmapped rather than assigned to a ticker reuse or acquirer.

Validation found zero duplicate fact-version keys, non-finite values, or facts
filed before their period end. Period ends run from 2012-01-01 through
2026-07-23, and filing dates run through 2026-07-24. The complete ticker list,
source digest, and validation result are
embedded in parquet metadata and are authoritative for later rebuilds.

The free SEC source is authoritative for filed facts, but it is not a normalized
Compustat-style database. Concept choice can vary across issuers and over time,
and Company Facts excludes custom taxonomy facts. Build economically equivalent
feature mappings explicitly and test them by industry before training a model.

Official sources:

- SEC EDGAR APIs: <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>
- SEC fair access: <https://www.sec.gov/about/developer-resources>
- EdgarTools: <https://dgunning.github.io/edgartools/>
