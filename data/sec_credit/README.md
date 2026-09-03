# SEC Credit

Builds one notebook-facing file, `data/sec_credit.parquet`, for Project 22. It
contains point-in-time credit facts and SEC filing-event metadata for a broad
U.S. nonfinancial operating-company universe. The existing historical S&P 500
issuers remain marked by `is_sp500_issuer` for the Merton subset.

The data layer preserves source facts and event metadata. It does not calculate
ratios, future-horizon labels, distance-to-default, or model features.

## Sources and bandwidth design

Run the S&P fundamentals pipeline first. Its extracted all-filer Company Facts
cache is reused directly; this folder never downloads `companyfacts.zip`.

The SEC also publishes a nightly all-filer `submissions.zip`, but that archive
was 1,560,992,008 bytes on 2026-08-29. This pipeline does not download it. The
first stage instead scans the local Company Facts cache once and keeps only
CIKs with:

- selected US-GAAP credit facts and at least one 10-K;
- subsequent 10-Q/10-K reporting;
- at least eight periodic filings and two years of history;
- at least four asset periods and four liability/equity periods.

Only the surviving CIKs are requested from the per-filer Submissions API. Their
small current JSON files are downloaded first. Authoritative SEC SIC and entity
type then remove SIC 6000-6999, investment vehicles, and other non-operating
filers before any older submission-history segments are requested. This gives
the broad estimation universe without paying for every filer in the bulk ZIP.

Official documentation:

https://www.sec.gov/search-filings/edgar-application-programming-interfaces

Set the SEC-required identifying User-Agent and run:

```powershell
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\sec_credit\download.py
.\.venv\Scripts\python.exe data\sec_credit\build.py
```

The first command performs the local accounting screen, downloads targeted
Submissions histories, and reviews Item 1.03 primary documents. It is resumable:
Company Facts, submissions, selected issuer facts, and event documents are all
cached under ignored `cache/` paths with manifests. No credential or User-Agent
identity is written to disk.

## One output, two source record types

`sec_credit.parquet` follows the long fundamentals style and distinguishes rows
with `record_type`:

- `fact` rows contain selected raw US-GAAP values from 10-K, 10-Q, and structured
  8-K XBRL filings;
- `filing` rows contain one source record per relevant SEC filing, including
  filings that have no useful XBRL fact.

Keeping both in one Parquet is necessary because distress 8-K filings often
contain no numeric fact. Facts are joined to their filing accession when
possible, adding the exact SEC acceptance timestamp.

The retained concepts cover balance sheet, liquidity, working capital, debt,
leases and maturity ladders, revenue and profitability, interest, operating
cash flow, capital expenditure, financing flows, impairments, and restructuring
charges. Unrelated Company Facts concepts are excluded.

## Distress events

Direct SEC flags include:

- 8-K Item 1.03 bankruptcy or receivership;
- 8-K Item 2.04 financial-obligation triggers;
- Items 2.05, 2.06, 3.01, and 4.02;
- late 10-K/10-Q notices and deregistration/delisting forms.

Item 1.03 metadata alone can include subsidiary cases, announced intent, plan
confirmation/emergence, and occasional miscoding. `review_events.py` therefore
downloads only the Item 1.03 primary documents and applies conservative,
auditable rules. `is_registrant_bankruptcy_event` is true only when the text
explicitly identifies the Company or registrant as the petition/receivership
subject. Ambiguous cases remain `bankruptcy_scope="unreviewed"`; they are not
silently promoted to positive labels. Reviewed scopes, rule notes, source URLs,
and excerpts remain in the ignored review cache, while the scope and note needed
by the notebook are included in the one ready Parquet.

These are source classifications, not future-horizon targets. The notebook is
responsible for episode rules and 6-, 12-, or 24-month labels. Item 2.04 is
broader than default and must not be treated as a bankruptcy flag without
context rules.

## Incremental update

```powershell
$env:EDGAR_IDENTITY = "Your Name your.email@example.com"
.\.venv\Scripts\python.exe data\sec_credit\update.py
```

The updater reads only recent SEC quarterly master indexes. It detects 10-K/10-Q
accessions absent from each retained issuer's selected-facts cache. Because the
SEC per-CIK Company Facts endpoint returns the issuer's entire history, ordinary
runs refresh at most 100 affected CIKs and persist the remaining accession
backlog. Repeated runs drain that backlog without bulk downloads or repeated
checks. Use `--accounting-batch-size 0` for an event-metadata-only run, or set a
different positive batch size to match the available bandwidth.

Every run refreshes Submissions only for affected CIKs in the final model
universe, reviews new Item 1.03 documents, and atomically rebuilds the single
Parquet from caches. It also keeps the existing point-in-time S&P fundamentals
mapping current unless `--skip-sp500-fundamentals` is supplied. Neither SEC bulk
ZIP is downloaded.
