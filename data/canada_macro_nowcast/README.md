# Canada Real-Time Macro Nowcasting Data

Builds the Canadian Project 23 repeat from official Statistics Canada and Bank
of Canada sources. The folder produces five notebook-facing files and keeps raw
downloads under the ignored `cache/` directory.

## Outputs

- `data/canada_statscan_realtime.parquet` — selected national real-time
  vintages from all mandatory StatsCan tables, plus the income-side GDP and
  current-account diagnostics;
- `data/canada_statscan_current.parquet` — non-revised raw CPI history and
  forward-only snapshots of current core CPI plus the manufacturing, wholesale,
  and retail successors to archived real-time tables;
- `data/canada_boc_market.parquet` — daily policy rate, USD/CAD, CORRA, 1/2/3/6
  month and 1 year bills, 2/5/10/long benchmark bonds, and the complete official
  120-maturity zero-coupon curve;
- `data/canada_boc_bos.csv` — raw Business Outlook Survey question responses and
  balances, with the re-estimated composite and the new 2026 activity/price
  indicators excluded;
- `data/canada_boc_mps.csv` — every public Market Participants Survey table in a
  source-faithful long format.

## Statistics Canada coverage

The real-time file includes tables 36-10-0491, 36-10-0431, 18-10-0259,
14-10-0331, 12-10-0165, 16-10-0014, 16-10-0015, 16-10-0118, 20-10-0005,
20-10-0019, 20-10-0020, 20-10-0081, and 20-10-0082. It also includes optional
diagnostics 36-10-0430 and 36-10-0042.

Every historical release remains separate. The builder keeps national totals,
top-level industries and product groups, and the source units/status fields. It
does not calculate growth, revisions, surprises, factors, lags, or targets.

Table 18-10-0004 supplies raw all-items CPI and major components. Statistics
Canada does not revise the headline non-seasonally-adjusted CPI, so this current
history does not need a vintage cube. The archived retail tables 20-10-0081 and
20-10-0082 end with their final published vintages. Current tables 20-10-0056
and 20-10-0067 are never used to manufacture historical vintages: each new
source release is stored only as a forward snapshot from the date this pipeline
first observed it.

At this build date, real-time core table 18-10-0259 ends at the March 2026
observation published on April 20. Regular table 18-10-0256 continues through
July 2026. Its five CPI-common/median/trim series are therefore captured as a
forward snapshot from the current source release, never inserted into the older
0259 vintage history. Later updates add a 0256 snapshot only when that source
release changes.

The manufacturing and wholesale real-time cubes were also retired during the
2026 table transition. Their current successors are stored with the same
forward-only rule: 16-10-0013, 16-10-0012, and 16-10-0047 continue the real,
capacity-utilization, and broad manufacturing blocks; 20-10-0003, 20-10-0074,
and 20-10-0076 continue wholesale price/volume, sales, and inventories. Their
revised histories remain inside the acquisition snapshot and are never spliced
into earlier vintage rows.

Official documentation and tables:

- https://www.statcan.gc.ca/en/developers/wds/user-guide
- https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610049101
- https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410033101
- https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810025901

## Bank of Canada coverage and timing

Valet identifiers are resolved from the live Lists endpoint, including CORRA
and the long benchmark used for the requested 30-year control; they are not
copied from a web page. The extra two-month bill is retained because it is free,
daily, and useful at the front of the curve. The zero-coupon file preserves all
120 maturities from 0.25 to 30 years and leaves yields in the Bank's published
decimal units.

BOS observations use their provider quarter labels. Exact release dates are
attached for the online publication archive (2004 onward); older rows remain
null rather than receiving invented dates. BOS Indicator, the 2026 activity
indicator, and the 2026 price indicator are excluded. The underlying raw
question responses and balances are retained.

MPS begins with the first public release in February 2023. The earlier pilot was
not publicly released as a historical dataset, so it is not backfilled.

Official sources:

- https://www.bankofcanada.ca/valet-api-how-to/
- https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/
- https://www.bankofcanada.ca/publications/bos/business-outlook-survey-data/
- https://www.bankofcanada.ca/publications/market-participants-survey/

## Run

Full first build:

```powershell
.\.venv\Scripts\python.exe data\canada_macro_nowcast\download.py
```

Routine update:

```powershell
.\.venv\Scripts\python.exe data\canada_macro_nowcast\update.py
```

The first build downloads the official bulk StatsCan archives once because that
is the only reliable way to recover every historical release, including several
newly archived tables. Routine updates do not repeat that download. They request
only previously unseen real-time release vectors through WDS, add a current-table
snapshot only when the source release changes, overlap the most recent two weeks
of Bank market data, and download only newly discovered MPS pages.

The Canadian notebook repeat should later reuse the existing U.S. point-in-time
Project 23 files for international predictors. This folder deliberately does
not duplicate or relabel U.S. data.
