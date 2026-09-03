# FINRA Credit

This folder combines FINRA's account-free structured-product archives with the
free Public Developer API. Endpoint-level downloads stay in the ignored cache;
the repository exposes only four notebook-facing tables with genuinely
different row meanings.

## Historical structured-product archives

FINRA publishes one public ZIP per month containing daily Structured Product
Activity Reports (`STAR`) and Structured Product Pricing Tables (`PXTABLES`).
The history starts in May 2011.

Official archive page:

https://www.finra.org/finra-data/browse-catalog/structured-product-activity-reports-and-tables/historic-reports

Download every listed archive without an account:

```powershell
.\.venv\Scripts\python.exe data\finra_credit\download_archives.py
```

Alternatively, place browser-downloaded files unchanged under:

```text
data/finra_credit/raw/HISTORIC_SPREPORTS-YYYYMM.zip
```

Each monthly ZIP is parsed once into an ignored cache. Later builds reuse every
unchanged month.

## Public Developer API

Create a free individual account at https://developer.finra.org/, open the API
Console, and create a **Public Credential**. A Mock Credential returns only
sample/randomized data.

Keep the Client ID and Client Secret outside the repository:

```powershell
$env:FINRA_CLIENT_ID = "your-client-id"
$env:FINRA_CLIENT_SECRET = "your-client-secret"
.\.venv\Scripts\python.exe data\finra_credit\download_api.py
```

Credentials and OAuth tokens are never written to disk. Source-native API
Parquets are cached under `data/finra_credit/cache/api/`, not placed beside the
ready datasets.

Project 22 uses these core API datasets:

- corporate and Rule 144A market breadth;
- corporate and Rule 144A market sentiment;
- corporate/agency capped volume;
- CBO/CDO/CLO pricing;
- securitized-product activity, capped volume, and errata.

The default download also keeps agency breadth/sentiment and Treasury daily and
monthly TRACE aggregates. They are small, interpretable controls for separating
corporate-specific deterioration from broad fixed-income trading conditions.

## Build and final files

```powershell
.\.venv\Scripts\python.exe data\finra_credit\build.py
```

The builder writes:

- `data/finra_credit_market.parquet`: a tidy long table containing corporate,
  Rule 144A, agency, and Treasury breadth/flow/capped-volume measures;
- `data/finra_structured_pricing.parquet`: the full archive pricing history,
  extended after the latest monthly ZIP with API CBO/CDO/CLO observations;
- `data/finra_structured_activity.parquet`: archive trading activity plus API
  updates and securitized capped-volume observations;
- `data/finra_structured_errata.parquet`: unique correction events.

Reported values and FINRA suppression markers remain separate. No suppressed
cell is imputed, and no imbalance, breadth index, spread signal, state, or model
feature is constructed in the data layer.

The explicit CBO/CDO/CLO rating/vintage table begins on 2024-12-09. The observed
production categories are AAA, non-AAA investment grade, and non-investment
grade; the source does not provide a long separate AA/A/BBB history.

The Public API products are market aggregates. They do not provide a scalable
CUSIP-level TRACE transaction history, issuer-specific bond spreads, or masked
dealer identifiers. Optional single-bond case studies can be retrieved from
FINRA's public Fixed Income Data Center, while research-scale transaction data
requires the delayed Academic Corporate Bond TRACE product and its separate
institutional agreement. The aggregate Project 22 workflow does not require
that paid product.

- https://www.finra.org/finra-data/fixed-income/about-cna-trade
- https://www.finra.org/industry/trace-historic-academic-data

## Incremental update

```powershell
$env:FINRA_CLIENT_ID = "your-client-id"
$env:FINRA_CLIENT_SECRET = "your-client-secret"
.\.venv\Scripts\python.exe data\finra_credit\update.py
```

The updater fetches only a newly listed monthly ZIP, requests a 14-day overlap
from each API table, deduplicates source rows, and rebuilds the four final
Parquets from cached history. The free Public Credential is capped at 10 GB per
month; these aggregate tables use only a small fraction of that allowance.
