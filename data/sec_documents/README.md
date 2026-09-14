# Selective SEC filings

Reuses the existing ticker/CIK mapping and `sec_credit.parquet` filing metadata. Targeted mode selects the latest 10-K and 10-Q, prior same-quarter/year comparables and at most six recent relevant 8-Ks. Only EX-99.1 is followed for earnings or relevant investor updates. No bulk filing-text or recursive exhibit crawl.

Official source: [Selective SEC filings](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

Install `quantfinlab[analyst]` and set `EDGAR_IDENTITY` to an identifying name and contact email. This identity is an HTTP User-Agent, not an API credential.

```powershell
python data/sec_documents/download.py --ticker NVDA
```

`update.py` refreshes discovery or issuer submission metadata and reuses unchanged raw downloads. SEC updates require explicit ticker arguments. `--limit` bounds discovery; `--historical` enables selective historical sampling. Use `--help` for the supported options.

A historical manifest is a JSON array of objects with `url`, optional `title`, `family`, `published_at` (timezone required), `report_period`, and `source_type`. CFTC uses `source_type="position_report"`. Only source-domain HTTPS URLs are fetched. Do not put inferred observation dates in `published_at`.

Raw responses and HTTP/hash provenance remain in this folder's ignored `raw/` cache. Normalized immutable `DocumentRecord` rows live in `workspace/financial_analyst/documents/source=sec/year=YYYY/`. The FTS5 index lives in `workspace/financial_analyst/index/documents.sqlite`; generated data and indexes are not committed. Requests are rate-limited and transient HTTP failures receive bounded retries. Per-document errors are written to `workspace/financial_analyst/updates/`; discovery failures raise an error and also yield a nonzero exit status.

The initial P24 corpus is selective research material. It is not an exhaustive source archive. Missing or unverifiable timestamps are explicit limitations, and historical eligibility requires verified source availability.
