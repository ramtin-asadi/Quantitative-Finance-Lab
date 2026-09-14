# EIA energy evidence

Automatically discovers a bounded set of Weekly Petroleum Status Report highlights from the official archive. A manifest can select additional official releases. Mutable pages and reports without a verified publication timestamp become available at first retrieval, never at the petroleum observation date.

Official source: [EIA energy evidence](https://www.eia.gov/petroleum/supply/weekly/archive/).

Install `quantfinlab[analyst]` and set `EDGAR_IDENTITY` to an identifying name and contact email. This identity is an HTTP User-Agent, not an API credential.

```powershell
python data/eia_energy/download.py --limit 1
```

`update.py` refreshes discovery or issuer submission metadata and reuses unchanged raw downloads. SEC updates require explicit ticker arguments. `--limit` bounds discovery; `--historical` enables selective historical sampling. Use `--help` for the supported options.

A historical manifest is a JSON array of objects with `url`, optional `title`, `family`, `published_at` (timezone required), `report_period`, and `source_type`. CFTC uses `source_type="position_report"`. Only source-domain HTTPS URLs are fetched. Do not put inferred observation dates in `published_at`.

Raw responses and HTTP/hash provenance remain in this folder's ignored `raw/` cache. Normalized immutable `DocumentRecord` rows live in `workspace/financial_analyst/documents/source=eia/year=YYYY/`. The FTS5 index lives in `workspace/financial_analyst/index/documents.sqlite`; generated data and indexes are not committed. Requests are rate-limited and transient HTTP failures receive bounded retries. Per-document errors are written to `workspace/financial_analyst/updates/`; discovery failures raise an error and also yield a nonzero exit status.

The initial P24 corpus is selective research material. It is not an exhaustive source archive. Missing or unverifiable timestamps are explicit limitations, and historical eligibility requires verified source availability.
