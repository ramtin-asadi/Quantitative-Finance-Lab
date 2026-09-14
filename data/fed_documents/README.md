# Federal Reserve documents

FOMC statements, minutes, monetary-policy releases and speech RSS discovery. Meeting dates are never treated as publication dates. Minutes need an official release-calendar or announcement timestamp; otherwise they remain eligible only from retrieval.

Official source: [Federal Reserve documents](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).

Install `quantfinlab[analyst]` and set `EDGAR_IDENTITY` to an identifying name and contact email. This identity is an HTTP User-Agent, not an API credential.

```powershell
python data/fed_documents/download.py
```

`update.py` refreshes discovery or issuer submission metadata and reuses unchanged raw downloads. SEC updates require explicit ticker arguments. `--limit` bounds discovery; `--historical` enables selective historical sampling. Use `--help` for the supported options.

A historical manifest is a JSON array of objects with `url`, optional `title`, `family`, `published_at` (timezone required), `report_period`, and `source_type`. CFTC uses `source_type="position_report"`. Only source-domain HTTPS URLs are fetched. Do not put inferred observation dates in `published_at`.

Raw responses and HTTP/hash provenance remain in this folder's ignored `raw/` cache. Normalized immutable `DocumentRecord` rows live in `workspace/financial_analyst/documents/source=fed/year=YYYY/`. The FTS5 index lives in `workspace/financial_analyst/index/documents.sqlite`; generated data and indexes are not committed. Requests are rate-limited and transient HTTP failures receive bounded retries. Per-document errors are written to `workspace/financial_analyst/updates/`; discovery failures raise an error and also yield a nonzero exit status.

The initial P24 corpus is selective research material. It is not an exhaustive source archive. Missing or unverifiable timestamps are explicit limitations, and historical eligibility requires verified source availability.
