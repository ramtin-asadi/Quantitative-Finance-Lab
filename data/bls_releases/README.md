# BLS releases

Employment Situation, CPI, PPI and JOLTS archived releases. The release banner supplies the actual publication date and Eastern time, including daylight-saving conversion. Observation months are retained in text and never substitute for availability.

Official source: [BLS releases](https://www.bls.gov/bls/news-release/home.htm).

Install `quantfinlab[analyst]` and set `EDGAR_IDENTITY` to an identifying name and contact email. This identity is an HTTP User-Agent, not an API credential.

```powershell
python data/bls_releases/download.py --start-year 2024 --limit 20
```

`update.py` refreshes discovery or issuer submission metadata and reuses unchanged raw downloads. SEC updates require explicit ticker arguments. `--limit` bounds discovery; `--historical` enables selective historical sampling. Use `--help` for the supported options.

A historical manifest is a JSON array of objects with `url`, optional `title`, `family`, `published_at` (timezone required), `report_period`, and `source_type`. CFTC uses `source_type="position_report"`. Only source-domain HTTPS URLs are fetched. Do not put inferred observation dates in `published_at`.

Raw responses and HTTP/hash provenance remain in this folder's ignored `raw/` cache. Normalized immutable `DocumentRecord` rows live in `workspace/financial_analyst/documents/source=bls/year=YYYY/`. The FTS5 index lives in `workspace/financial_analyst/index/documents.sqlite`; generated data and indexes are not committed. Requests are rate-limited and transient HTTP failures receive bounded retries. Per-document errors are written to `workspace/financial_analyst/updates/`; discovery failures raise an error and also yield a nonzero exit status.

The initial P24 corpus is selective research material. It is not an exhaustive source archive. Missing or unverifiable timestamps are explicit limitations, and historical eligibility requires verified source availability.
