# GDELT news discovery

Bounded recent discovery of headlines, URLs and metadata; no article-body crawling. Canonical URLs and similar headlines form duplicate groups. GDELT seen dates are retained as discovery metadata, not asserted as actual publication timestamps. Verify substantive facts against primary sources.

Official source: [GDELT news discovery](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/).

Install `quantfinlab[analyst]` and set `EDGAR_IDENTITY` to an identifying name and contact email. This identity is an HTTP User-Agent, not an API credential.

```powershell
python data/gdelt_news/download.py --transport gal --minutes 30 --limit 20
```

`update.py` refreshes discovery or issuer submission metadata and reuses unchanged raw downloads. SEC updates require explicit ticker arguments. `--limit` bounds discovery; `--historical` enables selective historical sampling. Use `--help` for the supported options.

A historical manifest is a JSON array of objects with `url`, optional `title`, `family`, `published_at` (timezone required), `report_period`, and `source_type`. CFTC uses `source_type="position_report"`. Only source-domain HTTPS URLs are fetched. Do not put inferred observation dates in `published_at`.

Raw responses and HTTP/hash provenance remain in this folder's ignored `raw/` cache. Normalized immutable `DocumentRecord` rows live in `workspace/financial_analyst/documents/source=gdelt/year=YYYY/`. The FTS5 index lives in `workspace/financial_analyst/index/documents.sqlite`; generated data and indexes are not committed. Requests are rate-limited and transient HTTP failures receive bounded retries. Per-document errors are written to `workspace/financial_analyst/updates/`; discovery failures raise an error and also yield a nonzero exit status.

The initial P24 corpus is selective research material. It is not an exhaustive source archive. Missing or unverifiable timestamps are explicit limitations, and historical eligibility requires verified source availability.

## Transport options

`--transport auto` first tries DOC and falls back to the official GDELT Article List (GAL) on failure. `--transport doc` keeps DOC semantics and fails if its API is unavailable. `--transport gal` directly downloads bounded recent minute files from `https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/gal/`.

Endpoint availability varies by network. When DOC times out or the custom data host is unavailable, the official Google Storage route provides a separate transport. Requests honor existing system proxy settings; the collector does not change those settings. The same collector can run in Colab when its network can reach the selected endpoint.

GAL searches headline and publisher-summary metadata only. It supports words, quoted phrases, AND, OR, parentheses and `sourcelang:english`; unsupported DOC operators raise an error rather than being silently ignored. `--minutes` is bounded to 1–120 and `--limit` to 250. Receipt metadata states the requested window, actual files scanned, matches and any DOC error. Missing minute files are common because publication is clustered. An empty successful result means no matching records in the scanned files, not no news in the world.

Repeated collection does not insert duplicate documents or chunks. The first-observed timestamp is preserved even if the same headline reappears in a later batch. Publisher/GDELT date fields remain metadata and do not establish historical availability. Only the headline and publisher link become retrievable text; article bodies are not crawled.

References: [GAL schema and timing](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/) and [official Google Storage download example](https://blog.gdeltproject.org/using-gemini-2-5-as-a-persona-based-news-recommender-service-summarizing-trends-from-a-day-of-global-tariff-trade-war-news/).
