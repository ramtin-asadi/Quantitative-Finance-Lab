import gzip
import json
from datetime import datetime, timezone

import pytest
import requests

from quantfinlab.analyst.corpus import macro_target, sentence, topic_interpretation
from quantfinlab.analyst.documents import DocumentStore, text_hash
from quantfinlab.analyst.news import discover_news, headline_query
from quantfinlab.analyst.schemas import DocumentRecord
from quantfinlab.analyst.sources import SourceClient, discover_fed, ingest_release, release_time


def test_gal_query_scope_is_explicit():
    match = headline_query('(earnings OR "Federal Reserve") sourcelang:english')
    assert match({'title': 'Federal Reserve statement', 'lang': 'en'})
    assert match({'title': 'Quarterly results', 'desc': 'Higher earnings reported', 'lang': 'en'})
    assert not match({'title': 'Learning about oil', 'lang': 'en'})
    assert not match({'title': 'Earnings report', 'lang': 'fr'})
    assert not headline_query('oil AND earnings')({'title': 'Oil falls'})
    with pytest.raises(ValueError, match='operators'):
        headline_query('tone:>5')


def test_gal_fallback_preserves_first_observation(tmp_path):
    class Client:
        root = tmp_path

        def fetch(self, url, **kwargs):
            if 'api.gdeltproject.org' in url:
                raise requests.Timeout('API unavailable')
            content = gzip.compress(json.dumps({'title': 'Oil earnings report', 'url': 'https://example.org/news?utm_source=x',
                'date': '2025-01-01T00:00:00Z', 'lang': 'en'}).encode())
            path = tmp_path / 'data/gdelt_news/raw/batch.raw'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return content, {'retrieved_at': datetime.now(timezone.utc).isoformat()}, path

    diagnostics = {}
    first = discover_news(Client(), query='oil sourcelang:english', limit=1, diagnostics=diagnostics)
    assert diagnostics['transport'] == 'gal' and diagnostics['doc_error']
    assert first[0].published_at is None and not first[0].metadata['historical_eligible']
    assert first[0].available_at.year > 2025
    store = DocumentStore(tmp_path / 'workspace/financial_analyst/documents')
    store.put(first)
    second = discover_news(Client(), query='oil sourcelang:english', limit=1)
    assert second[0].available_at == first[0].available_at
    assert store.put(second) == 0


def test_official_bucket_allowlist(tmp_path):
    client = SourceClient(tmp_path, 'gdelt', identity='Tester test@example.org')
    with pytest.raises(ValueError, match='official GDELT'):
        client.fetch('https://storage.googleapis.com/unrelated-bucket/file.json')


def test_minutes_use_publication_date_not_meeting_date(tmp_path):
    url = 'https://www.federalreserve.gov/monetarypolicy/fomcminutes20250129.htm'
    calendar = b'<div><a href="/monetarypolicy/fomcminutes20250129.htm">HTML</a> Released February 19, 2025</div>'

    class Client:
        source = 'fed'
        root = tmp_path

        def fetch(self, link, **kwargs):
            raw = tmp_path / 'release.raw'
            content = calendar if 'calendars' in link else b'<main>Meeting January 29, 2025 at 9:00 a.m. The participants discussed inflation.</main>'
            return content, {'retrieved_at': '2026-01-01T00:00:00Z', 'hash': text_hash(content.hex())}, raw

    record = next(r for r in discover_fed(Client(), start=2025, end=2025) if r['url'] == url)
    document = ingest_release(Client(), url, metadata={'publication_date': record['publication_date']})
    assert document.available_at.isoformat() == '2025-02-20T04:59:59+00:00'
    assert document.published_at is None
    assert release_time('January 29, 2025 at 9:00 a.m.', source='fed', url=url)[0] is None


def test_semantic_regressions():
    assert topic_interpretation('dilution', 'In May 2024, the company repurchased shares worth $10 million.')[1] == 'medium'
    assert topic_interpretation('debt', 'The estimated fair value of debt was $8 billion.')[0].startswith('The disclosure concerns the market value')
    assert topic_interpretation('debt', 'The company holds debt in its investment portfolio.')[1] == 'low'
    assert 'margin percentage' in topic_interpretation('margins', 'Gross margin increased $5 billion.')[0]
    assert 'Jerome H. Powell' in sentence('Voting for the action were Jerome H. Powell and other members of the Committee.')
    text = 'Total nonfarm payroll employment edged down by 92,000 in February.'
    document = DocumentRecord(document_id='bls-test', source='bls', source_type='official_release', title='Employment',
        available_at='2026-03-01T13:30:00Z', retrieved_at='2026-03-02T00:00:00Z', source_url='https://www.bls.gov/test',
        raw_path='test.raw', text=text, text_hash=text_hash(text), duplicate_group='test', metadata={'family': 'empsit'})
    target = macro_target(document, [{'text': text, 'evidence_id': 'bls-test'}])
    assert 'contraction' in target.conclusion
    growth = 'Total nonfarm payroll employment rose by 559,000 in May, and the unemployment rate declined to 5.8 percent.'
    assert 'hiring growth' in macro_target(document, [{'text': growth, 'evidence_id': 'bls-test'}]).conclusion
