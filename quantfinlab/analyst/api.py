"""Thin, stateful facade over the public financial-analysis operations."""

import json
import logging
from datetime import datetime, timedelta, timezone

from .config import AnalystConfig
from .context import ContextRegistry, load_market_prices
from .documents import DocumentStore, document_coverage
from .events import (
    EventStore,
    candidate_events,
    event_candidate,
    event_evidence,
    event_from_report,
    events_to_evidence,
)
from .inference import LlamaRuntime, save_json
from .macro import reaction_evidence, release_evidence, release_reaction, select_macro_release
from .reports import DailyBrief
from .retrieval import DocumentIndex, index_new_documents, retrieve_queries
from .routing import resolve_plan
from .schemas import QueryPlan, utc
from .sec import company_filings, compare_sections, select_change_evidence
from .workflows import analyze_packet, prepare_packet

logger = logging.getLogger(__name__)


class FinancialAnalyst:
    def __init__(self, config, runtime, *, as_of=None, identity=None):
        self.config, self.runtime = config, runtime
        self.explicit_cutoff = as_of is not None
        cutoff_path = config.workspace / "current_cutoff.json"
        saved_cutoff = json.loads(cutoff_path.read_text())["as_of"] if cutoff_path.exists() else None
        self.as_of = utc(as_of or saved_cutoff) if as_of or saved_cutoff else datetime.now(timezone.utc).replace(microsecond=0)
        if not self.explicit_cutoff and saved_cutoff is None:
            save_json(cutoff_path, {"as_of": self.as_of.isoformat()})
        self.identity = identity
        self.store = DocumentStore(config.workspace / "documents")
        self.index = DocumentIndex(config.workspace / "index/documents.sqlite")
        self.contexts = ContextRegistry(config.root)
        self.event_store = EventStore(config.workspace / "events/events.sqlite")
        self.last_packet = None

    @classmethod
    def from_repo(cls, path=".", *, as_of=None, identity=None, binary=None, runtime=None, start=True, port=8089):
        config = AnalystConfig.from_repo(path)
        runtime = runtime or LlamaRuntime.from_repo(path, binary=binary, port=port)
        analyst = cls(config, runtime, as_of=as_of, identity=identity)
        indexed, missing_dates = analyst.index.connection.execute("SELECT COUNT(*), SUM(content_date IS NULL) FROM chunks").fetchone()
        if not indexed or missing_dates:
            analyst.index_documents()
        if start and getattr(runtime, "process", None) is None:
            records = analyst.store.records(as_of=analyst.as_of)
            passages = []
            for document in records:
                passages.append(document.text[:24000])
                if sum(map(len, passages)) > 90000:
                    break
            text = "\n\n".join(passages) or "Revenue growth, operating margins, cash generation and leverage are distinct financial measures. " * 2000
            runtime.calibrate(text)
        return analyst

    def close(self):
        self.index.close()
        self.event_store.close()
        self.runtime.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def coverage(self):
        return document_coverage(self.store.records(as_of=self.as_of))

    def index_documents(self, documents=None):
        records = self.store.records(as_of=self.as_of) if documents is None else documents
        return index_new_documents(self.index, records)

    def plan(self, question, *, ticker=None, use_model=None):
        return resolve_plan(self.config, self.runtime, question, as_of=self.as_of, ticker=ticker, use_model=use_model)

    def retrieve(self, question, *, plan=None, ticker=None):
        plan = plan or self.plan(question, ticker=ticker)
        return retrieve_queries(self.index, plan.queries, as_of=self.as_of, sources=plan.sources, ticker=ticker,
            since=self.as_of - timedelta(days=plan.lookback_days), sections=plan.sections)

    def packet(self, question, *, task="market", ticker=None, plan=None, evidence=(), contexts=None):
        plan = plan or self.plan(question, ticker=ticker)
        packet, budget = prepare_packet(self.config, self.runtime, self.store, self.index, self.contexts,
            question, as_of=self.as_of, task=task, ticker=ticker, plan=plan, evidence=evidence, contexts=contexts)
        self.last_packet = packet
        logger.info("Packed %s evidence items, %s prompt tokens", len(packet["evidence"]), budget["prompt_tokens"])
        return packet

    def analyze(self, question, packet, *, task="market", use_cache=True, diagnostics=None):
        if utc(packet["as_of"]) != self.as_of:
            raise ValueError("Analysis packet and analyst cutoffs differ.")
        return analyze_packet(self.config, self.runtime, question, packet, task=task,
                              use_cache=use_cache, diagnostics=diagnostics)

    def ask(self, question, *, ticker=None, use_cache=True):
        plan = self.plan(question, ticker=ticker)
        ticker = ticker or (plan.entities[0] if len(plan.entities) == 1 else None)
        packet = self.packet(question, ticker=ticker, plan=plan)
        return self.analyze(question, packet, task="question", use_cache=use_cache,
                            diagnostics={"plan": plan.model_dump(mode="json")})

    def market(self, question="Which market signals agree, and which contradict a simple risk-on interpretation?"):
        names = ["market", "risk", "cross_asset", "rates", "volatility", "financial_conditions", "credit", "factors", "macro"]
        packet = self.packet(question, task="market", contexts=names)
        return self.analyze(question, packet, task="market")

    def company_filings(self, ticker, *, refresh=False):
        return company_filings(self.config.root, self.store, ticker, as_of=self.as_of,
                                identity=self.identity, refresh=refresh, index=self.index)

    def company(self, ticker, question=None, *, refresh=False):
        previous, current = self.company_filings(ticker, refresh=refresh)
        question = question or f"What material or non-obvious changes in {ticker}'s latest filing deserve attention?"
        changes = compare_sections(previous, current)
        evidence, selected = select_change_evidence(previous, current, changes, ticker=ticker, count=self.runtime.count, question=question)
        packet = self.packet(question, task="sec_change", ticker=ticker, evidence=evidence,
                              contexts=["fundamentals", "risk", "factors", "market", "credit"])
        return self.analyze(question, packet, task="sec_change", diagnostics={"previous": previous.document_id,
            "current": current.document_id, "selected_changes": selected, "change_candidates": len(changes)})

    def macro_release(self, family="cpi", question=None):
        current, previous = select_macro_release(self.store, family, as_of=self.as_of)
        question = question or f"What changed in the latest {family.upper()} release, and what matters for policy and markets?"
        evidence = release_evidence(current, previous, family=family)
        reaction = release_reaction(load_market_prices(self.config.root, self.as_of),
                                    published_at=current.published_at, as_of=self.as_of)
        evidence += reaction_evidence(reaction, current)
        packet = self.packet(question, task="macro", evidence=evidence,
                              contexts=["macro", "rates", "financial_conditions", "market"])
        return self.analyze(question, packet, task="macro", diagnostics={"current": current.document_id,
            "previous": previous.document_id if previous else None, "consensus_supplied": False,
            "daily_release_window": reaction})

    def event_candidates(self, *, days=7):
        return candidate_events(self.store.records(as_of=self.as_of), as_of=self.as_of, days=days)

    def analyze_event(self, document_id):
        document = self.store.get(document_id)
        candidate = event_candidate(document)
        if candidate is None or document.available_at > self.as_of:
            raise ValueError("Document is not an eligible event candidate.")
        question = f"What happened in this {candidate['family']} event, what changed and why does it matter?"
        evidence = event_evidence(document)
        plan = QueryPlan(entities=document.tickers, sources=[document.source], contexts=[], queries=[document.title], lookback_days=14)
        packet = self.packet(question, task="event", plan=plan, evidence=evidence, contexts=[])
        report = self.analyze(question, packet, task="event")
        event = event_from_report(report, document, model_sha=self.runtime.identity["sha256"], prompt_version=self.config.prompt_version)
        if event is not None:
            self.event_store.put(event)
        return report, event

    def events(self, *, days=7, limit=4, analyze=True):
        if analyze:
            for candidate in self.event_candidates(days=days)[:limit]:
                self.analyze_event(candidate["document_id"])
        return self.event_store.list(as_of=self.as_of, since=self.as_of - timedelta(days=days),
                                     model=self.runtime.identity["sha256"], prompt=self.config.prompt_version)[:limit]

    def daily_brief(self, *, days=7, limit=4):
        board = self.events(days=days, limit=limit)
        evidence = events_to_evidence(board, self.config.workspace / "responses")
        question = ("Which market moves and recent events matter in this daily brief, and which interpretations remain uncertain? "
                    "Compare equities, credit and the supplied event facts. Keep the observation dates distinct.")
        packet = self.packet(question, task="daily", evidence=evidence,
            contexts=["market", "cross_asset", "rates", "macro"])
        report = self.analyze(question, packet, task="daily", diagnostics={"events": len(board), "event_window_days": days})
        sections = {"summary": report.analysis.conclusion, "changes": report.analysis.what_changed,
                    "implications": report.analysis.why_it_matters} if report.analysis else {}
        return DailyBrief(report, board, sections)

    def update(self, *, sources=("fed", "bls", "bea", "eia", "cftc", "gdelt"), tickers=(), structured=(), limit=4):
        from .updates import update_sources

        result = update_sources(self.config, identity=self.identity, sources=sources, tickers=tickers,
                                structured=structured, limit=limit)
        if not self.explicit_cutoff and any(row["status"] == "updated" for row in result):
            self.as_of = utc(json.loads((self.config.workspace / "current_cutoff.json").read_text())["as_of"])
        return result

