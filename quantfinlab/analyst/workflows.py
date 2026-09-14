"""Explicit composition of evidence preparation and checked generation."""

import logging
from datetime import timedelta

from .caching import load_response, response_key
from .evidence import attach_sources, context_evidence, fit_evidence, merge_evidence
from .inference import save_json
from .prompts import analysis_messages, repair_messages, response_schema
from .reports import AnalysisReport
from .retrieval import pack_evidence, retrieve_queries
from .schemas import utc
from .validation import check_response, supported_subset

logger = logging.getLogger(__name__)


def prepare_packet(config, runtime, store, index, registry, question, *, as_of, task, ticker=None,
                   plan, evidence=(), contexts=None):
    names = contexts if contexts is not None else plan.contexts
    snapshots = [registry.build(name, as_of=as_of, ticker=ticker) for name in names]
    calculated, statuses = context_evidence(snapshots, count=runtime.count, ticker=ticker)
    results = retrieve_queries(index, plan.queries, as_of=as_of, sources=plan.sources, ticker=ticker,
        since=utc(as_of) - timedelta(days=plan.lookback_days), sections=plan.sections)
    if task == "event":
        origins = {row["document_id"] for row in evidence}
        results = [row for row in results if row[0].document_id not in origins]
    selected = pack_evidence(results, as_of=as_of, budget=config.evidence_tokens, count=runtime.count,
                              max_per_document=1 if task == "event" else 2, max_items=2 if task == "event" else 6)
    retrieved = attach_sources(selected["evidence"], store)
    priority = (calculated, evidence) if task in {"sec_change", "daily", "market", "question"} else (evidence, calculated)
    combined = merge_evidence(*priority, retrieved, as_of=as_of)
    return fit_evidence(question, combined, as_of=as_of, task=task, entities=plan.entities,
                        context_status=statuses, runtime=runtime, config=config)


def analyze_packet(config, runtime, question, packet, *, task="market", use_cache=True, diagnostics=None):
    key = response_key(runtime.identity["sha256"], config.prompt_version, question, packet,
                       task=task, generation_tokens=config.generation_tokens)
    path = config.workspace / "responses" / f"{key}.json"
    cached = load_response(path) if use_cache else None
    if cached is not None:
        logger.info("Reused saved %s analysis", task)
        return cached
    messages, attempts, target, errors = analysis_messages(question, packet), [], None, []
    for attempt in range(2):
        logger.info("Generating %s analysis%s", task, " (one repair)" if attempt else "")
        output = runtime.generate(messages, schema=response_schema(packet),
                                       max_tokens=config.generation_tokens)
        target, errors = check_response(output["text"], packet, stopped=output["stopped"] and not output["truncated"])
        attempts.append({**output, "errors": errors})
        if not errors:
            break
        messages = repair_messages(messages, output["text"], errors)
    if errors:
        target = supported_subset(target, packet)
    notes = diagnostics or {}
    notes["freshness"] = [f"{row['name']}: {row['freshness']}; latest {row['latest_data_at']}"
                          for row in packet.get("context_status", []) if row["freshness"] != "current"]
    notes["freshness"] += [note for row in packet.get("context_status", []) for note in row["notes"] if "stale" in note.lower()]
    report = AnalysisReport(question, task, packet["as_of"], target, packet, errors, attempts, key,
                            diagnostics=notes)
    save_json(path, report.to_dict())
    return report


