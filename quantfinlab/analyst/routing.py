import json
import logging
import re

from pydantic import ValidationError

from .documents import text_hash
from .inference import save_json
from .schemas import QueryPlan, utc
from .sec import resolve_ticker

logger = logging.getLogger(__name__)


def query_plan(question: str, *, tickers=()):
    text = question.lower()
    sources, contexts, terms = set(), set(), []
    rules = [
        (("10-q", "10-k", "guidance", "earnings", "fundamental", "liquidity"), ["sec"], ["fundamentals", "risk", "credit"]),
        (("cpi", "payroll", "employment", "inflation", "jolts", "ppi"), ["bls"], ["macro", "rates", "financial_conditions"]),
        (("gdp", "pce", "income", "outlays"), ["bea"], ["macro", "rates"]),
        (("fed", "fomc", "policy", "monetary"), ["fed"], ["rates", "macro"]),
        (("oil", "energy", "petroleum"), ["eia"], ["cross_asset"]),
        (("cot", "positioning"), ["cftc"], ["cross_asset"]),
        (("yield", "treasury", "curve", "rates"), ["fed"], ["rates"]),
        (("credit", "distress", "debt"), ["sec", "fed"], ["credit", "fundamentals"]),
        (("market", "risk", "cross-asset", "daily"), ["fed", "bls", "bea", "gdelt"], ["market", "risk", "cross_asset", "volatility", "factors"]),
    ]
    for triggers, source_names, builder_names in rules:
        if any(term in text for term in triggers):
            sources.update(source_names)
            contexts.update(builder_names)
            terms.extend(term for term in triggers if term in text)
    entities = sorted(set(t.upper() for t in tickers if re.search(r"\b" + re.escape(t) + r"\b", question, re.I)))
    if entities:
        sources.add("sec")
        contexts.update(["fundamentals", "risk", "factors", "credit"])
    return {"entities": entities, "sources": sorted(sources), "contexts": sorted(contexts),
            "queries": [question, *terms], "requires_model_plan": not bool(sources)}


def resolve_plan(config, runtime, question, *, as_of, ticker=None, use_model=None):
    if ticker:
        resolve_ticker(config.root, ticker, as_of=utc(as_of))
    candidates = [ticker] if ticker else re.findall(r"\b[A-Z][A-Z.\-]{0,5}\b", question)
    known = []
    for candidate in candidates:
        try:
            resolve_ticker(config.root, candidate, as_of=utc(as_of))
        except (ValueError, FileNotFoundError):
            continue
        known.append(candidate)
    routed = query_plan(question, tickers=known)
    if ticker:
        routed["sources"] = sorted(set([*routed["sources"], "sec"]))
        routed["contexts"] = sorted(set([*routed["contexts"], "fundamentals", "risk", "factors", "credit"]))
        routed["requires_model_plan"] = False
    entities = sorted(set(([ticker.upper()] if ticker else []) + routed["entities"]))
    priority = ["fundamentals", "market", "cross_asset", "rates", "macro", "risk", "volatility", "credit", "financial_conditions", "factors"]
    context_names = routed["contexts"] or ["market", "cross_asset", "rates"]
    fallback = QueryPlan(entities=entities, sources=routed["sources"],
        contexts=sorted(context_names, key=priority.index),
        queries=routed["queries"][:5])
    if use_model is False or (use_model is None and not routed["requires_model_plan"]):
        return fallback
    messages = [{"role": "system", "content": "Plan lexical financial evidence retrieval. Return only JSON matching the schema. "
                 "Use only the supplied entity identifiers. Do not answer the question or invent facts. "
                 "Return entities, sources, contexts, queries, sections and lookback_days. Retain the supplied entities. "
                 "Choose relevant sources and context builders, and two short keyword queries. The lexical plan is a useful starting point."},
                {"role": "user", "content": json.dumps({"question": question, "entities": entities,
                  "sources": ["sec", "fed", "bls", "bea", "eia", "cftc", "gdelt"],
                  "contexts": list(QueryPlan.model_json_schema()["properties"]["contexts"]["items"]["enum"]),
                  "lexical_plan": fallback.model_dump(), "as_of": utc(as_of).isoformat()})}]
    key = text_hash(json.dumps({"model": runtime.identity["sha256"], "messages": messages,
                               "schema": QueryPlan.model_json_schema(), "version": config.prompt_version}, sort_keys=True))
    path = config.workspace / "plans" / f"{key}.json"
    if path.exists():
        return QueryPlan.model_validate_json(path.read_text())
    output = runtime.generate(messages, schema=QueryPlan.model_json_schema(), max_tokens=500)
    try:
        planned = QueryPlan.model_validate_json(output["text"])
        if set(planned.entities) != set(entities) or not planned.sources or not planned.contexts or not output["stopped"]:
            raise ValueError("Planner omitted required routing, changed entities or did not finish.")
    except (ValidationError, ValueError) as error:
        logger.warning("Using lexical routing after invalid model plan: %s", error)
        planned = fallback
    save_json(path, planned.model_dump(mode="json"))
    return planned

