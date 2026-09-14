"""Explicit evidence records, numerical context, and model-token budgets."""

import json
import re

from .documents import bounded_excerpt, text_hash
from .prompts import analysis_messages
from .schemas import utc


def evidence_row(text, *, key, available_at, source, title, document_id=None, tickers=(), entities=(), url=""):
    return {"evidence_id": key, "document_id": document_id or key,
            "available_at": utc(available_at).isoformat(), "source": source, "title": title,
            "tickers": list(tickers), "entities": list(entities), "text": text,
            "text_hash": text_hash(text), "source_url": url}


def context_text(snapshot):
    summary = snapshot_summary(snapshot)
    lines = [summary] if summary else []
    lines.append(f"{snapshot.name}; observation {snapshot.latest_data_at}; {snapshot.freshness}.")
    percent_fields = {"roa", "roe", "accruals_assets", "debt_assets", "net_debt_assets", "capex_revenue",
                      "dividend_payout", "fcf_yield", "vol_change_5d", "vol_change_21d"}

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, [*path, key])
        elif isinstance(value, (float, int)):
            field = path[-1]
            label = ".".join(path).replace("_", " ")
            horizon = re.fullmatch(r"return_(\d+)d", field)
            if horizon:
                label = ".".join(path[:-1]) + f" return over {horizon[1]} trading days"
            ratio = (field.startswith(("return_", "realized_vol_", "spy_realized_vol_", "drawdown_"))
                     or "margin_ttm" in field or field in percent_fields
                     or "growth" in path or "proxy_spread_21d" in path
                     or field in {"alpha_daily", "residual_return_21d", "stress_breadth"}
                     or snapshot.name == "cross_asset" and len(path) == 1 and not field.startswith("rolling_avg_corr")
                     or "quarters" in path and any("margin" in item for item in path))
            if "percentile" in field:
                formatted = f"{value * 100:.2f} percentile"
            elif "capital_allocation" in path and field.endswith("_ttm"):
                formatted = f"${value / 1e9:.3f} billion"
            elif path[0] in {"UNRATE", "FEDFUNDS"} and field in {"value", "previous_value"}:
                formatted = f"{value:.2f} percent"
            elif path[0] in {"UNRATE", "FEDFUNDS"} and field == "change":
                formatted = f"{value:.2f} percentage points"
            elif field.endswith("_pp"):
                formatted = f"{value:.2f} percentage points"
            elif field.endswith("_bp"):
                formatted = f"{value:.2f} bp"
            elif "percent" in field:
                formatted = f"{value:.2f} percent"
            elif ratio:
                formatted = f"{value * 100:.2f} percent"
            else:
                formatted = f"{value:.2f}"
            lines.append(f"{label}: {formatted}")
        elif value is not None:
            lines.append(".".join(path).replace("_", " ") + ": " + str(value))

    visit(snapshot.measures, [])
    return "\n".join(lines)


def snapshot_summary(snapshot):
    """Put the most useful calculated comparison before the detailed measures."""
    values = snapshot.measures
    date = str(snapshot.latest_data_at.date()) if snapshot.latest_data_at else "unknown date"
    if snapshot.name == "market":
        returns = [f"{asset} {row['return_1d'] * 100:+.2f} percent" for asset, row in values.items()
                   if isinstance(row, dict) and "return_1d" in row]
        return f"Market close snapshot for {date}; returns over 1 trading day: " + "; ".join(returns) + "." if returns else ""
    if snapshot.name == "fundamentals" and values.get("metrics"):
        metrics, growth = values["metrics"], values.get("growth", {})
        quarters = values.get("margin_trends", {}).get("operating_margin", {}).get("quarters", {})
        date = max(quarters) if quarters else date
        parts = [f"{values.get('ticker', 'Issuer')} financial results through {date}"]
        if "cfo_net_income" in metrics:
            ratio = metrics["cfo_net_income"]
            relation = "below" if ratio < 1 else "above" if ratio > 1 else "equal to"
            parts.append(f"Operating cash flow was {ratio:.2f} times net income over the trailing 12 months, so cash generation was {relation} reported earnings")
        if {"cfo_qoq", "net_income_qoq"}.issubset(growth):
            cash_direction = "fell" if growth["cfo_qoq"] < 0 else "rose"
            income_direction = "fell" if growth["net_income_qoq"] < 0 else "rose"
            parts.append(f"In the latest quarter, operating cash flow {cash_direction} {abs(growth['cfo_qoq']) * 100:.2f} percent from the prior quarter while net income {income_direction} {abs(growth['net_income_qoq']) * 100:.2f} percent")
        for name, label in [("operating_margin_ttm", "operating margin"), ("net_margin_ttm", "net margin"), ("fcf_margin_ttm", "free cash flow margin")]:
            if name in metrics:
                parts.append(f"trailing 12-month {label} {metrics[name] * 100:.2f} percent")
        return ". ".join(parts) + "."
    if snapshot.name == "cross_asset" and "breadth_21" in values:
        parts = [f"Across the selected ETF universe, {values['breadth_21'] * 100:.2f} percent had positive returns over 21 trading days"]
        if "breadth_63" in values:
            parts.append(f"{values['breadth_63'] * 100:.2f} percent had positive returns over 63 trading days")
        for pair, row in sorted(values.get("relative_moves", {}).items(), key=lambda item: abs(item[1].get("gap_z", 0)), reverse=True)[:3]:
            parts.append(f"{pair.replace('_minus_', ' minus ')} return gap over 1 trading day {row['return_gap_1d_pp']:+.2f} percentage points")
        return ". ".join(parts) + "."
    return ""


def context_evidence(snapshots, *, count, budget=3500, ticker=None):
    rows, statuses = [], []
    for snapshot in snapshots:
        statuses.append({"name": snapshot.name, "freshness": snapshot.freshness,
                         "latest_data_at": snapshot.latest_data_at.isoformat() if snapshot.latest_data_at else None,
                         "notes": list(snapshot.notes)})
        if not snapshot.measures:
            continue
        if snapshot.freshness == "stale":
            statuses[-1]["notes"].append("Stale measures are omitted from the answer prompt.")
            continue
        text = bounded_excerpt(context_text(snapshot), count, budget=900)
        row = evidence_row(text, key="context-" + snapshot.name + "-" + text_hash(text)[:16],
            available_at=snapshot.available_at or snapshot.latest_data_at, source="structured_context",
            title=snapshot.name.replace("_", " ").title() + " · calculated Quantfinlab context",
            tickers=[ticker] if ticker and snapshot.name in {"fundamentals", "factors", "credit", "risk"} else [])
        if count(json.dumps([*rows, row])) <= budget:
            rows.append(row)
        else:
            statuses[-1]["notes"].append("Omitted from prompt because of the structured-context budget.")
    return rows, statuses


def attach_sources(evidence, store):
    rows = []
    for item in evidence:
        document = store.get(item["document_id"])
        rows.append({**item, "title": document.title, "source_url": document.source_url})
    return rows


def merge_evidence(*collections, as_of):
    rows, seen = [], set()
    for collection in collections:
        for row in collection:
            if row["evidence_id"] in seen:
                continue
            if utc(row["available_at"]) > utc(as_of):
                raise ValueError("Evidence was published after the cutoff.")
            if row["text_hash"] != text_hash(row["text"]):
                raise ValueError("Evidence hash does not match its text.")
            rows.append(dict(row))
            seen.add(row["evidence_id"])
    return rows


def fit_evidence(question, evidence, *, as_of, task, entities, context_status, runtime, config):
    rows = list(evidence)
    packet = {"as_of": utc(as_of).isoformat(), "task": task, "entities": list(entities),
              "evidence": rows, "context_status": list(context_status)}
    prompt_budget = min(config.evidence_tokens, config.context_tokens - 2 * config.generation_tokens - 1536)
    while rows:
        prompt = runtime.apply_template(analysis_messages(question, packet))
        prompt_tokens = runtime.count(prompt)
        if prompt_tokens <= prompt_budget:
            return packet, {"prompt_tokens": prompt_tokens, "prompt_budget": prompt_budget,
                            "generation_tokens": config.generation_tokens,
                            "remaining_tokens": config.context_tokens - prompt_tokens - config.generation_tokens}
        rows.pop()
    raise ValueError("No usable evidence fits this question. Refresh sources or narrow the request.")

