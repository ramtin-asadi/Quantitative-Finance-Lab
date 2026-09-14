import json
import re
from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from .documents import normalize_text, text_hash
from .schemas import AnalysisTarget, Claim, TrainingExample, utc


def numbers(text: str) -> set[str]:
    found = re.findall(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*%|\s*bps?)?", text)
    result = set()
    for item in found:
        value = re.sub(r"\s*(%|bps?)$", "", item).replace(",", "")
        unit = "percent" if "%" in item else "bp" if re.search(r"bp", item) else "number"
        try:
            result.add(f"{Decimal(value).normalize()}:{unit}")
        except InvalidOperation:
            continue
    return result


def financial_numbers(text: str) -> set[str]:
    pattern = r"(?<![\w])(?P<currency>\$\s*)?(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<scale>trillion|billion|million|thousand)?\s*(?P<unit>percentage points?|percent|%|basis points?|bps?\b|dollars?|USD\b)?"
    result = set()
    multipliers = {"trillion": 10**12, "billion": 10**9, "million": 10**6, "thousand": 1000}
    for match in re.finditer(pattern, text, re.I):
        scale = (match["scale"] or "").lower()
        unit = (match["unit"] or "").lower()
        value = Decimal(match["value"].replace(",", "")) * multipliers.get(scale, 1)
        kind = ("currency" if match["currency"] or unit in {"dollar", "dollars", "usd"}
                else "pp" if unit.startswith("percentage point") else "percent" if unit in {"percent", "%"}
                else "bp" if unit.startswith(("bp", "basis point")) else "number")
        result.add(f"{value.normalize()}:{kind}")
    return result


def validate_target(target: AnalysisTarget, packet: dict, *, extract_numbers=numbers) -> list[str]:
    errors = []
    cutoff = utc(packet["as_of"])
    evidence = {row["evidence_id"]: row for row in packet["evidence"]}
    if len(evidence) != len(packet["evidence"]):
        errors.append("Duplicate evidence IDs.")
    for row in evidence.values():
        if utc(row["available_at"]) > cutoff:
            errors.append(f"Future evidence: {row['evidence_id']}")
        if row.get("text_hash") != text_hash(row["text"]):
            errors.append(f"Evidence hash mismatch: {row['evidence_id']}")
    for claim in target.claims:
        missing = set(claim.evidence_ids) - evidence.keys()
        if missing:
            errors.append("Unknown evidence IDs: " + ", ".join(sorted(missing)))
            continue
        source_text = "\n".join(evidence[key]["text"] for key in claim.evidence_ids)
        unsupported = extract_numbers(claim.statement) - extract_numbers(source_text)
        if unsupported:
            errors.append("Untraceable claim numbers: " + ", ".join(sorted(unsupported)))
    prose = " ".join([target.conclusion, target.what_changed, target.why_it_matters, *target.uncertainty])
    unsupported = extract_numbers(prose) - extract_numbers("\n".join(row["text"] for row in evidence.values()))
    if unsupported:
        errors.append("Untraceable summary numbers: " + ", ".join(sorted(unsupported)))
    return errors


def validate_example(example: TrainingExample, *, store=None) -> list[str]:
    packet = json.loads(example.messages[1]["content"])
    target = AnalysisTarget.model_validate_json(example.messages[-1]["content"])
    errors = validate_target(target, packet)
    if utc(packet["as_of"]) != example.cutoff:
        errors.append("Example cutoff differs from visible packet cutoff.")
    source_ids = {row["document_id"] for row in packet["evidence"]}
    if source_ids != set(example.source_ids) or set(example.source_hashes) != source_ids:
        errors.append("Source ID/hash inventory does not match evidence.")
    for row in packet["evidence"]:
        if row.get("tickers") and not set(row["tickers"]).issubset(example.entities):
            errors.append("Evidence entity is absent from example metadata.")
        if store is not None and row.get("source") != "structured_context":
            document = store.get(row["document_id"])
            if example.source_hashes[document.document_id] != document.text_hash:
                errors.append("Source document hash mismatch.")
            if row.get("tickers", []) != document.tickers:
                errors.append("Source ticker metadata differs from the evidence packet.")
            if normalize_text(row["text"]) not in normalize_text(document.text):
                errors.append("Evidence excerpt is not contained in the source document.")
            if document.available_at > example.cutoff or utc(row["available_at"]) != document.available_at:
                errors.append("Source document availability mismatch.")
    return errors


def split_leakage(train, validation) -> list[str]:
    errors = []
    for name, getter in [
        ("example", lambda row: {row.example_id}),
        ("group", lambda row: {row.group_id}),
        ("source", lambda row: set(row.source_ids)),
        ("source hash", lambda row: set(row.source_hashes.values())),
    ]:
        left = set().union(*(getter(row) for row in train)) if train else set()
        right = set().union(*(getter(row) for row in validation)) if validation else set()
        if left & right:
            errors.append(f"Split leakage in {name}: {len(left & right)} shared keys.")
    if train and validation and max(row.cutoff for row in train) >= min(row.cutoff for row in validation):
        errors.append("Validation is not strictly later than training.")
    return errors


def check_response(text, packet, *, stopped=True):
    try:
        target = AnalysisTarget.model_validate_json(text)
    except ValidationError as error:
        return None, [str(error)]
    errors = validate_target(target, packet, extract_numbers=financial_numbers)
    prose = " ".join([target.conclusion, target.what_changed, target.why_it_matters])
    errors.extend(direction_conflicts(prose, packet))
    if not stopped:
        errors.append("Generation ended before a normal response stop.")
    statements = [re.sub(r"\W+", "", claim.statement).casefold() for claim in target.claims]
    if len(set(statements)) != len(statements):
        errors.append("Repeated claims.")
    lookup = {row["evidence_id"]: row for row in packet["evidence"]}
    for claim in target.claims:
        cited = [lookup[key] for key in claim.evidence_ids if key in lookup]
        horizons = return_horizons(claim.statement)
        supported = return_horizons(" ".join(row["text"] for row in cited))
        if horizons - supported:
            errors.append("Untraceable claim horizon: " + str(sorted(horizons - supported))
                          + "; cited evidence horizons: " + str(sorted(supported)))
        if cited and claim.kind == "fact" and all(row["source"] == "gdelt" for row in cited):
            errors.append("News discovery alone cannot establish a reported financial fact.")
        if cited and claim.kind == "fact" and not (fact_is_quoted(claim.statement, cited) or market_return_claim_supported(claim.statement, cited)):
            errors.append("Fact claim is not a traceable excerpt of its cited evidence: " + claim.statement[:180])
    return target, errors


def fact_is_quoted(statement, evidence):
    attribution = r"^(?:the )?(?:source|release|filing|report|current market snapshot)\s+(?:reports|states|shows|describes)[^:]{0,100}:\s*"
    statement = re.sub(attribution, "", statement, count=1, flags=re.I)
    words = " ".join(re.findall(r"\w+", statement.casefold()))
    return bool(words) and any(words in " ".join(re.findall(r"\w+", row["text"].casefold())) for row in evidence)


def market_return_claim_supported(statement, evidence):
    if return_horizons(statement) - {(1, "day")}:
        return False
    observed = {}
    for row in evidence:
        if row["source"] == "structured_context" and "returns over 1 trading day:" in row["text"]:
            observed.update({ticker: float(value) for ticker, value in re.findall(
                r"\b([A-Z]{2,4})\s+([-+]?\d+\.\d+) percent", row["text"])})
    if not observed:
        return False
    mentions = list(re.finditer(r"\b(?:" + "|".join(observed) + r")\b", statement))
    if not mentions:
        return False
    allowed = set("a the with of over trading selected day days return returns returned rose fell gained declined percent and while was were to source reports".split())
    words = set(re.findall(r"[a-z]+", re.sub(r"\b(?:" + "|".join(observed) + r")\b", "", statement).lower()))
    if words - allowed:
        return False
    for i, mention in enumerate(mentions):
        end = mentions[i + 1].start() if i + 1 < len(mentions) else len(statement)
        values = re.findall(r"([-+]?\d+(?:\.\d+)?)\s*(?:percent|%)", statement[mention.end():end])
        if len(values) != 1 or abs(float(values[0]) - observed[mention[0]]) > 0.000001:
            return False
    return not direction_conflicts(statement, {"evidence": evidence})


def direction_conflicts(text, packet):
    aliases = {"spy": "SPY", "qqq": "QQQ", "iwm": "IWM", "hyg": "HYG", "lqd": "LQD",
               "ief": "IEF", "tlt": "TLT", "gld": "GLD", "dbc": "DBC", "uup": "UUP",
               "high-yield bonds": "HYG", "gold": "GLD"}
    returns = {}
    for row in packet["evidence"]:
        if row["source"] == "structured_context" and "returns over 1 trading day:" in row["text"]:
            lead = row["text"].split("returns over 1 trading day:", 1)[1].split("\n", 1)[0]
            returns.update({ticker: float(value) for ticker, value in
                            re.findall(r"\b([A-Z]{2,4})\s+([-+]?\d+\.\d+) percent", lead)})
    errors = []
    for clause in re.split(r"[.;]|\b(?:while|but|whereas|with)\b", text.lower()):
        horizons = return_horizons(clause)
        if horizons and horizons != {(1, "day")}:
            continue
        previous_end = 0
        for match in re.finditer(r"\b(rose|gained|rallied|fell|declined|dropped)\b", clause):
            subject = clause[previous_end:match.start()]
            previous_end = match.end()
            sign = 1 if match[0] in {"rose", "gained", "rallied"} else -1
            for alias, ticker in aliases.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", subject) and ticker in returns:
                    if returns[ticker] * sign < 0:
                        errors.append(f"Summary direction conflicts with {ticker}'s one-day return ({returns[ticker]:+.2f} percent).")
    return list(dict.fromkeys(errors))


def return_horizons(text):
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
    pattern = r"\b(?:over|past|last)\s+(?:the\s+)?(\d+|" + "|".join(words) + r")\s+(?:trading\s+)?(day|week|month|year)s?\b"
    return {(int(value) if value.isdigit() else words[value], unit) for value, unit in re.findall(pattern, text.lower())}


def supported_subset(target, packet):
    if target is None:
        return None
    kept = []
    for claim in target.claims:
        candidate = AnalysisTarget(conclusion=claim.statement, materiality="uncertain", claims=[claim],
            what_changed="Only individually checked claims are retained.",
            why_it_matters="The complete analysis could not be validated.",
            uncertainty=["Unsupported or incomplete portions of the generated answer were removed."])
        _, errors = check_response(candidate.model_dump_json(), packet)
        cited = [row for row in packet["evidence"] if row["evidence_id"] in claim.evidence_ids]
        literal = fact_is_quoted(claim.statement, cited) or market_return_claim_supported(claim.statement, cited)
        if not errors and literal and claim.statement not in [item.statement for item in kept]:
            kept.append(claim)
    if not kept:
        candidates = [row for row in packet["evidence"] if row["source"] == "validated_event"]
        candidates = [packet["evidence"][0], *candidates] if packet["evidence"] else []
        for row in candidates[:3]:
            if utc(row["available_at"]) > utc(packet["as_of"]) or row["text_hash"] != text_hash(row["text"]):
                continue
            excerpt = " ".join(row["text"].split("\n", 1)[0].split()[:100])
            if excerpt and row["source"] != "gdelt":
                kept.append(Claim(statement=excerpt, evidence_ids=[row["evidence_id"]], kind="fact"))
        if not kept:
            return None
    return AnalysisTarget(conclusion="The generated interpretation did not pass all checks; traceable excerpts are retained below.",
        materiality="uncertain", claims=kept, what_changed="The retained claims are listed below.",
        why_it_matters="The complete generated interpretation did not pass validation.",
        uncertainty=["Unsupported portions were removed; review the original evidence before drawing a conclusion."])

