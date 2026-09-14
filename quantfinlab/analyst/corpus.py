import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .context import close_time
from .documents import DocumentStore, text_hash
from .prompts import training_messages
from .schemas import AnalysisTarget, Claim, TrainingExample
from .training import anchor_targets, write_examples
from .validation import validate_example

disclosure_topics = {
    "liquidity": (r"\b(?:liquidity|cash flows?|operating cash|cash and cash equivalents)\b",
        "Cash availability and cash generation determine how much of the operating plan can be funded internally.",
        "Separate recurring operating cash from working-capital timing and external financing before judging funding pressure."),
    "debt": (r"\b(?:debt|borrowings|credit facilit|covenant|maturities)\b",
        "Debt terms and funding access affect refinancing flexibility and the cash burden ahead of equity holders.",
        "The passage alone does not establish default risk; maturity timing, covenant headroom and available liquidity matter."),
    "capital_spending": (r"\b(?:capital expenditures?|capital spending|purchase commitments|capital investment)\b",
        "Investment commitments can absorb cash before the associated revenue arrives, making funding capacity relevant to the growth narrative.",
        "Spending is not evidence of attractive returns by itself; utilization, timing and incremental cash generation remain uncertain."),
    "concentration": (r"\b(?:customer concentration|major customers?|significant customers?|single customer|customers? accounted)\b",
        "Customer concentration can make reported growth sensitive to a small number of purchasing decisions.",
        "Concentration does not imply that a customer will leave; the disclosure needs to be read with contract and demand evidence."),
    "margins": (r"\b(?:gross margin|operating margin|gross profit|profitability)\b",
        "Margin information helps distinguish revenue growth from profitable growth and changes the amount of revenue that reaches earnings.",
        "Mix, pricing, utilization and exceptional costs can move margins differently; the excerpt does not isolate every contribution."),
    "restructuring": (r"\b(?:restructuring|impairment|reorganization|severance)\b",
        "Restructuring or impairment disclosures can separate current earnings charges from future cash needs and operating adjustments.",
        "An accounting charge and a cash outflow need not occur together; expected savings are not demonstrated realized savings."),
    "risk": (r"\b(?:export controls?|contingenc|litigation|uncertainty|supply constraints?|cybersecurity)\b",
        "The risk language identifies an exposure that can affect operations, cash flows or management's room to respond.",
        "A disclosed risk is not proof that a loss has occurred. Materiality depends on realization, magnitude and mitigation."),
    "dilution": (r"\b(?:repurchases?|stock-based compensation|share-based compensation|dilution|diluted shares)\b",
        "Share issuance and repurchases affect how company-level cash generation translates into value per share.",
        "A repurchase authorization is not an executed purchase, and gross buybacks alone do not establish a shrinking diluted share count."),
    "segments": (r"\b(?:segment revenue|segment operating|reportable segments?|data center|operating segments?)\b",
        "Segment detail can reveal where the aggregate result is being earned and whether the business mix is changing.",
        "A segment movement does not establish a durable company-wide trend; segment definitions and comparable periods must be checked."),
}


def paragraphs(document):
    return [p.strip() for p in document.text.split("\n\n") if 100 <= len(p.strip()) <= 2200
            and p.count("|") < 8 and not re.search(r"forward-looking statements|safe harbor|table of contents|copyright|media contact|following table|table below|non-GAAP measures include|to supplement.*statements", p, re.I)]


def sentence(text: str, limit=520) -> str:
    compact = re.sub(r"\s+", " ", text).strip().lstrip("•| ")
    sentences = re.split(r"(?<!U\.S\.)(?<!Inc\.)(?<!Corp\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+(?=[A-Z])", compact)
    eligible = [part for part in sentences if 45 <= len(part) <= limit]
    return eligible[0] if eligible else compact if len(compact) <= limit else ""


def salient_sentence(text: str, pattern: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip().lstrip("•| ")
    sentences = re.split(r"(?<!U\.S\.)(?<!Inc\.)(?<!Corp\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+(?=[A-Z])", compact)
    selected = [s for s in sentences if 45 <= len(s) <= 680 and re.search(pattern,s,re.I)]
    if not selected:
        return ""
    def score(value):
        return (bool(re.search(r"\d", value)) * 2 + bool(re.search(r"\b(?:increased|decreased|declined|grew|rose|fell|recorded|incurred|issued|repurchased|repaid|compared)\b", value,re.I)) * 3
                - bool(re.search(r"\b(?:may|could|might|if|potential)\b",value,re.I)) * 3)
    return max(selected,key=score)


def topic_interpretation(topic: str, fact: str):
    _, general, caveat = disclosure_topics[topic]
    lower = fact.lower()
    if not fact.rstrip().endswith((".", "!", "?", ";", ":")) and len(fact.split()) > 12:
        return "The selected excerpt ends mid-sentence. Its missing continuation prevents a complete assessment of the financial change.", "uncertain", "Retrieve the complete passage before assigning a direction or treating an omitted qualification as removed."
    if topic == "segments" and re.search(r"launch|introduc|announce",lower) and not re.search(r"revenue|sales|operating income|profit",lower):
        return "The product announcement does not quantify segment revenue or profitability. A product milestone should not be treated as a demonstrated financial contribution.", "low", "Commercial adoption, pricing and financial impact are not established by the launch alone."
    if re.search(r"industry-leading|industry leading|best.in.class",lower):
        return "The reported measures provide company-level context, but the comparative ranking is management's characterization. It needs independent peer data before being treated as verified relative leadership.", "medium", "No consistent peer comparison is supplied."
    if topic == "dilution" and "authorization" in lower:
        return "The board action changes permitted repurchases or planned distributions. An authorization is not an executed buyback, and a dividend increase does not establish a shrinking share count.", "medium" if "dividend" in lower else "low", caveat
    if re.search(r"shown below|as follows|following table|table below",lower) and not re.search(r"\$\s*\d|\d[\d.]*\s*(?:percent|%|billion|million)",lower):
        return "The excerpt introduces a table but does not supply its values. It cannot establish a numerical change without the referenced data.", "low", "Retrieve the table before assessing the financial movement."
    if "cash flow hedge" in lower or "cash-flow hedge" in lower:
        return "The passage concerns hedge accounting rather than operating cash generation. It does not establish a change in available liquidity.", "low", "Contract fair values and hedge designation do not by themselves measure the company's cash funding capacity."
    if topic == "debt" and re.search(r"marketable securities|investment portfolio|available-for-sale",lower):
        return "These debt instruments are investment assets. Their maturity or credit exposure should not be interpreted as a new issuer borrowing or refinancing event.", "low", "The asset portfolio and the company's own liabilities require separate analysis."
    if topic == "debt" and re.search(r"fair value.{0,30}(?:debt|instruments|notes)|classify the fair value",lower):
        return "The disclosure concerns the market value of existing debt, not a new financing flow. Changes in fair value do not establish repayment or a change in the carrying amount.", "medium", "Interest rates, credit spreads and instrument composition can affect fair value."
    if topic == "margins" and "gross margin" in lower and not re.search(r"percent|%|basis point|bps|margin rate",lower):
        return "The reported gross-margin amount and its operating explanation do not by themselves establish a higher margin percentage. Revenue and mix can change gross profit without the same change in the profit rate.", "medium", "A percentage margin or a consistent revenue denominator is needed to assess the profit rate."
    if re.search(r"accounting standard|impairment test|preparation of financial statements|estimates and assumptions|evaluate our estimates",lower):
        return "The excerpt describes accounting policy or estimation uncertainty rather than establishing a realized charge. A policy description should not be promoted to an operating event.", "uncertain", "The excerpt does not quantify a realized earnings or cash-flow impact."
    future = bool(re.search(r"\b(?:may(?!\s+\d)|could|might|would|if|potential|expected|expects|anticipates|outlook)\b",lower))
    subject = {"margins":r"(?:gross|operating) margin", "liquidity":r"(?:operating cash|cash (?:flow|provided)|cash generation)",
               "capital_spending":r"(?:capital expenditure|capital spending|capital investment)"}.get(topic,r"(?:revenue|profit|debt)")
    down = bool(re.search(subject + r"[^.;]{0,50}\b(?:decreased|declined|lower|fell|compressed)\b|\b(?:decrease|decline|lower|compression)\b[^.;]{0,35}" + subject,lower))
    up = bool(re.search(subject + r"[^.;]{0,50}\b(?:increased|higher|rose|expanded|grew)\b|\b(?:increase|higher|expansion)\b[^.;]{0,35}" + subject,lower))
    if re.search(r"did not result in an earnings charge", lower):
        return "The passage distinguishes the impairment from its effect on segment earnings. The stated allocation or guarantee prevents treating the impairment as that segment's earnings charge.", "medium", "Segment treatment does not by itself establish the consolidated earnings or cash impact."
    if re.search(r"(?:did not|have not|has not) (?:recognize|record|incur)|no (?:significant |material )?(?:impairment|restructuring|charge)",lower):
        return "The disclosure explicitly reports no material charge. It does not support presenting an impairment or restructuring loss as a realized event.", "low", "This statement is limited to the reported scope and period; it does not rule out future impairment."
    if future:
        if re.search(r"has been (?:impacted|affected)|have been (?:impacted|affected)", lower):
            return "The passage acknowledges an actual business impact while expressing uncertainty about the outlook. It does not quantify the impact or establish a realized accounting loss.", "medium", caveat
        if topic in {"capital_spending", "restructuring"} and re.search(r"expect|outlook",lower):
            return "The stated estimate informs the prospective funding or earnings burden. It is guidance rather than realized spending or a recognized charge, and any revision must be compared on the same basis.", "medium", caveat
        return "The passage describes a conditional exposure, so it does not establish a realized loss or operating change.", "uncertain", caveat
    if topic == "margins" and (down or up):
        return ("The disclosed margin pressure means revenue growth would provide an incomplete account of earnings quality. The cost and mix explanation matters for deciding whether pressure can persist." if down else
                "The disclosed margin improvement suggests more revenue is reaching profit, but sustainability depends on the operating drivers rather than the headline revenue growth alone."), "medium", caveat
    if topic == "liquidity" and "operating" in lower and (down or up):
        return ("Weaker operating cash generation would reduce the cash available for investment and distributions, even if accounting earnings remain positive. Working-capital timing should be separated from persistent cash weakness." if down else
                "Stronger operating cash generation expands internal funding capacity. The distinction between recurring cash earnings and working-capital releases is important before extrapolating it."), "medium", caveat
    if topic == "debt" and re.search(r"issued|borrowed|borrowings|refinanc",lower):
        return "Financing can extend funding capacity while changing interest expense and refinancing exposure. Gross issuance should not be treated as net leverage growth without the repayment and cash positions.", "medium", caveat
    if topic == "debt" and re.search(r"repaid|repayment|redeemed",lower):
        return "Debt repayment can reduce future financing obligations while using cash today. Balance-sheet improvement depends on whether the repayment was funded internally or replaced with new borrowing.", "medium", caveat
    if topic == "capital_spending" and "marketable securities" in lower:
        return "The investing-cash-flow movement includes financial-asset transactions, so it is not a clean measure of operating investment or capital intensity.", "medium", "Separate purchases and sales of securities from capital expenditure before inferring operating expansion."
    if topic == "capital_spending" and (down or up):
        return ("Lower disclosed investment can support near-term free cash flow, but may also reflect project timing or reduced expansion. It does not establish improved investment returns." if down else
                "Higher investment absorbs cash ahead of potential future revenue. The funding burden and the timing of returns qualify a growth-only reading."), "medium", caveat
    if topic == "dilution" and "authorization" in lower:
        return "The authorization changes management's permitted capital-allocation capacity; it does not establish that shares were actually repurchased.", "low", caveat
    if topic == "dilution" and re.search(r"repurchased|repurchases|bought back",lower):
        return "The repurchases represent a use of shareholder capital. Whether they offset employee issuance or reduce the diluted share base requires the share-count evidence as well as the gross spending figure.", "medium", caveat
    if topic == "concentration" and ("distributor" in lower or "channel" in lower):
        return "Sales concentration at a distributor or channel partner differs from concentration in final demand. It can still create counterparty and order-timing exposure, but does not prove dependence on a single end customer.", "medium", caveat
    if topic == "restructuring" and re.search(r"recorded|incurred|charge",lower):
        return "The reported charge affects earnings for the stated period, while cash payments and expected operating benefits can arrive on different schedules. Treating the whole charge as recurring operating weakness or as an immediate cash loss would both require more evidence.", "medium", caveat
    return general, "medium", caveat


def excerpt(document, text: str, label: str):
    return {"evidence_id": f"{document.document_id}:{label}:{text_hash(text)[:10]}",
        "document_id": document.document_id, "available_at": document.available_at.isoformat(),
        "source": document.source, "entities": document.entities, "tickers": document.tickers,
        "text": text, "text_hash": text_hash(text)}


def make_example(task, question, evidence, documents, target, *, cutoff=None, group_id=None, method="evidence_bound_draft_v1"):
    cutoff = cutoff or max(document.available_at for document in documents)
    packet = {"as_of": cutoff.isoformat(), "evidence": evidence}
    identity = text_hash(task + question + json.dumps(packet, sort_keys=True))[:24]
    hashes = {document.document_id: document.text_hash for document in documents}
    hashes.update({row["document_id"]: row["text_hash"] for row in evidence if row["source"] == "structured_context"})
    example = TrainingExample(example_id=f"{task}-{identity}", task=task,
        group_id=group_id or documents[-1].duplicate_group, cutoff=cutoff,
        entities=sorted({ticker for row in evidence for ticker in row["tickers"]}),
        source_ids=sorted(hashes), source_hashes=hashes, creation_method=method,
        messages=training_messages(question, packet, target))
    errors = validate_example(example)
    if errors:
        raise ValueError("; ".join(errors))
    return example


def macro_evidence(document):
    family = document.metadata.get("family", "")
    terms = {
        "cpi": [r"consumer price index.*(?:increased|rose|declined|unchanged)", r"all items less food and energy", r"shelter|energy index|food index"],
        "ppi": [r"producer price index.*final demand", r"final demand services", r"final demand goods"],
        "empsit": [r"total nonfarm payroll employment", r"average hourly earnings", r"revised.*(?:up|down)|revision", r"participation rate|unemployment rate"],
        "jolts": [r"number of job openings|job openings.*(?:increased|decreased|unchanged)", r"quits", r"layoffs and discharges"],
        "pce": [r"personal income.*(?:increased|decreased)", r"pce price index|excluding food and energy", r"personal saving|disposable personal"],
        "fomc": [r"decided to.*(?:target range|federal funds)", r"inflation|economic activity", r"balance sheet|holdings"],
        "minutes": [r"participants.*inflation", r"participants.*(?:labor|employment)", r"participants.*(?:policy|rate)"],
    }
    chosen = []
    for pattern in terms.get(family, [r"gross domestic product|personal consumption expenditures", r"revised|revision"]):
        matches = [p for p in paragraphs(document) if re.search(pattern, re.sub(r"\s+", " ", p), re.I)]
        found = next((p for p in matches if p not in chosen and sentence(p) and
                      (family in {"minutes", "fomc"} or re.search(r"\d", p)) and
                      not re.search(r"Total separations includes|are defined as|is defined as|quits are generally",p,re.I)), None)
        if found:
            chosen.append(found)
    return chosen


def macro_target(document, evidence, *, event=False):
    family = document.metadata.get("family", "")
    facts = [sentence(row["text"]) for row in evidence]
    if family == "fomc":
        decision = salient_sentence(evidence[0]["text"],r"Committee decided to")
        if decision:
            facts[0] = decision
    roles = {
        "cpi": ("inflation composition", "The headline price change and the underlying components need to be read together.",
            "A component-driven move may say less about persistent inflation than a broad change across core categories.",
            "No consensus forecast is supplied, so this cannot be labelled an upside or downside surprise."),
        "ppi": ("producer-price pressure", "The mix of producer-price changes matters for interpreting the aggregate release.",
            "Goods and service-price pressure can reach margins and consumer prices through different channels; pass-through is not automatic.",
            "Producer-price inflation does not map one-for-one into consumer inflation or company margins."),
        "empsit": ("labor-market balance", "Hiring, wages and revisions can tell different stories about labor demand.",
            "Payroll growth describes hiring while wage and household indicators qualify its implications for spending and policy.",
            "The surveys measure different populations, and no market consensus or causal market-reaction estimate is supplied."),
        "jolts": ("labor demand and turnover", "Openings and worker flows help distinguish desired hiring from realized employment changes.",
            "Vacancies, quits and layoffs describe different margins of labor-market adjustment; the headline alone cannot identify the full balance.",
            "Job openings are not completed hires, and this release does not establish a payroll forecast."),
        "pce": ("household spending and inflation", "Income, spending and the price measure need separate treatment.",
            "Nominal spending can move with prices as well as real demand; saving and disposable income help frame sustainability.",
            "No consensus estimate or independently measured market reaction is present in this release packet."),
        "fomc": ("monetary-policy decision", "The announced decision should be separated from conditional guidance about future policy.",
            "The policy setting affects financing conditions, while the statement's conditions determine what evidence could change the next decision.",
            "A policy statement is not a commitment to a fixed future rate path; the packet does not quantify what markets had priced."),
        "minutes": ("monetary-policy deliberations", "The minutes report deliberations from an earlier meeting, released at this later cutoff.",
            "Differences in participants' views matter for conditional policy interpretation, but are not a new policy decision.",
            "The discussion predates publication, and participant views are not a committee commitment."),
    }
    label, conclusion, significance, uncertainty = roles.get(family,
        ("growth composition", "Read the reported growth result alongside its composition and revisions.",
         "The composition distinguishes demand strength from volatile contributions and affects persistence.",
         "A reported release is not a forecast and does not establish a market surprise without expectations."))
    first = facts[0]
    if family == "fomc":
        if re.search(r"decided to (?:raise|increase)",first,re.I):
            conclusion = "The Committee tightened the policy setting. " + first
            significance = "A higher target range raises the policy hurdle for financing and valuation. The economic rationale and conditional guidance matter for the path ahead; the move alone does not reveal whether markets were surprised."
        elif re.search(r"decided to (?:lower|reduce)",first,re.I):
            conclusion = "The Committee eased the policy setting. " + first
            significance = "A lower target range eases the policy setting but may also signal concern about the outlook. The reason for easing matters before treating it as unambiguously positive for risky assets."
        elif re.search(r"decided to (?:maintain|keep)",first,re.I):
            conclusion = "The Committee held the policy setting steady. " + first
            significance = "An unchanged policy rate can coexist with a change in the policy bias. The accompanying outlook and conditions matter more for the next decision than the unchanged rate alone."
        else:
            conclusion = "The source describes monetary-policy information. " + first
            significance = "The reported information needs to be distinguished from an explicit target-rate decision. The selected text does not by itself establish a rate increase, cut or hold."
    elif family == "empsit":
        contraction = bool(re.search(r"\bemployment\b.{0,15}(?:fell|declined|decreased|edged down)",first,re.I))
        stable = bool(re.search(r"\bemployment\b.{0,70}(?:changed little|little change)|both total nonfarm.*changed little",first,re.I))
        conclusion = ("The headline reports a contraction in employment. " if contraction else "The headline reports little employment change. " if stable else "The headline reports hiring growth. ") + first
        significance = ("Job losses can weaken household income and signal demand pressure, but wages, participation and revisions determine how broadly that reading holds." if contraction else
                        "The release describes little employment change, so the point estimate should not be overstated as a strong directional signal. Wages and revisions provide separate information about labor income and the recent trend." if stable else
                        "Hiring supports labor income, but the wage and household details determine whether stronger employment also means stronger inflation pressure. A positive payroll headline alone does not settle that question.")
    elif family == "jolts":
        conclusion = first + " Openings measure desired hiring; worker flows test whether that demand is translating into activity."
        significance = "The vacancy and turnover evidence should be reconciled before calling the labor market tighter or looser. Openings can change without a comparable change in completed hires or layoffs."
    elif family in {"cpi", "ppi"}:
        conclusion = first + " The component evidence determines how much of the headline is persistent."
        if re.search(r"(?:fell|declined|decreased)",first,re.I) and any(re.search(r"(?:less food and energy|core).{0,80}(?:rose|increased)",row["text"],re.I|re.S) for row in evidence[1:]):
            significance = "A softer headline alongside rising underlying prices would qualify a broad disinflation reading. The component mix matters more than the sign of the headline alone."
    elif family == "pce":
        conclusion = first + " The spending, price and income measures answer different questions about household demand."
    claims = [Claim(statement="The release reports: " + fact, evidence_ids=[row["evidence_id"]], kind="fact")
              for row, fact in zip(evidence, facts, strict=True)]
    claims.append(Claim(statement=significance, evidence_ids=[row["evidence_id"] for row in evidence], kind="interpretation"))
    first = facts[0]
    materiality = "medium"
    if family == "fomc" and re.search(r"decided to (?:raise|increase|lower|reduce)",first,re.I):
        materiality = "high"
    if family == "jolts" and all(re.search(r"(?:unchanged|little changed|changed little)",fact,re.I) for fact in facts):
        materiality = "low"
    if family in {"cpi", "ppi"}:
        change = re.search(r"(?:rose|increased|decreased|fell|declined)\s+([\d.]+)\s+percent",first,re.I)
        if change and float(change[1]) >= (0.6 if family=="cpi" else 1.0):
            materiality = "high"
    if family == "empsit":
        change = re.search(r"(?:by\s+|\()([\d,]+)\s*(million)?",first)
        if change and float(change[1].replace(',','')) * (1e6 if change[2] else 1) >= 500000:
            materiality = "high"
    return AnalysisTarget(conclusion=(f"The event concerns {label}. " if event else "") + conclusion,
        materiality=materiality, claims=claims,
        what_changed=("The contemporaneous release records the following development: " if event else "The reported change to assess is: ") + first,
        why_it_matters=significance, uncertainty=[uncertainty, "The selected excerpts do not establish the sole driver of any asset-price move."])


def document_examples(document):
    rows = []
    if document.source in {"bls", "bea", "fed"} and document.metadata.get("historical_eligible"):
        selected = macro_evidence(document)
        if len(selected) >= 2:
            evidence = [excerpt(document, text, str(i)) for i, text in enumerate(selected)]
            for task in ["macro", "event"]:
                used = evidence if task == "macro" else evidence[:2]
                target = macro_target(document, used, event=task == "event")
                rows.append(make_example(task,
                    ("Analyze the material change, composition and policy relevance of this release." if task == "macro" else
                     "Extract the reported event and explain why it matters without inventing expectations."),
                    used, [document], target))
    if document.source == "sec" and document.form == "EX-99.1" and document.tickers:
        for topic, (pattern, significance, uncertainty) in disclosure_topics.items():
            matches = [p for p in paragraphs(document) if salient_sentence(p, pattern) and
                       re.search(r"\d",salient_sentence(p,pattern)) and not re.search(r"accounting standard|impairment test|cash flow hedge",p,re.I)]
            if not matches:
                continue
            matches.sort(key=lambda p: (bool(re.search(r"\b(?:increased|decreased|compared|recorded|incurred|issued|repaid|repurchased)\b",salient_sentence(p,pattern),re.I)),
                                       bool(re.search(r"\d",salient_sentence(p,pattern)))), reverse=True)
            selected = matches[0]
            evidence = [excerpt(document, selected, topic)]
            fact = salient_sentence(selected, pattern)
            significance, materiality, uncertainty = topic_interpretation(topic,fact)
            prospective = materiality == "uncertain"
            target = AnalysisTarget(conclusion=fact + " " + significance,
                materiality=materiality,
                claims=[Claim(statement="The filing states: " + fact, evidence_ids=[evidence[0]["evidence_id"]], kind="fact"),
                        Claim(statement=significance, evidence_ids=[evidence[0]["evidence_id"]], kind="interpretation")],
                what_changed=fact,
                why_it_matters=significance, uncertainty=[uncertainty,
                    "Conditional or forward-looking language is not a realized event." if prospective else "A prior comparable disclosure is needed to establish a change."])
            rows.append(make_example("event", f"Assess the {topic.replace('_', ' ')} event evidence in {document.tickers[0]}'s filing.",
                                     evidence, [document], target))
    return rows


def disclosure_examples(previous, current):
    if not current.tickers:
        return []
    old_paragraphs = paragraphs(previous)
    current_paragraphs = paragraphs(current)
    examples = []
    for topic, (pattern, significance, uncertainty) in disclosure_topics.items():
        left = [p for p in old_paragraphs if salient_sentence(p, pattern)]
        right = [p for p in current_paragraphs if salient_sentence(p, pattern)]
        pairs = []
        for after in right[:10]:
            scored = [(difflib.SequenceMatcher(None, before.split(), after.split(), autojunk=False).ratio(), before) for before in left[:15]]
            if scored:
                score, before = max(scored, key=lambda x: x[0])
                if score >= 0.45:
                    pairs.append((score, before, after))
        if not pairs:
            continue
        changed = [pair for pair in pairs if pair[0] < 0.995]
        selected = min(changed, key=lambda x: abs(x[0] - 0.8)) if changed else pairs[0]
        similarity, before, after = selected
        same = re.sub(r"\W", "", before).lower() == re.sub(r"\W", "", after).lower()
        evidence = [excerpt(previous, before, topic + "-prior"), excerpt(current, after, topic + "-current")]
        old_fact, new_fact = salient_sentence(before,pattern), salient_sentence(after,pattern)
        significance, materiality, uncertainty = topic_interpretation(topic,new_fact)
        if re.search(r"receivables",after,re.I) and topic=="concentration":
            significance = "The concentration concerns receivables and collection exposure, not necessarily revenue dependence. Aggregate exposure across several customers must not be confused with the largest single-customer share."
        if "allocat" in after.lower() and "segment" in after.lower():
            significance = "Segment allocation and reporting definitions can change measured segment profitability without the same change in the underlying business. Share-compensation allocation does not establish new share issuance."
        conclusion = (f"No material textual change is identified in the matched {topic.replace('_', ' ')} passage." if same else
                      f"The current {topic.replace('_', ' ')} disclosure says: {new_fact} {significance}")
        claims = [Claim(statement="Prior disclosure: " + old_fact, evidence_ids=[evidence[0]["evidence_id"]], kind="fact"),
                  Claim(statement="Current disclosure: " + new_fact, evidence_ids=[evidence[1]["evidence_id"]], kind="fact"),
                  Claim(statement=("The selected passage is unchanged; it does not support calling this a newly disclosed risk." if same else significance),
                        evidence_ids=[e["evidence_id"] for e in evidence], kind="interpretation")]
        target = AnalysisTarget(conclusion=conclusion, materiality="low" if same else materiality, claims=claims,
            what_changed="The compared wording is identical." if same else "The before-and-after passages differ. The cited language establishes the change, without proving every edit is economically material.",
            why_it_matters="Stable wording is evidence against presenting this selected disclosure as new." if same else significance,
            uncertainty=[uncertainty, "The comparison covers a matched passage, not every section of either filing."])
        examples.append(make_example("sec_change", f"Compare {current.tickers[0]}'s {topic.replace('_', ' ')} disclosure with its prior comparable filing. Identify whether a material change is supported.",
            evidence, [previous, current], target, group_id=f"sec-pair-{previous.accession}-{current.accession}"))
    return examples


def market_records(root: Path, dates):
    from quantfinlab.dataio import load_par_yield_curve, load_yfinance_panel
    from quantfinlab.ml.features import breadth, realized_vol, rolling_avg_corr

    price_path = root / "data/core_cross_asset_etfs.csv"
    rate_path = root / "data/us_treasury_yields.csv"
    prices = load_yfinance_panel(price_path, fields=["close"], lowercase=False)["close"]
    tickers = [t for t in ["SPY", "QQQ", "IWM", "HYG", "LQD", "TLT", "IEF", "GLD", "DBC", "UUP"] if t in prices]
    prices = prices[tickers]
    returns = prices.pct_change(fill_method=None)
    yields = load_par_yield_curve(rate_path, percent=True)
    vol = realized_vol(returns.SPY, 21)
    broad = breadth(prices, 21, tickers)
    avg_corr = rolling_avg_corr(returns, 63)
    move_z = (returns - returns.shift(1).rolling(252).mean()) / returns.shift(1).rolling(252).std()
    output = {}
    for date in sorted(set(pd.Timestamp(d).normalize() for d in dates)):
        eligible = prices.index[prices.index <= date]
        if not len(eligible):
            continue
        day = eligible[-1]
        if (date - day).days > 4 or pd.isna(vol.loc[day]):
            continue
        cutoff = close_time(day)
        moves = returns.loc[day]
        z = move_z.loc[day]
        rate_rows = yields.loc[:day]
        if len(rate_rows) < 2:
            continue
        ten_change = (rate_rows['10Y'].iloc[-1] - rate_rows['10Y'].iloc[-2]) * 10000
        credit_spread = (moves.HYG - moves.LQD) * 100
        equity_spread = (moves.QQQ - moves.SPY) * 100
        text = f"Market close snapshot for {day.date()}; available after 18:00 America/New_York.\n"
        text += "\n".join(f"{ticker}: daily return {moves[ticker]*100:+.2f} percent; standardized move {z[ticker]:+.2f}." for ticker in tickers)
        text += (f"\nSPY realized volatility over 21 trading days: {vol.loc[day]*100:.2f} percent annualized."
                 f"\nETF breadth over 21 trading days: {broad.loc[day]*100:.2f} percent; average correlation over 63 trading days: {avg_corr.loc[day]:.3f}."
                 f"\nTreasury 10Y daily change: {ten_change:+.2f} bp."
                 f"\nHYG minus LQD daily return: {credit_spread:+.2f} percentage points."
                 f"\nQQQ minus SPY daily return: {equity_spread:+.2f} percentage points."
                 "\nNo intraday prices, consensus estimates or causal driver model are supplied. Adjusted market histories are current-vendor reconstructions.")
        if re.search(r"\bnan\b|\binf\b", text):
            continue
        digest = text_hash(text)
        evidence = {"evidence_id": f"market-{day.date()}", "document_id": f"market-{day.date()}",
            "available_at": cutoff.isoformat(), "source": "structured_context", "entities": [], "tickers": tickers,
            "text": text, "text_hash": digest}
        output[date] = {"evidence": evidence, "cutoff": cutoff, "moves": moves,
                        "z": z, "ten_change": ten_change, "credit_spread": credit_spread,
                        "equity_spread": equity_spread, "breadth": broad.loc[day]}
    return output


def market_target(record, *, release=None):
    e = record["evidence"]
    moves = record["moves"]
    spy, hyg, tlt, gold = moves.SPY, moves.HYG, moves.TLT, moves.GLD
    quiet = record["z"].abs().max() < 1 and moves.abs().max() < 0.01 and abs(record["ten_change"]) < 5
    large = record["z"].abs().max() >= 2.5 or moves.abs().max() >= 0.03
    divergence = np.sign(spy) != np.sign(hyg)
    direction = "rose" if spy > 0 else "fell" if spy < 0 else "was unchanged"
    conclusion = ("Moves across the selected assets were modest; the packet does not establish a dominant market driver." if quiet else
                  "Equities and high-yield bonds moved in opposite directions, while other asset moves qualify the market reading." if divergence else
                  "Equities and high-yield bonds moved in the same direction; Treasury, commodity and currency moves provide separate information.")
    facts = [f"SPY {direction}, with a daily return of {spy*100:+.2f} percent; QQQ returned {moves.QQQ*100:+.2f} percent.",
             f"HYG returned {hyg*100:+.2f} percent and LQD returned {moves.LQD*100:+.2f} percent; their return gap was {record['credit_spread']:+.2f} percentage points.",
             f"TLT returned {tlt*100:+.2f} percent, while the Treasury yield change was {record['ten_change']:+.2f} bp and GLD returned {gold*100:+.2f} percent."]
    implication = ("Equity direction alone is insufficient: credit relative performance and Treasury sensitivity qualify the risk reading. ETF return gaps are not pure credit-spread or rate shocks." if divergence else
                   "Agreement across equities and credit supports a descriptive risk reading. It does not identify the news shock, and rate duration still influences bond ETF returns.")
    claims = [Claim(statement=fact,evidence_ids=[e["evidence_id"]],kind="fact") for fact in facts]
    claims.append(Claim(statement=implication,evidence_ids=[e["evidence_id"]],kind="interpretation"))
    uncertainty = ["Daily co-movement cannot establish causation or the exact intraday response to a release.",
                   "The selected ETF universe is a compact market proxy; it does not establish stock-level breadth."]
    if release is not None:
        fact = sentence(release["text"])
        claims.insert(0,Claim(statement="The official source reports: " + fact,evidence_ids=[release["evidence_id"]],kind="fact"))
        conclusion = "The official release establishes what was reported; the market snapshot shows the day's moves. A causal link between them is not established by timing alone."
        uncertainty.append("No pre-release consensus or intraday event window is provided, so a surprise-driven market attribution is unsupported.")
    return AnalysisTarget(conclusion=conclusion,materiality="high" if large else "low" if quiet else "medium",claims=claims,
        what_changed="The daily moves and cross-asset return gaps are reported in the cited market snapshot.",
        why_it_matters=implication,uncertainty=uncertainty)


def build_candidates(root: str | Path, destination: str | Path, *, anchors_only=True):
    root, destination = Path(root), Path(destination)
    documents = sorted(DocumentStore(root / "workspace/financial_analyst/documents").records(), key=lambda d: (d.available_at, d.document_id))
    if anchors_only:
        selected = []
        for source in ["bls", "bea", "fed"]:
            candidates = [d for d in documents if d.source == source]
            selected += [candidates[int(i)] for i in np.linspace(0, max(0,len(candidates)-1), min(30,len(candidates)),dtype=int)]
        selected += [d for d in documents if d.source == "sec" and d.tickers and d.tickers[0] in {"NVDA", "AAPL"}]
        documents = selected
    examples, errors = [], []
    for document in documents:
        try:
            examples.extend(document_examples(document))
        except ValueError as error:
            errors.append({"document_id": document.document_id, "error": str(error)})
    filing_groups = {}
    for document in documents:
        if document.form in {"10-K", "10-Q"}:
            filing_groups.setdefault((document.cik, document.form), []).append(document)
    for group in filing_groups.values():
        for current in group:
            if not current.report_period:
                continue
            prior = [d for d in group if d.available_at < current.available_at and d.report_period
                     and 300 <= (pd.Timestamp(current.report_period) - pd.Timestamp(d.report_period)).days <= 430]
            if prior:
                try:
                    examples.extend(disclosure_examples(prior[-1], current))
                except ValueError as error:
                    errors.append({"document_id": current.document_id, "error": str(error)})
    macro_documents = [d for d in documents if d.source in {"bls", "fed", "bea"} and d.metadata.get("historical_eligible") and macro_evidence(d)]
    dates = [d.available_at.date() for d in macro_documents]
    dates += list(pd.date_range("2016-01-04", "2026-08-31", freq="180D" if anchors_only else "7D"))
    markets = market_records(root, dates)
    for date, record in markets.items():
        evidence = record["evidence"]
        target = market_target(record)
        try:
            examples.append(make_example("market", "Assess the day's cross-asset moves, identify conflicting signals, and state what cannot be attributed to a driver.",
                [evidence], [], target, cutoff=record["cutoff"], group_id=f"market-day-{date.date()}"))
        except ValueError as error:
            errors.append({"date": str(date), "error": str(error)})
    for document in macro_documents:
        date = pd.Timestamp(document.available_at.date())
        record = markets.get(date)
        if record is None or record["cutoff"] < document.available_at:
            continue
        evidence = excerpt(document, macro_evidence(document)[0], "release")
        target = market_target(record, release=evidence)
        try:
            examples.append(make_example("reconciliation", "Reconcile the official release with the market close. Distinguish reported facts, observed moves and unsupported driver narratives.",
                [evidence, record["evidence"]], [document], target, cutoff=record["cutoff"], group_id=f"market-day-{date.date()}"))
        except ValueError as error:
            errors.append({"document_id": document.document_id, "error": str(error)})
    unique = {}
    for example in examples:
        unique.setdefault(example.example_id, example)
    examples = sorted(unique.values(), key=lambda row: (row.cutoff, row.example_id))
    anchors = []
    for task, count in anchor_targets.items():
        candidates = [row for row in examples if row.task == task]
        positions = np.linspace(0, max(0,len(candidates)-1), min(count,len(candidates)),dtype=int)
        anchors.extend(candidates[int(i)].model_copy(update={"anchor": True, "quality_status": "review"}) for i in positions)
    write_examples(destination / "anchors.jsonl", anchors)
    if not anchors_only:
        write_examples(destination / "candidates.jsonl", examples)
    summary = {"documents": len(documents), "candidate_counts": dict(Counter(row.task for row in examples)),
               "anchor_counts": dict(Counter(row.task for row in anchors)), "errors": errors,
               "status": "anchors_awaiting_individual_review"}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
