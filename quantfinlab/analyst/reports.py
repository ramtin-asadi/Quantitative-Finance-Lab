"""Readable reports with the original structured analysis and evidence attached."""

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .schemas import AnalysisTarget


def unique_passages(passages):
    seen, sentences_seen, selected = [], set(), []
    for text in passages:
        text = re.sub(r"\s+", " ", text).strip()
        key = text.casefold().strip(" .")
        if key and not any(key == old or key in old for old in seen):
            sentences = re.split(r"(?<!U\.S\.)(?<!Inc\.)(?<!Corp\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+(?=[A-Z])", text)
            kept = []
            for sentence in sentences:
                normalized = sentence.casefold().strip(" .")
                if normalized not in sentences_seen:
                    kept.append(sentence)
                    sentences_seen.add(normalized)
            if kept:
                selected.append(" ".join(kept))
            seen.append(key)
    return selected


@dataclass
class AnalysisReport:
    question: str
    task: str
    as_of: str
    analysis: AnalysisTarget | None
    packet: dict
    errors: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    cache_key: str = ""
    cached: bool = False
    diagnostics: dict = field(default_factory=dict)

    @property
    def validated(self):
        return self.analysis is not None and not self.errors

    @property
    def sources(self):
        used = {key for claim in self.analysis.claims for key in claim.evidence_ids} if self.analysis else set()
        return [row for row in self.packet.get("evidence", []) if row["evidence_id"] in used]

    def to_dict(self):
        return {"question": self.question, "task": self.task, "as_of": self.as_of,
                "analysis": self.analysis.model_dump(mode="json") if self.analysis else None,
                "packet": self.packet, "errors": self.errors, "attempts": self.attempts,
                "cache_key": self.cache_key, "cached": self.cached, "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("analysis") is not None:
            value["analysis"] = AnalysisTarget.model_validate(value["analysis"])
        return cls(**value)

    def to_markdown(self):
        if self.analysis is None:
            return "No supported analysis could be produced.\n\n" + "\n".join(self.errors)
        answer = self.analysis
        observations = unique_passages([claim.statement for claim in answer.claims if claim.kind == "fact"])
        interpretations = unique_passages([claim.statement for claim in answer.claims if claim.kind == "interpretation"])
        lines = [" ".join(observations) if observations else answer.conclusion]
        if interpretations:
            lines += ["", "Model interpretation: " + " ".join(interpretations)]
        lines += ["", "Supporting evidence:", ""]
        source_numbers = {row["evidence_id"]: str(i) for i, row in enumerate(self.sources, 1)}
        for claim in answer.claims:
            refs = ", ".join(source_numbers[key] for key in claim.evidence_ids if key in source_numbers)
            lines.append(f"- {claim.statement} [{refs}]")
        lines += ["", "Uncertainty: " + " ".join(unique_passages(answer.uncertainty)), "",
                  f"As of {self.as_of} · Materiality: {answer.materiality} · "
                  + ("Automatic checks passed; interpretation needs review" if self.validated else "Incomplete answer; see validation issues")]
        for i, row in enumerate(self.sources, 1):
            label = row.get("title", row.get("source", "Evidence"))
            url = row.get("source_url", "")
            lines.append(f"\n[{i}] {label}" + (f" — {url}" if url else "") + f"; available {row['available_at']}")
        if self.errors:
            lines += ["", "Validation issues: " + "; ".join(self.errors)]
        return "\n".join(lines)

    def _repr_html_(self):
        escape = html.escape
        card_style = ("max-width:920px;line-height:1.65;margin:1rem 0;padding:1.15rem 1.3rem;"
                      "border:1px solid #2d3339;border-radius:12px;background:#111416;color:#e7ece9;"
                      "box-shadow:0 8px 24px rgba(0,0,0,.2);font-family:Inter,ui-sans-serif,system-ui,sans-serif")
        if self.analysis is None:
            return (f'<article style="{card_style};border-color:#8a6428">'
                    '<div style="font-size:.72rem;font-weight:700;letter-spacing:.08em;color:#e9b866">QUANTFINLAB ANALYST</div>'
                    '<p><strong>No supported analysis could be produced.</strong></p><p>'
                    + escape("; ".join(self.errors)) + "</p></article>")
        answer = self.analysis
        observations = unique_passages([claim.statement for claim in answer.claims if claim.kind == "fact"])
        interpretations = unique_passages([claim.statement for claim in answer.claims if claim.kind == "interpretation"])
        header = ('<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.9rem">'
                  '<span style="display:flex;gap:.65rem;align-items:center">'
                  '<span style="display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;'
                  'border-radius:8px;background:#10a37f;color:#071a15;font-size:1rem;font-weight:800">Q</span>'
                  '<span><strong style="display:block;font-size:.95rem">QuantFinLab Analyst</strong>'
                  '<small style="color:#8d9993">local model response</small></span></span>'
                  f'<span style="font-size:.68rem;letter-spacing:.07em;color:#78cdb7">{escape(self.task.replace("_", " ").upper())}</span></div>'
                  '<div style="background:#202427;border:1px solid #30363b;border-radius:9px;padding:.65rem .8rem;margin-bottom:1rem">'
                  '<small style="display:block;color:#8d9993;font-weight:650;letter-spacing:.04em;margin-bottom:.2rem">USER QUERY</small>'
                  f'{escape(self.question)}</div>')
        paragraphs = ['<div style="background:#171b1d;border:1px solid #252b2e;border-radius:9px;'
                      'padding:.8rem .95rem;margin-bottom:.75rem"><small style="display:block;color:#78cdb7;'
                      'font-weight:650;letter-spacing:.04em;margin-bottom:.25rem">EVIDENCE-BACKED RESPONSE</small>'
                      + escape(" ".join(observations) if observations else answer.conclusion) + "</div>"]
        if interpretations:
            paragraphs.append('<div style="background:rgba(16,163,127,.13);border-left:3px solid #10a37f;border-radius:7px;'
                              'padding:.75rem .9rem;margin-bottom:.85rem"><strong>Model interpretation</strong><br>'
                              + escape(" ".join(interpretations)) + "</div>")
        references = {row["evidence_id"]: i for i, row in enumerate(self.sources, 1)}
        details = []
        for claim in answer.claims:
            refs = ", ".join(str(references[key]) for key in claim.evidence_ids if key in references)
            details.append("<li>" + escape(claim.statement) + f" <small>[{refs}] · {escape(claim.kind)}</small></li>")
        sources = []
        for row in self.sources:
            title = escape(row.get("title", row.get("source", "Evidence")))
            url = row.get("source_url", "")
            if urlparse(url).scheme in {"https", "http"}:
                title = f'<a style="color:#78cdb7" href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
            sources.append(f"<li>{title}<br><small>Available {escape(row['available_at'])}</small></li>")
        status = "Automatic checks passed; interpretation needs review" if self.validated else "Incomplete answer"
        stamp = f"As of {self.as_of} · {answer.materiality.capitalize()} materiality · {status}"
        if self.cached:
            stamp += " · Saved answer reused"
        warnings = '<p style="color:#f2b7b5"><strong>Validation issues:</strong> ' + escape("; ".join(self.errors)) + "</p>" if self.errors else ""
        freshness = self.diagnostics.get("freshness", [])
        freshness_html = "<p><small>" + escape("; ".join(freshness)) + "</small></p>" if freshness else ""
        generated = "".join("<p>" + escape(passage) + "</p>" for passage in
                            unique_passages([answer.conclusion, answer.what_changed, answer.why_it_matters]))
        return (f'<article style="{card_style}">' + header + "".join(paragraphs)
                + f'<details style="margin:.55rem 0;color:#cbd4cf"><summary style="cursor:pointer;font-weight:600">Supporting evidence ({len(details)} claims)</summary><ul>' + "".join(details) + "</ul></details>"
                + '<div style="background:rgba(217,160,67,.12);border-left:3px solid #d9a043;border-radius:7px;'
                'padding:.7rem .85rem;margin:.85rem 0"><strong>Uncertainty</strong><br>'
                + escape(" ".join(unique_passages(answer.uncertainty))) + "</div>"
                + '<details style="margin:.55rem 0;color:#cbd4cf"><summary style="cursor:pointer;font-weight:600">Sources and availability</summary><ol>' + "".join(sources) + "</ol></details>"
                + '<details style="margin:.55rem 0;color:#cbd4cf"><summary style="cursor:pointer;font-weight:600">Full generated summary</summary>' + generated + "</details>"
                + freshness_html + warnings + '<div style="display:flex;flex-wrap:wrap;gap:.4rem;border-top:1px solid #2b3235;'
                'margin-top:.95rem;padding-top:.7rem;color:#8d9993;font-size:.76rem">'
                + "".join(f'<span style="background:#202427;border-radius:999px;padding:.18rem .5rem">{escape(part.strip())}</span>'
                          for part in stamp.split("·")) + "</div></article>")

    def __str__(self):
        return self.to_markdown()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(self.to_dict(), indent=2, ensure_ascii=False) if path.suffix == ".json"
                   else self._repr_html_() if path.suffix == ".html" else self.to_markdown())
        path.write_text(content, encoding="utf-8")
        return path


@dataclass
class DailyBrief:
    report: AnalysisReport
    events: list
    sections: dict[str, str]

    def _repr_html_(self):
        return self.report._repr_html_()

    def __str__(self):
        return str(self.report)
