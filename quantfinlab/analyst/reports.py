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
        if self.analysis is None:
            return "<p><strong>No supported analysis could be produced.</strong></p><p>" + escape("; ".join(self.errors)) + "</p>"
        answer = self.analysis
        observations = unique_passages([claim.statement for claim in answer.claims if claim.kind == "fact"])
        interpretations = unique_passages([claim.statement for claim in answer.claims if claim.kind == "interpretation"])
        paragraphs = ["<p>" + escape(" ".join(observations) if observations else answer.conclusion) + "</p>"]
        if interpretations:
            paragraphs.append("<p><strong>Model interpretation:</strong> " + escape(" ".join(interpretations)) + "</p>")
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
                title = f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
            sources.append(f"<li>{title}<br><small>Available {escape(row['available_at'])}</small></li>")
        status = "Automatic checks passed; interpretation needs review" if self.validated else "Incomplete answer"
        stamp = f"As of {self.as_of} · {answer.materiality.capitalize()} materiality · {status}"
        if self.cached:
            stamp += " · Saved answer reused"
        warnings = "<p><strong>Validation issues:</strong> " + escape("; ".join(self.errors)) + "</p>" if self.errors else ""
        freshness = self.diagnostics.get("freshness", [])
        freshness_html = "<p><small>" + escape("; ".join(freshness)) + "</small></p>" if freshness else ""
        generated = "".join("<p>" + escape(passage) + "</p>" for passage in
                            unique_passages([answer.conclusion, answer.what_changed, answer.why_it_matters]))
        return ('<article style="max-width:920px;line-height:1.65">' + "".join(paragraphs)
                + f"<details><summary>Supporting evidence ({len(details)} claims)</summary><ul>" + "".join(details) + "</ul></details>"
                + "<p><em>Uncertainty:</em> " + escape(" ".join(unique_passages(answer.uncertainty))) + "</p>"
                + "<details><summary>Sources and availability</summary><ol>" + "".join(sources) + "</ol></details>"
                + "<details><summary>Full generated summary</summary>" + generated + "</details>"
                + freshness_html + warnings + "<p><small>" + escape(stamp) + "</small></p></article>")

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
