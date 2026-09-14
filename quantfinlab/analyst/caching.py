"""Response identities and loading completed, inspectable analysis records."""

import json
from pathlib import Path

from .documents import text_hash
from .prompts import response_schema
from .reports import AnalysisReport


def response_key(model_sha, prompt_version, question, packet, *, task, generation_tokens=2300):
    identity = {"model": model_sha, "prompt": prompt_version, "task": task, "question": question,
                "packet": packet, "schema": response_schema(packet), "validator": "analyst-validation-v7",
                "settings": {"temperature": 0, "seed": 3407, "max_tokens": generation_tokens}}
    return text_hash(json.dumps(identity, sort_keys=True, ensure_ascii=False))


def load_response(path):
    path = Path(path)
    if not path.exists():
        return None
    report = AnalysisReport.from_dict(json.loads(path.read_text(encoding="utf-8")))
    report.cached = True
    return report
