import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .documents import text_hash
from .schemas import TrainingExample, utc
from .validation import split_leakage, validate_example

task_targets = {"event": 900, "sec_change": 750, "macro": 600, "reconciliation": 450, "market": 300}
anchor_targets = {"event": 50, "sec_change": 35, "macro": 30, "reconciliation": 20, "market": 15}


def read_examples(path: str | Path) -> list[TrainingExample]:
    return [TrainingExample.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_examples(path: str | Path, examples) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(row.model_dump_json() + "\n" for row in examples), encoding="utf-8")
    temporary.replace(path)


def record_review(example: TrainingExample, *, reviewer: str, decision: str, notes: str,
                  method: str, target_hash: str) -> TrainingExample:
    if decision not in {"accepted", "rejected"} or not reviewer.strip() or not notes.strip():
        raise ValueError("A named reviewer, explicit decision and substantive notes are required.")
    actual = text_hash(example.messages[-1]["content"])
    if target_hash != actual:
        raise ValueError("The target changed since it was reviewed.")
    return example.model_copy(update={"quality_status": decision, "review": {
        "reviewer": reviewer, "method": method, "notes": notes, "target_hash": actual,
        "reviewed_at": datetime.now(timezone.utc).isoformat()}})


def chronological_split(examples, *, validation_start, validation_size=300):
    start = utc(validation_start)
    candidates = sorted([row for row in examples if row.cutoff >= start], key=lambda x: (x.cutoff, x.example_id))
    groups = {}
    for row in candidates:
        groups.setdefault(row.group_id, []).append(row)
    validation = []
    for group in groups.values():
        validation.extend(group)
        if len(validation) >= validation_size:
            break
    used_ids = set().union(*(set(row.source_ids) for row in validation)) if validation else set()
    used_hashes = set().union(*(set(row.source_hashes.values()) for row in validation)) if validation else set()
    used_groups = {row.group_id for row in validation}
    train, embargo = [], []
    for row in sorted(examples, key=lambda x: (x.cutoff, x.example_id)):
        if row.cutoff >= start:
            continue
        shared = set(row.source_ids) & used_ids or set(row.source_hashes.values()) & used_hashes or row.group_id in used_groups
        (embargo if shared else train).append(row)
    return train, validation, embargo


def freeze_dataset(train, validation, destination: str | Path, *, store=None,
                   minimum_examples=2700, minimum_anchors=100) -> dict:
    examples = [*train, *validation]
    errors = split_leakage(train, validation)
    if len(examples) < minimum_examples:
        errors.append(f"Only {len(examples)} examples; need at least {minimum_examples}.")
    if len(validation) < 250:
        errors.append("At least 250 later validation examples are required.")
    anchors = [row for row in examples if row.anchor]
    if len(anchors) < minimum_anchors:
        errors.append("Anchor review is incomplete.")
    ids, prompts, answers = set(), set(), set()
    counts = Counter(row.task for row in examples)
    for task, target in task_targets.items():
        if counts[task] < target * 0.7:
            errors.append(f"Task coverage below minimum: {task}.")
    low = sum(json.loads(row.messages[-1]["content"])["materiality"] in {"low", "uncertain"} for row in examples)
    if examples and low / len(examples) < 0.10:
        errors.append("Fewer than 10% low-materiality or ambiguous examples.")
    for example in examples:
        errors.extend(f"{example.example_id}: {error}" for error in validate_example(example, store=store))
        if example.quality_status != "accepted":
            errors.append(f"Unaccepted example: {example.example_id}")
        if example.anchor or example in validation:
            if example.review.get("method") not in {"human_read", "agent_read"}:
                errors.append(f"Individual review required: {example.example_id}")
        if example.review and example.review.get("target_hash") != text_hash(example.messages[-1]["content"]):
            errors.append(f"Stale review: {example.example_id}")
        prompt_hash = text_hash(example.messages[1]["content"])
        answer_hash = text_hash(example.messages[-1]["content"])
        if example.example_id in ids or prompt_hash in prompts or answer_hash in answers:
            errors.append(f"Duplicate example, prompt or answer: {example.example_id}")
        ids.add(example.example_id)
        prompts.add(prompt_hash)
        answers.add(answer_hash)
    if errors:
        raise ValueError("Dataset is not ready to freeze:\n" + "\n".join(errors[:60]))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_examples(destination / "train.jsonl", train)
    write_examples(destination / "validation.jsonl", validation)
    manifest = {"status": "frozen", "template_version": "analyst-v1", "task_counts": dict(counts),
        "counts": {"train": len(train), "validation": len(validation), "anchors": len(anchors), "low_materiality": low},
        "files": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in ["train.jsonl", "validation.jsonl"]},
        "cutoffs": {"train_max": max(row.cutoff for row in train).isoformat(),
                    "validation_min": min(row.cutoff for row in validation).isoformat()},
        "review_methods": dict(Counter(row.review.get("method", "unchecked") for row in examples)),
        "created_at": datetime.now(timezone.utc).isoformat()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
