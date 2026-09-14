import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantfinlab.analyst.corpus import make_example
from quantfinlab.analyst.documents import text_hash
from quantfinlab.analyst.schemas import AnalysisTarget, Claim, utc
from quantfinlab.analyst.training import chronological_split, freeze_dataset, record_review
from quantfinlab.analyst.validation import split_leakage, validate_example


def example(day, source="context-a", group="day-a"):
    text = "SPY returned 1.5 percent; causation is not identified."
    evidence = {"evidence_id":source,"document_id":source,"source":"structured_context",
        "available_at":day,"tickers":["SPY"],"entities":["SPY"],"text":text,"text_hash":text_hash(text)}
    target = AnalysisTarget(conclusion="Equities rose.",materiality="medium",
        claims=[Claim(statement="SPY returned 1.5 percent.",evidence_ids=[source],kind="fact")],
        what_changed="The observed return was positive.",why_it_matters="A positive return does not identify its cause.",
        uncertainty=["No event window is supplied."])
    return make_example("market","Analyze the market.",[evidence],[],target,cutoff=utc(day),group_id=group)


def test_review_is_bound_to_target_hash():
    row = example("2024-01-01T23:00:00Z")
    with pytest.raises(ValueError,match="changed"):
        record_review(row,reviewer="Reviewer",decision="accepted",notes="Inspected evidence.",method="human_read",target_hash="old")
    reviewed = record_review(row,reviewer="Reviewer",decision="accepted",notes="Inspected evidence and attribution.",method="human_read",target_hash=text_hash(row.messages[-1]["content"]))
    assert reviewed.quality_status == "accepted"


def test_chronological_split_embargoes_shared_evidence():
    earlier = example("2024-01-01T23:00:00Z")
    later = example("2025-01-01T23:00:00Z",group="day-b")
    train, validation, embargo = chronological_split([earlier,later],validation_start="2025-01-01T00:00:00Z",validation_size=1)
    assert train == [] and embargo == [earlier] and validation == [later]
    assert any("source" in error for error in split_leakage([earlier],[later]))


def test_freeze_never_promotes_drafts(tmp_path):
    row = example("2024-01-01T23:00:00Z")
    with pytest.raises(ValueError,match="not ready"):
        freeze_dataset([row],[],tmp_path,minimum_examples=1,minimum_anchors=0)
    assert not (tmp_path/"train.jsonl").exists()


def test_packet_cutoff_and_hash_are_checked():
    row = example("2024-01-01T23:00:00Z")
    packet = json.loads(row.messages[1]["content"])
    packet["as_of"] = "2023-01-01T23:00:00Z"
    row.messages[1]["content"] = json.dumps(packet)
    errors = validate_example(row)
    assert any("cutoff" in error for error in errors)
    assert any("Future" in error for error in errors)


def notebook_definitions(names, namespace):
    path = Path(__file__).resolve().parents[2] / "models/qwen_train_lora.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    definitions = [node for cell in notebook["cells"] if cell["cell_type"] == "code"
                   for node in ast.parse("".join(cell["source"])).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_training_masks_prompt_and_preserves_complete_answer_and_eos():
    prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    tokenizer = SimpleNamespace(eos_token_id=2, encode=lambda text, **kwargs: list(text.encode()),
                                apply_chat_template=lambda *args, **kwargs: prefix)
    namespace = notebook_definitions({"prompt_text", "encode_example"}, {"tokenizer": tokenizer, "training_context": 1000})
    row = {"example_id": "one", "messages": [{"role": "system", "content": "system"},
           {"role": "user", "content": "question"}, {"role": "assistant", "content": '{"answer":"complete"}'}]}
    encoded = namespace["encode_example"](row)
    assert encoded["labels"][:len(prefix)] == [-100] * len(prefix)
    assert bytes(encoded["labels"][len(prefix):-1]).decode() == row["messages"][-1]["content"]
    assert encoded["labels"][-1] == tokenizer.eos_token_id
    assert len(encoded["labels"]) == len(encoded["input_ids"]) == len(encoded["attention_mask"])
    namespace["training_context"] = len(encoded["input_ids"]) - 1
    with pytest.raises(ValueError, match="no silent truncation"):
        namespace["encode_example"](row)


def test_amp_skips_are_tolerated_but_persistent_overflow_stops_training():
    accelerator = SimpleNamespace(optimizer_step_was_skipped=True, scaler=SimpleNamespace(get_scale=lambda: 512))
    namespace = notebook_definitions({"TrainingProgress"}, {"TrainerCallback": object, "time": time,
        "trainer": SimpleNamespace(accelerator=accelerator), "log": lambda *args, **kwargs: None})
    progress = namespace["TrainingProgress"]()
    progress.accelerator = accelerator
    for step in range(7):
        progress.on_step_end(None, SimpleNamespace(global_step=step), None)
    accelerator.optimizer_step_was_skipped = False
    progress.on_step_end(None, SimpleNamespace(global_step=8), None)
    assert progress.consecutive_skips == 0 and progress.skipped_updates == 7
    accelerator.optimizer_step_was_skipped = True
    for step in range(7):
        progress.on_step_end(None, SimpleNamespace(global_step=step + 9), None)
    with pytest.raises(FloatingPointError, match="Eight consecutive"):
        progress.on_step_end(None, SimpleNamespace(global_step=16), None)


def test_training_loss_guard_checks_train_and_eval_return_formats():
    class ParentTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return (self.loss, {}) if return_outputs else self.loss

    namespace = notebook_definitions({"FiniteLossTrainer"}, {"Trainer": ParentTrainer,
                                     "torch": SimpleNamespace(isfinite=np.isfinite)})
    trainer = namespace["FiniteLossTrainer"]()
    trainer.loss = SimpleNamespace(detach=lambda: np.asarray(1.2))
    assert trainer.compute_loss(None, {}) is trainer.loss
    assert trainer.compute_loss(None, {}, return_outputs=True)[0] is trainer.loss
    trainer.loss = SimpleNamespace(detach=lambda: np.asarray(float("nan")))
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        trainer.compute_loss(None, {})
