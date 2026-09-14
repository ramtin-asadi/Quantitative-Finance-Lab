import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from quantfinlab.analyst import FinancialAnalyst
from quantfinlab.analyst.config import AnalystConfig
from quantfinlab.analyst.documents import text_hash
from quantfinlab.analyst.events import EventStore
from quantfinlab.analyst.evidence import evidence_row
from quantfinlab.analyst.inference import LlamaRuntime, download_model, save_json
from quantfinlab.analyst.reports import AnalysisReport
from quantfinlab.analyst.schemas import AnalysisTarget, Claim, EventRecord, utc
from quantfinlab.analyst.validation import check_response


@pytest.fixture
def packet():
    return {"as_of": "2026-09-01T23:00:00+00:00", "evidence": [evidence_row(
        "Revenue increased from 100 to 120 million dollars. Operating margin was 25 percent.", key="e1",
        available_at="2026-09-01T12:00:00Z", source="sec", title="Issuer quarterly release",
        tickers=["NVDA"], url="https://www.sec.gov/Archives/test.htm")]}


@pytest.fixture
def answer():
    return AnalysisTarget(conclusion="Revenue increased.", materiality="medium",
        claims=[Claim(statement="Revenue increased from 100 to 120 million dollars.", evidence_ids=["e1"], kind="fact")],
        what_changed="Operating margin was 25 percent.", why_it_matters="Higher revenue increases the operating scale.",
        uncertainty=["A causal market reaction is not established."])


class FakeRuntime:
    identity = {"sha256": "a" * 64}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return {"text": next(self.responses), "stopped": True, "truncated": False, "seconds": 0.01}

    def stop(self):
        pass


def test_response_cache_tracks_evidence_and_settings(tmp_path, packet, answer):
    runtime = FakeRuntime([answer.model_dump_json(), answer.model_dump_json()])
    analyst = FinancialAnalyst(AnalystConfig(tmp_path), runtime, as_of=packet["as_of"])
    first = analyst.analyze("What changed?", packet)
    second = analyst.analyze("What changed?", packet)
    assert first.validated and second.cached and runtime.calls == 1
    packet["evidence"][0]["text"] += " Cash generation also increased."
    packet["evidence"][0]["text_hash"] = text_hash(packet["evidence"][0]["text"])
    third = analyst.analyze("What changed?", packet)
    assert third.cache_key != first.cache_key and runtime.calls == 2
    analyst.close()


def test_one_repair_then_validated_subset(tmp_path, packet, answer):
    wrong = answer.model_copy(deep=True)
    wrong.claims.append(Claim(statement="Revenue was 999 million dollars.", evidence_ids=["e1"], kind="fact"))
    runtime = FakeRuntime([wrong.model_dump_json(), wrong.model_dump_json()])
    analyst = FinancialAnalyst(AnalystConfig(tmp_path), runtime, as_of=packet["as_of"])
    report = analyst.analyze("What changed?", packet)
    assert runtime.calls == 2 and not report.validated and len(report.analysis.claims) == 1
    assert "999" not in report.analysis.model_dump_json()
    assert report.errors and len(report.attempts) == 2
    analyst.close()


def test_failed_json_is_not_presented_as_analysis(tmp_path, packet):
    runtime = FakeRuntime(['{"conclusion":', "not json"])
    analyst = FinancialAnalyst(AnalystConfig(tmp_path), runtime, as_of=packet["as_of"])
    report = analyst.analyze("Explain this.", packet)
    assert report.analysis is None and runtime.calls == 2
    assert "No supported analysis" in report._repr_html_()
    analyst.close()


def test_claim_cannot_change_trading_days_into_months(packet, answer):
    text = "TLT returned -5.60 percent over 63 trading days."
    packet["evidence"][0].update(text=text, text_hash=text_hash(text))
    answer.claims = [Claim(statement="TLT returned -5.60 percent over six months.", evidence_ids=["e1"], kind="fact")]
    answer.what_changed = "The return is negative."
    _, errors = check_response(answer.model_dump_json(), packet)
    assert any("horizon" in error for error in errors)
    answer.claims[0].statement = text
    assert not check_response(answer.model_dump_json(), packet)[1]


def test_report_rendering_escapes_evidence_and_preserves_citations(packet, answer):
    answer.conclusion = '<script>alert("x")</script>'
    packet["evidence"][0]["source_url"] = "javascript:alert(1)"
    report = AnalysisReport("Question", "event", packet["as_of"], answer, packet)
    output = report._repr_html_()
    assert "<script>" not in output and "&lt;script&gt;" in output
    assert 'href="javascript:' not in output and "[1]" in output
    assert "Revenue increased from 100 to 120" in output
    assert AnalysisReport.from_dict(report.to_dict()).analysis == answer


def test_summary_direction_matches_the_stated_market_horizon(packet, answer):
    from quantfinlab.analyst.validation import direction_conflicts

    packet["evidence"][0].update(source="structured_context",
        text="Market close snapshot; returns over 1 trading day: SPY +0.85 percent; HYG -0.03 percent; GLD +0.61 percent.")
    assert direction_conflicts("Equities rose while high-yield bonds and gold gained.", packet)
    assert not direction_conflicts("SPY rose and HYG fell, while gold gained.", packet)
    assert not direction_conflicts("HYG gained over 21 trading days.", packet)


def test_partial_answer_does_not_promote_unchecked_paraphrases(packet, answer):
    from quantfinlab.analyst.validation import supported_subset

    answer.claims.append(Claim(statement="The company faces a cash-constrained growth narrative.",
                               evidence_ids=["e1"], kind="fact"))
    partial = supported_subset(answer, packet)
    assert len(partial.claims) == 1
    assert "cash-constrained" not in partial.model_dump_json()


def test_unsupported_fact_without_numbers_cannot_pass(packet, answer):
    answer.claims = [Claim(statement="Broad equity indexes fell slightly but were near all-time highs.",
                           evidence_ids=["e1"], kind="fact")]
    assert any("traceable excerpt" in error for error in check_response(answer.model_dump_json(), packet)[1])


def test_structured_returns_allow_checked_paraphrases_but_not_swapped_values(packet, answer):
    text = "Market close snapshot; returns over 1 trading day: SPY +0.85 percent; HYG -0.03 percent; GLD +0.61 percent."
    packet["evidence"][0].update(source="structured_context", text=text, text_hash=text_hash(text))
    answer.conclusion = answer.what_changed = answer.why_it_matters = "The daily returns differ."
    answer.claims = [Claim(statement="SPY rose, with a return over the trading days of 0.85 percent; HYG returned -0.03 percent.",
                           evidence_ids=["e1"], kind="fact")]
    assert not check_response(answer.model_dump_json(), packet)[1]
    answer.claims[0].statement = "SPY returned -0.03 percent; HYG returned 0.85 percent."
    assert any("traceable excerpt" in error for error in check_response(answer.model_dump_json(), packet)[1])


def test_evidence_time_hash_and_discovery_failures(packet, answer):
    packet["evidence"][0]["available_at"] = "2026-09-02T00:00:00Z"
    _, errors = check_response(answer.model_dump_json(), packet)
    assert any("Future" in error for error in errors)
    packet["evidence"][0]["source"] = "gdelt"
    packet["evidence"][0]["text_hash"] = "wrong"
    _, errors = check_response(answer.model_dump_json(), packet)
    assert any("discovery" in error for error in errors) and any("hash" in error for error in errors)


def test_event_store_filters_versions_and_cutoffs(tmp_path):
    store = EventStore(tmp_path / "events.sqlite")
    stamp = utc("2026-09-01T00:00:00Z")
    event = EventRecord(event_id="event1", available_at=stamp, family="labor", event_type="release",
        what_happened="Payroll release", what_changed="Growth slowed", importance="high",
        document_ids=["doc1"], evidence_ids=["e1"], model_version="model1", prompt_version="v2")
    store.put(event)
    store.put(event)
    assert len(store.list(as_of=stamp, since=stamp - timedelta(days=1), model="model1", prompt="v2")) == 1
    assert not store.list(as_of=stamp - timedelta(seconds=1), since=stamp - timedelta(days=1))
    assert not store.list(as_of=stamp, since=stamp - timedelta(days=1), model="model2")
    store.close()


def test_existing_model_never_redownloaded(tmp_path, monkeypatch):
    import huggingface_hub

    config = AnalystConfig(tmp_path)
    model = tmp_path / "models/local/test.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF fixture")
    from quantfinlab.analyst.inference import file_sha256

    identity = {"filename": "test.gguf", "sha256": file_sha256(model), "size_bytes": model.stat().st_size,
                "context_tokens": 24576, "revision": "fixed", "gguf_repo": "fixture/model"}
    save_json(tmp_path / "models/model_config.json", identity)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: pytest.fail("Model was downloaded again"))
    assert download_model(config) == model
    assert download_model(config) == model
    model.write_bytes(b"corrupted model")
    with pytest.raises(ValueError, match="checksum"):
        download_model(config)


def test_generation_rejects_context_overflow():
    runtime = object.__new__(LlamaRuntime)
    runtime.config = SimpleNamespace(context_tokens=24576, generation_tokens=2300)
    runtime.count = lambda text: 24000
    with pytest.raises(ValueError, match="exceeds"):
        runtime.complete("long prompt")


def test_asof_mismatch_cannot_reuse_analysis(tmp_path, packet, answer):
    runtime = FakeRuntime([answer.model_dump_json()])
    analyst = FinancialAnalyst(AnalystConfig(tmp_path), runtime, as_of="2026-08-01T00:00:00Z")
    with pytest.raises(ValueError, match="cutoffs"):
        analyst.analyze("What changed?", packet)
    assert runtime.calls == 0
    analyst.close()


@pytest.mark.parametrize("stop_type,stopped,truncated", [("eos", True, False), ("limit", False, True)])
def test_native_stream_stop_protocol(stop_type, stopped, truncated):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield b'data: {"content":"A supported answer."}'
            yield b"data: " + json.dumps({"content": "", "stop": True, "stop_type": stop_type}).encode()

    runtime = object.__new__(LlamaRuntime)
    runtime.config = SimpleNamespace(context_tokens=24576, generation_tokens=2300)
    runtime.count = lambda text: 100
    runtime.progress = None
    runtime.url = "http://127.0.0.1:1"
    runtime.session = SimpleNamespace(post=lambda *args, **kwargs: Response())
    output = runtime.complete("Prompt")
    assert output["text"] == "A supported answer."
    assert output["stopped"] == stopped and output["truncated"] == truncated


def test_number_units_and_scales_are_not_interchangeable():
    from quantfinlab.analyst.validation import financial_numbers

    assert financial_numbers("$3.63 billion") == financial_numbers("3630 million USD")
    assert financial_numbers("3.63 million USD") != financial_numbers("3.63 billion USD")
    assert financial_numbers("25%") == financial_numbers("25 percent")
    assert financial_numbers("25 percentage points") != financial_numbers("25 percent")
