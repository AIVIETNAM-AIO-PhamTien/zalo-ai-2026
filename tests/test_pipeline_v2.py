import json

import pytest

from datamate.pipeline import BusinessCache, Checkpoint, PreparedBusiness, prepare_business, run_questions
from test_semantics_v2 import warehouse
from test_verifier_policy_v2 import verified_fixture


def test_explicit_projection_retains_raw_and_rejects_silent_null(tmp_path):
    warehouse(tmp_path)
    prepared = prepare_business(tmp_path, "", {"facts.csv": {"Measurement": {"type": "number", "locale": "en"}}})
    assert prepared.schemas["facts.csv"]["Measurement"] == "number"
    assert prepared.warehouse.catalog.foundation.tables["facts.csv"].rows[1][1] == "10"
    assert len(prepared.conversions) == 2
    (tmp_path / "facts.csv").write_text("Ref,Measurement\na,10\nb,not-a-number\n")
    bad = prepare_business(tmp_path, "", {"facts.csv": {"Measurement": {"type": "number", "locale": "en"}}})
    assert "Measurement" not in bad.frames["facts.csv"]
    assert "Ref" in bad.frames["facts.csv"]
    assert len(bad.frames["facts.csv"]) == 2
    assert bad.diagnostics[-1]["code"] == "conversion"


def test_cache_is_source_rule_projection_and_logic_sensitive(tmp_path):
    warehouse(tmp_path)
    cache = BusinessCache(max_businesses=1)
    first = cache.get(tmp_path, "rule1")
    assert cache.get(tmp_path, "rule1") is first
    second = cache.get(tmp_path, "rule2")
    assert second is not first
    (tmp_path / "facts.csv").write_text("Ref,Measurement\na,33\n")
    assert cache.get(tmp_path, "rule2") is not second
    assert len(cache.values) == 1


def test_checkpoint_recovery_rebuilds_submission_and_rejects_changed_inputs(tmp_path):
    checkpoint = Checkpoint(tmp_path, ["b", "a"], {"config": "one"})
    checkpoint.record("a", {"type": "number", "answer": 0})
    (tmp_path / "submission.jsonl").write_text("partial crash garbage")
    recovered = Checkpoint(tmp_path, ["b", "a"], {"config": "one"})
    assert json.loads((tmp_path / "submission.jsonl").read_text()) == {"id": "a", "type": "number", "answer": 0}
    recovered.record("b", {"type": "abstain"})
    assert [json.loads(l)["id"] for l in (tmp_path / "submission.jsonl").read_text().splitlines()] == ["b", "a"]
    with pytest.raises(ValueError, match="differ"):
        Checkpoint(tmp_path, ["b", "a"], {"config": "two"})


def test_question_failure_isolated_and_gold_never_written(tmp_path):
    business = tmp_path / "business"
    business.mkdir()
    w, p, frames, schemas, req, _ = verified_fixture(business)
    prepared = PreparedBusiness(w, frames, schemas)
    checkpoint = Checkpoint(tmp_path / "out", ["bad", "good"], {})
    def requirements(question):
        if question["id"] == "bad":
            raise ValueError("fixture failure")
        return req
    questions = [{"id": "bad", "question": "bad", "gold": "SECRET_GOLD"},
                 {"id": "good", "question": "duration", "gold": "SECRET_GOLD"}]
    traces = run_questions(questions, prepared, checkpoint, requirements=requirements)
    assert checkpoint.rows["bad"]["type"] == "abstain"
    assert checkpoint.rows["good"]["answer"] == 30
    assert run_questions(questions, prepared, checkpoint, requirements=requirements) == []
    for path in (tmp_path / "out").rglob("*.json*"):
        assert "SECRET_GOLD" not in path.read_text()


def test_checkpoint_rejects_extra_output_fields(tmp_path):
    checkpoint = Checkpoint(tmp_path, ["q"], {})
    with pytest.raises(ValueError, match="unexpected"):
        checkpoint.record("q", {"type": "abstain", "gold": 123})


def test_submission_write_failure_preserves_canonical_progress(tmp_path, monkeypatch):
    import datamate.pipeline as pipeline
    checkpoint = Checkpoint(tmp_path, ["q"], {})
    original = pipeline.os.replace
    def fail_submission(source, destination):
        if str(destination).endswith("submission.jsonl"):
            raise OSError("simulated interrupted submission write")
        return original(source, destination)
    monkeypatch.setattr(pipeline.os, "replace", fail_submission)
    with pytest.raises(OSError):
        checkpoint.record("q", {"type": "number", "answer": 0})
    assert checkpoint.rows["q"]["answer"] == 0
    monkeypatch.setattr(pipeline.os, "replace", original)
    restored = Checkpoint(tmp_path, ["q"], {})
    assert restored.rows["q"]["answer"] == 0


@pytest.mark.parametrize("answer", [{"type": "number", "answer": True}, {"type": "number", "answer": "12"},
    {"type": "date", "answer": "2026-02-30"}, {"type": "percent", "answer": float("nan")},
    {"type": "table", "answer": {"columns": ["x"], "rows": [[1, 2]]}},
    {"type": "list", "answer": [{"raw": 1}]}])
def test_checkpoint_rejects_malformed_values(tmp_path, answer):
    checkpoint = Checkpoint(tmp_path, ["q"], {})
    with pytest.raises(ValueError, match="typed answer"):
        checkpoint.record("q", answer)
