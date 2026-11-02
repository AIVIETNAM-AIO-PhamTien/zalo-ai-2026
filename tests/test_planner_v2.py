import json
import pandas as pd

from datamate.model import BackendError, LocalModel, ScriptedModel
from datamate.planner import solve
from datamate.policy import Budget
from test_engine_v2 import plan
from test_verifier_policy_v2 import verified_fixture

import pytest


def test_verified_macro_uses_no_model_calls(tmp_path):
    w, p, frames, schemas, requirements, _ = verified_fixture(tmp_path)
    model = ScriptedModel([])
    result = solve({"question": "total duration"}, w, frames, schemas, model, requirements)
    assert result.selection.answer == {"type": "number", "answer": 30}
    assert result.llm_calls == 0
    assert model.calls == []


def test_invalid_json_targeted_repair_then_missing_semantics_stops(tmp_path):
    w, p, frames, schemas, _, _ = verified_fixture(tmp_path)
    model = ScriptedModel(["invalid", p, p])
    result = solve({"question": "duration"}, w, frames, schemas, model)
    assert result.selection.answer["type"] == "abstain"
    assert result.llm_calls == 2
    assert any(d["code"] == "invalid_json" for d in result.diagnostics)
    assert any(d["code"] == "missing_grounding" for d in result.diagnostics)
    assert len(model.calls[-1]) == 4


def test_nonexistent_source_is_repaired_with_schema_diagnostic(tmp_path):
    w, p, frames, schemas, _, _ = verified_fixture(tmp_path)
    model = ScriptedModel([plan({"kind": "scan", "table": "made_up"}), p])
    result = solve({"question": "duration"}, w, frames, schemas, model)
    assert result.llm_calls == 2
    assert "unknown_table" in model.calls[1][-1]["content"]


def test_planner_context_contains_only_candidate_value_hits(tmp_path):
    w, p, frames, schemas, _, _ = verified_fixture(tmp_path)
    w.catalog.frames["facts.csv"]["Ref"] = ["Coca Cola", "Other"]
    model = ScriptedModel([p])
    solve({"question": "Coca Cola duration"}, w, frames, schemas, model)
    context = json.loads(model.calls[0][1]["content"])
    assert context["value_candidates"]["matches"][0]["value"] == "Coca Cola"
    assert context["value_candidates"]["matches"][0]["status"] == "candidate"


def test_value_hit_can_seed_schema_selection_before_fallback(tmp_path):
    w, _, frames, schemas, _, _ = verified_fixture(tmp_path)
    w.catalog.frames["Z"] = pd.DataFrame({"name": ["Coca Cola"]}, dtype=object)
    schemas = {**{name: {"name": "text"} for name in "ABCD"}, "Z": {"name": "text"}}
    model = ScriptedModel(["invalid"])
    solve({"question": "Coca Cola"}, w, frames, schemas, model,
          budget=Budget(max_repairs=0, max_candidates=1))
    context = json.loads(model.calls[0][1]["content"])
    assert "Z" in context["schema"]
    assert context["value_candidates"]["matches"][0]["table"] == "Z"


def test_scalar_scan_is_rejected_before_execution(tmp_path):
    w, _, frames, schemas, _, _ = verified_fixture(tmp_path)
    model = ScriptedModel([plan({"kind": "scan", "table": "facts.csv"})])
    result = solve({"question": "duration"}, w, frames, schemas, model,
                   budget=Budget(max_repairs=0, max_candidates=1))
    assert not result.candidates
    assert result.diagnostics[0]["code"] == "output_shape"


@pytest.mark.parametrize("message", ["timeout", "GPU out of memory", "server stopped"])
def test_backend_failure_is_bounded(tmp_path, message):
    w, _, frames, schemas, _, _ = verified_fixture(tmp_path)
    model = ScriptedModel([BackendError(message)])
    result = solve({"question": "duration"}, w, frames, schemas, model)
    assert result.llm_calls == 1
    assert result.selection.answer["type"] == "abstain"


def test_repair_budget_and_oversized_context(tmp_path):
    w, _, frames, schemas, _, _ = verified_fixture(tmp_path)
    model = ScriptedModel(["bad"] * 10)
    result = solve({"question": "duration"}, w, frames, schemas, model, budget=Budget(max_repairs=1))
    assert result.llm_calls == 2
    result = solve({"question": "duration"}, w, frames, schemas, model, context_bytes=1)
    assert result.llm_calls == 0
    assert result.diagnostics[-1]["code"] == "context_budget"


@pytest.mark.parametrize("url", ["https://api.example.com/v1", "http://localhost:8000/v1", "http://127.0.0.1.evil/v1",
                                "http://user:pass@127.0.0.1/v1", "http://192.168.1.1/v1"])
def test_offline_model_rejects_external_or_dns_resolved_endpoints(url):
    with pytest.raises(ValueError):
        LocalModel(url, "local", "pinned", 1_000_000)


def test_model_parameter_cap():
    with pytest.raises(ValueError):
        LocalModel("http://127.0.0.1:8000/v1", "local", "pinned", 15_000_000_001)


def test_invalid_reference_date_stops_before_macro_or_model(tmp_path):
    w, _, frames, schemas, requirements, _ = verified_fixture(tmp_path)
    model = ScriptedModel([])
    result = solve({"question": "duration", "as_of_date": "2026-02-30"}, w, frames, schemas, model, requirements)
    assert result.selection.answer["type"] == "abstain"
    assert result.diagnostics[0]["code"] == "temporal_reference"
    assert model.calls == []
