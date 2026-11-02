from dataclasses import asdict, replace
from decimal import Decimal

import pandas as pd
import pytest

from datamate.engine import execute
from datamate.policy import Budget, Candidate, cell, select
from datamate.verifier import ExactEvaluator, verify
from test_engine_v2 import SCAN, SCHEMA, aggregate, binary, col, facts, lit, named, plan
from test_semantics_v2 import grounded_contract, warehouse


def verified_fixture(tmp_path):
    w = warehouse(tmp_path)
    contract = grounded_contract(w)
    w.create_view("v_duration", contract)
    frames = {"facts.csv": pd.DataFrame({"Ref": ["a", "b"],
        "Measurement": [Decimal("10"), Decimal("20")]}, dtype=object)}
    schemas = {"facts.csv": {"Ref": "text", "Measurement": "number"}}
    p = plan(aggregate({"kind": "scan", "table": "facts.csv"},
                       {"kind": "aggregate", "op": "sum", "value": col("Measurement")}))
    p["contract_ids"] = ["v_duration"]
    req = {k: p[k] for k in ("relation", "answer_type", "contract_ids")}
    result = execute(p, frames, schemas)
    verified = verify(p, result, frames, schemas, w, req, independent=True)
    return w, p, frames, schemas, req, Candidate(p, result, verified, "macro")


def test_fully_grounded_candidate_and_no_extra_inference(tmp_path):
    _, _, _, _, _, candidate = verified_fixture(tmp_path)
    assert candidate.verification.accepted
    assert select([candidate]).answer == {"type": "number", "answer": 30}
    assert Budget().next_action([candidate], repairs=0, elapsed=1) == "answer"


def test_independent_sum_does_not_establish_semantics():
    p = plan(aggregate(SCAN, {"kind": "aggregate", "op": "sum", "value": col("amount")}))
    result = execute(p, facts(), SCHEMA)
    report = verify(p, result, facts(), SCHEMA, independent=True)
    assert report.exact_result == [(Decimal("3.30"),)]
    assert not report.accepted
    assert select([Candidate(p, result, report, "llm")]).answer == {"type": "abstain"}


def test_same_wrong_semantics_in_multiple_candidates_still_abstains(tmp_path):
    w, p, frames, schemas, req, candidate = verified_fixture(tmp_path)
    wrong = {**p, "relation": aggregate({"kind": "scan", "table": "facts.csv"},
             {"kind": "aggregate", "op": "count", "value": col("Measurement")})}
    result = execute(wrong, frames, schemas)
    report = verify(wrong, result, frames, schemas, w, req, independent=True)
    assert any(d.code == "metric_obligation" for d in report.diagnostics)
    wrong_candidate = Candidate(wrong, result, report, "llm")
    assert select([wrong_candidate, wrong_candidate]).answer["type"] == "abstain"


def test_arithmetic_corruption_and_mutated_contract_are_detected(tmp_path):
    w, p, frames, schemas, req, candidate = verified_fixture(tmp_path)
    result = replace(candidate.result, rows=[(Decimal(999),)])
    report = verify(p, result, frames, schemas, w, req, independent=True)
    assert any(d.code == "independent_mismatch" for d in report.diagnostics)
    w.views["v_duration"].contract.units = "VND"
    report = verify(p, candidate.result, frames, schemas, w, req)
    assert any(d.code == "mutated_contract" for d in report.diagnostics)


def test_exact_percentage_rounding_and_null_division():
    rel = {"kind": "project", "input": SCAN,
           "columns": [named("ratio", binary("divide", col("amount"), lit(3)))]}
    _, rows = ExactEvaluator(facts()).evaluate(rel)
    assert cell(rows[0][0], percent=True) == 0.37
    assert rows[-1] == (None,)
    assert cell(Decimal("1.005"), percent=True) == 1.01
    assert cell(Decimal("-1.005"), percent=True) == -1.01


def test_exact_evaluator_case_sum_and_grouping():
    expr = {"kind": "case", "when": binary("eq", col("id"), lit("b", "text")),
            "then": col("amount"), "else": lit(0)}
    rel = aggregate(SCAN, {"kind": "aggregate", "op": "sum", "value": expr})
    assert ExactEvaluator(facts()).evaluate(rel)[1] == [(Decimal("2.20"),)]


def test_output_scalar_shape_bool_and_precision_guards(tmp_path):
    _, _, _, _, _, candidate = verified_fixture(tmp_path)
    wrong = replace(candidate, result=replace(candidate.result, rows=[(True,)]),
                    verification=replace(candidate.verification, exact_result=[(True,)]))
    assert select([wrong]).answer["type"] == "abstain"
    with pytest.raises(ValueError, match="exact decimal"):
        cell(Decimal("1.1234567890123456789012345"))
    assert Budget(max_seconds=1).next_action([], 0, 2) == "abstain"


def test_budget_does_not_turn_missing_semantics_into_more_votes(tmp_path):
    _, _, _, _, _, candidate = verified_fixture(tmp_path)
    bad = replace(candidate, verification=replace(candidate.verification, semantic_valid=False))
    assert Budget(max_repairs=1).next_action([bad], 0, 1) == "repair"
    assert Budget(max_repairs=1).next_action([bad], 1, 1) == "abstain"


def test_independent_work_budget_scales_past_fifty_thousand_source_rows():
    count = 50_001
    frames = {"large": pd.DataFrame({"amount": list(range(count))}, dtype=object)}
    schemas = {"large": {"amount": "number"}}
    p = plan(aggregate({"kind": "scan", "table": "large"},
                       {"kind": "aggregate", "op": "sum", "value": col("amount")}))
    result = execute(p, frames, schemas)
    verified = verify(p, result, frames, schemas, independent=True)
    assert verified.exact_result == [(count * (count - 1) // 2,)]
    assert not any(d.level == "numerical" for d in verified.diagnostics)
    assert not verified.semantic_valid
