"""Grounded relational obligations are checked independently of execution."""
from datamate.semantic_obligations import compare_relation
from datamate.engine import execute
from datamate.verifier import verify
from test_verifier_policy_v2 import verified_fixture


def column(name):
    return {"kind": "column", "name": name}


def literal(value, kind="number"):
    return {"kind": "literal", "type": kind, "value": value}


def binary(op, left, right):
    return {"kind": "binary", "op": op, "left": left, "right": right}


def test_adjacent_filters_and_boolean_order_are_equivalent():
    paid = binary("eq", column("state"), literal("paid", "text"))
    positive = binary("gt", column("amount"), literal(0))
    scan = {"kind": "scan", "table": "sales"}
    reference = {"kind": "filter", "input": {"kind": "filter", "input": scan,
                 "predicate": paid}, "predicate": positive}
    candidate = {"kind": "filter", "input": scan,
                 "predicate": binary("and", positive,
                     binary("eq", literal("paid", "text"), column("state")))}
    assert compare_relation(candidate, reference) == []


def test_wrong_metric_filter_grain_and_source_get_targeted_diagnostics():
    scan = {"kind": "scan", "table": "sales"}
    paid = binary("eq", column("state"), literal("paid", "text"))
    base = {"kind": "filter", "input": scan, "predicate": paid}
    reference = {"kind": "aggregate", "input": base,
                 "groups": [{"name": "state", "expression": column("state")}],
                 "metrics": [{"name": "total", "expression": {"kind": "aggregate",
                             "op": "sum", "value": column("amount")}}]}
    wrong = {**reference, "input": {"kind": "filter", "input":
             {"kind": "scan", "table": "refunds"}, "predicate":
             binary("eq", column("state"), literal("open", "text"))},
             "groups": [], "metrics": [{"name": "total", "expression":
             {"kind": "aggregate", "op": "count", "value": column("amount")}}]}
    failures = compare_relation(wrong, reference)
    assert {f.code for f in failures} >= {
        "source_obligation", "filter_obligation", "grain_obligation", "metric_obligation"}
    assert all(f.path.startswith("/relation/") for f in failures)


def test_extra_predicate_and_changed_boolean_operator_are_rejected():
    scan = {"kind": "scan", "table": "sales"}
    paid = binary("eq", column("state"), literal("paid", "text"))
    positive = binary("gt", column("amount"), literal(0))
    reference = {"kind": "filter", "input": scan, "predicate": paid}
    extra = {"kind": "filter", "input": scan, "predicate": binary("and", paid, positive)}
    wrong_or = {"kind": "filter", "input": scan, "predicate": binary("or", paid, positive)}
    assert any(f.code == "filter_obligation" for f in compare_relation(extra, reference))
    assert any(f.code == "filter_obligation" for f in compare_relation(wrong_or, extra))


def test_verifier_accepts_equivalent_filters_but_rejects_wrong_measure(tmp_path):
    warehouse, plan, frames, schemas, requirements, _ = verified_fixture(tmp_path)
    positive = binary("gt", column("Measurement"), literal(0))
    nonnegative = binary("ge", column("Measurement"), literal(0))
    scan = {"kind": "scan", "table": "facts.csv"}
    expected_relation = {"kind": "filter", "input": {"kind": "filter",
                         "input": scan, "predicate": positive},
                         "predicate": nonnegative}
    requirements = {**requirements, "relation": {**plan["relation"],
                    "input": expected_relation}}
    equivalent = {**plan, "relation": {**plan["relation"],
                  "input": {"kind": "filter", "input": scan,
                            "predicate": binary("and", nonnegative, positive)}}}
    result = execute(equivalent, frames, schemas)
    report = verify(equivalent, result, frames, schemas, warehouse, requirements, independent=True)
    assert report.accepted

    wrong = {**equivalent, "relation": {**equivalent["relation"],
             "metrics": [{"name": "answer", "expression":
             {"kind": "aggregate", "op": "count", "value": column("Measurement")}}]}}
    wrong_result = execute(wrong, frames, schemas)
    rejected = verify(wrong, wrong_result, frames, schemas, warehouse, requirements)
    assert not rejected.semantic_valid
    assert any(d.code == "metric_obligation" for d in rejected.diagnostics)
