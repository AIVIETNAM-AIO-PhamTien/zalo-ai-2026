"""Trusted direct-data question adapter, shared by macros and typed planning.

Full grammar coverage is required. This adapter authorizes raw-column operations
explicitly requested by a question; it does not infer business metrics from names.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from datamate.ir import PlanError
from datamate.semantics import Evidence, GroundedRule, SemanticContract
from datamate.temporal import reference_date, resolve_window
from datamate.temporal_rules import grounded_window


DIRECT = re.compile(
    r"(?P<op>tổng|trung bình|đếm) cột `(?P<column>[^`]+)` của tất cả bản ghi trong bảng `(?P<table>[^`]+)`"
    r"(?: theo cột ngày `(?P<time>[^`]+)` trong (?P<period>[^?]+))?\??", re.IGNORECASE)
ROWS = re.compile(r"đếm tất cả bản ghi trong bảng `(?P<table>[^`]+)`"
                  r"(?: theo cột ngày `(?P<time>[^`]+)` trong (?P<period>[^?]+))?\??", re.IGNORECASE)


def register_contract(warehouse, contract, rule_evidence):
    expected = {key: value for key, value in asdict(contract).items() if key in {
        "concept", "sources", "grain", "expression", "joins", "status_policy", "time_column", "units",
        "transformations", "applicability"}}
    for rule_id in contract.rule_ids:
        warehouse.rules[rule_id] = GroundedRule(rule_id, rule_evidence, expected)
    view_id = "v_" + contract.fingerprint()
    if view_id not in warehouse.views:
        warehouse.add_candidate(contract)
        warehouse.create_view(view_id, contract)
    return view_id


def direct_requirements(question, prepared, period_policy=None):
    text = question.get("question", "").strip()
    match = DIRECT.fullmatch(text) or ROWS.fullmatch(text)
    if match is None:
        return None
    fields = match.groupdict()
    table, column = fields["table"], fields.get("column")
    schemas, warehouse = prepared.schemas, prepared.warehouse
    if table not in schemas:
        raise PlanError("adapter_source", "explicit table is unavailable or has unresolved projections")
    if column is not None and column not in schemas[table]:
        raise PlanError("adapter_source", "explicit source column does not exist")
    op = {"tổng": "sum", "trung bình": "avg", "đếm": "count"}[fields.get("op", "đếm").lower()]
    if op in {"sum", "avg"} and schemas[table][column] != "number":
        raise PlanError("adapter_type", "raw-column arithmetic needs an evidenced numeric projection")
    relation = {"kind": "scan", "table": table}
    time_column = fields.get("time")
    time_evidence = []
    if time_column:
        if schemas[table].get(time_column) != "date":
            raise PlanError("adapter_type", "explicit time column needs a date projection")
        reference = reference_date(question=question["as_of_date"]) if question.get("as_of_date") is not None else None
        if period_policy is None:
            window, time_evidence = grounded_window(fields["period"].strip(), reference, warehouse.catalog.foundation.documents)
        else:
            window = resolve_window(fields["period"].strip(), reference, period_policy)
        relation = window.apply(relation, time_column)
    value = {"kind": "column", "name": column} if column else {"kind": "literal", "type": "number", "value": 1}
    relation = {"kind": "aggregate", "input": relation, "groups": [],
                "metrics": [{"name": "value", "expression": {"kind": "aggregate", "op": op, "value": value}}]}
    digest = hashlib.sha256((text + json.dumps(relation, sort_keys=True)).encode()).hexdigest()
    document, rule_id = "$direct_question:" + digest, "direct:" + digest
    warehouse.catalog.foundation.documents[document] = text
    evidence = Evidence(document, 0, len(text), text)
    # A row count consumes physical records, including valid duplicate rows.
    columns = [column] if column else list(schemas[table])
    if time_column and time_column not in columns:
        columns.append(time_column)
    contract = SemanticContract(
        concept="explicit raw-column " + op, sources=[(table, c) for c in columns],
        grain=[(table, "$physical_row")], expression=relation,
        time_column=(table, time_column) if time_column else None,
        units="records" if op == "count" else "source-column units (no conversion requested)",
        evidence=[evidence] + time_evidence, applicability={"exact_request": text, "all_records": True}, rule_ids=[rule_id])
    view_id = register_contract(warehouse, contract, evidence)
    return {"relation": relation, "answer_type": "number", "contract_ids": [view_id]}
