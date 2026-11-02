"""Document-grounded named metrics for an explicit, machine-checkable rule subset.

Rules refer to documented meanings, not physical column names or industry schemas.
Unrecognized rule suffixes are rejected as a whole. Natural-language rules outside
this subset need another tested adapter; citations alone cannot authorize them.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from datamate.adapters import register_contract
from datamate.ir import PlanError
from datamate.semantics import Evidence, SemanticContract
from datamate.temporal import reference_date, resolve_window
from datamate.temporal_rules import grounded_window


RULE = re.compile(r"(?P<operation>tổng|trung bình|đếm|đếm phân biệt) cột có nghĩa `(?P<meaning>[^`]+)`; "
                  r"đơn vị `(?P<unit>[^`]+)`; tất cả bản ghi"
                  r"(?:; thời gian có nghĩa `(?P<time>[^`]+)`)?", re.IGNORECASE)


@dataclass(frozen=True)
class MetricRule:
    concept: str
    operation: str
    meaning: str
    unit: str
    time_meaning: str | None
    evidence: Evidence


def metric_rules(warehouse):
    text = warehouse.catalog.foundation.documents["$rulebook"]
    rules, offset = [], 0
    for line in text.splitlines(keepends=True):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and len(cells) == 2:
            match = RULE.fullmatch(cells[1])
            if match and cells[0]:
                rules.append(MetricRule(cells[0], {"tổng": "sum", "trung bình": "avg", "đếm": "count",
                    "đếm phân biệt": "count_distinct"}[match["operation"].lower()], match["meaning"],
                    match["unit"], match["time"], Evidence("$rulebook", offset, offset + len(line), line)))
        offset += len(line)
    return rules


def metric_requirements(question, prepared, period_policy=None):
    text = question.get("question", "").strip().removesuffix("?").strip()
    warehouse = prepared.warehouse
    matches = []
    for rule in metric_rules(warehouse):
        # The named concept comes from rulebook evidence, not a fixed metric registry.
        if text.casefold() == rule.concept.casefold():
            matches.append((rule, None))
        elif text.casefold().startswith(rule.concept.casefold() + " trong "):
            matches.append((rule, text[len(rule.concept) + len(" trong "):]))
    if not matches:
        return None
    if len(matches) != 1:
        raise PlanError("metric_ambiguity", "multiple rule definitions match the requested metric")
    rule, period = matches[0]
    definitions = [line.strip().strip("|").split("|")[0].strip()
                   for line in warehouse.catalog.foundation.documents["$rulebook"].splitlines()
                   if line.lstrip().startswith("|")]
    if sum(concept.casefold() == rule.concept.casefold() for concept in definitions) != 1:
        raise PlanError("metric_ambiguity", "conflicting or repeated metric definitions require resolution")
    bindings = [b for b in warehouse.bindings if b.description.casefold() == rule.meaning.casefold()]
    if len(bindings) != 1:
        raise PlanError("metric_binding", "metric meaning must bind to exactly one documented source")
    binding = bindings[0]
    table, column = binding.table, binding.column
    if table not in prepared.schemas or column not in prepared.schemas[table]:
        raise PlanError("metric_source", "bound metric source has no valid typed projection")
    if rule.operation in {"sum", "avg"} and prepared.schemas[table][column] != "number":
        raise PlanError("metric_type", "metric requires evidenced numeric conversion")
    relation = {"kind": "scan", "table": table}
    sources, evidence = [(table, column)], [rule.evidence, binding.evidence]
    time_column = None
    if period:
        dates = [b for b in warehouse.bindings if b.table == table and rule.time_meaning is not None
                 and b.description.casefold() == rule.time_meaning.casefold()]
        if len(dates) != 1 or prepared.schemas[table].get(dates[0].column) != "date":
            raise PlanError("metric_time", "metric period needs one documented typed time column")
        time_column = (table, dates[0].column)
        sources.append(time_column)
        evidence.append(dates[0].evidence)
        reference = reference_date(question=question["as_of_date"]) if question.get("as_of_date") else None
        if period_policy is None:
            window, time_evidence = grounded_window(period, reference, warehouse.catalog.foundation.documents)
            evidence.extend(time_evidence)
        else:
            window = resolve_window(period, reference, period_policy)
        relation = window.apply(relation, dates[0].column)
    relation = {"kind": "aggregate", "input": relation, "groups": [], "metrics": [
        {"name": "value", "expression": {"kind": "aggregate", "op": rule.operation,
                                           "value": {"kind": "column", "name": column}}}]}
    key = hashlib.sha256((text + rule.evidence.quote + json.dumps(relation, sort_keys=True)).encode()).hexdigest()
    contract = SemanticContract(rule.concept, sources, [(table, "$physical_row")], relation,
        status_policy=[{"scope": "all_records", "evidence": rule.evidence.quote}], time_column=time_column,
        units=rule.unit, evidence=evidence, applicability={"exact_request": text, "operation": rule.operation},
        rule_ids=["named_metric:" + key])
    view_id = register_contract(warehouse, contract, rule.evidence)
    return {"relation": relation, "answer_type": "number", "contract_ids": [view_id]}
