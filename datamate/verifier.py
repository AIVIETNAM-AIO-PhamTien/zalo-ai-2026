"""Structured verification. Independent arithmetic does not establish semantics."""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, localcontext

from datamate.engine import Compiler, ExecutionResult
from datamate.ir import PlanError
from datamate.semantics import SemanticWarehouse
from datamate.semantic_obligations import compare_relation


@dataclass
class Diagnostic:
    level: str
    code: str
    message: str
    path: str = ""


@dataclass
class Verification:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    exact_result: list[tuple] | None = None
    structural_valid: bool = False
    semantic_valid: bool = False
    execution_valid: bool = False

    @property
    def accepted(self):
        return self.structural_valid and self.semantic_valid and self.execution_valid and not self.diagnostics


def nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


class ExactEvaluator:
    """Small independent row evaluator for bounded financial/grouping checks.

    Unsupported operators fail closed. It shares IR semantics, not DuckDB SQL or
    DuckDB intermediate values; no claim about question interpretation is made.
    """
    def __init__(self, frames, max_rows=50000):
        self.frames, self.max_rows = frames, max_rows
        self.work = 0

    def charge(self, count):
        self.work += count
        if self.work > self.max_rows:
            raise PlanError("independent_budget", "independent computation work budget exceeded")

    def expression(self, node, row, group=None):
        self.charge(1)
        kind = node["kind"]
        if kind == "column":
            return row[node["name"]]
        if kind == "literal":
            value = node["value"]
            if node["type"] == "number":
                return Decimal(str(value))
            if node["type"] == "date":
                return dt.datetime.fromisoformat(value)
            return value
        if kind == "is_null":
            return self.expression(node["value"], row) is None
        if kind == "not":
            value = self.expression(node["value"], row)
            return None if value is None else not value
        if kind == "case":
            branch = "then" if self.expression(node["when"], row, group) is True else "else"
            return self.expression(node[branch], row, group)
        if kind == "date_part":
            value = self.expression(node["value"], row)
            return None if value is None else getattr(value, node["part"])
        if kind == "binary":
            left = self.expression(node["left"], row, group)
            right = self.expression(node["right"], row, group)
            op = node["op"]
            if op == "and":
                return False if left is False or right is False else None if left is None or right is None else True
            if op == "or":
                return True if left is True or right is True else None if left is None or right is None else False
            if left is None or right is None:
                return None
            if op in {"add", "subtract", "multiply", "divide"}:
                left, right = Decimal(left), Decimal(right)
            operations = {"add": lambda: left + right, "subtract": lambda: left - right,
                          "multiply": lambda: left * right, "divide": lambda: None if right == 0 else left / right,
                          "eq": lambda: left == right, "ne": lambda: left != right, "lt": lambda: left < right,
                          "le": lambda: left <= right, "gt": lambda: left > right, "ge": lambda: left >= right}
            return operations[op]()
        if kind == "aggregate":
            if group is None:
                raise PlanError("independent", "aggregate lacks group")
            values = [self.expression(node["value"], item) for item in group]
            values = [v for v in values if v is not None]
            op = node["op"]
            if op == "count":
                return len(values)
            if op == "count_distinct":
                return len(set(values))
            if not values:
                return None
            if op in {"sum", "avg"}:
                total = sum((Decimal(v) for v in values), Decimal(0))
                return total if op == "sum" else total / len(values)
            return min(values) if op == "min" else max(values)
        raise PlanError("independent_unsupported", kind)

    def relation(self, node):
        kind = node["kind"]
        if kind == "scan":
            frame = self.frames[node["table"]]
            self.charge(len(frame))
            return list(frame.columns), frame.to_dict("records")
        if kind == "union":
            inputs = [self.relation(child) for child in node["inputs"]]
            columns = inputs[0][0]
            if any(names != columns for names, _ in inputs):
                raise PlanError("independent_schema", "union columns differ")
            rows = [row for _, records in inputs for row in records]
            self.charge(len(rows))
            return columns, rows  # UNION ALL: duplicates are evidence, not automatically removed.
        if kind == "join":
            left_columns, left_rows = self.relation(node["left"])
            right_columns, right_rows = self.relation(node["right"])
            pairs = node["keys"]
            def join_key(row, side):
                values = tuple(row[pair[side]] for pair in pairs)
                return None if any(value is None for value in values) else values
            lookup, left_counts = defaultdict(list), defaultdict(int)
            for row in right_rows:
                self.charge(1)
                key = join_key(row, "right")
                if key is not None:
                    lookup[key].append(row)
            if node["cardinality"].endswith("one") and any(len(records) > 1 for records in lookup.values()):
                raise PlanError("independent_cardinality", "right join keys are not unique")
            for row in left_rows:
                self.charge(1)
                key = join_key(row, "left")
                if key is not None:
                    left_counts[key] += 1
            if node["cardinality"].startswith("one") and any(n > 1 for n in left_counts.values()):
                raise PlanError("independent_cardinality", "left join keys are not unique")
            output = []
            for row in left_rows:
                matches = lookup.get(join_key(row, "left"), [])
                if node["how"] == "anti":
                    if not matches:
                        self.charge(1)
                        output.append(dict(row))
                elif matches:
                    for match in matches:
                        self.charge(1)
                        output.append({**row, **match})
                elif node["how"] == "left":
                    self.charge(1)
                    output.append({**row, **dict.fromkeys(right_columns)})
            columns = left_columns if node["how"] == "anti" else left_columns + right_columns
            return columns, output
        columns, rows = self.relation(node["input"])
        if kind == "filter":
            return columns, [r for r in rows if self.expression(node["predicate"], r) is True]
        if kind == "project":
            return [c["name"] for c in node["columns"]], [
                {c["name"]: self.expression(c["expression"], r) for c in node["columns"]} for r in rows]
        if kind == "aggregate":
            groups = defaultdict(list)
            for row in rows:
                group_key = tuple(self.expression(g["expression"], row) for g in node["groups"])
                groups[group_key].append(row)
            if not node["groups"] and not groups:
                groups[()] = []
            output = []
            for group_key, values in groups.items():
                record = {g["name"]: v for g, v in zip(node["groups"], group_key)}
                for metric in node["metrics"]:
                    record[metric["name"]] = self.expression(metric["expression"], values[0] if values else {}, values)
                output.append(record)
            return [v["name"] for v in node["groups"] + node["metrics"]], output
        if kind == "sort":
            for spec in reversed(node["by"]):
                name = spec["column"]
                present = [r for r in rows if r[name] is not None]
                absent = [r for r in rows if r[name] is None]
                rows = sorted(present, key=lambda r: r[name], reverse=spec["descending"]) + absent
            return columns, rows
        if kind == "limit":
            return columns, rows[:node["count"]]
        if kind == "distinct":
            seen, out = set(), []
            for row in rows:
                key = tuple(row[c] for c in columns)
                if key not in seen:
                    seen.add(key)
                    out.append(row)
            return columns, out
        raise PlanError("independent_unsupported", kind)

    def evaluate(self, relation):
        with localcontext() as context:
            context.prec = 80
            columns, rows = self.relation(relation)
            return columns, [tuple(r[c] for c in columns) for r in rows]


def verify(plan: dict, result: ExecutionResult, frames, schemas, warehouse: SemanticWarehouse | None = None,
           requirements: dict | None = None, independent=False) -> Verification:
    report = Verification()
    try:
        compiled = Compiler(schemas).compile(plan)
        report.structural_valid = True
    except PlanError as exc:
        report.diagnostics.append(Diagnostic("structural", exc.code, str(exc), exc.path))
        return report
    if result.columns != list(compiled.columns) or sorted(result.sources) != sorted(compiled.sources):
        report.diagnostics.append(Diagnostic("execution", "provenance_mismatch", "execution result does not match compiled plan"))
    else:
        report.execution_valid = True
    # Requirements are supplied by a tested rule/question adapter, not model JSON.
    if warehouse is None or requirements is None:
        report.diagnostics.append(Diagnostic("semantic", "missing_grounding", "independent question requirements required"))
    else:
        if requirements.get("answer_type") != plan["answer_type"]:
            report.diagnostics.append(Diagnostic("semantic", "answer_type", "question output type mismatch"))
        if requirements.get("relation") is None:
            report.diagnostics.append(Diagnostic("semantic", "missing_grounding", "grounded relation obligations required"))
        else:
            for mismatch in compare_relation(plan["relation"], requirements["relation"]):
                report.diagnostics.append(Diagnostic("semantic", mismatch.code, mismatch.message, mismatch.path))
        if requirements.get("contract_ids") != plan["contract_ids"] or not plan["contract_ids"]:
            report.diagnostics.append(Diagnostic("semantic", "contracts", "required contracts missing or changed"))
        if plan["assumptions"]:
            report.diagnostics.append(Diagnostic("semantic", "assumptions", "unresolved plan assumptions"))
        for contract_id in plan["contract_ids"]:
            view = warehouse.views.get(contract_id)
            if view is None:
                report.diagnostics.append(Diagnostic("semantic", "unknown_contract", contract_id))
                continue
            if view.contract.fingerprint() != view.fingerprint:
                report.diagnostics.append(Diagnostic("semantic", "mutated_contract", contract_id))
            for error in warehouse.validate_contract(view.contract):
                report.diagnostics.append(Diagnostic("semantic", "contract", error, contract_id))
        report.semantic_valid = not any(d.level == "semantic" for d in report.diagnostics)
    if independent:
        try:
            # Scale work to the selected source size, with a hard cap. The old
            # fixed 50k cap rejected even a scan of a 56k-row training source.
            source_rows = sum(len(frames[name]) for name in compiled.sources)
            work_budget = min(2_000_000, max(50_000, source_rows * 32))
            columns, exact = ExactEvaluator(frames, max_rows=work_budget).evaluate(plan["relation"])
            report.exact_result = exact
            if columns != result.columns or len(exact) != len(result.rows):
                raise PlanError("independent_mismatch", "independent output shape differs")
            for expected_row, actual_row in zip(exact, result.rows):
                for expected, actual in zip(expected_row, actual_row):
                    if isinstance(actual, float) and isinstance(expected, (int, Decimal)):
                        # Detect gross errors; exact result is retained for final decimal rounding.
                        different = abs(Decimal(str(actual)) - Decimal(expected)) > max(Decimal("1e-12"), abs(Decimal(expected)) * Decimal("1e-14"))
                    else:
                        different = expected != actual
                    if different:
                        raise PlanError("independent_mismatch", "independent arithmetic disagrees")
        except (PlanError, ArithmeticError, KeyError, TypeError) as exc:
            report.diagnostics.append(Diagnostic("numerical", getattr(exc, "code", "independent"), str(exc)))
    elif result.diagnostics:
        report.diagnostics.extend(Diagnostic("numerical", d["code"], d["message"]) for d in result.diagnostics)
    if any(v is None for row in (report.exact_result if report.exact_result is not None else result.rows) for v in row):
        report.diagnostics.append(Diagnostic("execution", "missing_value", "NULL result requires explicit question semantics"))
    return report
