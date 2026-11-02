"""Closed JSON Schema for compositional relational plans; no SQL/code escape hatch."""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


def variant(kind, fields):
    return obj({"kind": {"const": kind}, **fields})


def array(items, minimum=0, maximum=100):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


TEXT = {"type": "string", "minLength": 1, "maxLength": 256}
EXPR = {"$ref": "#/$defs/expression"}
REL = {"$ref": "#/$defs/relation"}
NAMED = obj({"name": TEXT, "expression": EXPR})
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$defs": {
        "expression": {"oneOf": [
            variant("column", {"name": TEXT}),
            variant("literal", {"type": {"enum": ["text", "number", "date", "boolean", "null"]},
                                "value": {"type": ["string", "integer", "number", "boolean", "null"]}}),
            variant("binary", {"op": {"enum": ["add", "subtract", "multiply", "divide", "eq", "ne", "lt", "le", "gt", "ge", "and", "or"]},
                               "left": EXPR, "right": EXPR}),
            variant("is_null", {"value": EXPR}),
            variant("not", {"value": EXPR}),
            variant("case", {"when": EXPR, "then": EXPR, "else": EXPR}),
            variant("aggregate", {"op": {"enum": ["sum", "count", "count_distinct", "min", "max", "avg"]}, "value": EXPR}),
            variant("date_part", {"part": {"enum": ["year", "month", "day", "hour"]}, "value": EXPR}),
        ]},
        "relation": {"oneOf": [
            variant("scan", {"table": TEXT}),
            variant("filter", {"input": REL, "predicate": EXPR}),
            variant("project", {"input": REL, "columns": array(NAMED, 1)}),
            variant("aggregate", {"input": REL, "groups": array(NAMED), "metrics": array(NAMED, 1)}),
            variant("join", {"left": REL, "right": REL, "how": {"enum": ["inner", "left", "anti"]},
                             "keys": array(obj({"left": TEXT, "right": TEXT}), 1, 8),
                             "cardinality": {"enum": ["one-to-one", "one-to-many", "many-to-one"]}}),
            variant("union", {"inputs": array(REL, 2, 16)}),
            variant("distinct", {"input": REL}),
            variant("sort", {"input": REL, "by": array(obj({"column": TEXT, "descending": {"type": "boolean"}}), 1)}),
            variant("limit", {"input": REL, "count": {"type": "integer", "minimum": 1, "maximum": 10000}}),
        ]},
    },
    **obj({"relation": REL, "answer_type": {"enum": ["number", "percent", "string", "date", "boolean", "list", "table"]},
           "contract_ids": array(TEXT, maximum=32), "assumptions": array(TEXT, maximum=32)}),
}


class PlanError(ValueError):
    def __init__(self, code: str, message: str, path: str = ""):
        self.code, self.path = code, path
        super().__init__(message)

    def diagnostic(self):
        return {"code": self.code, "path": self.path, "message": str(self)}


def validate_plan(plan: dict, max_bytes: int = 65536, max_nodes: int = 256, max_depth: int = 24):
    try:
        encoded = json.dumps(plan, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise PlanError("invalid_json", str(exc)) from exc
    if len(encoded.encode()) > max_bytes:
        raise PlanError("plan_budget", "plan exceeds byte budget")
    pending, count = [(plan, 0)], 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > max_nodes or depth > max_depth:
            raise PlanError("plan_budget", "plan exceeds node/depth budget")
        if isinstance(value, dict):
            pending.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            pending.extend((v, depth + 1) for v in value)
    error = next(Draft202012Validator(SCHEMA).iter_errors(plan), None)
    if error:
        raise PlanError("schema", error.message[:1000], "/".join(map(str, error.absolute_path)))
    return plan
