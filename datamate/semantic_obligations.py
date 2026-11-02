"""Compare a candidate with independently grounded relational obligations.

This is a conservative equivalence check, not a natural-language judge. The
reference relation must come from a trusted question/rule adapter. Only SQL
three-valued AND/OR associativity and commutativity, comparison symmetry, and
adjacent filter composition are normalized; other rewrites remain unproved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ObligationMismatch:
    code: str
    path: str
    message: str


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _flatten_boolean(node, op):
    if isinstance(node, dict) and node.get("kind") == "binary" and node.get("op") == op:
        return _flatten_boolean(node["left"], op) + _flatten_boolean(node["right"], op)
    return [_canonical(node)]


def _canonical(value):
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("kind")
    if kind == "filter":
        predicates = []
        current = value
        while isinstance(current, dict) and current.get("kind") == "filter":
            predicates.extend(_flatten_boolean(current["predicate"], "and"))
            current = current["input"]
        return {"kind": "filter", "input": _canonical(current),
                "predicate": {"kind": "all", "terms": sorted(predicates, key=_key)}}
    if kind == "binary" and value.get("op") in {"and", "or"}:
        op = value["op"]
        return {"kind": op, "terms": sorted(_flatten_boolean(value, op), key=_key)}
    if kind == "binary" and value.get("op") in {"eq", "ne", "add", "multiply"}:
        left, right = _canonical(value["left"]), _canonical(value["right"])
        if _key(left) > _key(right):
            left, right = right, left
        return {**value, "left": left, "right": right}
    return {key: _canonical(child) for key, child in value.items()}


def _category(path):
    if "/predicate" in path:
        return "filter_obligation"
    if "/groups" in path:
        return "grain_obligation"
    if "/metrics" in path:
        return "metric_obligation"
    if "/keys" in path or "/cardinality" in path or "/how" in path:
        return "join_obligation"
    if path.endswith("/table") or "/sources" in path:
        return "source_obligation"
    if "/by" in path or "/count" in path:
        return "ordering_obligation"
    if "/columns" in path:
        return "output_obligation"
    return "relation_obligation"


def compare_relation(candidate: dict, reference: dict, max_diagnostics: int = 8) -> list[ObligationMismatch]:
    """Return bounded, path-specific failures against a trusted relation.

    This comparison never treats execution agreement or a model citation as
    evidence that the reference encodes the question correctly.
    """
    actual, expected = _canonical(candidate), _canonical(reference)
    failures: list[ObligationMismatch] = []

    def walk(left, right, path):
        if len(failures) >= max_diagnostics or left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                if len(failures) >= max_diagnostics:
                    return
                child_path = f"{path}/{key}"
                if key not in left or key not in right:
                    failures.append(ObligationMismatch(_category(child_path), child_path,
                        f"{'unexpected' if key in left else 'missing'} {key} in candidate"))
                else:
                    walk(left[key], right[key], child_path)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                failures.append(ObligationMismatch(_category(path), path,
                    f"candidate has {len(left)} items; grounded requirement has {len(right)}"))
            for index, (item, required) in enumerate(zip(left, right)):
                walk(item, required, f"{path}/{index}")
        else:
            failures.append(ObligationMismatch(_category(path), path,
                "candidate value differs from grounded requirement"))

    walk(actual, expected, "/relation")
    return failures
