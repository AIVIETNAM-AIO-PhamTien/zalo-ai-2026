"""Business preparation, bounded question execution and atomic recovery.

Experimental v2 orchestration; legacy remains default until valid PA ablations.
"""
from __future__ import annotations

import hashlib
import datetime as dt
import json
import math
import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from decimal import Decimal

import pandas as pd

from datamate.foundation import date_value, decimal_value, load_foundation
from datamate.planner import solve
from datamate.policy import Budget
from datamate.semantics import SemanticWarehouse
from datamate.adapters import direct_requirements
from datamate.ir import PlanError
from datamate.metric_rules import metric_requirements
from datamate.revenue_rules import revenue_requirements
from datamate.projections import infer_spec, documented_type


def atomic_json(path: Path, data, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            if jsonl:
                for row in data:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            else:
                json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_answer(answer):
    if not isinstance(answer, dict):
        raise ValueError("answer must be an object")
    typ = answer.get("type")
    if typ not in {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}:
        raise ValueError("invalid answer type")
    allowed = {"type", "reason"} if typ == "unanswerable" else {"type"} if typ == "abstain" else {"type", "answer"}
    if set(answer) != allowed:
        raise ValueError("unexpected output fields")
    def scalar(value):
        return isinstance(value, (str, bool, int)) or isinstance(value, float) and math.isfinite(value)
    value = answer.get("answer")
    valid = True
    if typ in {"number", "percent"}:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    elif typ == "string":
        valid = isinstance(value, str) and bool(value.strip())
    elif typ == "date":
        valid = isinstance(value, str)
        if valid:
            try:
                parsed = dt.date.fromisoformat(value)
                valid = parsed.isoformat() == value
            except ValueError:
                valid = False
    elif typ == "boolean":
        valid = isinstance(value, bool)
    elif typ == "list":
        valid = isinstance(value, list) and all(scalar(v) for v in value)
    elif typ == "table":
        valid = (isinstance(value, dict) and set(value) == {"columns", "rows"}
                 and isinstance(value["columns"], list) and all(isinstance(c, str) for c in value["columns"])
                 and isinstance(value["rows"], list)
                 and all(isinstance(r, list) and len(r) == len(value["columns"]) and all(scalar(v) for v in r)
                         for r in value["rows"]))
    elif typ == "unanswerable":
        valid = isinstance(answer["reason"], str)
    if not valid:
        raise ValueError("invalid typed answer value")


@dataclass
class PreparedBusiness:
    warehouse: SemanticWarehouse
    frames: dict
    schemas: dict
    conversions: list[dict] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)


def prepare_business(path, rulebook, projections=None):
    warehouse = SemanticWarehouse.build(load_foundation(path, rulebook))
    projections = projections or {}
    prepared = PreparedBusiness(warehouse, {}, {})
    for table, frame in warehouse.catalog.frames.items():
        raw = warehouse.catalog.foundation.tables[table]
        if raw.columns is None and frame.attrs["header"] is None:
            prepared.diagnostics.append({"code": "unresolved_header", "table": table})
            continue
        projected, schema, errors = frame.copy(), {}, []
        for column in frame.columns:
            explicit = projections.get(table, {}).get(column)
            descriptions = [b.description for b in warehouse.bindings if b.table == table and b.column == column]
            spec = explicit or infer_spec(frame[column].tolist(), descriptions)
            if not spec and documented_type(descriptions) and any(v is not None for v in frame[column]):
                errors.append({"code": "documented_projection_unresolved", "table": table, "column": column,
                               "expected_type": documented_type(descriptions)})
            if spec:
                converted = []
                column_failed = False
                conversion_cache = {}
                for physical, value in zip(frame.attrs["physical_rows"], frame[column]):
                    cache_key = type(value), repr(value)
                    if cache_key not in conversion_cache:
                        if spec["type"] == "number":
                            conversion_cache[cache_key] = decimal_value(value, spec.get("locale"))
                        elif spec["type"] == "date":
                            conversion_cache[cache_key] = date_value(value, format=spec.get("format"))
                        else:
                            raise ValueError("unsupported explicit projection")
                    conversion = conversion_cache[cache_key]
                    if explicit or conversion.error:
                        prepared.conversions.append({"table": table, "column": column, "physical_row": physical,
                                                  "rule": conversion.rule, "error": conversion.error,
                                                  "basis": "explicit" if explicit else spec["basis"],
                                                  "documented_meanings": descriptions})
                    if conversion.error:
                        column_failed = True
                        errors.append({"code": "conversion", "table": table, "column": column,
                                       "physical_row": physical, "message": conversion.error})
                    converted.append(conversion.value)
                projected[column] = pd.Series(converted, dtype=object)
                if not column_failed:
                    schema[column] = spec["type"]
                    if not explicit:
                        prepared.conversions.append({"table": table, "column": column, "rule": str(spec),
                            "basis": spec["basis"], "documented_meanings": descriptions,
                            "physical_rows": frame.attrs["physical_rows"], "error": None})
            else:
                values = [v for v in frame[column] if v is not None]
                types = {"boolean" if isinstance(v, bool) else "number" if isinstance(v, (int, Decimal))
                         else "date" if isinstance(v, (dt.datetime, dt.date)) else "text" if isinstance(v, str)
                         else "unknown" for v in values}
                if not types:
                    schema[column] = "null"
                elif len(types) != 1 or "unknown" in types:
                    errors.append({"code": "explicit_projection_required", "table": table, "column": column})
                else:
                    schema[column] = next(iter(types))
        prepared.diagnostics.extend(errors)
        if schema:
            # Only the typed view omits unconvertible columns; raw catalog/source
            # keeps them. Row count, physical indices, NULLs and duplicates survive.
            prepared.frames[table], prepared.schemas[table] = projected[list(schema)].copy(), schema
    return prepared


class BusinessCache:
    def __init__(self, max_businesses=2):
        if max_businesses < 1:
            raise ValueError("positive cache size required")
        self.max_businesses, self.values = max_businesses, OrderedDict()

    def get(self, path, rulebook, projections=None):
        root = Path(path).resolve()
        digest = hashlib.sha256(rulebook.encode())
        digest.update(json.dumps(projections or {}, sort_keys=True).encode())
        for source in sorted(p for p in root.rglob("*") if p.is_file()):
            if source.suffix.lower() not in {".csv", ".xlsx", ".xlsm", ".db", ".sqlite", ".md", ".txt"}:
                continue
            digest.update(source.relative_to(root).as_posix().encode())
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        for module in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(module.name.encode())
            digest.update(module.read_bytes())
        key = str(root), digest.hexdigest()
        if key not in self.values:
            self.values[key] = prepare_business(root, rulebook, projections)
            while len(self.values) > self.max_businesses:
                self.values.popitem(last=False)
        self.values.move_to_end(key)
        return self.values[key]


class Checkpoint:
    """Canonical state is atomic; submission is a recoverable projection of it."""
    def __init__(self, directory, ids, manifest):
        self.directory = Path(directory)
        self.ids = list(ids)
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("duplicate question IDs")
        self.path = self.directory / "checkpoint.json"
        self.manifest = manifest
        self.rows = {}
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state["manifest"] != manifest or state["ids"] != self.ids:
                raise ValueError("restart inputs/config differ from checkpoint")
            self.rows = state["rows"]
            if set(self.rows) - set(self.ids):
                raise ValueError("checkpoint contains unknown IDs")
            for qid, row in self.rows.items():
                if row.get("id") != qid:
                    raise ValueError("checkpoint row ID mismatch")
                validate_answer({k: v for k, v in row.items() if k != "id"})
        elif (self.directory / "submission.jsonl").exists():
            raise ValueError("submission exists without canonical checkpoint")
        self.flush()

    def record(self, qid, answer):
        if qid not in self.ids or qid in self.rows:
            raise ValueError("unknown or already completed question")
        validate_answer(answer)
        self.rows[qid] = {"id": qid, **answer}
        try:
            self.flush()
        except Exception:
            # Canonical state may already be durable when submission replacement fails.
            # Reconcile memory to disk so a same-process retry cannot erase that progress.
            if self.path.exists():
                self.rows = json.loads(self.path.read_text(encoding="utf-8"))["rows"]
            else:
                self.rows.pop(qid)
            raise

    def flush(self):
        atomic_json(self.path, {"manifest": self.manifest, "ids": self.ids, "rows": self.rows})
        atomic_json(self.directory / "submission.jsonl", [self.rows[qid] for qid in self.ids if qid in self.rows], jsonl=True)


def run_questions(questions, prepared, checkpoint, model=None, requirements=None, budget=None):
    """Run one prepared business; caller separates gold/evaluation from this API."""
    traces = []
    for question in questions:
        qid = question["id"]
        if qid in checkpoint.rows:
            continue
        try:
            result = solve({k: question[k] for k in ("question", "as_of_date") if k in question},
                           prepared.warehouse, prepared.frames, prepared.schemas, model,
                           requirements(question) if requirements else
                           (direct_requirements(question, prepared) or metric_requirements(question, prepared)
                            or revenue_requirements(question, prepared)), budget or Budget())
            answer = result.selection.answer
            trace = {"id": qid, "llm_calls": result.llm_calls, "tokens": result.tokens,
                     "runtime_seconds": result.runtime_seconds, "diagnostics": result.diagnostics,
                     "reason": result.selection.reason}
        except Exception as exc:
            answer = {"type": "abstain"}
            trace = {"id": qid, "diagnostics": [exc.diagnostic() if isinstance(exc, PlanError) else
                                                {"code": "question_failure", "error_type": type(exc).__name__}]}
        # Writing failures propagate: never lose persistence failures in a per-question catch.
        checkpoint.record(qid, answer)
        atomic_json(checkpoint.directory / "traces" / (hashlib.sha256(qid.encode()).hexdigest() + ".json"), trace)
        traces.append(trace)
    return traces
