"""Typed IR to parameterized, allowlisted DuckDB SELECTs with bounded execution."""
from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal

import duckdb
import pandas as pd

from datamate.foundation import decimal_value
from datamate.ir import PlanError, validate_plan


def ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass
class Compiled:
    sql: str
    columns: dict[str, str]
    params: list = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    checks: list[tuple[str, list, str]] = field(default_factory=list)
    # Output fields whose source rows can have been replicated by a declared
    # one-to-many join. They may not feed additive measures until a roll-up
    # establishes a new grain.
    duplicated: set[str] = field(default_factory=set)


def _columns_in(expression) -> set[str]:
    """Physical output fields referenced by an expression (not aliases)."""
    if not isinstance(expression, dict):
        return set()
    if expression.get("kind") == "column":
        return {expression["name"]}
    return set().union(*(_columns_in(value) for value in expression.values()))


def _has_additive_aggregate(expression) -> bool:
    """Aggregates for which a replicated input changes the result."""
    if not isinstance(expression, dict):
        return False
    if expression.get("kind") == "aggregate" and expression.get("op") in {"sum", "avg", "count"}:
        return True
    return any(_has_additive_aggregate(value) for value in expression.values())


class Compiler:
    def __init__(self, schemas: dict[str, dict[str, str]]):
        self.schemas = schemas

    def check_metric_grain(self, expression, groups):
        if expression in groups or expression["kind"] in {"aggregate", "literal"}:
            return
        if expression["kind"] == "column":
            raise PlanError("grain", "metric references an ungrouped row-level column: " + expression["name"])
        for value in expression.values():
            if isinstance(value, dict) and "kind" in value:
                self.check_metric_grain(value, groups)

    def expression(self, node, columns, aggregate=False):
        kind = node["kind"]
        if kind == "column":
            name = node["name"]
            if name not in columns:
                raise PlanError("unknown_column", name)
            return ident(name), columns[name], []
        if kind == "literal":
            value, typ = node["value"], node["type"]
            if typ == "number":
                converted = decimal_value(value, "en")
                if converted.error or value is None:
                    raise PlanError("conversion", converted.error or "NULL number")
                value = converted.value
            elif typ == "date":
                try:
                    value = dt.datetime.fromisoformat(value)
                except (ValueError, TypeError) as exc:
                    raise PlanError("conversion", "invalid ISO date literal") from exc
            elif (typ == "text" and not isinstance(value, str) or
                  typ == "boolean" and not isinstance(value, bool) or
                  typ == "null" and value is not None):
                raise PlanError("type", "literal type mismatch")
            return "?", typ, [value]
        if kind == "binary":
            left, lt, lp = self.expression(node["left"], columns, aggregate)
            right, rt, rp = self.expression(node["right"], columns, aggregate)
            op = node["op"]
            if op in {"add", "subtract", "multiply", "divide"}:
                if lt != "number" or rt != "number":
                    raise PlanError("type", "arithmetic requires explicit numeric projections")
                typ = "number"
            elif op in {"and", "or"}:
                if lt != "boolean" or rt != "boolean":
                    raise PlanError("type", "logical operator requires booleans")
                typ = "boolean"
            else:
                if lt != rt or lt == "null":
                    raise PlanError("type", "comparison type mismatch; use is_null for NULL")
                typ = "boolean"
            symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "eq": "=",
                       "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "and": "AND", "or": "OR"}
            if op == "divide":
                right = f"NULLIF({right}, 0)"
            return f"({left} {symbols[op]} {right})", typ, lp + rp
        if kind in {"is_null", "not", "date_part", "aggregate"}:
            value, typ, params = self.expression(node["value"], columns, False)
            if kind == "is_null":
                return f"({value} IS NULL)", "boolean", params
            if kind == "not":
                if typ != "boolean":
                    raise PlanError("type", "NOT requires boolean")
                return f"(NOT {value})", "boolean", params
            if kind == "date_part":
                if typ != "date":
                    raise PlanError("type", "date_part requires date")
                return f"EXTRACT({node['part']} FROM {value})", "number", params
            if not aggregate:
                raise PlanError("grain", "aggregate expression outside aggregate stage")
            op = node["op"]
            if op in {"sum", "avg"} and typ != "number":
                raise PlanError("type", "numeric aggregate requires numbers")
            sql = f"COUNT(DISTINCT {value})" if op == "count_distinct" else f"{op.upper()}({value})"
            return sql, ("number" if op in {"count", "count_distinct", "sum", "avg"} else typ), params
        if kind == "case":
            condition, ct, cp = self.expression(node["when"], columns, aggregate)
            yes, yt, yp = self.expression(node["then"], columns, aggregate)
            no, nt, np = self.expression(node["else"], columns, aggregate)
            if ct != "boolean" or yt != nt and "null" not in {yt, nt}:
                raise PlanError("type", "CASE types do not match")
            return f"CASE WHEN {condition} THEN {yes} ELSE {no} END", nt if yt == "null" else yt, cp + yp + np
        raise PlanError("schema", "unknown expression")

    def compile(self, plan) -> Compiled:
        validate_plan(plan)
        return self.relation(plan["relation"])

    def relation(self, node) -> Compiled:
        kind = node["kind"]
        if kind == "scan":
            name = node["table"]
            if name not in self.schemas:
                raise PlanError("unknown_table", name)
            return Compiled(f"SELECT * FROM {ident(name)}", dict(self.schemas[name]), sources={name})
        if kind == "join":
            left, right = self.relation(node["left"]), self.relation(node["right"])
            if set(left.columns) & set(right.columns) and node["how"] != "anti":
                raise PlanError("schema", "join column collisions require explicit project aliases")
            on = []
            for pair in node["keys"]:
                lc, rc = pair["left"], pair["right"]
                if lc not in left.columns or rc not in right.columns:
                    raise PlanError("unknown_column", "join key not present")
                if left.columns[lc] != right.columns[rc]:
                    raise PlanError("type", "join keys have incompatible types")
                on.append(f"l.{ident(lc)} = r.{ident(rc)}")
            checks = left.checks + right.checks
            for side, keys, required_unique in (
                (left, [k["left"] for k in node["keys"]], node["cardinality"].startswith("one")),
                (right, [k["right"] for k in node["keys"]], node["cardinality"].endswith("one"))):
                if required_unique:
                    cols = ", ".join(map(ident, keys))
                    nonnull = " AND ".join(f"{ident(k)} IS NOT NULL" for k in keys)
                    checks.append((f"SELECT 1 FROM ({side.sql}) AS s WHERE {nonnull} GROUP BY {cols} HAVING COUNT(*) > 1 LIMIT 1",
                                   side.params, "join cardinality violated"))
            how = {"inner": "INNER", "left": "LEFT", "anti": "ANTI"}[node["how"]]
            selected = "l.*" if how == "ANTI" else "l.*, r.*"
            cols = left.columns if how == "ANTI" else {**left.columns, **right.columns}
            duplicated = set(left.duplicated) if how == "ANTI" else set(left.duplicated) | set(right.duplicated)
            if how != "ANTI":
                # A one-to-many join repeats the parent side; a many-to-one
                # join repeats the right side.  Do not let a later SUM/AVG/
                # COUNT silently turn that repetition into a measure.
                if node["cardinality"] == "one-to-many":
                    duplicated.update(left.columns)
                elif node["cardinality"] == "many-to-one":
                    duplicated.update(right.columns)
            return Compiled(f"SELECT {selected} FROM ({left.sql}) AS l {how} JOIN ({right.sql}) AS r ON {' AND '.join(on)}",
                            cols, left.params + right.params, left.sources | right.sources, checks, duplicated)
        if kind == "union":
            inputs = [self.relation(n) for n in node["inputs"]]
            if any(list(c.columns.items()) != list(inputs[0].columns.items()) for c in inputs[1:]):
                raise PlanError("type", "UNION schema mismatch")
            # Verification must separately authorize coverage/grain/units/duplicate partitions.
            return Compiled(" UNION ALL ".join(f"({c.sql})" for c in inputs), inputs[0].columns,
                            [p for c in inputs for p in c.params], set.union(*(c.sources for c in inputs)),
                            [check for c in inputs for check in c.checks], set.union(*(c.duplicated for c in inputs)))
        child = self.relation(node["input"])
        base = f"({child.sql}) AS s"
        if kind == "filter":
            predicate, typ, params = self.expression(node["predicate"], child.columns)
            if typ != "boolean":
                raise PlanError("type", "filter predicate must be boolean")
            return Compiled(f"SELECT * FROM {base} WHERE {predicate}", child.columns,
                            child.params + params, child.sources, child.checks, set(child.duplicated))
        if kind in {"project", "aggregate"}:
            entries = node["columns"] if kind == "project" else node["groups"] + node["metrics"]
            names, expressions, params = {}, [], []
            for i, entry in enumerate(entries):
                name = entry["name"]
                if name in names:
                    raise PlanError("schema", "duplicate output name")
                is_metric = kind == "aggregate" and i >= len(node["groups"])
                if is_metric:
                    self.check_metric_grain(entry["expression"], [g["expression"] for g in node["groups"]])
                    if (_has_additive_aggregate(entry["expression"]) and
                            _columns_in(entry["expression"]) & child.duplicated):
                        raise PlanError("fanout", "additive metric reads a join-replicated field; roll up before join")
                sql, typ, values = self.expression(entry["expression"], child.columns, is_metric)
                names[name] = typ
                expressions.append(f"{sql} AS {ident(name)}")
                params.extend(values)
            group = " GROUP BY " + ", ".join(str(i + 1) for i in range(len(node["groups"]))) if kind == "aggregate" and node["groups"] else ""
            # A GROUP BY/aggregate is an explicit grain transition.  Its
            # output measures are at the declared group grain, not the input
            # join grain.  Project retains provenance for its derived fields.
            duplicated = set() if kind == "aggregate" else {
                entry["name"] for entry in node["columns"]
                if _columns_in(entry["expression"]) & child.duplicated
            }
            return Compiled(f"SELECT {', '.join(expressions)} FROM {base}{group}", names,
                            params + child.params, child.sources, child.checks, duplicated)
        if kind == "sort":
            if any(b["column"] not in child.columns for b in node["by"]):
                raise PlanError("unknown_column", "sort column missing")
            order = ", ".join(ident(b["column"]) + (" DESC" if b["descending"] else " ASC") + " NULLS LAST" for b in node["by"])
            sql = f"SELECT * FROM {base} ORDER BY {order}"
        elif kind == "limit":
            sql = f"SELECT * FROM {base} LIMIT {node['count']}"
        elif kind == "distinct":
            sql = f"SELECT DISTINCT * FROM {base}"
        else:
            raise PlanError("schema", "unknown relation")
        return Compiled(sql, child.columns, child.params, child.sources, child.checks, set(child.duplicated))


@dataclass
class ExecutionResult:
    columns: list[str]
    rows: list[tuple]
    types: list[str]
    sources: list[str]
    diagnostics: list[dict] = field(default_factory=list)


def execute(plan: dict, frames: dict[str, pd.DataFrame], schemas: dict[str, dict[str, str]],
            timeout: float = 5, memory_mb: int = 256, max_rows: int = 1000) -> ExecutionResult:
    if timeout <= 0 or memory_mb < 32 or max_rows < 1:
        raise ValueError("invalid execution budget")
    compiled = Compiler(schemas).compile(plan)
    deadline = time.monotonic() + timeout
    con = duckdb.connect(":memory:")
    timer = None
    try:
        con.execute(f"SET memory_limit = '{int(memory_mb)}MB'")
        con.execute("SET threads = 1")
        con.execute("SET max_temp_directory_size = '0B'")
        for name in compiled.sources:
            if name not in frames or list(frames[name].columns) != list(schemas[name]):
                raise PlanError("schema", "registered frame/schema mismatch")
            for column, typ in schemas[name].items():
                # Runtime cannot silently promote raw text into numeric/date projections.
                for value in frames[name][column]:
                    if value is None:
                        continue
                    valid = {"number": isinstance(value, (int, Decimal)) and not isinstance(value, bool),
                             "text": isinstance(value, str), "date": isinstance(value, (dt.date, dt.datetime)),
                             "boolean": isinstance(value, bool)}.get(typ, False)
                    if not valid:
                        raise PlanError("conversion", f"{name}.{column}: explicit {typ} projection required")
                    if typ == "number" and decimal_value(value, "en").error:
                        raise PlanError("precision", f"{name}.{column}: numeric value not representable")
            # DuckDB otherwise samples only the first 1,000 Pandas object rows
            # when inferring DECIMAL width. A later larger value can fail at
            # execution even though the complete typed projection is valid.
            con.execute(f"SET pandas_analyze_sample = {max(1000, len(frames[name]))}")
            con.register(name, frames[name])
        con.execute("SET enable_external_access = false")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlanError("timeout", "input validation exhausted execution budget")
        timer = threading.Timer(remaining, con.interrupt)
        timer.daemon = True
        timer.start()
        for sql, params, reason in compiled.checks:
            if time.monotonic() >= deadline:
                raise PlanError("timeout", "execution timed out between checks")
            if con.execute(sql, params).fetchone() is not None:
                raise PlanError("join_cardinality", reason)
        if time.monotonic() >= deadline:
            raise PlanError("timeout", "execution timed out before final query")
        cursor = con.execute(compiled.sql, compiled.params)
        description = cursor.description
        rows = cursor.fetchmany(max_rows + 1)
        if time.monotonic() >= deadline:
            raise PlanError("timeout", "execution exceeded time budget")
        if len(rows) > max_rows:
            raise PlanError("output_budget", "result exceeds bounded output")
        diagnostics = []
        if any(str(d[1]) in {"FLOAT", "DOUBLE"} for d in description):
            diagnostics.append({"code": "inexact_arithmetic", "message": "independent exact computation required"})
        return ExecutionResult([d[0] for d in description], rows, [str(d[1]) for d in description],
                               sorted(compiled.sources), diagnostics)
    except duckdb.InterruptException as exc:
        raise PlanError("timeout", "execution timed out") from exc
    except duckdb.Error as exc:
        raise PlanError("execution", str(exc)) from exc
    finally:
        if timer:
            timer.cancel()
            timer.join()
        con.close()
