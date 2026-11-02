from decimal import Decimal

import pandas as pd
import pytest

from datamate.engine import Compiler, execute
from datamate.ir import PlanError, validate_plan


def col(name):
    return {"kind": "column", "name": name}


def lit(value, typ="number"):
    return {"kind": "literal", "type": typ, "value": value}


def binary(op, left, right):
    return {"kind": "binary", "op": op, "left": left, "right": right}


def named(name, expr):
    return {"name": name, "expression": expr}


def plan(relation, typ="number"):
    return {"relation": relation, "answer_type": typ, "contract_ids": [], "assumptions": []}


def aggregate(source, expression, groups=None):
    return {"kind": "aggregate", "input": source, "groups": groups or [], "metrics": [named("value", expression)]}


SCAN = {"kind": "scan", "table": "facts"}
SCHEMA = {"facts": {"id": "text", "amount": "number"}}


def facts():
    return {"facts": pd.DataFrame({"id": ["a", "b", "b"], "amount": [Decimal("1.10"), Decimal("2.20"), None]}, dtype=object)}


def test_parameterized_exact_sum_and_null_preservation():
    rel = {"kind": "filter", "input": SCAN, "predicate": binary("eq", col("id"), lit("b", "text"))}
    p = plan(aggregate(rel, {"kind": "aggregate", "op": "sum", "value": col("amount")}))
    compiled = Compiler(SCHEMA).compile(p)
    assert compiled.params == ["b"]
    assert "'b'" not in compiled.sql
    assert execute(p, facts(), SCHEMA).rows == [(Decimal("2.20"),)]


def test_multistage_group_ranking_and_parameter_order():
    rel = aggregate(SCAN, {"kind": "aggregate", "op": "sum", "value": col("amount")}, [named("id", col("id"))])
    rel = {"kind": "filter", "input": rel, "predicate": binary("gt", col("value"), lit(1))}
    rel = {"kind": "project", "input": rel, "columns": [named("id", col("id")),
           named("scaled", binary("multiply", col("value"), lit(3)))]}
    rel = {"kind": "sort", "input": rel, "by": [{"column": "scaled", "descending": True}, {"column": "id", "descending": False}]}
    rel = {"kind": "limit", "input": rel, "count": 1}
    assert execute(plan(rel, "table"), facts(), SCHEMA).rows == [("b", Decimal("6.60"))]


def test_zero_count_and_empty_sum_are_distinct():
    rel = {"kind": "filter", "input": SCAN, "predicate": binary("eq", col("id"), lit("missing", "text"))}
    p = plan(aggregate(rel, {"kind": "aggregate", "op": "sum", "value": col("amount")}))
    assert execute(p, facts(), SCHEMA).rows == [(None,)]
    p["relation"]["metrics"][0]["expression"]["op"] = "count"
    assert execute(p, facts(), SCHEMA).rows == [(0,)]


def test_sql_and_identifier_injection_cannot_escape_allowlist():
    for relation in ({"kind": "sql", "query": "DROP TABLE facts"},
                     {"kind": "scan", "table": 'facts"; DROP TABLE facts; --'},
                     {"kind": "project", "input": SCAN, "columns": [named("x", col("missing"))]}):
        with pytest.raises(PlanError):
            execute(plan(relation), facts(), SCHEMA)
    rel = {"kind": "filter", "input": SCAN, "predicate": binary("eq", col("id"), lit("'; DROP TABLE facts; --", "text"))}
    assert execute(plan(rel, "table"), facts(), SCHEMA).rows == []


def test_join_actual_cardinality_is_checked_even_if_model_says_safe():
    frames = {**facts(), "names": pd.DataFrame({"key": ["b", "b"], "name": ["Bob", "Ben"]}, dtype=object)}
    schema = {**SCHEMA, "names": {"key": "text", "name": "text"}}
    rel = {"kind": "join", "left": SCAN, "right": {"kind": "scan", "table": "names"},
           "how": "inner", "keys": [{"left": "id", "right": "key"}], "cardinality": "many-to-one"}
    with pytest.raises(PlanError, match="cardinality"):
        execute(plan(rel, "table"), frames, schema)
    frames["names"] = frames["names"].iloc[:1]
    assert len(execute(plan(rel, "table"), frames, schema).rows) == 2
    rel["how"] = "anti"
    assert execute(plan(rel, "table"), frames, schema).rows == [("a", Decimal("1.10"))]


def test_parent_measure_must_be_rolled_up_before_one_to_many_join():
    frames = {
        "orders": pd.DataFrame({"id": ["o1"], "total": [Decimal("100")]}, dtype=object),
        "lines": pd.DataFrame({"order": ["o1", "o1"], "sku": ["a", "b"]}, dtype=object),
    }
    schemas = {"orders": {"id": "text", "total": "number"},
               "lines": {"order": "text", "sku": "text"}}
    joined = {"kind": "join", "left": {"kind": "scan", "table": "orders"},
              "right": {"kind": "scan", "table": "lines"}, "how": "inner",
              "keys": [{"left": "id", "right": "order"}], "cardinality": "one-to-many"}
    unsafe = aggregate(joined, {"kind": "aggregate", "op": "sum", "value": col("total")})
    with pytest.raises(PlanError, match="roll up before join"):
        execute(plan(unsafe), frames, schemas)

    # The child is first reduced to one row per order, so the join is 1:1 and
    # the parent total is no longer multiplied.
    lines = {"kind": "aggregate", "input": {"kind": "scan", "table": "lines"},
             "groups": [named("order", col("order"))],
             "metrics": [named("line_count", {"kind": "aggregate", "op": "count", "value": col("sku")})]}
    joined["right"], joined["cardinality"] = lines, "one-to-one"
    safe = aggregate(joined, {"kind": "aggregate", "op": "sum", "value": col("total")})
    assert execute(plan(safe), frames, schemas).rows == [(Decimal("100"),)]


def test_numeric_projection_cannot_be_assumed():
    frames = facts()
    frames["facts"]["amount"] = ["1.10", "2.20", None]
    with pytest.raises(PlanError, match="projection"):
        execute(plan(SCAN, "table"), frames, SCHEMA)


def test_decimal_width_is_inferred_from_all_rows_not_first_thousand():
    values = [Decimal("1")] * 1001 + [Decimal("1056000")]
    frames = {"facts": pd.DataFrame({"id": ["x"] * len(values), "amount": values}, dtype=object)}
    relation = aggregate(SCAN, {"kind": "aggregate", "op": "sum", "value": col("amount")})
    assert execute(plan(relation), frames, SCHEMA).rows == [(Decimal("1057001"),)]


def test_output_and_plan_budgets():
    with pytest.raises(PlanError, match="bounded output"):
        execute(plan(SCAN, "table"), facts(), SCHEMA, max_rows=1)
    rel = SCAN
    for _ in range(30):
        rel = {"kind": "distinct", "input": rel}
    with pytest.raises(PlanError, match="budget"):
        validate_plan(plan(rel))


def test_division_is_not_misrepresented_as_exact_financial_result():
    rel = {"kind": "project", "input": SCAN, "columns": [named("ratio", binary("divide", col("amount"), lit(3)))]}
    assert execute(plan(rel, "list"), facts(), SCHEMA).diagnostics[0]["code"] == "inexact_arithmetic"


def test_unknown_extra_model_fields_are_rejected():
    p = plan(SCAN)
    p["verified"] = True
    with pytest.raises(PlanError):
        validate_plan(p)
