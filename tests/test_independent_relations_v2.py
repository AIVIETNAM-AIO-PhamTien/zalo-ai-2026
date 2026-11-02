from decimal import Decimal

import pandas as pd
import pytest

from datamate.engine import execute
from datamate.ir import PlanError
from datamate.verifier import ExactEvaluator
from test_engine_v2 import aggregate, col, named, plan
from test_engine_v2 import binary, lit


def sources():
    frames = {
        "visits": pd.DataFrame({"visit": ["v1", "v2", "v3", "v4"], "staff": ["a", "b", None, "unknown"],
                                "fee": [Decimal("10.25"), Decimal("20.50"), Decimal("1.10"), None]}, dtype=object),
        "employees": pd.DataFrame({"employee": ["a", "b", None], "name": ["An", "Binh", "Missing"]}, dtype=object)}
    schemas = {"visits": {"visit": "text", "staff": "text", "fee": "number"},
               "employees": {"employee": "text", "name": "text"}}
    return frames, schemas


def joined(how):
    return {"kind": "join", "left": {"kind": "scan", "table": "visits"},
            "right": {"kind": "scan", "table": "employees"}, "how": how,
            "keys": [{"left": "staff", "right": "employee"}], "cardinality": "many-to-one"}


@pytest.mark.parametrize("how,expected", [("inner", 2), ("left", 4), ("anti", 2)])
def test_independent_join_null_semantics_and_unmatched_rows(how, expected):
    frames, schemas = sources()
    rel = {"kind": "sort", "input": joined(how), "by": [{"column": "visit", "descending": False}]}
    columns, rows = ExactEvaluator(frames).evaluate(rel)
    sql = execute(plan(rel, "table"), frames, schemas)
    assert sql.columns == columns
    assert sql.rows == rows
    assert len(rows) == expected
    if how == "left":
        assert rows[2][-2:] == (None, None)  # NULL never joins to NULL.


def test_independent_financial_sum_after_join():
    frames, schemas = sources()
    rel = aggregate(joined("inner"), {"kind": "aggregate", "op": "sum", "value": col("fee")})
    assert ExactEvaluator(frames).evaluate(rel)[1] == [(Decimal("30.75"),)]
    assert execute(plan(rel), frames, schemas).rows == [(Decimal("30.75"),)]


def test_independent_cardinality_and_work_limit_detect_amplification():
    frames, _ = sources()
    frames["employees"] = pd.concat([frames["employees"], frames["employees"].iloc[:1]], ignore_index=True)
    with pytest.raises(PlanError, match="not unique"):
        ExactEvaluator(frames).evaluate(joined("inner"))
    with pytest.raises(PlanError, match="budget"):
        ExactEvaluator(frames, max_rows=2).evaluate(joined("left"))


def test_union_all_preserves_partition_duplicates_and_nulls():
    frames, schemas = sources()
    rel = {"kind": "union", "inputs": [{"kind": "scan", "table": "visits"}] * 2}
    columns, rows = ExactEvaluator(frames).evaluate(rel)
    assert len(rows) == 8
    assert rows[:4] == rows[4:]
    assert execute(plan(rel, "table"), frames, schemas).rows == rows
    unique = {"kind": "distinct", "input": rel}
    assert len(ExactEvaluator(frames).evaluate(unique)[1]) == 4


def test_preaggregation_restores_join_grain_for_financial_sum():
    frames, schemas = sources()
    # Multiple employee records must be collapsed explicitly before declaring a unique key.
    frames["employees"] = pd.concat([frames["employees"], frames["employees"].iloc[:1]], ignore_index=True)
    rel = joined("inner")
    rel["right"] = {"kind": "distinct", "input": rel["right"]}
    total = aggregate(rel, {"kind": "aggregate", "op": "sum", "value": col("fee")})
    assert ExactEvaluator(frames).evaluate(total)[1] == execute(plan(total), frames, schemas).rows


def test_ungrouped_column_in_metric_is_structural_grain_error():
    frames, schemas = sources()
    rel = aggregate({"kind": "scan", "table": "visits"},
                    binary("add", {"kind": "aggregate", "op": "sum", "value": col("fee")}, col("fee")))
    with pytest.raises(PlanError) as error:
        execute(plan(rel), frames, schemas)
    assert error.value.code == "grain"


def test_grouped_column_expression_can_be_reused_inside_metric():
    frames = {"counts": pd.DataFrame({"bucket": [1, 1, 2], "n": [10, 20, 30]}, dtype=object)}
    schemas = {"counts": {"bucket": "number", "n": "number"}}
    rel = aggregate({"kind": "scan", "table": "counts"},
                    binary("add", {"kind": "aggregate", "op": "sum", "value": col("n")}, col("bucket")),
                    [named("bucket", col("bucket"))])
    rel = {"kind": "sort", "input": rel, "by": [{"column": "bucket", "descending": False}]}
    assert ExactEvaluator(frames).evaluate(rel)[1] == execute(plan(rel, "table"), frames, schemas).rows == [(1, 31), (2, 32)]
