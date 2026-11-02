from decimal import Decimal
import pytest
from datamate.sql_bridge import sql_to_plan
from datamate.engine import execute
from datamate.ir import PlanError
from datamate.model import ScriptedModel
from datamate.planner import solve
from test_engine_v2 import facts, SCHEMA
from test_verifier_policy_v2 import verified_fixture


def test_lowering_uses_parameters_and_preserves_exact_aggregate():
    plan = sql_to_plan("SELECT SUM(amount) AS total FROM facts WHERE id='b'", SCHEMA, "number")
    assert execute(plan, facts(), SCHEMA).rows == [(Decimal("2.20"),)]


def test_group_sort_limit_and_count_nulls():
    plan = sql_to_plan("SELECT id, COUNT(amount) AS n FROM facts GROUP BY id ORDER BY n DESC NULLS LAST LIMIT 2", SCHEMA)
    assert sorted(execute(plan, facts(), SCHEMA).rows) == [("a", 1), ("b", 1)]


def test_between_and_count_distinct_lower_to_closed_ir():
    plan = sql_to_plan(
        "SELECT COUNT(DISTINCT id) AS n FROM facts WHERE amount BETWEEN 1 AND 3",
        SCHEMA, "number")
    assert plan["relation"]["input"]["metrics"][0]["expression"]["op"] == "count_distinct"
    assert execute(plan, facts(), SCHEMA).rows == [(2,)]


def test_unique_unaccented_column_resolves_but_collision_fails():
    schema = {"facts": {"Ngày tháng": "text", "amount": "number"}}
    frame = {"facts": facts()["facts"].rename(columns={"id": "Ngày tháng"})}
    plan = sql_to_plan('SELECT COUNT("Ngay thang") AS n FROM facts', schema, "number")
    assert execute(plan, frame, schema).rows == [(3,)]
    ambiguous = {"facts": {"Ngày tháng": "text", "Ngay thang": "text"}}
    with pytest.raises(PlanError) as error:
        sql_to_plan('SELECT COUNT("NGAY THANG") AS n FROM facts', ambiguous, "number")
    assert error.value.code == "unknown_column"


def test_bridge_explicitly_targets_duckdb_division_semantics():
    plan = sql_to_plan("SELECT 5 / 2 AS ratio FROM facts LIMIT 1", SCHEMA)
    assert execute(plan, facts(), SCHEMA).rows == [(2.5,)]


def test_ast_work_budget_is_bounded_before_lowering():
    with pytest.raises(PlanError):
        sql_to_plan("SELECT " + "+".join(["1"] * 200) + " AS n FROM facts", SCHEMA)


@pytest.mark.parametrize("sql", ["DROP TABLE facts", "SELECT id FROM facts; DELETE FROM facts",
    "SELECT * FROM facts", "SELECT x FROM facts", "SELECT id FROM read_csv('secret.csv')",
    "SELECT id FROM facts JOIN other ON 1=1", "SELECT id FROM facts OFFSET 1",
    "WITH x AS (SELECT id FROM facts) SELECT id FROM x", "SELECT RANDOM() AS x FROM facts",
    "SELECT COUNT(DISTINCT id, amount) AS n FROM facts", "SELECT id FROM facts ORDER BY id NULLS FIRST",
    "SELECT id FROM facts LIMIT 10 PERCENT", "SELECT id FROM facts TABLESAMPLE(10 ROWS)",
    "SELECT id FROM facts AS f(renamed)"])
def test_unsupported_or_dangerous_sql_is_rejected_whole(sql):
    with pytest.raises(PlanError):
        sql_to_plan(sql, SCHEMA)


def test_sql_parse_diagnostics_enter_bounded_repair_without_semantic_bypass(tmp_path):
    w, _, frames, schemas, _, _ = verified_fixture(tmp_path)
    model = ScriptedModel([
        {"sql": "DELETE FROM facts", "answer_type": "number", "contract_ids": []},
        {"sql": 'SELECT SUM("Measurement") AS total FROM "facts.csv"', "answer_type": "number", "contract_ids": []}])
    result = solve({"question": "duration"}, w, frames, schemas, model)
    assert result.llm_calls == 2
    assert result.selection.answer["type"] == "abstain"
    assert any(d["code"] == "sql_unsupported" for d in result.diagnostics)
    assert any(d["code"] == "missing_grounding" for d in result.diagnostics)
