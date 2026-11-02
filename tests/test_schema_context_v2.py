import pytest

from datamate.catalog import Relation
from datamate.ir import PlanError
from datamate.schema_context import schema_context


def edge(left, right):
    return Relation(left, "key", right, "key", "many-to-one", 1, 1, 1)


def test_bridge_tables_and_keys_survive_retrieval_selection():
    schemas = {t: {"key": "text", "amount": "number"} for t in "ABCD"}
    result = schema_context(schemas, ["A", "C"], [edge("A", "B"), edge("B", "C")])
    assert list(result["schema"]) == ["A", "C", "B"]
    assert result["candidate_join_paths"] == [["A", "B", "C"]]
    assert all(not r["verified"] for r in result["candidate_relations"])
    assert "D" not in result["schema"]


def test_alternative_paths_are_retained_without_semantic_selection():
    schemas = {t: {"key": "text"} for t in "ABCD"}
    result = schema_context(schemas, ["A", "D"], [edge("A", "B"), edge("B", "D"), edge("A", "C"), edge("C", "D")])
    assert result["candidate_join_paths"] == [["A", "B", "D"], ["A", "C", "D"]]
    with pytest.raises(PlanError):
        schema_context(schemas, ["A", "D"], [edge("A", "B"), edge("B", "D")], max_tables=2)


def test_unavailable_keys_cannot_connect_tables_and_unknown_seeds_fail():
    schemas = {"A": {"key": "text"}, "B": {"other": "text"}}
    result = schema_context(schemas, ["A", "B"], [edge("A", "B")])
    assert result["candidate_relations"] == []
    assert result["unconnected_table_pairs"] == [["A", "B"]]
    with pytest.raises(PlanError):
        schema_context(schemas, ["invented"], [])
