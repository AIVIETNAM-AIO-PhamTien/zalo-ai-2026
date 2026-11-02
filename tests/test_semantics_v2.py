from dataclasses import asdict, replace

import pytest

from datamate.foundation import load_foundation
from datamate.semantics import Evidence, GroundedRule, SemanticContract, SemanticWarehouse


def warehouse(tmp_path, domain="clinic"):
    (tmp_path / "facts.csv").write_text("Ref,Measurement\na,10\nb,20\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        f"# {domain}\n## Columns in `facts.csv`\n- **Ref**: visit identifier\n"
        "- **Measurement**: duration in minutes\n", encoding="utf-8")
    return SemanticWarehouse.build(load_foundation(tmp_path, "Sum duration at visit grain; no status filter."))


def grounded_contract(w):
    text = w.catalog.foundation.documents["$rulebook"]
    evidence = Evidence("$rulebook", 0, len(text), text)
    contract = SemanticContract("duration", [("facts.csv", "Measurement")], [("facts.csv", "Ref")],
        {"kind": "aggregate", "op": "sum", "column": "Measurement"}, units="minute", evidence=[evidence],
        applicability={"operation": "sum", "filters": []}, rule_ids=["duration_rule"])
    # Test-only independent rule adapter. Production planner cannot add grounded rules.
    expected = {k: v for k, v in asdict(contract).items() if k in {
        "concept", "sources", "grain", "expression", "joins", "status_policy", "time_column",
        "units", "transformations", "applicability"}}
    w.rules["duration_rule"] = GroundedRule("duration_rule", evidence, expected)
    return contract


def test_unseen_domain_preserves_candidates_and_has_no_sales_schema(tmp_path):
    w = warehouse(tmp_path)
    assert list(w.catalog.frames["facts.csv"].columns) == ["Ref", "Measurement"]
    assert len(w.catalog.frames["facts.csv"]) == 2
    assert w.roles["facts.csv"][0]["role"] == "unknown"
    assert "duration in minutes" in w.contracts
    assert not w.views
    assert w.retrieve("duration")[0]["column"] == "Measurement"
    assert all(b.evidence.valid(w.catalog.foundation.documents) for b in w.bindings)


def test_only_grounded_contract_gets_view_and_lineage(tmp_path):
    w = warehouse(tmp_path)
    c = grounded_contract(w)
    assert w.validate_contract(c) == []
    view = w.create_view("v_visits", c)
    assert view.lineage[0]["physical_rows"] == [2, 3]
    assert view.fingerprint == c.fingerprint()
    with pytest.raises(ValueError, match="already exists"):
        w.create_view("v_visits", c)


@pytest.mark.parametrize("mutation", [
    {"units": "VND"}, {"expression": {"kind": "aggregate", "op": "count", "column": "Ref"}},
    {"status_policy": [{"column": "status", "eq": "paid"}]},
    {"time_column": ("facts.csv", "made_up")}, {"grain": [("facts.csv", "Measurement")]},
    {"concept": "cash received"}, {"applicability": {"operation": "sum"}},
])
def test_shared_semantic_mistakes_cannot_be_verified_by_quote(tmp_path, mutation):
    w = warehouse(tmp_path)
    c = grounded_contract(w)
    c = replace(c, **mutation)
    assert w.validate_contract(c)
    with pytest.raises(ValueError):
        w.create_view("wrong", c)


def test_model_citation_alone_and_stale_evidence_are_not_grounding(tmp_path):
    w = warehouse(tmp_path)
    c = grounded_contract(w)
    w.rules.clear()
    assert any("unknown or stale" in error for error in w.validate_contract(c))
    c = grounded_contract(w)
    w.catalog.foundation.documents["$rulebook"] = "Changed: exclude refunded visits"
    assert any("evidence" in error for error in w.validate_contract(c))


def test_repeated_header_stays_ambiguous_and_no_rows_disappear(tmp_path):
    w = warehouse(tmp_path)
    (tmp_path / "facts.csv").write_text("Ref,Measurement\na,10\nRef,Measurement\nb,20\n")
    w = SemanticWarehouse.build(load_foundation(tmp_path))
    assert len(w.catalog.frames["facts.csv"]) == 4
    assert w.catalog.frames["facts.csv"].attrs["header"] is None
    assert any("unique header" in a for a in w.catalog.ambiguities)


def test_same_column_name_in_two_tables_needs_document_scope(tmp_path):
    warehouse(tmp_path)
    (tmp_path / "cash.csv").write_text("Ref,Measurement\na,100\n")
    w = SemanticWarehouse.build(load_foundation(tmp_path))
    assert not any(b.table == "cash.csv" for b in w.bindings)
    assert w.catalog.frames["cash.csv"].attrs["header"] is None


def test_duplicate_header_labels_do_not_silently_bind_first_column(tmp_path):
    warehouse(tmp_path)
    (tmp_path / "facts.csv").write_text("Ref,Measurement,Measurement\na,10,100\n")
    w = SemanticWarehouse.build(load_foundation(tmp_path))
    assert w.catalog.frames["facts.csv"].attrs["header"] is None
    assert any("duplicate column labels" in a for a in w.catalog.ambiguities)
