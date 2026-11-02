import sqlite3

import pytest

from datamate.ir import PlanError
from datamate.metric_rules import metric_requirements
from datamate.pipeline import Checkpoint, prepare_business, run_questions
from datamate.planner import solve


def business(tmp_path, value="elapsed_x", date="finished_y", operation="tổng"):
    with sqlite3.connect(tmp_path / "world.db") as con:
        con.execute(f'CREATE TABLE facts ("{value}" INTEGER, "{date}" TEXT, person TEXT)')
        con.executemany("INSERT INTO facts VALUES (?,?,?)", [(10, "2026-02-01", "a"),
            (10, "2026-02-01", "a"), (40, "2026-03-01", "b"), (None, "2026-02-20", "c")])
    (tmp_path / "README.md").write_text(
        f"## Columns in `world.db::facts`\n- **{value}**: thời lượng thực hiện\n"
        f"- **{date}**: ngày hoàn tất\n- **person**: người phụ trách\n", encoding="utf-8")
    rules = f"| Thời lượng xử lý | {operation} cột có nghĩa `thời lượng thực hiện`; đơn vị `phút`; tất cả bản ghi; thời gian có nghĩa `ngày hoàn tất` |\n"
    return prepare_business(tmp_path, rules, {"world.db::facts": {date: {"type": "date"}}})


@pytest.mark.parametrize("value,date", [("elapsed_x", "finished_y"), ("so_phut", "hoan_tat")])
def test_named_metric_links_documented_meaning_across_new_column_names(tmp_path, value, date):
    prepared = business(tmp_path, value, date)
    question = {"question": "Thời lượng xử lý trong tháng 2/2026?"}
    requirements = metric_requirements(question, prepared)
    result = solve(question, prepared.warehouse, prepared.frames, prepared.schemas, requirements=requirements)
    assert result.selection.answer == {"type": "number", "answer": 20}
    view = prepared.warehouse.views[requirements["contract_ids"][0]]
    assert view.contract.time_column == ("world.db::facts", date)
    assert len(view.contract.evidence) == 3
    assert result.llm_calls == 0


@pytest.mark.parametrize("operation,expected", [("tổng", 60), ("trung bình", 20), ("đếm", 3), ("đếm phân biệt", 2)])
def test_named_metric_operations_keep_null_and_duplicate_semantics(tmp_path, operation, expected):
    prepared = business(tmp_path, operation=operation)
    checkpoint = Checkpoint(tmp_path / "out", ["q"], {})
    run_questions([{"id": "q", "question": "Thời lượng xử lý"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["answer"] == expected


def test_unrecognized_rule_suffix_is_never_ignored(tmp_path):
    prepared = business(tmp_path)
    docs = prepared.warehouse.catalog.foundation.documents
    docs["$rulebook"] = docs["$rulebook"].replace(" |", "; chỉ người đang làm việc |")
    assert metric_requirements({"question": "Thời lượng xử lý"}, prepared) is None


def test_ambiguous_meaning_and_unknown_question_conditions_block_execution(tmp_path):
    prepared = business(tmp_path)
    bindings = prepared.warehouse.bindings
    bindings.append(bindings[0])
    with pytest.raises(PlanError, match="exactly one"):
        metric_requirements({"question": "Thời lượng xử lý"}, prepared)
    bindings.pop()
    assert metric_requirements({"question": "Thời lượng xử lý của khách An"}, prepared) is None


def test_relative_period_without_rule_policy_is_not_guessed(tmp_path):
    prepared = business(tmp_path)
    with pytest.raises(PlanError, match="policy"):
        metric_requirements({"question": "Thời lượng xử lý trong tháng này", "as_of_date": "2026-02-15"}, prepared)


def test_stale_document_evidence_invalidates_existing_contract(tmp_path):
    prepared = business(tmp_path)
    requirements = metric_requirements({"question": "Thời lượng xử lý"}, prepared)
    view = prepared.warehouse.views[requirements["contract_ids"][0]]
    prepared.warehouse.catalog.foundation.documents["README.md"] = "Changed meanings"
    assert any("evidence" in e for e in prepared.warehouse.validate_contract(view.contract))


def test_conflicting_unsupported_definition_is_not_silently_ignored(tmp_path):
    prepared = business(tmp_path)
    prepared.warehouse.catalog.foundation.documents["$rulebook"] += "| Thời lượng xử lý | chỉ tính ca đêm theo quy tắc khác |\n"
    with pytest.raises(PlanError, match="conflicting"):
        metric_requirements({"question": "Thời lượng xử lý"}, prepared)


def test_same_relative_question_on_two_reference_dates_keeps_both_contracts_valid(tmp_path):
    prepared = business(tmp_path)
    contracts = []
    for date in ("2026-02-01", "2026-03-01"):
        requirements = metric_requirements({"question": "Thời lượng xử lý trong hôm nay", "as_of_date": date}, prepared)
        contracts.append(prepared.warehouse.views[requirements["contract_ids"][0]].contract)
    assert contracts[0].fingerprint() != contracts[1].fingerprint()
    assert all(not prepared.warehouse.validate_contract(contract) for contract in contracts)
