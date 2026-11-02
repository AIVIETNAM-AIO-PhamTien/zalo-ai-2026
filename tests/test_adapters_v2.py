import sqlite3

import pytest

from datamate.adapters import direct_requirements
from datamate.ir import PlanError
from datamate.pipeline import Checkpoint, prepare_business, run_questions
from datamate.planner import solve


@pytest.fixture
def prepared(tmp_path):
    business = tmp_path / "business"
    business.mkdir()
    with sqlite3.connect(business / "clinic.db") as connection:
        connection.execute('CREATE TABLE appointments (person TEXT, minutes INTEGER, occurred TEXT)')
        connection.executemany('INSERT INTO appointments VALUES (?,?,?)', [
            ("a", 10, "2026-02-01"), ("a", 10, "2026-02-01"),
            ("b", 40, "2026-03-01"), ("c", None, "2026-02-28")])
    return prepare_business(business, "", {"clinic.db::appointments": {"occurred": {"type": "date"}}})


@pytest.mark.parametrize("prompt_text,expected", [
    ("Tổng cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments`", 60),
    ("Trung bình cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments`", 20),
    ("Đếm cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments`", 3),
    ("Đếm tất cả bản ghi trong bảng `clinic.db::appointments`", 4),
    ("Tổng cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments` theo cột ngày `occurred` trong tháng 2/2026?", 20),
    ("Tổng cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments` theo cột ngày `occurred` trong hôm nay", 40),
])
def test_direct_source_question_is_grounded_and_computed_without_model(prepared, prompt_text, expected):
    question = {"question": prompt_text, "as_of_date": "2026-03-01"}
    requirements = direct_requirements(question, prepared)
    result = solve(question, prepared.warehouse, prepared.frames, prepared.schemas, requirements=requirements)
    assert result.selection.answer == {"type": "number", "answer": expected}
    assert result.llm_calls == 0
    contract = prepared.warehouse.views[requirements["contract_ids"][0]].contract
    assert contract.grain == [("clinic.db::appointments", "$physical_row")]
    assert not prepared.warehouse.validate_contract(contract)


def test_pipeline_automatically_uses_adapter_and_resumes(prepared, tmp_path):
    questions = [{"id": "one", "question": "Đếm tất cả bản ghi trong bảng `clinic.db::appointments`"}]
    checkpoint = Checkpoint(tmp_path / "out", ["one"], {})
    run_questions(questions, prepared, checkpoint)
    assert checkpoint.rows["one"]["answer"] == 4
    assert run_questions(questions, prepared, checkpoint) == []


@pytest.mark.parametrize("prompt_text", [
    "Doanh thu của clinic là bao nhiêu?",
    "Tổng cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments` chỉ với khách a",
    "Tổng cột `minutes` của tất cả bản ghi trong bảng `clinic.db::appointments` và bỏ đơn hủy",
])
def test_incomplete_grammar_coverage_does_not_drop_conditions(prepared, prompt_text):
    assert direct_requirements({"question": prompt_text}, prepared) is None


@pytest.mark.parametrize("prompt_text", [
    "Tổng cột `person` của tất cả bản ghi trong bảng `clinic.db::appointments`",
    "Tổng cột `unknown` của tất cả bản ghi trong bảng `clinic.db::appointments`",
    "Đếm tất cả bản ghi trong bảng `missing`",
    "Đếm tất cả bản ghi trong bảng `clinic.db::appointments` theo cột ngày `person` trong hôm nay",
    "Đếm tất cả bản ghi trong bảng `clinic.db::appointments` theo cột ngày `occurred` trong hôm nay",
])
def test_missing_binding_or_temporal_evidence_fails_closed(prepared, prompt_text):
    with pytest.raises(PlanError):
        direct_requirements({"question": prompt_text}, prepared)


def test_adapter_reentry_is_idempotent_and_rule_mutation_detected(prepared):
    question = {"question": "Đếm tất cả bản ghi trong bảng `clinic.db::appointments`"}
    first = direct_requirements(question, prepared)
    assert direct_requirements(question, prepared) == first
    view = prepared.warehouse.views[first["contract_ids"][0]]
    view.contract.status_policy.append({"ignore_null": True})
    assert prepared.warehouse.validate_contract(view.contract)
