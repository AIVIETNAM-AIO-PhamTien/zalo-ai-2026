import pytest
from datamate.ir import PlanError

from datamate.question_text import normalize_question
from datamate.revenue_rules import revenue_requirements
from datamate.pipeline import Checkpoint, run_questions
from test_revenue_pipeline_v2 import prepared_source


def test_alias_preprocessing_preserves_original_and_quoted_entities():
    raw = 'dthu hqua cua cua hang la bao nhiu, sản phẩm "hqua" mã `dthu`'
    result = normalize_question(raw)
    assert result.original == raw
    assert result.canonical == 'doanh thu hôm qua của cửa hàng là bao nhiêu, sản phẩm "hqua" mã `dthu`'
    assert len(result.replacements) == 4
    assert normalize_question(result.canonical).canonical == result.canonical
    assert normalize_question("ID_dthu_123").canonical == "ID_dthu_123"


def test_teencode_revenue_reaches_verified_pipeline_without_dropping_constraints(tmp_path):
    prepared = prepared_source(tmp_path)
    checkpoint = Checkpoint(tmp_path / "out", ["q"], {})
    run_questions([{"id": "q", "question": "dthu hqua cua cua hang la bao nhiu", "as_of_date": "2026-02-02"}], prepared, checkpoint)
    assert checkpoint.rows["q"] == {"id": "q", "type": "number", "answer": 30}
    contract = next(iter(prepared.warehouse.views.values())).contract
    assert contract.applicability["question"] == "dthu hqua cua cua hang la bao nhiu"


@pytest.mark.parametrize("question", [
    "Doanh thu của Coca Cola 330ml trong tháng 5 năm 2026 là bao nhiêu?",
    "Doanh thu tháng 2/2026 của khách An",
    "dthu hqua của sản phẩm X la bao nhiu",
])
def test_entity_specific_revenue_is_not_a_whole_business_macro(tmp_path, question):
    prepared = prepared_source(tmp_path)
    if " trong " in question:
        with pytest.raises(PlanError) as error:
            revenue_requirements({"question": question, "as_of_date": "2026-02-02"}, prepared)
        assert error.value.code == "entity_binding"
    else:
        assert revenue_requirements({"question": question, "as_of_date": "2026-02-02"}, prepared) is None
