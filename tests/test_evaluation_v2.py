import pytest

from datamate.evaluation import business_split, evaluate, reliability


def test_pa_groups_costs_and_regression_accounting():
    questions = [{"id": str(i), "business_id": "a" if i < 2 else "b"} for i in range(4)]
    refs = [{"id": str(i), "type": "number", "answer": i} for i in range(4)]
    predictions = [refs[0], {"id": "1", "type": "number", "answer": 99}, {"id": "2", "type": "abstain"}]
    result = evaluate(predictions, refs, questions, refs, [{"tokens": 10, "llm_calls": 1, "runtime_seconds": 0.1}])
    assert result["metrics"] == {"total": 4, "correct": 1, "wrong": 1, "abstain": 2, "pa": 0,
                                  "precision": 0.5, "coverage": 0.5}
    assert result["correct_to_wrong"] == 1
    assert result["by_business"]["b"]["abstain"] == 2
    assert result["costs"]["tokens"] == 10
    assert result["errors"]["missing_prediction"] == 1


def test_business_split_is_disjoint_and_stable():
    qs = [{"id": f"{biz}-{i}", "business_id": str(biz)} for biz in range(20) for i in range(3)]
    split = business_split(qs)
    assert split == business_split(qs)
    assert not set(split["development"]) & set(split["validation"])
    for biz in range(20):
        assert all(f"{biz}-{i}" in split["development"] for i in range(3)) or all(f"{biz}-{i}" in split["validation"] for i in range(3))


def test_evaluation_rejects_duplicate_or_mismatched_ids():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([{"id": "a"}, {"id": "a"}], [], [])
    with pytest.raises(ValueError, match="scope"):
        evaluate([], [{"id": "a"}], [])


def test_reliability_reports_measurements_not_fake_calibration():
    result = reliability([0.8, 0.8], [True, False])
    assert result["ece"] == pytest.approx(0.3)
    assert result["brier"] == pytest.approx(0.34)
    with pytest.raises(ValueError):
        reliability([2], [True])
