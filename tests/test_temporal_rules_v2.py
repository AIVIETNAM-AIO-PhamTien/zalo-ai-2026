import datetime as dt
from pathlib import Path

import pytest

from datamate.ir import PlanError
from datamate.temporal import reference_date
from datamate.temporal_rules import SUPPORTED_PERIODS, grounded_window
from datamate.metric_rules import metric_requirements
from datamate.planner import solve
from test_metric_rules_v2 import business


def rulebook(*labels):
    return {"$rulebook": "\n".join(f"| {label} | {SUPPORTED_PERIODS[label]} |" for label in labels)}


@pytest.mark.parametrize("phrase,label,reference,start,end", [
    ("tháng này", "Tháng này", "2026-02-15", "2026-02-01", "2026-02-16"),
    ("cùng kỳ tháng trước", "Tháng này", "2024-03-31", "2024-02-01", "2024-03-01"),
    ("tuần này", "Tuần", "2026-10-29", "2026-10-26", "2026-10-30"),
    ("tuần trước", "Tuần", "2026-10-29", "2026-10-19", "2026-10-26"),
    ("từ đầu năm", "Từ đầu năm, từ đầu năm tới nay", "2026-02-15", "2026-01-01", "2026-02-16"),
])
def test_only_applicable_rule_is_required_and_preserves_evidence(phrase, label, reference, start, end):
    docs = rulebook(label)
    window, evidence = grounded_window(phrase, reference_date(question=reference), docs)
    assert (window.start.isoformat(), window.end_exclusive.isoformat()) == (start, end)
    assert len(evidence) == 1 and evidence[0].valid(docs)


def test_missing_changed_duplicated_policy_is_not_defaulted():
    reference = reference_date(question="2026-10-29")
    docs = rulebook("Tuần")
    with pytest.raises(PlanError):
        grounded_window("tháng này", reference, docs)
    docs["$rulebook"] = docs["$rulebook"].replace("thứ Hai", "Chủ nhật")
    with pytest.raises(PlanError):
        grounded_window("tuần này", reference, docs)
    docs = rulebook("Tháng này")
    docs["$rulebook"] *= 2
    with pytest.raises(PlanError):
        grounded_window("tháng này", reference, docs)


def test_named_metric_automatically_uses_policy_and_detects_stale_rulebook(tmp_path):
    prepared = business(tmp_path)
    docs = prepared.warehouse.catalog.foundation.documents
    docs["$rulebook"] += "\n" + rulebook("Tháng này")["$rulebook"]
    question = {"question": "Thời lượng xử lý trong tháng này", "as_of_date": "2026-02-01"}
    requirements = metric_requirements(question, prepared)
    result = solve(question, prepared.warehouse, prepared.frames, prepared.schemas, requirements=requirements)
    assert result.selection.answer == {"type": "number", "answer": 20}
    contract = prepared.warehouse.views[requirements["contract_ids"][0]].contract
    assert any("Tháng này" in e.quote for e in contract.evidence)
    docs["$rulebook"] = docs["$rulebook"].replace("hết hôm nay", "hết tháng")
    assert prepared.warehouse.validate_contract(contract)


def test_train_rulebook_is_supported_without_answers():
    path = Path("data/train/rulebook.md")
    if not path.exists():
        pytest.skip("training rulebook unavailable")
    docs = {"$rulebook": path.read_text(encoding="utf-8")}
    window, evidence = grounded_window("tháng này", reference_date(question="2026-02-15"), docs)
    assert window.end_exclusive == dt.date(2026, 2, 16)
    assert evidence[0].valid(docs)
