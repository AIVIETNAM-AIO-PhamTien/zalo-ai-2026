import sqlite3

import pandas as pd
import pytest

from datamate.entities import resolve_entity
from datamate.revenue_rules import revenue_requirements
from datamate.ir import PlanError
from datamate.pipeline import prepare_business, Checkpoint, run_questions
from test_revenue_pipeline_v2 import prepared_source


def product_source(tmp_path):
    original = prepared_source(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.db") as con:
        con.execute("ALTER TABLE lines ADD COLUMN product TEXT")
        con.execute("ALTER TABLE lines ADD COLUMN sku TEXT")
        for rowid, product in enumerate(["Target", "Other", "Target", "Other", "Target"], 1):
            con.execute("UPDATE lines SET product=?,sku=? WHERE rowid=?", (product, product + "-ID", rowid))
    path = tmp_path / "README.md"
    path.write_text(path.read_text(encoding="utf-8") + "- **product**: tên hàng\n- **sku**: mã hàng\n", encoding="utf-8")
    return prepare_business(tmp_path, original.warehouse.catalog.foundation.documents["$rulebook"])


def test_product_filter_is_after_whole_order_return_exclusion(tmp_path):
    prepared = product_source(tmp_path)
    checkpoint = Checkpoint(tmp_path / "out", ["q"], {})
    traces = run_questions([{"id": "q", "question": "Doanh thu của target trong tháng 2/2026 là bao nhiêu?"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["answer"] == 50  # Not 150: another item returned on order b.
    assert traces[0]["diagnostics"] == []
    contract = next(iter(prepared.warehouse.views.values())).contract
    assert contract.applicability["entity"] == {"column": "product", "value": "Target"}
    assert ("ledger.db::lines", "product") in contract.sources
    assert ("ledger.db::lines", "sku") in contract.sources


@pytest.mark.parametrize("problem", ["unknown", "multiple_ids", "missing_name"])
def test_unknown_or_incomplete_entity_does_not_become_unfiltered_revenue(tmp_path, problem):
    prepared = product_source(tmp_path)
    frame = prepared.frames["ledger.db::lines"]
    requested = "Target"
    if problem == "unknown":
        requested = "Unknown"
    elif problem == "multiple_ids":
        frame.loc[2, "sku"] = "another-id"
    else:
        frame.loc[1, "product"] = None
    checkpoint = Checkpoint(tmp_path / "out", ["q"], {})
    traces = run_questions([{"id": "q", "question": f"Doanh thu của {requested} trong tháng 2/2026"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["type"] == "abstain"
    assert traces[0]["diagnostics"][0]["code"].startswith("entity_")


def test_zero_for_resolved_product_with_only_returned_orders(tmp_path):
    prepared = product_source(tmp_path)
    frame = prepared.frames["ledger.db::lines"]
    frame.loc[[0, 1, 4], "state"] = "Huỷ"
    checkpoint = Checkpoint(tmp_path / "out", ["q"], {})
    run_questions([{"id": "q", "question": "Doanh thu của Target trong tháng 2/2026"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["answer"] == 0


def test_exact_resolver_preserves_ids_and_diacritics():
    frame = pd.DataFrame({"name": ["Khánh", "Khanh"], "id": ["001", "002"]}, dtype=object)
    binding = resolve_entity(frame, "name", "KHÁNH", "id")
    assert binding.value == "Khánh"
    assert frame["id"].tolist() == ["001", "002"]
    with pytest.raises(PlanError):
        resolve_entity(pd.DataFrame({"name": ["Item", "item"]}), "name", "ITEM")


def test_explicit_customer_question_is_not_interpreted_as_product(tmp_path):
    prepared = product_source(tmp_path)
    assert revenue_requirements({"question": "Doanh thu của khách Target trong tháng 2/2026"}, prepared) is None


def test_same_label_for_customer_and_product_is_ambiguous(tmp_path):
    original = product_source(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.db") as con:
        con.execute("ALTER TABLE lines ADD COLUMN customer TEXT")
        con.execute("UPDATE lines SET customer='Target'")
    path = tmp_path / "README.md"
    path.write_text(path.read_text(encoding="utf-8") + "- **customer**: tên khách\n", encoding="utf-8")
    prepared = prepare_business(tmp_path, original.warehouse.catalog.foundation.documents["$rulebook"])
    with pytest.raises(PlanError) as error:
        revenue_requirements({"question": "Doanh thu của Target trong tháng 2/2026"}, prepared)
    assert error.value.code == "entity_kind_ambiguity"
