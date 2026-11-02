import sqlite3

import pytest

from datamate.pipeline import Checkpoint, prepare_business, run_questions
from datamate.revenue_rules import SUPPORTED


def prepared_source(tmp_path, columns=("o", "net", "placed", "state")):
    order, amount, date, status = columns
    with sqlite3.connect(tmp_path / "ledger.db") as con:
        con.execute(f'CREATE TABLE lines ("{order}" TEXT, "{amount}" INTEGER, "{date}" TEXT, "{status}" TEXT)')
        con.executemany("INSERT INTO lines VALUES (?,?,?,?)", [
            ("a", 10, "2026-02-01", "Hoàn tất"), ("a", 20, "2026-02-01", "Hoàn tất"),
            ("b", 100, "2026-02-02", "Hoàn tất"), ("b", 5, "2026-02-02", "Trả một phần"),
            ("c", 40, "2026-02-03", "Đang giao")])
    (tmp_path / "README.md").write_text(
        "Mỗi dòng là một dòng hàng trong đơn.\nTrạng thái đơn đã phản ánh mọi huỷ/trả hàng.\n"
        f"## Columns in `ledger.db::lines`\n- **{order}**: mã đơn\n"
        f"- **{amount}**: thành tiền dòng hàng sau giảm\n- **{date}**: ngày đặt đơn\n- **{status}**: trạng thái đơn\n",
        encoding="utf-8")
    rules = "\n".join(f"| {k} | {v} |" for k, v in SUPPORTED.items())
    return prepare_business(tmp_path, rules, {"ledger.db::lines": {date: {"type": "date"}}})


@pytest.mark.parametrize("columns", [("o", "net", "placed", "state"), ("ma", "tien", "ngay", "ket_qua")])
def test_revenue_question_automatically_runs_end_to_end(tmp_path, columns):
    prepared = prepared_source(tmp_path, columns)
    checkpoint = Checkpoint(tmp_path / "result", ["q"], {})
    traces = run_questions([{"id": "q", "question": "Tổng doanh thu tháng 2/2026?"}], prepared, checkpoint)
    assert checkpoint.rows["q"] == {"id": "q", "type": "number", "answer": 70}
    assert traces[0]["llm_calls"] == 0
    assert traces[0]["diagnostics"] == []


def test_source_completeness_must_be_documented(tmp_path):
    prepared = prepared_source(tmp_path)
    docs = prepared.warehouse.catalog.foundation.documents
    docs["README.md"] = docs["README.md"].replace("Trạng thái đơn đã phản ánh mọi huỷ/trả hàng.", "")
    checkpoint = Checkpoint(tmp_path / "result", ["q"], {})
    traces = run_questions([{"id": "q", "question": "Doanh thu tháng 2/2026"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["type"] == "abstain"
    assert traces[0]["diagnostics"][0]["code"] == "revenue_completeness"


def test_changed_column_meaning_invalidates_role_binding(tmp_path):
    prepared = prepared_source(tmp_path)
    from dataclasses import replace
    bindings = prepared.warehouse.bindings
    bindings[:] = [replace(b, description="tiền khách đã thanh toán") if b.column == "net" else b for b in bindings]
    checkpoint = Checkpoint(tmp_path / "result", ["q"], {})
    traces = run_questions([{"id": "q", "question": "Doanh thu tháng 2/2026"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["type"] == "abstain"
    assert traces[0]["diagnostics"][0]["code"] == "revenue_binding"


def test_extra_customer_condition_is_not_dropped(tmp_path):
    prepared = prepared_source(tmp_path)
    checkpoint = Checkpoint(tmp_path / "result", ["q"], {})
    run_questions([{"id": "q", "question": "Doanh thu tháng 2/2026 của khách An"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["type"] == "abstain"


def test_train_style_documentation_and_rulebook_status_authority(tmp_path):
    prepared = prepared_source(tmp_path)
    from dataclasses import replace
    docs = prepared.warehouse.catalog.foundation.documents
    docs["README.md"] = docs["README.md"].replace(
        "Mỗi dòng là một dòng hàng trong đơn.\nTrạng thái đơn đã phản ánh mọi huỷ/trả hàng.",
        "Một file phẳng, mỗi dòng một bản ghi (một dòng hàng trong đơn).")
    docs["$rulebook"] += "\nRiêng trạng thái huỷ, trả hàng của đơn lấy theo trạng thái ghi trên đơn, kể cả khi phiếu trả ghi ngày sau hôm nay."
    # Rebuild bindings so evidence points at the actual changed document.
    from datamate.semantics import document_bindings
    docs["README.md"] = docs["README.md"].replace("ngày đặt đơn", "ngày bán").replace("thành tiền dòng hàng sau giảm", "thành tiền sau giảm")
    prepared.warehouse.bindings = document_bindings(prepared.warehouse.catalog.foundation)
    checkpoint = Checkpoint(tmp_path / "result", ["q"], {})
    traces = run_questions([{"id": "q", "question": "Cho tôi biết doanh thu tháng 2 năm 2026."}], prepared, checkpoint)
    assert checkpoint.rows["q"]["answer"] == 70
    assert traces[0]["diagnostics"] == []


def test_rulebook_status_authority_cannot_be_fabricated_from_partial_quote(tmp_path):
    prepared = prepared_source(tmp_path)
    docs = prepared.warehouse.catalog.foundation.documents
    docs["README.md"] = docs["README.md"].replace("Trạng thái đơn đã phản ánh mọi huỷ/trả hàng.", "")
    docs["$rulebook"] += "\nTrạng thái đơn có thể chưa phản ánh phiếu trả hàng riêng."
    checkpoint = Checkpoint(tmp_path / "result", ["q"], {})
    traces = run_questions([{"id": "q", "question": "Doanh thu tháng 2/2026"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["type"] == "abstain"
    assert traces[0]["diagnostics"][0]["code"] == "revenue_completeness"
