import datetime as dt

import pandas as pd
import pytest

from datamate.baseline_sem import SemRules
from datamate.loader import Shop
from datamate.semantic import OrderView


@pytest.fixture
def solver():
    readme = """## Cột trong `sales.csv`
- **id**: mã đơn
- **qty**: số lượng
- **amount**: thành tiền sau giảm
- **Giá vốn**: giá vốn

## Cột trong `headers.csv`
- **id**: mã đơn
- **staff**: mã nhân viên bán
- **group**: nhóm khách

## Quy ước
- **Giá vốn**: **Giá vốn** là giá vốn một đơn vị bán.
- **Sổ chi phí**: mọi khoản chi vận hành, thiết bị.
"""
    sales = pd.DataFrame({"id": ["1", "2"], "qty": [2, 1], "Giá vốn": [10, 10], "amount": [100, 90]})
    headers = pd.DataFrame({"id": ["1", "2"], "staff": ["NV2", "NV1"], "group": ["Thân thiết", "Thân thiết"]})
    expenses = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "So tien": [5]})
    tables = {"sales": sales, "headers": headers, "so_chi_phi": expenses}
    meta = {k: {"sources": [f"{k}.csv"]} for k in tables}
    shop = Shop("synthetic", readme, tables, meta)
    orders = pd.DataFrame({"order_id": ["1", "2"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                           "cls": ["completed", "returned"]})
    view = OrderView(pd.DataFrame({"order_id": ["1", "2"], "amount": [100, 90]}), orders,
                     customers=pd.DataFrame({"order_id": ["1"], "customer": ["A"], "display": ["An"]}),
                     source={"table": "sales", "order_id": "id", "amount": "amount"})
    return SemRules(shop, view)


TODAY = dt.date(2026, 1, 31)


@pytest.mark.parametrize("q,expected", [
    ("Tổng lãi gộp tháng 1/2026 là bao nhiêu?", {"type": "number", "answer": 80}),
    ("Lợi nhuận ròng tháng 1/2026 là bao nhiêu?", {"type": "number", "answer": 75}),
    ("Chi phí tháng 1/2026 là bao nhiêu tiền?", {"type": "number", "answer": 5}),
    ("Nhân viên nào bán giỏi nhất tháng 1/2026?", {"type": "string", "answer": "NV2"}),
    ("Nhân viên bán giỏi nhất tháng 1/2026 là ai?", {"type": "string", "answer": "NV2"}),
    ("Nhân viên mã NV2 tháng 1/2026 bán được bao nhiêu tiền?", {"type": "number", "answer": 100}),
    ("Trừ hết chi phí rồi thì tháng 1/2026 còn lãi ròng bao nhiêu?", {"type": "number", "answer": 75}),
    ("Có bao nhiêu khách đã mua tháng 1/2026?", {"type": "number", "answer": 1}),
    ("Số đơn trả hàng tháng 1/2026?", {"type": "number", "answer": 1}),
    ("Doanh thu nhóm khách Thân thiết tháng 1/2026?", {"type": "number", "answer": 100}),
    ("Số đơn trung bình mỗi ngày tháng 1/2026?", {"type": "number", "answer": 0.03}),
])
def test_answers_without_llm(solver, q, expected):
    assert solver.answer(q, TODAY) == expected


@pytest.mark.parametrize("q", [
    "Lãi gộp của khách An tháng 1/2026?", "Lãi gộp tháng 1/2026 so với tháng 2/2026?",
    "Lãi gộp tháng 1/2026 chiếm bao nhiêu %?", "Lãi gộp tháng 1/2026 không tính Trà?",
    "Lãi gộp riêng ca sáng tháng 1/2026?", "Lãi gộp tháng 1/2026 và lãi ròng?",
    "Lãi gộp tháng 1/2026 của đơn có từ 5 sản phẩm?", "Nhân viên nào bán nhiều đơn nhất tháng 1/2026?",
    "Tổng lãi gộp tháng 12/2025?", "Tổng lãi gộp tháng 2/2026?",
])
def test_unrepresented_conditions_fall_through(solver, q):
    assert solver.answer(q, TODAY) is None


def test_changed_rules_and_ambiguous_columns_fall_through(solver):
    solver.rulebook = "| Lãi gộp | luật mới |"
    assert solver.answer("Lãi gộp tháng 1/2026?", TODAY) is None
    solver.rulebook = ""
    solver.shop.tables["sales"]["GiaVonLucBan"] = 1
    other = SemRules(solver.shop, solver.view)
    assert other.answer("Lãi gộp tháng 1/2026?", TODAY) is None


def test_preparation_keeps_only_required_raw_tables(solver):
    solver.shop.tables["unrelated"] = pd.DataFrame({"payload": [1, 2, 3]})
    other = SemRules(solver.shop, solver.view)
    assert "unrelated" not in other.shop.tables
    assert other.answer("Lãi gộp tháng 1/2026?", TODAY) == {"type": "number", "answer": 80}
