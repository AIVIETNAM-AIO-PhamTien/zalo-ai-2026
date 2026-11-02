"""Hợp đồng luật public, dữ liệu tự tạo hoàn toàn, không có id/đáp án public."""
import datetime as dt

import pandas as pd
import pytest

from datamate.baseline import answer, prepare_sem
from datamate.helpers import Periods
from datamate.loader import Shop
from datamate.semantic import OrderView


@pytest.fixture
def view():
    frame = pd.DataFrame({"id": ["A", "A", "B", "C", "D", "E"],
        "Phí ship": [5, 5, 100, 100, 100, 3], "Tiền giảm voucher": [-10, -10, -100, -100, -100, -2],
        "Tiền giảm điểm": [8, 8, 100, 100, 100, 2], "Thu máy cũ": [50, 50, 100, 100, 100, 0]})
    registry = pd.DataFrame({"Mã khách": ["new", "never_bought", "future"],
                              "Ngày đăng ký": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"])})
    tables = {"orders": frame, "registry": registry}
    shop = Shop("synthetic", "", tables, {t: {"sources": [f"{t}.csv"]} for t in tables})
    orders = pd.DataFrame({"order_id": list("ABCDE"),
                           "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-04", "2026-01-03"]),
                           "cls": ["delivering", "returned", "cancelled", "completed", "completed"]})
    v = OrderView(pd.DataFrame({"order_id": list("AABCDE"), "amount": [100, 50, 100, 100, 100, 100]}), orders,
                  source={"table": "orders", "order_id": "id", "amount": "amount"})
    prepare_sem(v, shop)
    return v


TODAY = dt.date(2026, 1, 3)


@pytest.mark.parametrize("question,value", [
    ("Tổng phí ship tháng 1/2026 là bao nhiêu?", 8),
    ("Tổng tiền giảm voucher tháng 1/2026?", 12),
    ("Tháng 1/2026 tiền giảm nhờ dùng điểm là bao nhiêu?", 10),
    ("Tiền thu máy cũ tháng 1/2026 là bao nhiêu?", 50),
    ("Tháng 1/2026 có bao nhiêu khách thành viên mới?", 2),
])
def test_public_rules_in_the_same_baseline(view, question, value):
    assert answer(question, TODAY, view, Periods(TODAY)) == {"type": "number", "answer": value}


def test_inconsistent_invoice_values_fail_instead_of_summing_lines(view):
    view._sem_rules.shop.tables["orders"].loc[1, "Phí ship"] = 7
    assert answer("Tổng phí ship tháng 1/2026?", TODAY, view, Periods(TODAY)) is None


def test_balances_or_unknown_filters_are_not_money_discounts(view):
    frame = view._sem_rules.shop.tables["orders"]
    del frame["Tiền giảm điểm"]
    frame["Điểm hiện có"] = 100
    old = view._sem_rules
    prepare_sem(view, old.shop)
    assert answer("Tiền giảm nhờ dùng điểm tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    assert answer("Phí ship cho đơn COD tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    assert answer("Tiền giảm voucher của khách An tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    assert answer("Có bao nhiêu khách mới tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    assert answer("Thu mấy máy cũ tháng 1/2026?", TODAY, view, Periods(TODAY)) is None


def test_changed_public_rules_block_auto_execution(view):
    view._sem_rules.rulebook = "| Tiền giảm voucher, tiền giảm nhờ dùng điểm, phí ship | luật mới |"
    assert answer("Phí ship tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    view._sem_rules.rulebook = "| Khách thành viên mới trong tháng | luật mới |"
    assert answer("Có bao nhiêu thành viên mới tháng 1/2026?", TODAY, view, Periods(TODAY)) is None


def test_registry_name_fallback_requires_unambiguous_identity(view):
    old = view._sem_rules
    table = old.shop.tables["registry"]
    table["Tên KH"] = ["An", "Bình", "Chi"]
    del table["Mã khách"]
    prepare_sem(view, old.shop)
    assert answer("Có bao nhiêu thành viên mới tháng 1/2026?", TODAY, view, Periods(TODAY))["answer"] == 2
    table.loc[1, "Tên KH"] = "an"
    prepare_sem(view, old.shop)
    assert answer("Có bao nhiêu thành viên mới tháng 1/2026?", TODAY, view, Periods(TODAY)) is None


def test_public_ablation_disables_only_the_new_auto_handlers(view):
    old = view._sem_rules
    prepare_sem(view, old.shop, public_rules=False)
    assert answer("Phí ship tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    assert answer("Có bao nhiêu thành viên mới tháng 1/2026?", TODAY, view, Periods(TODAY)) is None
    assert answer("Doanh thu tháng 1/2026?", TODAY, view, Periods(TODAY)) == {"type": "number", "answer": 250}


@pytest.mark.parametrize("question,expected", [
    ("Khách nào trung thành nhất tháng 1/2026?", {"type": "string", "answer": "An"}),
    ("Khách sộp nhất tháng 1/2026 là ai?", {"type": "string", "answer": "An"}),
    ("Khách nào hay đổi ý nhất tháng 1/2026?", {"type": "string", "answer": "An"}),
    ("Mặt hàng được ưa chuộng nhất tháng 1/2026?", {"type": "string", "answer": "Trà"}),
    ("Ngày nào đắt khách nhất tháng 1/2026?", {"type": "date", "answer": "2026-01-03"}),
    ("Giờ cao điểm tháng 1/2026 là giờ nào?", {"type": "number", "answer": 9}),
    ("Khách nào mua nhiều đơn nhất tháng 1/2026?", {"type": "string", "answer": "An"}),
    ("Khách chi tiêu nhiều nhất tháng 1/2026 là ai?", {"type": "string", "answer": "An"}),
])
def test_rankings_use_public_definitions_not_familiar_synonyms(view, question, expected):
    view.orders = pd.concat([view.orders, pd.DataFrame({"order_id": ["X", "Y"],
        "date": pd.to_datetime(["2026-01-02", "2026-01-03"]), "cls": ["completed", "completed"]})], ignore_index=True)
    view.lines = pd.concat([view.lines, pd.DataFrame({"order_id": ["X", "Y"], "amount": [10, 10]})], ignore_index=True)
    view.customers = pd.DataFrame({"order_id": ["A", "B", "C", "D", "E", "X", "Y"],
        "customer": ["C1", "C2", "C1", "C2", "C2", "C1", "C1"],
        "display": ["An", "Bình", "An", "Bình", "Bình", "An", "An"]})
    view.items = pd.DataFrame({"order_id": ["A", "E", "X"], "item": ["tra", "tra", "banh"],
                               "name": ["Trà", "Trà", "Bánh"], "bqty": [1, 1, 1], "bunit": ["ly", "ly", "cai"]})
    view.orders["ts"] = view.orders.date + pd.to_timedelta([9, 10, 10, 10, 10, 9, 9], unit="h")
    assert answer(question, TODAY, view, Periods(TODAY)) == expected


def test_big_spender_uses_money_half_up_before_ranking():
    orders = pd.DataFrame({"order_id": list("abcdef"), "date": pd.to_datetime(["2026-01-01"] * 6),
                           "cls": ["completed"] * 6})
    customers = pd.DataFrame({"order_id": list("abcdef"), "customer": ["Z"] * 3 + ["A"] * 3,
                              "display": ["Zed"] * 3 + ["An"] * 3})
    lines = pd.DataFrame({"order_id": list("abcdef"), "amount": [100, 100, 101.5, 100, 100, 100]})
    v = OrderView(lines, orders, customers=customers)
    # Z: TB 100.5 → 101; A: 100. Không dùng round() nửa về số chẵn để tạo hoà giả.
    assert v.big_spender((dt.date(2026, 1, 1), dt.date(2026, 1, 31)), TODAY) == "Zed"
