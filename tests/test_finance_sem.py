import datetime as dt

import pandas as pd
import pytest

from datamate.loader import Shop
from datamate.semantic import OrderView
from datamate.sem_api import SemAPI

KY = (dt.date(2026, 1, 1), dt.date(2026, 1, 31))


@pytest.fixture
def sem():
    sales = pd.DataFrame({"id": ["1", "1", "1", "2", "3", "4"],
                          "item": ["Trà", "Bánh", "Bánh", "Trà", "Trà", "Trà"],
                          "qty": [2, 3, 3, 10, 10, 10], "cost": [10., 5., 5., 10., 10., 10.],
                          "amount": [100, 60, 60, 999, 999, 999]})
    expenses = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"] * 3 + ["2026-02-01"]),
                             "money": [10, 10, 50, 100],
                             "kind": ["điện", "điện", "thay thiết bị", "khác"]})
    shop = Shop("synthetic", "", {"sales": sales, "expenses": expenses})
    orders = pd.DataFrame({"order_id": ["1", "2", "3", "4"],
                           "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-21"]),
                           "cls": ["delivering", "cancelled", "returned", "completed"]})
    lines = pd.DataFrame({"order_id": ["1", "1", "2", "3", "4"], "amount": [100, 60, 999, 999, 999]})
    v = OrderView(lines, orders, source={"table": "sales", "order_id": "id", "amount": "amount"})
    api = SemAPI(shop, dt.date(2026, 1, 20))
    api._built, api._view = True, v
    return api


def test_profit_uses_recorded_unit_cost_excludes_cancel_returns_future(sem):
    assert sem.gross_profit(KY, "qty", "cost") == 125  # 100-2*10 + 60-3*5
    assert sem.gross_profit(KY, "qty", "cost", "item", "Trà") == 80
    assert sem.gross_profit(KY, "qty", "cost", "item", "missing") is None
    assert sem.gross_profit(KY, "qty", "cost", mat_hang="Trà") is None


def test_expenses_preserve_duplicate_rows_and_equipment(sem):
    assert sem.expense_total(KY, "expenses", "date", "money") == 70
    assert sem.net_profit(KY, "qty", "cost", "expenses", "date", "money") == 55
    assert sem.expense_total((dt.date(2025, 1, 1), dt.date(2025, 1, 31)),
                             "expenses", "date", "money") is None


def test_profit_checks_per_order_agreement_and_missing_cost(sem):
    sem._shop.tables["sales"].loc[0, "amount"] += 1
    assert sem.gross_profit(KY, "qty", "cost") is None
    sem._shop.tables["sales"].loc[0, "amount"] -= 1
    sem._shop.tables["sales"].loc[0, "cost"] = float("nan")
    assert sem.gross_profit(KY, "qty", "cost") is None


def test_finance_guard_and_absent_inputs(sem):
    assert sem.gross_profit(KY, "missing", "cost") is None
    assert sem.expense_total(KY, "missing", "date", "money") is None
    assert sem.net_profit(KY, "qty", "cost", "missing", "date", "money") is None
    sem._rulebook = "| Lãi gộp | luật khác |"
    del sem._common_guard
    assert sem.gross_profit(KY, "qty", "cost") is None
    assert sem.net_profit(KY, "qty", "cost", "expenses", "date", "money") is None


def test_profit_fractional_money_uses_half_up(sem):
    sem._shop.tables["sales"].loc[0, "cost"] = 10.25
    assert sem.gross_profit(KY, "qty", "cost") == 125  # 124.5 làm tròn lên


def test_staff_ranks_by_revenue_not_lines_and_conflicts_abstain(sem):
    sem._shop.tables["headers"] = pd.DataFrame({"id": ["1", "2", "3", "4"], "staff": ["Z", "A", "A", "A"]})
    assert sem.staff_best(KY, "headers", "id", "staff") == "Z"
    sem._shop.tables["headers"] = pd.DataFrame({"id": ["1", "1", "2", "3", "4"],
                                                "staff": ["Z", "A", "A", "A", "A"]})
    assert sem.staff_best(KY, "headers", "id", "staff") is None


def test_walkins_never_in_customer_group_even_if_labelled_retail(sem):
    sem._shop.tables["headers"] = pd.DataFrame({"id": ["1", "2", "3", "4"], "group": ["lẻ"] * 4})
    sem.view.customers = pd.DataFrame({"order_id": ["2"], "customer": ["A"], "display": ["An"]})
    assert sem.customer_group_revenue(KY, "headers", "id", "group", "lẻ") == 0
    assert sem.revenue(KY) == 160
    sem.view.customers = pd.DataFrame({"order_id": ["1"], "customer": ["A"], "display": ["An"]})
    assert sem.customer_group_revenue(KY, "headers", "id", "group", "lẻ") == 160


def test_profitable_item_uses_total_profit_not_revenue_or_unit_margin(sem):
    frame = sem._shop.tables["sales"]
    frame["code"] = frame["item"].map({"Trà": "T", "Bánh": "B"})
    assert sem.best_profit_item(KY, "qty", "cost", "item", "code") == "T"
    # Lãi Trà = 80, Bánh = 45; đổi giá vốn Trà để bằng 45 thì lấy B trước T.
    frame.loc[frame["item"] == "Trà", "cost"] = 27.5
    assert sem.best_profit_item(KY, "qty", "cost", "item", "code") == "B"
    frame.loc[frame["item"] == "Trà", "cost"] = -1
    assert sem.best_profit_item(KY, "qty", "cost", "item", "code") is None


def test_unsold_requires_valid_history_and_no_valid_sale_in_period(sem):
    frame = sem._shop.tables["sales"]
    frame["code"] = frame["item"].map({"Trà": "T", "Bánh": "B"})
    sem.view.items = pd.DataFrame({"order_id": frame.id, "item": frame.item.map({"Trà": "tra", "Bánh": "banh"})})
    # Đơn 1 trước kỳ có cả hai hàng; đơn 2 trong kỳ mua trà.
    sem.view.orders.loc[sem.view.orders.order_id == "2", "date"] = pd.Timestamp("2026-01-02")
    sem.view.orders.loc[sem.view.orders.order_id == "2", "cls"] = "completed"
    p = (dt.date(2026, 1, 2), dt.date(2026, 1, 20))
    assert sem.unsold_items(p, "item", "code") == ["B"]
    # Huỷ đơn 2 thì Trà cũng không có đơn hợp lệ trong kỳ.
    sem.view.orders.loc[sem.view.orders.order_id == "2", "cls"] = "cancelled"
    assert sem.unsold_items(p, "item", "code") == ["B", "T"]
