"""Số liệu tự tạo; không dùng câu hỏi/đáp án của public."""
import datetime as dt

import pandas as pd
import pytest

from datamate.loader import Shop
from datamate.semantic import OrderView
from datamate.sem_api import SemAPI


@pytest.fixture
def sem():
    orders = pd.DataFrame({
        "order_id": list("1234567"),
        "date": pd.to_datetime(["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-02",
                                "2026-03-03", "2026-03-04", "2026-03-02"]),
        "cls": ["completed", "delivering", "cancelled", "returned", "completed", "completed", "completed"],
    })
    lines = pd.DataFrame({"order_id": list("11234567"), "amount": [100, 50, 200, 999, 888, 90, 60, 30]})
    customers = pd.DataFrame({"order_id": list("123467"), "customer": ["A", "A", "B", "B", "B", "C"],
                              "display": ["An", "An", "Bình", "Bình", "Bình", "An"]})
    items = pd.DataFrame({"order_id": list("1123"), "item": ["tra"] * 4})
    attrs = pd.DataFrame({"order_id": list("1234567"),
                         "pay": ["cash", "debt", "cash", "cash", "transfer", "cash", "cash"],
                         "channel": ["store", "zalo", "store", "store", "store", "store", "phone"]})
    api = SemAPI(Shop("synthetic", ""), dt.date(2026, 3, 3))
    api._built = True
    api._view = OrderView(lines, orders, customers=customers, items=items, attrs=attrs)
    return api


KY = (dt.date(2026, 3, 1), dt.date(2026, 3, 31))


def test_customer_totals_do_not_multiply_line_amounts(sem):
    assert sem.customer_revenue(KY, "A") == 350
    assert sem.customer_order_count(KY, "A") == 2
    assert sem.customer_avg_order_value(KY, "A") == 175
    assert sem.customer_count(KY) == 2  # A và C; khách lẻ không tính
    assert sem.customer_revenue(KY, "Bình") == 0  # chỉ huỷ/trả, đơn bán sau hôm nay
    assert sem.customer_avg_order_value(KY, "B") is None


def test_customer_identity_is_not_guessed_from_duplicate_names(sem):
    assert sem.customer_revenue(KY, "An") is None  # hai mã cùng tên
    assert sem.customer_order_count(KY, "không tồn tại") is None
    sem.view.customers = None
    assert sem.customer_count(KY) is None


def test_item_counts_orders_instead_of_lines(sem):
    assert sem.item_order_count(KY, "Trà") == 2
    assert sem.item_order_count(KY, "Bánh") is None
    sem.view.items = None
    assert sem.item_order_count(KY, "Trà") is None


def test_calendar_days_include_zero_sales_and_stop_at_today(sem):
    assert sem.active_days(KY) == 3
    assert sem.avg_daily_revenue(KY) == 157  # 470 / 3, làm tròn đồng
    assert sem.avg_daily_orders(KY) == 1.33  # 4 / 3
    # Ngày 2 có một đơn hợp lệ, ngày 3 có một đơn; ngày 1 không bị tính lại.
    assert sem.avg_daily_orders((dt.date(2026, 3, 2), dt.date(2026, 3, 3))) == 1
    sem._as_of = dt.date(2026, 3, 5)
    assert sem.avg_daily_orders((dt.date(2026, 3, 1), dt.date(2026, 3, 4))) == 1.25


def test_returns_and_channel_payment_counts(sem):
    assert sem.returned_count(KY) == 1
    assert sem.payment_order_count(KY, "cash") == 2
    assert sem.payment_order_count(KY, "debt") == 1
    assert sem.channel_order_count(KY, "store") == 2
    assert sem.channel_order_count(KY, "zalo") == 1
    assert sem.channel_order_count(KY, "unknown") is None
    assert sem.payment_order_count(KY, "unknown") is None
    sem.view.attrs.loc[sem.view.attrs.order_id == "1", "pay"] = None
    assert sem.payment_order_count(KY, "cash") is None


@pytest.mark.parametrize("method,args", [
    ("customer_count", ()), ("customer_revenue", ("A",)), ("customer_order_count", ("A",)),
    ("customer_avg_order_value", ("A",)), ("item_order_count", ("Trà",)),
    ("active_days", ()), ("avg_daily_revenue", ()), ("avg_daily_orders", ()),
    ("returned_count", ()), ("payment_order_count", ("cash",)), ("channel_order_count", ("store",)),
])
def test_common_sem_abstains_for_unavailable_periods_and_changed_core(sem, method, args):
    fn = getattr(sem, method)
    assert fn((dt.date(2025, 1, 1), dt.date(2025, 1, 31)), *args) is None
    assert fn((dt.date(2026, 3, 4), dt.date(2026, 3, 5)), *args) is None
    assert fn(tuple(reversed(KY)), *args) is None
    sem._rulebook = "| Đơn tính vào doanh thu | luật khác hoàn toàn |"
    del sem._common_guard
    assert fn(KY, *args) is None


def test_direct_agent_calls_honor_changed_customer_rules(sem):
    sem._rulebook = "| Xác định khách | luật mới |"
    assert sem.customer_revenue(KY, "A") is None
    assert sem.customer_count(KY) is None
    assert sem.active_days(KY) == 3  # không phụ thuộc cách xác định khách


def test_rank_ties_use_customer_code_date_and_hour(sem):
    o = pd.DataFrame({"customer": ["B", "A"], "display": ["An", "Zed"]})
    assert sem.view._top_name(pd.Series({"B": 3, "A": 3}), o) == "Zed"
    # Trong kỳ ngày 2–3 mỗi ngày có một đơn hợp lệ.
    ky = (dt.date(2026, 3, 2), dt.date(2026, 3, 3))
    assert sem.busiest_day(ky) == "2026-03-02"
    sem.view.orders["ts"] = sem.view.orders["date"] + pd.to_timedelta([9, 10, 9, 9, 10, 12, 9], unit="h")
    # Đơn hợp lệ giờ 9: 1,7; giờ 10: 2,5 → hoà, lấy 9.
    assert sem.peak_hour(KY) == 9
