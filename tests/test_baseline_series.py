import datetime as dt

import pandas as pd
import pytest

from datamate.baseline import answer
from datamate.helpers import Periods
from datamate.semantic import OrderView


@pytest.fixture
def view():
    orders = pd.DataFrame({"order_id": ["A", "B", "C", "D"],
                           "date": pd.to_datetime(["2025-11-01", "2025-12-01", "2026-01-01", "2026-01-02"]),
                           "cls": ["completed", "completed", "completed", "cancelled"]})
    return OrderView(pd.DataFrame({"order_id": ["A", "B", "C", "D"], "amount": [100, 200, 300, 9999]}), orders)


TODAY = dt.date(2026, 2, 1)


@pytest.mark.parametrize("question,expected", [
    ("Doanh thu 3 tháng liền nhau kết thúc ở tháng 1/2026 là bao nhiêu mỗi tháng, từ tháng cũ đến tháng mới?",
     {"type": "list", "answer": [100, 200, 300]}),
    ("Trong các tháng 11/2025, 12/2025 và 1/2026, doanh thu cao nhất vào tháng nào?",
     {"type": "date", "answer": "2026-01"}),
    ("Từ tháng 11/2025 tới tháng 1/2026, tháng nào cửa hàng bán được nhiều tiền nhất?",
     {"type": "date", "answer": "2026-01"}),
    ("Từ tháng 11/2025 qua tháng 12/2025 rồi tháng 1/2026, doanh thu có tăng liên tục không?",
     {"type": "boolean", "answer": True}),
    ("Xét 3 tháng 11/2025, 12/2025 và 1/2026, doanh thu tháng nào lớn nhất?",
     {"type": "date", "answer": "2026-01"}),
    ("Doanh thu từng tháng của 3 tháng 11/2025, 12/2025, 1/2026 lần lượt là bao nhiêu?",
     {"type": "list", "answer": [100, 200, 300]}),
])
def test_month_compositions_use_existing_revenue(view, question, expected):
    assert answer(question, TODAY, view, Periods(TODAY)) == expected


def test_ties_and_strict_increasing(view):
    view.lines.loc[view.lines.order_id == "C", "amount"] = 200
    q = "Doanh thu cao nhất vào tháng nào trong các tháng 11/2025, 12/2025 và 1/2026?"
    assert answer(q, TODAY, view, Periods(TODAY)) == {"type": "date", "answer": "2025-12"}
    q = "Từ tháng 11/2025 qua tháng 12/2025 rồi tháng 1/2026, doanh thu có tăng liên tục không?"
    assert answer(q, TODAY, view, Periods(TODAY)) == {"type": "boolean", "answer": False}


def test_explicit_month_prefixes_do_not_turn_a_year_into_a_count(view):
    q = 'Cho tôi doanh thu của tháng 11/2025, tháng 12/2025 và tháng 1/2026, ghi lần lượt theo thứ tự đó.'
    assert answer(q, TODAY, view, Periods(TODAY)) == {'type': 'list', 'answer': [100, 200, 300]}
    q = 'Doanh thu của tháng 12/2025 và tháng 1/2026 lần lượt bao nhiêu?'
    assert answer(q, TODAY, view, Periods(TODAY)) == {'type': 'list', 'answer': [200, 300]}


@pytest.mark.parametrize("question", [
    "Doanh thu nhóm bánh 3 tháng liền nhau kết thúc ở tháng 1/2026 là bao nhiêu mỗi tháng?",
    "Doanh thu 30 tháng liền nhau kết thúc ở tháng 1/2026 là bao nhiêu mỗi tháng?",
    "Doanh thu 3 tháng liền nhau kết thúc ở tháng 1/2027 là bao nhiêu mỗi tháng?",
    "Doanh thu 3 tháng liền nhau kết thúc ở tháng 1/2025 là bao nhiêu mỗi tháng?",
    "Tháng nào lợi nhuận cao nhất từ tháng 11/2025 tới tháng 1/2026?",
    "Xét 4 tháng 11/2025, 12/2025 và 1/2026, doanh thu tháng nào lớn nhất?",
])
def test_unsupported_conditions_and_missing_periods_fall_through(view, question):
    assert answer(question, TODAY, view, Periods(TODAY)) is None
