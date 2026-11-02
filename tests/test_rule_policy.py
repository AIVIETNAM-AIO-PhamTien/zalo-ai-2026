from pathlib import Path

import pytest

from datamate.rule_guard import Guard
from datamate.rule_policy import Policy, infer

TRAIN = Path("data/train/rulebook.md").read_text(encoding="utf-8")


def _swap(rule, new, base=TRAIN):
    out = []
    for line in base.splitlines():
        if line.startswith(f"| {rule} |"):
            line = f"| {rule} | {new} |"
        out.append(line)
    return "\n".join(out)


def test_train_rulebook_is_default():
    assert infer(TRAIN) == (Policy(), set())


def test_r1_rules_become_parameters():
    rb = _swap("Đơn tính vào doanh thu", "đơn **hoàn thành**. Đơn huỷ, đơn trả hàng (**kể cả chỉ trả một phần**) "
               "và đơn **đang giao** đều **không** tính")
    rb = rb.replace("| N ngày gần nhất | N ngày liền nhau **kể cả hôm nay**, ngày cuối là hôm nay |",
                    "| N ngày gần nhất | N ngày liền nhau **trước hôm nay**, ngày cuối là **hôm qua**; không tính hôm nay |")
    policy, resolved = infer(rb)
    assert policy == Policy(delivering_valid=False, last_n_include_today=False)
    assert resolved == {"Đơn tính vào doanh thu", "N ngày gần nhất"}
    assert not Guard(rb, policy_aware=True).view
    assert Guard(rb).core  # SEM / kế hoạch / agent chưa theo Policy: vẫn chặn


def test_unreadable_change_stays_blocked():
    rb = _swap("Đơn tính vào doanh thu", "đơn đã thu đủ tiền")
    assert infer(rb)[1] == set()
    assert Guard(rb, policy_aware=True).core


R2 = Path("outputs/synth/r2/rulebook.md")


def _r2_rules():
    rb = TRAIN
    for name in ("Doanh thu một đơn", "Giá trị trung bình một đơn"):
        row = next(l for l in R2.read_text(encoding="utf-8").splitlines() if l.startswith(f"| {name} |"))
        rb = _swap(name, row.split(" | ", 1)[1].rstrip(" |"), rb)
    return rb


@pytest.mark.skipif(not R2.exists(), reason="cần outputs/synth/r2 (sinh bằng datamate.synth)")
def test_r2_order_level_rules_become_parameters():
    policy, resolved = infer(_r2_rules())
    assert policy == Policy(order_discount=True, ship_fee=True, aov_floor=True)
    assert resolved == {"Doanh thu một đơn", "Giá trị trung bình một đơn"}


def test_order_revenue_needs_every_clause():
    rb = _swap("Doanh thu một đơn", "tổng thành tiền các dòng hàng, trừ giảm giá trên cả đơn; không cộng thuế VAT")
    assert infer(rb)[1] == set()  # không nói phí giao: không đoán


def test_apply_policy_adds_order_level_line_or_abstains():
    import pandas as pd
    from types import SimpleNamespace
    from datamate.semantic import OrderView

    lines = pd.DataFrame({"order_id": ["a", "a", "b"], "amount": [100, 50, 70]})
    v = SimpleNamespace(lines=lines, order_extras=pd.DataFrame({"order_id": ["a", "b"], "disc": [20, 0],
                                                                "ship": [15, 0]}))
    assert OrderView.apply_policy(v, Policy(order_discount=True, ship_fee=True))
    assert v.lines.groupby("order_id")["amount"].sum().to_dict() == {"a": 145, "b": 70}
    assert OrderView.apply_policy(v, Policy())
    assert len(v.lines) == 3  # bỏ policy: về lại dòng hàng
    v2 = SimpleNamespace(lines=lines, order_extras=pd.DataFrame({"order_id": ["a"], "disc": [20]}),
                         orders=pd.DataFrame({"order_id": ["a", "b", "c"]}))
    assert not OrderView.apply_policy(v2, Policy(ship_fee=True))
    # thiếu cột phí giao: tiền mọi đơn chưa biết, dòng hàng (doanh thu nhóm / mặt hàng) giữ nguyên
    assert OrderView.order_money(v2).isna().all()
    assert v2._goods_lines.groupby("order_id")["amount"].sum().to_dict() == {"a": 150, "b": 70}
    assert OrderView.apply_policy(v2, Policy())
    assert OrderView.order_money(v2).to_dict() == {"a": 150, "b": 70}


def test_unknown_order_money_blocks_money_answers_not_counts():
    import datetime as dt
    import pandas as pd
    from datamate.semantic import OrderView

    v = OrderView.__new__(OrderView)
    v.orders = pd.DataFrame({"order_id": ["a", "b", "c"], "cls": ["completed"] * 3, "customer": ["x", "x", "y"],
                             "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"])})
    v.lines = pd.DataFrame({"order_id": ["a", "b", "c"], "amount": [100, 50, 70]})
    v.order_extras = pd.DataFrame({"order_id": ["a", "b", "c"]})
    assert not v.apply_policy(Policy(order_discount=True))
    period, as_of = (dt.date(2026, 1, 1), dt.date(2026, 1, 31)), dt.date(2026, 1, 31)
    assert v.order_count(period, as_of) == 3
    assert v.best_revenue_day(period, as_of) is None
    assert v.order_count_min(period, as_of, 60) is None
    assert v.days_revenue_min(period, as_of, 60) is None
    assert v.revenue(period, as_of) is None
    assert v.avg_order_value(period, as_of) is None
