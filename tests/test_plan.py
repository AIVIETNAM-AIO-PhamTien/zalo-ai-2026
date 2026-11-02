import datetime as dt

import pandas as pd

from datamate.loader import Shop
from datamate.plan import check_plan, execute
from datamate.semantic import OrderView
from datamate.sem_api import SemAPI


def _ky(a, b):
    return {"tu": a, "den": b}


def _plan(op, *calls):
    return {"op": op, "calls": [{"fn": f, "args": a} for f, a in calls], "bo_sot": []}


def test_check_plan_period_must_match_question():
    today = dt.date(2026, 2, 8)
    week = _plan("value", ("revenue", {"ky": _ky("2026-02-01", "2026-02-08")}))
    assert check_plan(week, "bua nay cua hang ban duoc bao nhieu tien", today) is None
    day = _plan("value", ("revenue", {"ky": _ky("2026-02-08", "2026-02-08")}))
    assert check_plan(day, "bua nay cua hang ban duoc bao nhieu tien", today) == day
    ytd = _plan("value", ("order_count", {"ky": _ky("2026-01-01", "2026-02-08")}))
    assert check_plan(ytd, "Từ 1/1 năm nay đến hết hôm nay đã bán được bao nhiêu đơn?", today) == ytd
    may = _plan("list", ("revenue", {"ky": _ky("2025-05-01", "2025-05-05")}))
    assert check_plan(may, "doanh thu tháng 5/2025", today) is None


def test_check_plan_concepts_and_pct_order():
    today = dt.date(2026, 8, 5)
    rev = _plan("value", ("revenue", {"ky": _ky("2025-07-01", "2025-07-31")}))
    assert check_plan(rev, "tháng 7/2025 lãi gộp của cửa hàng là bao nhiêu?", today) is None
    grp = _plan("share", ("group_revenue", {"ky": _ky("2025-07-01", "2025-07-31"), "nhom": "bia"}),
                ("revenue", {"ky": _ky("2025-07-01", "2025-07-31")}))
    assert check_plan(grp, "tháng 7/2025 nhóm bia đóng góp bao nhiêu % doanh thu", today) == grp
    rev = _plan("pct_change", ("revenue", {"ky": _ky("2025-05-01", "2025-05-31")}),
                ("revenue", {"ky": _ky("2026-05-01", "2026-05-31")}))
    fixed = check_plan(rev, "So cùng kỳ năm trước, doanh thu tháng 5/2026 tăng mấy %", today)
    assert fixed["calls"][0]["args"]["ky"]["tu"] == "2026-05-01"


def _shop():
    df = pd.DataFrame({"HĐ": ["1", "1", "2", "3", "4"], "Ngày": pd.to_datetime(
        ["2026-01-05", "2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"]),
        "Khách": ["An", "An", "An", "Bình", "An"], "Tên hàng": ["Trà", "Bánh", "Trà", "Trà", "Bánh"],
        "SL": [1, 1, 2, 1, 1], "TT": [100, 50, 200, 100, 50], "Ghi chú": [None] * 5})
    shop = Shop("b", "", {"ban": df}, {"ban": {}})
    shop.card = {"ban": {"HĐ": "mã đơn", "Ngày": "ngày bán", "SL": "số lượng", "TT": "thành tiền sau giảm",
                         "Tên hàng": "tên hàng"}}
    return shop


def test_new_sem_functions():
    shop = _shop()
    v = OrderView.from_shop(shop)
    assert v is not None
    sem = SemAPI(shop, dt.date(2026, 1, 31), "")
    ky = ("2026-01-01", "2026-01-31")
    assert sem.item_revenue(ky, "trà") == 400
    assert sem.item_revenue(ky, "cà phê") is None
    assert sem.orders_min(ky, 150) == 2
    assert sem.days_revenue_min(ky, 300) == 1
    plan = _plan("value", ("item_revenue", {"ky": _ky("2026-01-01", "2026-01-31"), "mat_hang": "Bánh"}))
    assert execute(plan, sem) == {"type": "number", "answer": 100}


def test_rules_per_function():
    today = dt.date(2026, 8, 5)
    ky = _ky("2026-01-01", "2026-01-31")
    big = _plan("value", ("big_spender", {"ky": ky}))
    assert check_plan(big, "khach mua nhieu tien nhat thang 1/2026 la ai", today) is None
    top = _plan("value", ("top_spender", {"ky": ky}))
    assert check_plan(top, "khach mua nhieu tien nhat thang 1/2026 la ai", today) == top
    rev = _plan("value", ("item_revenue", {"ky": ky, "mat_hang": "cơm trắng"}))
    assert check_plan(rev, "tháng 1/2026 cơm trắng bán được bao nhiêu phần", today) is None
    assert check_plan(rev, "tháng 1/2026 cơm trắng bán được bao nhiêu tiền", today) == rev
    lst = _plan("list", ("revenue", {"ky": _ky("2025-03-01", "2025-03-31")}),
                ("revenue", {"ky": _ky("2025-04-01", "2025-04-30")}))
    assert check_plan(lst, "doanh thu 3 tháng liền kết thúc ở tháng 4/2025", today) is None


def test_same_period_needs_same_length():
    today = dt.date(2026, 6, 10)
    p = _plan("pct_change", ("revenue", {"ky": _ky("2026-06-01", "2026-06-10")}),
              ("revenue", {"ky": _ky("2026-05-01", "2026-05-31")}))
    assert check_plan(p, "từ đầu tháng này đến hết hôm nay dthu hơn kém cùng kỳ th trước mấy %", today) is None


def test_anchored_windows_must_match_code():
    q = ("So số ly trà bán ra trong 30 ngày kể từ ngày 05/02/2026, ngày tăng giá, với 30 ngày ngay trước ngày đó: "
         "tăng hay giảm bao nhiêu %")
    today = dt.date(2026, 9, 14)

    def qty(a, b, c, d):
        return _plan("pct_change", ("qty_sold", {"ky": _ky(a, b), "don_vi": "ly"}),
                     ("qty_sold", {"ky": _ky(c, d), "don_vi": "ly"}))
    assert check_plan(qty("2026-02-05", "2026-03-06", "2026-01-06", "2026-02-04"), q, today) is not None
    assert check_plan(qty("2026-02-05", "2026-03-07", "2026-01-06", "2026-02-04"), q, today) is None
    rev = _plan("pct_change", ("revenue", {"ky": _ky("2026-02-05", "2026-03-06")}),
                ("revenue", {"ky": _ky("2026-01-06", "2026-02-04")}))
    assert check_plan(rev, q, today) is None  # 'tăng giá' chỉ bỏ qua cho so số lượng


def test_compare_hint_resolves_reference_periods():
    from datamate.agent import compare_hint
    h = compare_hint("so với cùng kỳ th trước, dthu từ đầu th này đến hết hôm nay tăng bn %", dt.date(2026, 2, 11), None)
    assert "2026-02-01 → 2026-02-11" in h and "2026-01-01 → 2026-01-11" in h
    h = compare_hint("doanh thu thang 2/2025 hon hay kem th truoc bao nhiu phan tram", dt.date(2026, 8, 19), None)
    assert "2025-01-01 → 2025-01-31" in h
    h = compare_hint("Doanh thu tháng 7/2026 so với tháng 7 năm rồi", dt.date(2026, 8, 19), None)
    assert "2025-07-01 → 2025-07-31" in h
    assert compare_hint("dthu tháng 3/2026 bao nhiêu", dt.date(2026, 8, 19), None) == ""


def test_value_hint_links_typed_names_to_real_values():
    from datamate.plan import value_hint
    items = pd.DataFrame({"Tên món": ["Gạnbá chiên nước mắm", "Mực chiên nước mắm", "Trà sữa ô long (M)"],
                          "Mã": ["M01", "M02", "M03"]})
    shop = Shop("biz_v", "", {"mon": items}, {"mon": {}})
    h = value_hint("ngày 16/03/2025 shop bán gạnbá chiên nước mắm với đơn giá bnhiu một đĩa", shop)
    assert h.index('"Gạnbá chiên nước mắm"') < h.index('"Mực chiên nước mắm"')
    assert "M01" not in h
    assert '"Trà sữa ô long (M)"' in value_hint("thang 2 ban duoc may ly tra sua o long (m)", shop)
    assert value_hint("doanh thu tháng 3", shop) == ""
