import datetime as dt

import pytest

from datamate.baseline import answer, extract_periods, norm
from datamate.helpers import Periods

D = dt.date


class FakeView:
    """Ghi lại lời gọi metric để kiểm tra parser chọn đúng ý định và kỳ."""

    def __init__(self, lo=D(2025, 1, 1), hi=D(2026, 9, 30)):
        self.lo, self.hi, self.calls = lo, hi, []

    def covers(self, p):
        return not (p[1] < self.lo or p[0] > self.hi)

    def revenue(self, p, as_of):
        self.calls.append(("revenue", p))
        return 100 if p[0].month % 2 else 80

    def order_count(self, p, as_of):
        self.calls.append(("order_count", p))
        return 7

    def cancelled_count(self, p, as_of):
        self.calls.append(("cancelled", p))
        return 3

    def cancel_rate(self, p, as_of):
        self.calls.append(("cancel_rate", p))
        return 1.5

    def avg_order_value(self, p, as_of):
        self.calls.append(("avg", p))
        return 123456

    def best_revenue_day(self, p, as_of):
        self.calls.append(("best_day", p))
        return "2025-04-27"


AS_OF = D(2026, 6, 16)


def run(q, view=None):
    view = view or FakeView()
    return answer(q, AS_OF, view, Periods(AS_OF)), view.calls


@pytest.mark.parametrize("q,period", [
    ("Cho tôi biết doanh thu tháng 6/2025.", (D(2025, 6, 1), D(2025, 6, 30))),
    ("thang 5/2026 dthu la bnhiu", (D(2026, 5, 1), D(2026, 5, 31))),
    ("tong doanh thu thang 4 nam 2026 la bao nhieu ha", (D(2026, 4, 1), D(2026, 4, 30))),
    ("dthu từ đầu năm tới hết hnay là bao nhiêu?", (D(2026, 1, 1), AS_OF)),
    ("Doanh thu hôm nay là bao nhiêu?", (AS_OF, AS_OF)),
    ("doanh thu tuan trc cua cua hang la bnhiu", (D(2026, 6, 8), D(2026, 6, 14))),
    ("Cho tôi biết doanh thu 30 ngày gần nhất.", (D(2026, 5, 18), AS_OF)),
    ("7 ngày gần nhất cửa hàng bán được nhiêu tiền vậy", (D(2026, 6, 10), AS_OF)),
])
def test_revenue_questions(q, period):
    res, calls = run(q)
    assert res == {"type": "number", "answer": 100 if period[0].month % 2 else 80}
    assert calls == [("revenue", period)]


def test_counts_and_rates():
    assert run("thang 5/2026 bi huy may don")[0] == {"type": "number", "answer": 3}
    assert run("Hôm qua cửa hàng bán đc mấy đơn?")[0] == {"type": "number", "answer": 7}
    assert run("tỉ lệ huỷ đơn tháng 6/2025 là nhiêu %")[0] == {"type": "percent", "answer": 1.5}


def test_pct_change_vs_previous_month():
    res, calls = run("thang 10/2025 dthu thay doi bao nhieu % so vs thang lien trc z")
    assert calls == [("revenue", (D(2025, 10, 1), D(2025, 10, 31))), ("revenue", (D(2025, 9, 1), D(2025, 9, 30)))]
    assert res == {"type": "percent", "answer": -20.0}  # tháng 10: 80, tháng 9: 100


@pytest.mark.parametrize("q", [
    "Tháng 12/2025 mặt hàng Oresol 245 DHG hộp 20 gói thu về bao nhiêu tiền?",
    "check gium dthu nhom hang sinh to chiem bnhiu % tong doanh thu thang 3 nam 2025",
    "Tháng này, dthu các đơn bán tại quầy là bao nhiêu",
    "Khách nhóm thân thiết tháng 6/2025 mua trung bình bao nhiêu tiền hàng mỗi đơn?",
    "ngày nào trong tháng 4/2025 dthu thấp nhất",
    "ngày nào trong tháng 4/2025 dthu ít nhất",
    "số lượng trung bình mỗi đơn tháng 4/2025 là bao nhiêu",
    "trung bình mỗi đơn tháng 4/2025 có mấy món",
    "Tổng tiền bán hàng của ca sáng tuần trước là bao nhiêu?",
])
def test_unknown_qualifiers_abstain(q):
    assert run(q)[0] is None


@pytest.mark.parametrize("q", [
    "Trung bình mỗi đơn tháng 5/2025 mang về bao nhiêu doanh thu?",
    "dthu binh quan moi don hang thang 5/2025 la bnhiu",
    "giá trị trung bình một đơn tháng 5/2025 là bao nhiêu vậy",
])
def test_average_order_value(q):
    res, calls = run(q)
    assert res == {"type": "number", "answer": 123456}
    assert calls == [("avg", (D(2025, 5, 1), D(2025, 5, 31)))]


@pytest.mark.parametrize("q", [
    "tháng 4/2025 doanh thu ngày nào cao nhất",
    "ngay nao trong thang 4/2025 dthu cao nhat vay",
])
def test_best_revenue_day(q):
    res, calls = run(q)
    assert res == {"type": "date", "answer": "2025-04-27"}
    assert calls == [("best_day", (D(2025, 4, 1), D(2025, 4, 30)))]


def test_quarter_with_slash_is_not_read_as_month():
    res, calls = run("Tổng doanh thu quý 2/2026 là bao nhiêu?")
    assert calls == [("revenue", (D(2026, 4, 1), D(2026, 6, 30)))]
    res, calls = run("cho hỏi số đơn quý 1/2026 là bao nhiêu")
    assert calls == [("order_count", (D(2026, 1, 1), D(2026, 3, 31)))]


@pytest.mark.parametrize("q,call", [
    ("Tháng 5/2026 cửa hàng bán được tổng cộng bao nhiêu tiền?", "revenue"),
    ("cho tôi biết doanh thu bữa qua", "revenue"),
    ("Tính từ ngày 1/1 đến hết hôm nay, doanh thu năm nay là bao nhiêu?", "revenue"),
    ("giúp mình đếm số đơn đã hủy trong tháng 5/2026", "cancelled"),
    ("tháng 5/2026 cửa hàng có bao nhiêu % đơn bị hủy", "cancel_rate"),
])
def test_more_phrasings(q, call):
    res, calls = run(q)
    assert res is not None and [c[0] for c in calls] == [call]


def test_this_month_vs_same_days_any_order():
    res, calls = run("Từ đầu tháng này đến hết hôm nay, doanh thu hơn hay kém cùng kỳ tháng trước mấy %?")
    assert [c[1] for c in calls] == [(D(2026, 6, 1), AS_OF), (D(2026, 5, 1), D(2026, 5, 16))]


def test_period_outside_data_is_unanswerable():
    res, _ = run("Cho tôi biết doanh thu tháng 3/2025.", FakeView(lo=D(2025, 4, 1)))
    assert res["type"] == "unanswerable"


def test_norm_and_periods():
    s = norm("Tháng 5-2025 dthu?")
    periods, rest = extract_periods(s, Periods(AS_OF))
    assert periods == [("month", (D(2025, 5, 1), D(2025, 5, 31)))]
    assert "DOANHTHU" in rest


class QtyView(FakeView):
    """Có bảng mặt hàng: Cà phê sữa (L) và Trà đào (M) bán theo ly, Thép phi 6 theo kg."""

    def __init__(self):
        super().__init__()
        import pandas as pd
        self.items = pd.DataFrame({"item": ["ca phe sua l", "tra dao m", "thep phi 6"],
                                   "name": ["Cà phê sữa (L)", "Trà đào (M)", "Thép phi 6"]})

    def units(self):
        return {"ly", "kg"}

    def qty_sold(self, p, as_of, unit, item=None, size=None):
        self.calls.append(("qty", p, unit, item, size))
        return 42


@pytest.mark.parametrize("q,unit,item,size", [
    ("Trong tháng 1 năm 2026 cửa hàng bán đc mấy ly Cà phê sữa (L)?", "ly", "ca phe sua l", None),
    ("coi gium trong thang 5 nam 2025 cua hang ban duoc may kg thep phi 6", "kg", "thep phi 6", None),
    ("Hôm qua bán được bao nhiêu ly?", "ly", None, None),
    ("tháng 8/2025 quán bán đc bn ly size m", "ly", None, "m"),
    ("check gium thang 11/2025 quan ban ra total may ly size m, cong tat ca cac mon", "ly", None, "m"),
])
def test_quantity_sold(q, unit, item, size):
    res, calls = run(q, QtyView())
    if unit is None:  # 'total' là từ lạ: không trả lời
        assert res is None
        return
    assert res == {"type": "number", "answer": 42}
    assert calls[0][2:] == (unit, item, size)


@pytest.mark.parametrize("q", [
    "Trong 14 ngày gần nhất có bao nhiêu mặt hàng không bán được cái nào?",
    "Tháng 5/2025 Cà phê sữa (L) bán được bao nhiêu tiền?",
    "Tháng 5/2025 món nào bán được nhiều ly nhất?",
    "Tháng 5/2025 trung bình mỗi ngày bán được bao nhiêu ly?",
    "Tháng 5/2025 ca chiều bán được bao nhiêu ly?",
])
def test_quantity_lookalikes_rejected(q):
    res, calls = run(q, QtyView())
    assert not any(c[0] == "qty" for c in calls)


class TopView(QtyView):
    def top_items(self, p, as_of, k):
        self.calls.append(("top", p, k))
        return ["Bia Tiger lon", "Bia Sài Gòn", "Coca"][:k]


@pytest.mark.parametrize("q,res,k", [
    ("Hàng nào bán chạy nhất tháng 1/2025?", {"type": "string", "answer": "Bia Tiger lon"}, 1),
    ("xem giúp tháng 9/2025 hàng nào chạy nhất", {"type": "string", "answer": "Bia Tiger lon"}, 1),
    ("Tháng 3/2026 top 2 hàng bán chạy nhất là gì?", {"type": "list", "answer": ["Bia Tiger lon", "Bia Sài Gòn"]}, 2),
    ("cho toi 3 mon hot nhat trong thang 3/2026", {"type": "list", "answer": ["Bia Tiger lon", "Bia Sài Gòn", "Coca"]}, 3),
])
def test_best_sellers(q, res, k):
    out, calls = run(q, TopView())
    assert out == res and calls[0][2] == k


@pytest.mark.parametrize("q", [
    "tháng 7/2025 hàng nào lời nhất hả",
    "Tháng 1/2026 nhóm nào bán chạy nhất?",
    "Tháng 1/2026 mặt hàng nào bán được nhiều tiền nhất?",
    "Tháng 1/2026 hàng nào bán chậm nhất?",
])
def test_best_seller_lookalikes_rejected(q):
    _, calls = run(q, TopView())
    assert not any(c[0] == "top" for c in calls)


class GroupView(FakeView):
    groups = True

    def group_names(self):
        return {"thuc pham": "Thực phẩm", "sat thep": "Sắt thép", "thuc pham chuc nang": "Thực phẩm chức năng"}

    def group_revenue(self, p, as_of, group):
        self.calls.append(("group", p, group))
        return 25


@pytest.mark.parametrize("q,res,group", [
    ("nhom thuc pham thang 7/2025 ban duoc bao nhieu tien", {"type": "number", "answer": 25}, "thuc pham"),
    ("cho hỏi tháng 8/2025 nhóm sắt thép bán đc bao nhiêu tiền", {"type": "number", "answer": 25}, "sat thep"),
    ("Tháng 10/2025 nhóm thực phẩm chức năng bán được bao nhiêu tiền?", {"type": "number", "answer": 25},
     "thuc pham chuc nang"),
    ("Xem giúp tháng 2/2026 nhóm thực phẩm chiếm bao nhiêu % doanh thu?", {"type": "percent", "answer": 31.25},
     "thuc pham"),
])
def test_group_revenue(q, res, group):
    out, calls = run(q, GroupView())
    assert out == res and calls[0][2] == group


@pytest.mark.parametrize("q", [
    "Tháng 5/2025 nhóm thực phẩm bán được bao nhiêu kg?",
    "Tháng 5/2025 doanh thu từ khách nhóm thân thiết là bao nhiêu?",
    "Tháng 5/2025 nhóm đồ uống bán được bao nhiêu tiền?",
])
def test_group_lookalikes_rejected(q):
    _, calls = run(q, GroupView())
    assert not any(c[0] == "group" for c in calls)


class AttrView(FakeView):
    attrs = True

    def attr_labels(self):
        return {"pay": {"cod", "cash", "transfer", "ewallet", "debt"}, "channel": {"store", "phone", "zalo"}}

    def attr_revenue(self, p, as_of, kind, label):
        self.calls.append(("attr", kind, label))
        return {"cod": 10, "cash": 40, "phone": 5, "store": 60, "zalo": 7}.get(label, 1)


@pytest.mark.parametrize("q,res", [
    ("thang 5/2026 dthu cac don ban tai quay duoc bao nhieu", {"type": "number", "answer": 60}),
    ("tháng này, dthu các đơn trả bằng cod và các đơn trả bằng tm mỗi loại bao nhiêu",
     {"type": "list", "answer": [10, 40]}),
    ("Cho mình hỏi doanh thu các đơn bán qua Zalo OA và doanh thu các đơn bán tại quầy tháng này lần lượt là bao nhiêu",
     {"type": "list", "answer": [7, 60]}),
    ("Doanh thu các đơn trả bằng tiền mặt tháng 5/2026 bằng bao nhiêu % tổng doanh thu tháng 5/2026?", None),
    ("tháng 5/2026, dthu các đơn bán qua điện thoại chiếm bao nhiêu % tổng doanh thu",
     {"type": "percent", "answer": 5.0}),
])
def test_channel_and_payment_revenue(q, res):
    out, _ = run(q, AttrView())
    assert out == res


def test_channel_lookalike_rejected():
    _, calls = run("Tháng 5/2026 kênh nào bán được nhiều tiền hơn, tại quầy hay qua điện thoại?", AttrView())
    assert not any(c[0] == "attr" for c in calls)


class IngredientView(FakeView):
    ingredients = True

    def ingredient_names(self):
        return {"paracetamol": "Paracetamol", "natri clorid": "Natri clorid"}

    def ingredient_revenue(self, p, as_of, key):
        self.calls.append(("ing", key))
        return 77


def test_ingredient_revenue():
    out, calls = run("Thuốc có hoạt chất Paracetamol tháng 5/2026 bán được tổng cộng bao nhiêu tiền?", IngredientView())
    assert out == {"type": "number", "answer": 77} and calls == [("ing", "paracetamol")]
    out, calls = run("Tháng 5/2026 thuốc có hoạt chất Ibuprofen bán được bao nhiêu tiền?", IngredientView())
    assert out is None and not calls


class ShiftView(FakeView):
    shifts = {"sang": (360, 839), "chieu": (840, 1319), "_base": "sang"}

    def shift_revenue(self, p, as_of, shift):
        self.calls.append(("shift", shift))
        return {"sang": 100, "chieu": 125}[shift]

    def shift_order_count(self, p, as_of, shift):
        self.calls.append(("shift_n", shift))
        return 9


def test_shift_questions():
    assert run("Tháng 4/2026 ca chiều bán được bao nhiêu tiền?", ShiftView())[0] == {"type": "number", "answer": 125}
    assert run("tuan roi ca chieu co bnhiu don hang", ShiftView())[0] == {"type": "number", "answer": 9}
    out, _ = run("Trong tháng 5/2025, doanh thu ca chiều hơn hay kém ca sáng bao nhiêu phần trăm vậy?", ShiftView())
    assert out == {"type": "percent", "answer": 25.0}


def test_parse_shifts_from_rulebook():
    from datamate.semantic import parse_shifts
    rb = "| Ca sáng, ca chiều | theo giờ tạo đơn: ca sáng 06:00–13:59, ca chiều 14:00–21:59. mốc là **ca sáng** |"
    assert parse_shifts(rb) == {"sang": (360, 839), "chieu": (840, 1319), "_base": "sang"}


class PriceView(FakeView):
    """Red Bull 250ml bán theo lon: 49 lon trong 7 ngày kể từ ngày tăng giá, 92 lon 7 ngày trước đó."""

    def __init__(self):
        super().__init__()
        import pandas as pd
        self.items = pd.DataFrame({"item": ["red bull 250ml", "tra dao m"], "name": ["Red Bull 250ml", "Trà đào (M)"],
                                   "bunit": ["lon", "ly"]})

    def qty_sold(self, p, as_of, unit, item=None, size=None):
        self.calls.append(("qty", p, unit, item))
        return 49 if p[0] == D(2026, 1, 5) else 92


@pytest.mark.parametrize("q", [
    "So số lon Red Bull 250ml bán ra trong 7 ngày kể từ ngày 05/01/2026, ngày tăng giá lên 13.000đ, "
    "với 7 ngày ngay trước ngày đó: tăng hay giảm bao nhiêu %?",
    "red bull 250ml đổi giá từ 12.000đ lên 13.000đ vào ngày 05/01/2026. 7 ngày kể từ ngày đó bán được nhiều hay "
    "ít hơn 7 ngày liền trước bao nhiêu %, tính theo lon",
])
def test_price_effect(q):
    res, calls = run(q, PriceView())
    assert res == {"type": "percent", "answer": -46.74}
    assert [c[1] for c in calls] == [(D(2026, 1, 5), D(2026, 1, 11)), (D(2025, 12, 29), D(2026, 1, 4))]
    assert all(c[2:] == ("lon", "red bull 250ml") for c in calls)


@pytest.mark.parametrize("q", [
    "Red Bull 250ml tăng giá lên 13.000đ: so 7 ngày kể từ ngày tăng với 7 ngày ngay trước thì tăng bao nhiêu %?",
    "So doanh thu Red Bull 250ml 7 ngày kể từ ngày 05/01/2026, ngày tăng giá, với 7 ngày trước: tăng bao nhiêu %?",
    "So số lon Red Bull 250ml bán trong 7 ngày kể từ ngày 05/01/2026, ngày tăng giá, với 14 ngày trước: bao nhiêu %?",
    "So số lon Red Bull 250ml bán ra trong 7 ngày kể từ ngày 12/06/2026, ngày tăng giá, với 7 ngày trước: bao nhiêu %?",
])
def test_price_effect_lookalikes_rejected(q):
    """Không ghi ngày, hỏi tiền, hai độ dài kỳ khác nhau, kỳ sau vượt ngày hỏi: để agent làm."""
    res, _ = run(q, PriceView())
    assert res is None


def test_item_keys_merge_spacing_variants():
    import pandas as pd
    from datamate.semantic import M_ITEM_NAME, M_QTY, M_UNIT, _items
    df = pd.DataFrame({"ma": ["1", "2", "3"], "ten": ["Red Bull 250ml", "Red Bull 250 ml", "RED BULL 250ML"],
                       "dvt": ["lon"] * 3, "sl": [1, 2, 3]})
    present = {"ten": M_ITEM_NAME, "dvt": M_UNIT, "sl": M_QTY}
    it = _items(df, "ma", present, {"coca": ("lon", {})})
    assert set(it["item"]) == {"red bull 250ml"} and it["bqty"].sum() == 6


class CustView(FakeView):
    customers = True

    def customers_with_orders(self, p, as_of, n):
        self.calls.append(("cust", p, n))
        return 30, 120

    def order_count_min(self, p, as_of, x):
        self.calls.append(("big", p, x))
        return 288


@pytest.mark.parametrize("q,res,n", [
    ("Tháng 4/2026 có bao nhiêu khách mua từ 3 đơn trở lên nhỉ?", {"type": "number", "answer": 30}, 3),
    ("Trong tháng 12 năm 2025, số khách mua ít nhất 4 đơn là bao nhiêu?", {"type": "number", "answer": 30}, 4),
    ("Giúp mình đếm số khách đã mua ≥ 3 đơn trong tháng 3/2026.", {"type": "number", "answer": 30}, 3),
    ("tháng 1/2026 khách mua từ 3 đơn trở lên chiếm bao nhiu % số khách đã mua", {"type": "percent", "answer": 25.0}, 3),
    ("trong so khach co mua thang 2 nam 2026, ti le khach mua it nhat 6 don la bao nhieu %",
     {"type": "percent", "answer": 25.0}, 6),
])
def test_customers_with_n_orders(q, res, n):
    out, calls = run(q, CustView())
    assert out == res and calls[0][2] == n


@pytest.mark.parametrize("q", [
    "Tháng 4/2026 khách nào mua từ 3 đơn trở lên nhiều tiền nhất?",
    "Tháng 4/2026 có bao nhiêu khách mua từ 3 đơn trở lên mà chưa trả nợ?",
])
def test_customers_lookalikes_rejected(q):
    out, calls = run(q, CustView())
    assert out is None and not calls


@pytest.mark.parametrize("q", [
    "Tháng 6/2025 có bao nhiêu đơn hàng có dthu từ 270.000 đồng trở lên?",
    "Đếm số đơn hàng tháng 11/2025 có tiền hàng từ 270.000 đồng trở lên.",
])
def test_orders_from_amount(q):
    out, calls = run(q, CustView())
    assert out == {"type": "number", "answer": 288} and calls[0][2] == 270000


def test_order_customers_by_code_then_name():
    import pandas as pd
    from datamate.semantic import _order_customers
    df = pd.DataFrame({"ma": ["1", "2", "3", "4"], "Mã KH": ["K1", "K1", None, "K2"]})
    c = _order_customers([(df, "ma")])
    assert dict(zip(c["order_id"], c["customer"])) == {"1": "K1", "2": "K1", "4": "K2"}
    df = pd.DataFrame({"ma": ["1", "2", "3", "4"], "Khách hàng": ["Chị Lan", "chị  lan", "Khách lẻ", "Chị Lân"]})
    c = _order_customers([(df, "ma")])
    assert c["customer"].nunique() == 2 and "3" not in set(c["order_id"])
