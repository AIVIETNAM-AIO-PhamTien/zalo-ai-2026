import datetime as dt

import pandas as pd

import pytest

from datamate.loader import Shop, load_shop
from datamate.semantic import OrderView, classify_status, parse_readme_columns

README_FLAT = """# Dữ liệu biz_s

## Cột trong `ban_hang.csv`
- **Kết quả**: trạng thái đơn
- **Nội dung**: tên hàng
- **Ngày tháng**: ngày bán
- **Thanh tien**: thành tiền sau giảm
- **Mã hóa đơn**: mã đơn

## Quy ước
- Dữ liệu có đến hết 31/03/2025.
"""

CSV_FLAT = """Mã hóa đơn,Ngày tháng,Kết quả,Nội dung,Thanh tien
O1,01/03/2025,Xong,Trà,"10,000"
O1,01/03/2025,Xong,Cà phê,20000
O1,01/03/2025,Xong,Cà phê,20000
O2,01/03/2025,Đã hủy,Trà,10000
O3,02/03/2025,Đổi trả,Trà,15000
O4,02/03/2025,Đang giao,Bánh,5000
O5,15/03/2025,Xong,Trà,7000
O6,20/03/2025,Xong,Trà,9000
"""


def test_classify_status():
    assert classify_status("Đã hủy đơn") == "cancelled"
    assert classify_status("khách hủy") == "cancelled"
    assert classify_status("cancelled") == "cancelled"
    assert classify_status("Không giao được") == "cancelled"
    assert classify_status("Hoàn trả") == "returned"
    assert classify_status("Đổi trả") == "returned"
    assert classify_status("hàng trả về") == "returned"
    assert classify_status("Đã hoàn hàng") == "returned"
    assert classify_status("Đang giao hàng") == "delivering"
    assert classify_status("Đang vận chuyển") == "delivering"  # 'chuyen' chứa 'huy'
    assert classify_status("shipping") == "delivering"
    assert classify_status("Xong") == "completed"
    assert classify_status("ok") == "completed"
    assert classify_status(None) is None


def test_parse_readme_columns():
    cols = parse_readme_columns(README_FLAT)
    assert cols["ban_hang.csv"]["Thanh tien"] == "thành tiền sau giảm"
    assert cols["ban_hang.csv"]["Kết quả"] == "trạng thái đơn"


@pytest.fixture
def flat(tmp_path):
    biz = tmp_path / "biz_s"
    biz.mkdir()
    (biz / "README.md").write_text(README_FLAT, encoding="utf-8")
    (biz / "ban_hang.csv").write_text(CSV_FLAT, encoding="utf-8")
    return OrderView.from_shop(load_shop(biz))


def test_flat_metrics(flat):
    assert flat is not None
    p = (dt.date(2025, 3, 1), dt.date(2025, 3, 2))
    asof = dt.date(2025, 3, 31)
    # O1 dòng trùng tính một lần: 10000 + 20000; O4 đang giao vẫn tính; O2 huỷ, O3 trả bị loại
    assert flat.revenue(p, asof) == 35000
    assert flat.order_count(p, asof) == 2
    assert flat.cancelled_count(p, asof) == 1
    assert flat.cancel_rate(p, asof) == 25.0
    assert flat.date_range() == (dt.date(2025, 3, 1), dt.date(2025, 3, 20))


def test_after_today_excluded_and_outside_range(flat):
    march = (dt.date(2025, 3, 1), dt.date(2025, 3, 31))
    assert flat.revenue(march, dt.date(2025, 3, 15)) == 42000
    assert flat.covers((dt.date(2025, 2, 1), dt.date(2025, 2, 28))) is False
    assert flat.covers(march) is True


README_SPLIT = """# Dữ liệu biz_j

## Cột trong `don_hang.csv`
- **Số CT**: mã đơn
- **Date**: ngày bán
- **Xử lý**: trạng thái đơn
- **Mã thẻ**: mã khách (trống nếu khách lẻ)

## Cột trong `chi_tiet_don.csv`
- **Mã bill**: mã đơn
- **Cộng**: thành tiền sau giảm
"""


def test_split_order_and_lines(tmp_path):
    biz = tmp_path / "biz_j"
    biz.mkdir()
    (biz / "README.md").write_text(README_SPLIT, encoding="utf-8")
    (biz / "don_hang.csv").write_text(
        "Mã thẻ,Số CT,Date,Xử lý\nKH1,O1,01/03/2025,completed\n,O2,01/03/2025,cancelled\n,O3,02/03/2025,returned\n",
        encoding="utf-8")
    (biz / "chi_tiet_don.csv").write_text(
        "Mã bill,Cộng\nO1,100\nO1,50\nO2,70\nO3,30\n", encoding="utf-8")
    v = OrderView.from_shop(load_shop(biz))
    p = (dt.date(2025, 3, 1), dt.date(2025, 3, 2))
    assert v.revenue(p, dt.date(2025, 3, 31)) == 150
    assert v.order_count(p, dt.date(2025, 3, 31)) == 1
    assert v.cancel_rate(p, dt.date(2025, 3, 31)) == 33.33


def test_continuation_rows_ffilled(tmp_path):
    biz = tmp_path / "biz_f"
    biz.mkdir()
    (biz / "README.md").write_text(README_FLAT.replace("ban_hang.csv", "DonHang.csv"), encoding="utf-8")
    (biz / "DonHang.csv").write_text(
        "Mã hóa đơn,Ngày tháng,Kết quả,Nội dung,Thanh tien\n"
        "O1,01/03/2025,Xong,Trà,100\n,,,Bánh,50\nO2,01/03/2025,Đã hủy,Trà,70\n,,,Bánh,30\n", encoding="utf-8")
    v = OrderView.from_shop(load_shop(biz))
    p = (dt.date(2025, 3, 1), dt.date(2025, 3, 1))
    assert v.revenue(p, dt.date(2025, 3, 31)) == 150
    assert v.cancelled_count(p, dt.date(2025, 3, 31)) == 1


def test_no_mapping_returns_none(tmp_path):
    biz = tmp_path / "biz_n"
    biz.mkdir()
    (biz / "README.md").write_text("# trống", encoding="utf-8")
    (biz / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert OrderView.from_shop(load_shop(biz)) is None



def test_order_total_repeated_on_every_line_disables_view(tmp_path):
    header = "Mã hóa đơn,Ngày tháng,Kết quả,Nội dung,Thanh tien"

    def view(name, lines):
        biz = tmp_path / name
        biz.mkdir()
        (biz / "README.md").write_text(README_FLAT, encoding="utf-8")
        (biz / "ban_hang.csv").write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
        return OrderView.from_shop(load_shop(biz))
    # 'Thanh tien' là tổng cả đơn lặp trên từng dòng → không dùng bộ giải tất định
    assert view("biz_f1", ["O1,01/03/2025,Xong,Trà,300", "O1,01/03/2025,Xong,Bánh,300",
                           "O2,02/03/2025,Xong,Trà,500", "O2,02/03/2025,Xong,Bánh,500"]) is None
    assert view("biz_f2", ["O1,01/03/2025,Xong,Trà,100", "O1,01/03/2025,Xong,Bánh,200",
                           "O2,02/03/2025,Xong,Trà,250", "O2,02/03/2025,Xong,Bánh,250"]) is not None


def test_readme_file_name_template_matches_monthly_files():
    from datamate.semantic import _file_pattern
    pat = _file_pattern("BanHang_T<tháng>_<năm>.xlsx")
    assert pat.match("BanHang_T05_2025.xlsx") and pat.match("BanHang_T12_2026.xlsx")
    assert not pat.match("BanHang_2025.xlsx") and not pat.match("DanhSachQuyDoi.xlsx")
    assert _file_pattern("ban_hang.csv").match("ban_hang.csv")
    assert not _file_pattern("ban_hang.csv").match("ban_hangXcsv")  # dấu chấm không phải ký tự đại diện


def test_infer_meanings_when_readme_is_silent():
    import pandas as pd
    from datamate.semantic import infer_meanings
    df = pd.DataFrame({
        "Mã đơn hàng": ["A", "A", "B"], "Ngày": pd.to_datetime(["2025-01-01"] * 3),
        "Trạng thái": ["Xong", "Xong", "Đã hủy"], "Tên hàng hóa": ["X", "Y", "X"],
        "Số lượng": [2, 1, 3], "Đơn giá": [100, 50, 100], "Thành tiền": [180, 50, 300],
    })
    m = infer_meanings(df)
    assert m["Mã đơn hàng"] == "mã đơn" and m["Ngày"] == "ngày bán" and m["Trạng thái"] == "trạng thái đơn"
    assert m["Tên hàng hóa"] == "tên hàng" and m["Thành tiền"] == "thành tiền sau giảm"


def test_infer_meanings_rejects_pre_discount_amount():
    import pandas as pd
    from datamate.semantic import infer_meanings
    df = pd.DataFrame({"Số lượng": [2, 1], "Đơn giá": [100, 50], "Giảm giá": [20, 0], "Thành tiền": [200, 50]})
    assert "Thành tiền" not in infer_meanings(df)  # có cột giảm giá mà thành tiền chưa trừ → không phải doanh thu


def test_goods_agreement_cross_check():
    import pandas as pd
    from datamate.loader import Shop
    from datamate.semantic import goods_agreement
    ids = [f"HD{i}" for i in range(30)]
    orders = pd.DataFrame({"Mã hóa đơn": ids, "Tổng tiền hàng": [100] * 30})
    shop = Shop("b", "", {"hd": orders}, {"hd": {"sources": ["HoaDon.xlsx"]}})
    cols = {"HoaDon.xlsx": {"Mã hóa đơn": "mã hoá đơn", "Tổng tiền hàng": "tổng tiền hàng của đơn (cộng thành tiền)"}}
    ok = pd.DataFrame({"order_id": ids * 2, "amount": [40] * 30 + [60] * 30})
    assert goods_agreement(shop, cols, ok, "ct") == 1.0
    bad = ok.assign(amount=ok["amount"] * 2)  # cột tiền đọc nhầm (vd. gồm thuế / nhân số lượng hai lần)
    assert goods_agreement(shop, cols, bad, "ct") == 0.0


def test_public_rulebook_superlatives():
    import datetime as dt

    import pandas as pd

    from datamate.semantic import OrderView
    ts = pd.to_datetime(["2026-01-05 09:10", "2026-01-05 09:40", "2026-01-06 10:05", "2026-01-12 09:20",
                         "2026-01-12 15:00", "2026-01-13 09:30"])
    orders = pd.DataFrame({"order_id": list("abcdef"), "date": ts.normalize(), "status": "x",
                           "cls": ["completed"] * 5 + ["cancelled"], "ts": ts})
    lines = pd.DataFrame({"order_id": list("abcdef"), "amount": [100, 200, 300, 400, 500, 600]})
    cust = pd.DataFrame({"order_id": list("abcdef"), "customer": ["k1", "k1", "k1", "k2", "k2", "k2"],
                         "display": ["An", "An", "An", "Bình", "Bình", "Bình"]})
    v = OrderView(lines, orders, customers=cust)
    p, d = (dt.date(2026, 1, 1), dt.date(2026, 1, 31)), dt.date(2026, 1, 31)
    assert v.customer_most_orders(p, d) == "An"          # An 3 đơn hợp lệ, Bình 2 (1 huỷ)
    assert v.loyal_customer(p, d) == "An"                # An mua 2 ngày (05, 06), Bình 1 ngày (12)
    assert v.fickle_customer(p, d) == "Bình"
    assert v.big_spender(p, d) == "An"                   # chỉ An đủ 3 đơn
    assert v.busiest_day(p, d) == "2026-01-05"           # hoà → ngày tăng dần, rulebook §9
    assert v.peak_hour(p, d) == 9
    assert v.weekday_avg(p, d, 0, "orders") == 1.0       # thứ Hai: 4 đơn / 4 ngày thứ Hai (5, 12, 19, 26/1)


def _notes_shop():
    df = pd.DataFrame({"HĐ": ["1", "1", "1", "2", "3", "3"],
                       "Ngày": pd.to_datetime(["2026-01-05"] * 3 + ["2026-01-06"] + ["2026-01-07"] * 2),
                       "SL": [1, 2, None, 1, 1, 1], "TT": [100, 200, -50, 70, 30, 20],
                       "Ghi chú": [None, None, None, "khách mang trả", None, None]})
    shop = Shop("b", "", {"ban": df}, {"ban": {}})
    shop.card = {"ban": {"HĐ": "mã đơn", "Ngày": "ngày bán", "SL": "số lượng", "TT": "thành tiền sau giảm"}}
    return shop


def test_notes_column_as_status_and_nongoods_rows_dropped():
    from datamate.semantic import view_report
    v = OrderView.from_shop(_notes_shop())
    assert v is not None
    jan = (dt.date(2026, 1, 1), dt.date(2026, 1, 31))
    # đơn 2 'khách mang trả' = trả hàng; dòng voucher -50 (SL trống) không phải doanh thu hàng
    assert v.revenue(jan, dt.date(2026, 1, 31)) == 350
    assert v.order_count(jan, dt.date(2026, 1, 31)) == 2
    rep = view_report(v)
    assert "`HĐ`" in rep and "1 dòng không phải hàng" in rep
    OrderView.why = "x"
    assert "Chưa dựng được" in view_report(None)
