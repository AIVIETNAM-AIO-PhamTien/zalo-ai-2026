import datetime as dt

import pandas as pd

from datamate.baseline import import_price_answer, norm
from datamate.helpers import Periods
from datamate.imports import ImportView
from datamate.loader import Shop

D = dt.date
README = ("# Dữ liệu\n## Quy ước\n- **Nhập hàng**: `chi_tiet_nhap.DonGiaNhap` là giá nhập một đơn vị, theo đơn vị "
          "nhập của dòng; `phieu_nhap.TongGiaTri` là tổng tiền của cả phiếu nhập.\n")


def shop():
    lines = pd.DataFrame({"MaPhieuNhap": ["P1", "P1", "P2", "P3", "P4"],
                          "MaHangHoa": ["SP1", "SP2", "SP1", "SP1", "SP1"],
                          "DonViTinh": ["thùng", "hộp", "thùng", "thùng", "thùng"],
                          "SL": [10, 5, 30, 100, 7], "DonGiaNhap": [200_000, 50_000, 210_000, 1, 300_000]})
    heads = pd.DataFrame({"MaPhieuNhap": ["P1", "P2", "P3", "P4"],
                          "NgayNhapKho": pd.to_datetime(["2026-01-03", "2026-01-20", "2026-01-21", "2026-02-02"]),
                          "TinhTrang": ["Đã nhập", "Đã nhập", "Đã huỷ", "Đã nhập"]})
    products = pd.DataFrame({"MaHangHoa": ["SP1", "SP2"], "TenGoi": ["Bia Tiger lon", "Bánh quy"]})
    return Shop("b", README, {"chi_tiet_nhap": lines, "phieu_nhap": heads, "san_pham": products}, {})


def test_weighted_by_quantity_and_skips_cancelled():
    iv = ImportView.from_shop(shop())
    # (10·200 000 + 30·210 000) / 40; phiếu huỷ P3 và phiếu tháng 2 không tính
    assert iv.avg_price("bia tiger lon", "thung", (D(2026, 1, 1), D(2026, 1, 31))) == 207500
    assert iv.avg_price("bia tiger lon", "hop", (D(2026, 1, 1), D(2026, 1, 31))) is None


def test_question_parsing():
    iv = ImportView.from_shop(shop())
    as_of = D(2026, 3, 1)
    for q in ["Giá nhập bình quân một thùng Bia Tiger lon trong tháng 1 năm 2026 là bao nhiêu?",
              "tháng 1/2026 cửa hàng nhập bia tiger lon bình quân bao nhiêu tiền một thùng"]:
        assert import_price_answer(norm(q), as_of, iv, Periods(as_of)) == {"type": "number", "answer": 207500}
    for q in ["Giá nhập bình quân một thùng Bia Tiger lon tháng 1 và tháng 2 năm 2026?",
              "Giá nhập bình quân một két Bia Tiger lon tháng 1/2026 là bao nhiêu?",
              "Giá nhập bình quân một thùng hàng tháng 1/2026 là bao nhiêu?"]:
        assert import_price_answer(norm(q), as_of, iv, Periods(as_of)) is None


def test_lots_expiry():
    from datamate.baseline import lot_answer
    from datamate.lots import LotView
    lots = pd.DataFrame({"MaLo": ["L1", "L2", "L3", "L4"], "MaHangHoa": ["SP1", "SP1", "SP2", "SP2"],
                         "HanSuDung": pd.to_datetime(["2026-07-10", "2027-01-01", "2026-07-20", "2026-08-30"]),
                         "SLNhapQuyDoi": [10, 10, 5, 5], "MaPhieuNhap": ["P1", "P1", "P2", "P2"]})
    heads = pd.DataFrame({"MaPhieuNhap": ["P1", "P2"], "NgayNhapKho": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                          "TinhTrang": ["received", "received"]})
    orders = pd.DataFrame({"MaHoaDon": ["H1", "H2"], "NgayLap": pd.to_datetime(["2026-03-01", "2026-03-02"]),
                           "TinhTrang": ["completed", "cancelled"]})
    moves = pd.DataFrame({"MaHoaDon": ["H1", "H1", "H2"], "MaLo": ["L1", "L3", "L4"],
                          "SLQuyDoi": [10, 5, 5], "SLTraQuyDoi": [2, 0, 0]})
    products = pd.DataFrame({"MaHangHoa": ["SP1", "SP2"], "TenGoi": ["Oresol", "Siro ho"]})
    shop = Shop("b", "", {"lo_hang": lots, "phieu_nhap": heads, "don_hang": orders, "chi_tiet_don_lo": moves,
                          "san_pham": products}, {})
    lv = LotView.from_shop(shop)
    as_of = D(2026, 7, 1)
    # L1 còn 2 (khách trả về lô), L3 hết, L4 còn 5 (đơn huỷ không lấy hàng)
    assert lot_answer("Lô Oresol gần hết hạn nhất có hạn dùng ngày nào?", as_of, lv) == \
        {"type": "date", "answer": "2026-07-10"}
    assert lot_answer("Có mấy lô hàng còn tồn sẽ hết hạn từ nay đến hết ngày 31/08/2026?", as_of, lv) == \
        {"type": "number", "answer": 2}
    assert lot_answer("Hàng nào hết hạn trong 30 ngày tới mà hiện vẫn còn tồn kho?", as_of, lv) == \
        {"type": "list", "answer": ["Oresol"]}
    assert lot_answer("Tiền vốn hàng hết hạn trong 30 ngày tới là bao nhiêu?", as_of, lv) is None
