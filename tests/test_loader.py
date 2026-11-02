import datetime as dt
import sqlite3

import openpyxl
import pandas as pd

from datamate.loader import detect_header, load_shop


def _xlsx(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for r in rows:
            ws.append(list(r))
    wb.save(path)


def test_detect_header_skips_title_and_group_rows():
    rows = [
        ("SỔ CHI TIẾT BÁN HÀNG",),
        ("Kỳ báo cáo: toàn bộ", "Kênh bán: Tất cả"),
        (),
        ("Ngày", "Mã đơn", "Tên hàng", "Số lượng", "Thành tiền"),
        ("01/04/2025", "ORD1", "Trà", "1 ly", "45,000"),
    ]
    assert detect_header(rows) == 3
    two_level = [
        ("Hàng", None, None, "Tiền", None),
        ("Hàng", "so_luong", "Unit", "Trị giá", "Giờ"),
        ("Trà", 1, "ly", "55.000đ", dt.datetime(2025, 2, 1, 6)),
    ]
    assert detect_header(two_level) == 1


def test_monthly_sheets_concat_with_types(tmp_path):
    biz = tmp_path / "biz_x"
    biz.mkdir()
    header = ("Ngày", "Mã đơn", "Tên hàng", "Số lượng", "Thành tiền", "SĐT")
    _xlsx(biz / "So_ban_hang_1.xlsx", {
        "T01_2025": [("SỔ BÁN HÀNG",), (), header,
                     ("01/01/2025", "ORD1", "Trà", "2 ly", "90,000", "0912345678"),
                     ("1 thg 1 2025", "ORD2", "Cà phê", 1, "25.000đ", None),
                     (None, "Tổng ngày 01/01", None, None, "115,000", None)],
        "T02_2025": [header, (45689, "ORD3", "Trà", 1, "45k", "0987654321")],
    })
    _xlsx(biz / "So_ban_hang_2026.xlsx", {
        "T01_2026": [header, ("05/01/2026", "ORD4", "Trà", 3, "135,000", None)],
    })
    shop = load_shop(biz)
    assert list(shop.tables) == ["So_ban_hang"]
    df = shop.tables["So_ban_hang"]
    assert len(df) == 4  # dòng "Tổng ngày" bị bỏ
    assert shop.meta["So_ban_hang"]["subtotal_rows_dropped"] == 1
    assert df["Thành tiền"].tolist() == [90000, 25000, 45000, 135000]
    assert df["Số lượng"].tolist() == [2, 1, 1, 3]
    assert pd.api.types.is_datetime64_any_dtype(df["Ngày"])
    assert df["Ngày"].dt.date.tolist() == [dt.date(2025, 1, 1), dt.date(2025, 1, 1),
                                           dt.date(2025, 2, 1), dt.date(2026, 1, 5)]
    assert df["SĐT"].iloc[0] == "0912345678"  # giữ số 0 đầu
    assert set(df["_sheet"]) == {"T01_2025", "T02_2025", "T01_2026"}


def test_subtotal_pattern_spares_person_names():
    from datamate.loader import _SUBTOTAL
    from datamate.parsing import fold
    for s in ["Tổng ngày 01/04", "Cộng ngày 01/05/2025", "Cộng", "Tổng cộng", "Tổng tháng 3"]:
        assert _SUBTOTAL.match(fold(s)), s
    for s in ["Tống Thắng Thu", "Tống Ngày Lan", "Công Phượng", "Tổng kho"]:
        assert not _SUBTOTAL.match(fold(s)), s


def test_different_sheets_become_separate_tables(tmp_path):
    biz = tmp_path / "biz_y"
    biz.mkdir()
    _xlsx(biz / "SoBanHang.xlsx", {
        "Ghi tắt": [("Ghi tắt", "Tên hàng"), ("Cáp C", "Cáp Type-C 1m")],
        "T7-2025": [("Ngày", "STT", "Hàng", "Giá"), ("01/07/2025", 1, "Cáp C", "70k")],
        "T8-2025": [("Ngày", "STT", "Hàng", "Giá"), ("01/08/2025", 2, "Cáp C", "70.000đ")],
    })
    shop = load_shop(biz)
    assert set(shop.tables) == {"SoBanHang.Ghi tắt", "SoBanHang"}
    assert shop.tables["SoBanHang"]["Giá"].tolist() == [70000, 70000]


def test_csv_and_sqlite(tmp_path):
    biz = tmp_path / "biz_z"
    biz.mkdir()
    (biz / "ban_hang.csv").write_text(
        "﻿Mã hóa đơn,Ngày tháng,Thanh tien,IMEI\n"
        "ORD1,1 thg 6 2025,\"560,000\",352628764420750\n"
        "ORD2,02/06/2025,\"10,000\",352628764420751\n",
        encoding="utf-8",
    )
    con = sqlite3.connect(biz / "pos.sqlite")
    con.execute("create table don_hang (MaHoaDon text, NgayLap text, TongThanhToan integer)")
    con.execute("insert into don_hang values ('HD1', '2025-06-01 08:00:00', 100000)")
    con.commit()
    con.close()
    shop = load_shop(biz)
    bh = shop.tables["ban_hang"]
    assert bh["Thanh tien"].tolist() == [560000, 10000]
    assert bh["IMEI"].tolist() == ["352628764420750", "352628764420751"]
    assert pd.api.types.is_datetime64_any_dtype(bh["Ngày tháng"])
    dh = shop.tables["don_hang"]
    assert dh["TongThanhToan"].tolist() == [100000]
    assert pd.api.types.is_datetime64_any_dtype(dh["NgayLap"])


def test_continuation_rows_are_kept_and_counted(tmp_path):
    biz = tmp_path / "biz_c"
    biz.mkdir()
    _xlsx(biz / "DonHang.xlsx", {"Sheet1": [
        ("Mã đơn", "Ngày", "Khách", "Tên sản phẩm", "Tổng tiền"),
        ("ORD1", "01/04/2025", "Lan", "Dầu ăn", "156000"),
        (None, None, None, "Chocopie", "165,000"),
        ("Ghi chú: số liệu ghi tay",),
    ]})
    shop = load_shop(biz)
    assert len(shop.tables["DonHang"]) == 2  # dòng ghi chú một ô bị bỏ
    assert shop.meta["DonHang"]["continuation_rows"] == 1
    assert shop.meta["DonHang"]["note_rows_dropped"] == 1


def test_mostly_empty_first_column_is_not_continuation(tmp_path):
    biz = tmp_path / "biz_e"
    biz.mkdir()
    _xlsx(biz / "KhachHang.xlsx", {"Sheet1": [
        ("Email", "Mã KH", "Tên"), (None, "KH1", "Lan"), (None, "KH2", "Tú"), (None, "KH4", "Mai"),
        (None, "KH5", "Hoa"), ("a@b.c", "KH3", "Hà")]})
    assert load_shop(biz).meta["KhachHang"]["continuation_rows"] == 0


def test_money_columns_never_in_ffill_list(tmp_path):
    biz = tmp_path / "biz_n"
    biz.mkdir()
    _xlsx(biz / "SoNo.xlsx", {"Sổ nợ": [
        ("Ngày", "Khách", "Nợ", "Đã trả", "Ngày trả", "Ghi chú"),
        ("02/07/2025", "Lan", "240.000đ", "80k", "17/07/2025", "STT 34"),
        (None, None, None, "80k", "20/07/2025", "STT 34"),
        ("03/07/2025", "Tú", "100k", "100k", "05/07/2025", "STT 40"),
    ]})
    m = load_shop(biz).meta["SoNo"]
    assert m["continuation_rows"] == 1
    assert m["continuation_cols"] == ["Ngày", "Khách"]


def test_empty_customer_code_is_not_continuation(tmp_path):
    biz = tmp_path / "biz_k"
    biz.mkdir()
    (biz / "don_hang.csv").write_text(
        "Số KH,Mã đơn,Ngày,Tổng\nKH1,O1,01/01/2025,100\n,O2,01/01/2025,200\nKH2,O3,02/01/2025,300\n",
        encoding="utf-8")
    assert load_shop(biz).meta["don_hang"]["continuation_rows"] == 0


def test_excel_serial_date_column_becomes_date(tmp_path):
    biz = tmp_path / "biz_s"
    biz.mkdir()
    _xlsx(biz / "SoBanHang.xlsx", {"Sheet1": [
        ("ngay_ban", "Số ngày được nợ", "Tổng tiền"),
        (45717, 30, 1000),
        (45718, 60, 2000),
    ]})
    df = load_shop(biz).tables["SoBanHang"]
    assert list(df["ngay_ban"].dt.strftime("%Y-%m-%d")) == ["2025-03-01", "2025-03-02"]
    assert list(df["Số ngày được nợ"]) == [30, 60]  # số ngày, không phải ngày
def test_numeric_column_with_missing_values_preserves_large_integer():
    import pandas as pd
    from datamate.loader import _type_column
    result, kind, _ = _type_column(pd.Series(['9007199254740993', '8', None], name='Giá vốn'), None)
    assert kind == 'int'
    assert result.iloc[0] == 9007199254740993
    assert pd.isna(result.iloc[2])


def test_map_unique_reuses_parser_result_for_repeated_values():
    import pandas as pd
    from datamate.loader import _map_unique
    calls = []
    def parse(value):
        calls.append(value)
        return int(value)
    result = _map_unique(pd.Series(['2', '2', '3', None]), parse, preserve_object=True)
    assert result.tolist() == [2, 2, 3, None]
    assert sorted(calls) == ['2', '3']


def test_clean_text_normalises_any_input():
    from datamate.loader import clean_text
    assert clean_text("  Ho\u0300a\u00a0 \u200bxinh\t") == "Hòa xinh"
    assert clean_text("\ufeff \u00ad ") is None
    assert clean_text(12.5) == 12.5
    assert clean_text("Trà Sữa") == "Trà Sữa"


def test_quality_report_measures_without_changing():
    from datamate.loader import quality_report
    df = pd.DataFrame({"ten": ["A", "a", None, "A"], "so": [1, 2, 3, 1]})
    df.loc[3, "ten"] = "A"
    rep = quality_report(df, {"ten": "text", "so": "number"}, {"so": 1})
    assert rep["columns"]["ten"]["completeness"] == 0.75
    assert rep["columns"]["so"]["validity"] < 1
    assert rep["exact_duplicate_rows"] == 1
    assert len(df) == 4


def test_quality_report_format_conformity():
    from datamate.loader import quality_report
    df = pd.DataFrame({"ma": [f"ORD{i:06d}" for i in range(19)] + ["ord-x"], "ghi": [f"ghi chú {i} lần" if i % 2 else "ok" for i in range(20)]})
    rep = quality_report(df, {"ma": "text", "ghi": "text"}, {})
    assert rep["columns"]["ma"]["conformity"] == 0.95 and rep["columns"]["ghi"]["conformity"] is None


def test_address_with_house_number_stays_text():
    from datamate.loader import _type_column
    s = pd.Series([f"{i} Đường số {i % 50}, Phường {i % 9}" for i in range(1, 200)], name="Địa chỉ")
    out, kind, bad = _type_column(s, None)
    assert kind == "text" and out.iloc[0] == "1 Đường số 1, Phường 1"
    q = pd.Series(["2 ly", "1 ly", "3 hộp", "5 kg"] * 20, name="SL")
    assert _type_column(q, None)[1] == "int(qty)"


def test_repeated_long_sentinel_is_number_but_phones_stay_text():
    from datamate.loader import _type_column
    assert _type_column(pd.Series(["999999999"] * 50, name="Tồn lớn nhất"), None)[1] == "int"
    phones = pd.Series([f"91234{i:04d}" for i in range(50)], name="Liên hệ")
    assert _type_column(phones, None)[1] == "text"


def test_sqlite_null_text_stays_none(tmp_path):
    biz = tmp_path / "biz_sql"
    biz.mkdir()
    con = sqlite3.connect(biz / "pos.sqlite")
    con.execute("create table don (ma text, khach text)")
    con.executemany("insert into don values (?, ?)", [(f"D{i}", None if i % 2 else f"KH{i}") for i in range(20)])
    con.commit()
    con.close()
    df = load_shop(biz).tables["don"]
    assert "nan" not in set(df["khach"].dropna()) and df["khach"].isna().sum() == 10


def test_csv_encodings_and_duplicate_files(tmp_path):
    import unicodedata
    from datamate.loader import load_shop
    text = "Mã đơn,Tên hàng,Thành tiền\r\nHD1,Cà phê sữa đá,50000\r\nHD2,Trà đào,35000\r\n"
    # cp1258 (Windows tiếng Việt): chữ có sẵn trong bảng mã, dấu thanh tổ hợp
    def cp1258(t):
        out = b""
        for ch in unicodedata.normalize("NFC", t):
            try:
                out += ch.encode("cp1258")
            except UnicodeEncodeError:
                base, *marks = unicodedata.normalize("NFD", ch)
                keep = "".join(m for m in marks if m in "\u031b\u0306\u0302")
                out += unicodedata.normalize("NFC", base + keep).encode("cp1258")
                out += "".join(m for m in marks if m not in keep).encode("cp1258")
        return out
    for name, data in {"u16": text.encode("utf-16"), "cp": cp1258(text)}.items():
        d = tmp_path / name
        d.mkdir()
        (d / "ban.csv").write_bytes(data)
        t = load_shop(d).tables["ban"]
        assert list(t.columns) == ["Mã đơn", "Tên hàng", "Thành tiền"] and t["Tên hàng"].iloc[0] == "Cà phê sữa đá"
    d = tmp_path / "dup"
    d.mkdir()
    (d / "ban.csv").write_text(text, encoding="utf-8")
    (d / "ban (1).csv").write_text(text, encoding="utf-8")
    shop = load_shop(d)
    assert list(shop.tables) == ["ban"] and shop.duplicate_files == [("ban (1).csv", "ban.csv")]  # (bỏ, giữ)
