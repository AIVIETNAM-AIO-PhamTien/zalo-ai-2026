import csv

from datamate.loader import load_shop


def _csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def test_unparsed_cells_keep_text(tmp_path):
    rows = [("Ngày", "Mã đơn", "Tên hàng", "Thành tiền")]
    rows += [(f"0{1 + i % 9}/04/2025", f"D{i}", "Trà", "45,000") for i in range(40)]
    rows.append(("02/04/2025", "D99", "Trà", "tặng"))
    _csv(tmp_path / "ban.csv", rows)
    shop = load_shop(tmp_path)
    m = shop.meta["ban"]
    assert m["types"]["Thành tiền"] == "int"
    assert list(m["unparsed_cells"]["Thành tiền"].values()) == ["tặng"]
    i = next(iter(m["unparsed_cells"]["Thành tiền"]))
    assert shop.tables["ban"]["Mã đơn"][i] == "D99"


def test_subtotal_rows_kept_and_checked(tmp_path):
    rows = [("Ngày", "Mã đơn", "Tên hàng", "Thành tiền"),
            ("01/04/2025", "D1", "Trà", "10,000"), ("01/04/2025", "D2", "Cà phê", "20,000"),
            ("Tổng ngày 01/04", None, None, "30,000"),
            ("02/04/2025", "D3", "Trà", "10,000"),
            ("Tổng ngày 02/04", None, None, "10,000"),
            ("Tổng cộng", None, None, "40,000")]
    _csv(tmp_path / "ban.csv", rows)
    m = load_shop(tmp_path).meta["ban"]
    assert len(m["subtotal_rows"]) == 3
    assert m["subtotal_check"]["tong ngay"]["Thành tiền"] == {"sub": 40000.0, "detail": 40000.0, "ok": True}


def test_subtotal_mismatch_flagged(tmp_path):
    rows = [("Ngày", "Mã đơn", "Tên hàng", "Thành tiền"),
            ("01/04/2025", "D1", "Trà", "10,000"), ("01/04/2025", "D2", "Cà phê", "20,000"),
            ("Tổng cộng", None, None, "50,000")]
    _csv(tmp_path / "ban.csv", rows)
    assert load_shop(tmp_path).meta["ban"]["subtotal_check"]["tong cong"]["Thành tiền"]["ok"] is False
