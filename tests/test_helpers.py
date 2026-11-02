import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from datamate.helpers import Periods, find_names, pct_change, round2, round_money, share, strip_phone

D = dt.date
CAL = Path(__file__).resolve().parents[1] / "data" / "train" / "calendar.csv"


@pytest.fixture
def p():
    return Periods(D(2026, 3, 31), CAL if CAL.exists() else None)


def test_day_and_yesterday(p):
    assert p.today() == (D(2026, 3, 31), D(2026, 3, 31))
    assert p.yesterday() == (D(2026, 3, 30), D(2026, 3, 30))


def test_weeks_start_monday():
    q = Periods(D(2026, 3, 4))  # thứ Tư
    assert q.this_week() == (D(2026, 3, 2), D(2026, 3, 4))
    assert q.last_week() == (D(2026, 2, 23), D(2026, 3, 1))


def test_this_month_and_prev_month_same_days(p):
    assert p.this_month() == (D(2026, 3, 1), D(2026, 3, 31))
    # tháng trước không có ngày 31 → đến hết tháng trước
    assert p.prev_month_same_days() == (D(2026, 2, 1), D(2026, 2, 28))
    q = Periods(D(2026, 3, 15))
    assert q.prev_month_same_days() == (D(2026, 2, 1), D(2026, 2, 15))


def test_month_without_year_rolls_back_if_after_today():
    q = Periods(D(2026, 3, 15))
    assert q.month(3) == (D(2026, 3, 1), D(2026, 3, 31))
    assert q.month(5) == (D(2025, 5, 1), D(2025, 5, 31))
    assert q.month(5, 2026) == (D(2026, 5, 1), D(2026, 5, 31))
    assert q.prev_month(1, 2026) == (D(2025, 12, 1), D(2025, 12, 31))


def test_quarter_ytd_last_n_days():
    q = Periods(D(2026, 3, 15))
    assert q.quarter(1) == (D(2026, 1, 1), D(2026, 3, 31))
    assert q.quarter(3) == (D(2025, 7, 1), D(2025, 9, 30))
    assert q.year_to_date() == (D(2026, 1, 1), D(2026, 3, 15))
    assert q.last_n_days(7) == (D(2026, 3, 9), D(2026, 3, 15))
    assert q.year(2025) == (D(2025, 1, 1), D(2025, 12, 31))


@pytest.mark.skipif(not CAL.exists(), reason="cần data/train/calendar.csv")
def test_tet_window_and_before(p):
    assert p.tet(2026) == (D(2026, 2, 10), D(2026, 2, 26))
    assert p.before_tet(2026) == (D(2026, 1, 24), D(2026, 2, 9))
    assert p.before_tet(2026, 7) == (D(2026, 2, 3), D(2026, 2, 9))
    assert p.tet(2025) == (D(2025, 1, 22), D(2025, 2, 7))


def test_mask():
    q = Periods(D(2026, 3, 15))
    s = pd.Series(pd.to_datetime(["2026-03-01 08:00", "2026-03-15 23:59", "2026-03-16 00:01", None]))
    assert q.mask(s, q.this_month()).tolist() == [True, True, False, False]


def test_rounding():
    assert round_money(1234.5) == 1235
    assert round_money(1234.49) == 1234
    assert round2(2.675) == 2.68
    assert round2(-12.455) == -12.46
    assert pct_change(110, 100) == 10.0
    assert pct_change(87.5, 100) == -12.5
    assert pct_change(5, 0) is None
    assert share(1, 3) == 33.33


def test_strip_phone():
    assert strip_phone("Đào Dương Giang - 0912 365 003") == "Đào Dương Giang"
    assert strip_phone("Đàm Bảo Bảo (0797929008)") == "Đàm Bảo Bảo"
    assert strip_phone("Trịnh Cường Quân 0930103677") == "Trịnh Cường Quân"
    assert strip_phone("Chị Lan") == "Chị Lan"


def test_find_names_accent_insensitive():
    cands = ["Bia Tiger lon", "Bia Tiger Bạc lon", "Sting dâu 330ml", "Cà phê sữa (M)"]
    assert find_names("bia tiger lon", cands)[0][0] == "Bia Tiger lon"
    assert find_names("sting dau", cands)[0][0] == "Sting dâu 330ml"
    assert find_names("ca phe sua M", cands)[0][0] == "Cà phê sữa (M)"


def test_ts_periods_compare_with_datetime_columns():
    from datamate.helpers import TsPeriods
    q = TsPeriods(Periods(D(2026, 3, 15)))
    start, end = q.month(2)
    assert start == pd.Timestamp("2026-02-01") and end.date() == D(2026, 2, 28) and end.hour == 23
    s = pd.Series(pd.to_datetime(["2026-02-28 21:00", "2026-03-01 00:00"]))
    assert ((s >= start) & (s <= end)).tolist() == [True, False]
    assert q.n_days(q.this_month()) == 15
    assert q.mask(s, q.month(2)).tolist() == [True, False]


def test_find_values_and_name_mask():
    from datamate.helpers import find_values, name_mask
    t = {"ban": pd.DataFrame({"hang": ["Bộ van dây gas Namilux", "BỘ  VAN DÂY GAS NAMILUX", "Gas Namilux 12kg"],
                              "sl": [1, 2, 3]}),
         "nv": pd.DataFrame({"ma": ["NV001", "NV002"]})}
    top = find_values(t, "bo van day gas namilux")[0]
    assert top[0] in ("Bộ van dây gas Namilux", "BỘ  VAN DÂY GAS NAMILUX") and top[1:3] == ("ban", "hang")
    assert int(t["ban"].loc[name_mask(t["ban"]["hang"], top[0]), "sl"].sum()) == 3
    assert find_values(t, "nv001")[0][:3] == ("NV001", "nv", "ma")


def test_valid_order_understands_status_spellings():
    from datamate.helpers import order_status, valid_order
    s = pd.Series(["Đã hoàn thành", "Đã hủy", "Đã hoàn hàng", "Đang giao dịch", "xong", None])
    # ô trống: đơn không ghi huỷ / trả → hợp lệ (rulebook §1)
    assert valid_order(s).tolist() == [True, False, False, True, True, True]
    assert order_status(s).tolist() == ["completed", "cancelled", "returned", "delivering", "completed", "completed"]


def test_blank_note_column_counts_as_normal_order():
    """Sổ chỉ ghi chú khi có chuyện: mọi đơn để trống là đơn bình thường."""
    from datamate.helpers import valid_order
    s = pd.Series([None, None, "Huỷ", "cancel", None, "đang giao"], dtype="string")
    assert valid_order(s).tolist() == [True, True, False, False, True, True]


def test_blank_lines_take_status_of_their_order():
    """Đơn huỷ ghi ở dòng hàng; dòng Voucher / phí ship cùng đơn để trống vẫn là đơn huỷ."""
    from datamate.helpers import fill_order_status, valid_order
    df = pd.DataFrame({
        "HĐ": ["A", "A", "A", "B", "B"],
        "Tên hàng": ["Tã", "Voucher", "Phí ship", "Sữa", "Voucher"],
        "Ghi chú": pd.Series(["cancel", None, None, None, None], dtype="string"),
    })
    out = fill_order_status(df, "HĐ")
    assert valid_order(out["Ghi chú"]).tolist() == [False, False, False, True, True]
    assert df["Ghi chú"].isna().sum() == 4  # không sửa bảng gốc
