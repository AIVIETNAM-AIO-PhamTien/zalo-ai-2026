import pandas as pd

from datamate.lint import LintHints, build_hints, lint
from datamate.loader import Shop

H = LintHints(status_cols={"Trạng thái"}, timed_date_cols={"Thời gian"}, name_cols={"Tên hàng"})


def test_hand_listed_status_is_flagged():
    notes = lint("df = df[df['Trạng thái'].isin(['Xong', 'Đang giao'])]", H)
    assert len(notes) == 1 and "valid_order" in notes[0]


def test_valid_order_is_not_flagged():
    assert lint("df = df[valid_order(df['Trạng thái'])]", H) == []


def test_counting_cancelled_with_one_value_is_not_flagged():
    assert lint("n = (df['Trạng thái'] == 'Đã hủy').sum()", H) == []
    assert lint("df = df[df['Trạng thái'] != 'Đã hủy']", H)  # loại một giá trị: dễ sót đơn trả


def test_equality_on_timed_date_is_flagged():
    assert any("P.mask" in n for n in lint("x = df[df['Thời gian'] == TODAY]", H))
    assert lint("x = df[df['Thời gian'].dt.normalize() == TODAY]", H) == []  # đã bỏ giờ: không nhắc


def test_name_equality_is_flagged_but_name_mask_is_not():
    assert any("name_mask" in n for n in lint("x = df[df['Tên hàng'] == 'Cà phê sữa']", H))
    assert lint("x = df[name_mask(df['Tên hàng'], 'Cà phê sữa')]", H) == []


def test_comments_are_ignored():
    assert lint("# df['Trạng thái'].isin(['Xong'])", H) == []


def test_build_hints_from_data():
    df = pd.DataFrame({
        "Trạng thái": ["Xong", "Đã hủy", "Xong"],
        "Thời gian": pd.to_datetime(["2025-01-01 08:30", "2025-01-02 00:00", "2025-01-03 00:00"]),
        "Ngày": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    })
    h = build_hints(Shop("b", "", {"t": df}, {"t": {}}))
    assert h.status_cols == {"Trạng thái"} and h.timed_date_cols == {"Thời gian"}
