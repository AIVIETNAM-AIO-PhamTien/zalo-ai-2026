import datetime as dt

import pytest

from datamate.parsing import fold, is_walkin, parse_date, parse_number, parse_quantity


@pytest.mark.parametrize("raw,expected", [
    ("120000", 120000),
    (120000, 120000),
    (37453.0, 37453),
    ("5,000", 5000),
    ("115,000.0000", 115000),
    ("1,550,000.0000", 1550000),
    ("270.000đ", 270000),
    ("10.000đ", 10000),
    ("1.234.567", 1234567),
    ("1.234,5", 1234.5),
    ("250k", 250000),
    ("16490k", 16490000),
    ("16,5tr", 16500000),
    ("16.5tr", 16500000),
    ("2 triệu", 2000000),
    ("1,2 tỷ", 1200000000),
    ("18.6", 18.6),
    ("0,5", 0.5),
    ("-12.45", -12.45),
    ("(50,000)", -50000),
    ("50.000 VND", 50000),
    ("50 000 ₫", 50000),
    (" 1500 ", 1500),
    ("409tr663", 409663000),
    ("1tr2", 1200000),
    ("1tr25", 1250000),
    ("16k5", 16500),
    ("4,38tr", 4380000),
    ("-2tr250", -2250000),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "  ", "abc", "Coca 330ml", "nan"])
def test_parse_number_invalid(raw):
    assert parse_number(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("4 chai", (4, "chai")),
    ("1", (1, None)),
    (3, (3, None)),
    ("1 m3", (1, "m3")),
    ("2,5 kg", (2.5, "kg")),
    ("10 cái", (10, "cái")),
    ("409tr663", (409663000, None)),
    ("1 xe 4m3", (1, "xe 4m3")),
    ("3 bao 25kg", (3, "bao 25kg")),
    ("2 hộp 30", (2, "hộp 30")),
])
def test_parse_quantity_valid(raw, expected):
    assert parse_quantity(raw) == expected


@pytest.mark.parametrize("raw", ["1 thg 6 2025", "02/06/2025", "chai", None])
def test_parse_quantity_rejects_dates_and_text(raw):
    assert parse_quantity(raw) is None


@pytest.mark.parametrize("raw,expected", [("1", (1, None))])
def test_parse_quantity(raw, expected):
    assert parse_quantity(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("01/07/2025", dt.datetime(2025, 7, 1)),
    ("1/7/2025", dt.datetime(2025, 7, 1)),
    ("1 thg 7 2025", dt.datetime(2025, 7, 1)),
    ("1 thg 7, 2025", dt.datetime(2025, 7, 1)),
    ("45870", dt.datetime(2025, 8, 1)),
    (45717, dt.datetime(2025, 3, 1)),
    ("45717.5", dt.datetime(2025, 3, 1, 12, 0)),
    ("2025-07-01", dt.datetime(2025, 7, 1)),
    ("2025-07-01 08:30:00", dt.datetime(2025, 7, 1, 8, 30)),
    ("01/07/2025 14:05", dt.datetime(2025, 7, 1, 14, 5)),
    ("01-07-2025", dt.datetime(2025, 7, 1)),
    ("ngày 1 tháng 7 năm 2025", dt.datetime(2025, 7, 1)),
    (dt.datetime(2025, 7, 1, 9), dt.datetime(2025, 7, 1, 9)),
    (dt.date(2025, 7, 1), dt.datetime(2025, 7, 1)),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_parse_date_without_year_uses_default_year():
    assert parse_date("05/03", default_year=2026) == dt.datetime(2026, 3, 5)
    assert parse_date("05/03") is None


@pytest.mark.parametrize("raw", [None, "", "abc", "32/13/2025", "123"])
def test_parse_date_invalid(raw):
    assert parse_date(raw) is None


def test_fold_removes_diacritics():
    assert fold("  Đường Biên HÒA 1kg ") == "duong bien hoa 1kg"
    assert fold("Cà phê sữa (M)") == "ca phe sua m"


@pytest.mark.parametrize("name", ["Khách lẻ", "khách lẻ", "kl", "KL", "khach le", "", None, "  "])
def test_is_walkin(name):
    assert is_walkin(name)


@pytest.mark.parametrize("name", ["Kiều Thanh Giang", "Khách Lan", "Chị Lan"])
def test_is_not_walkin(name):
    assert not is_walkin(name)
def test_integer_parsing_preserves_values_beyond_float_precision():
    from datamate.parsing import parse_number
    assert parse_number('9007199254740993') == 9007199254740993
    assert parse_number('9.007.199.254.740.993') == 9007199254740993
    assert parse_number('-9007199254740993') == -9007199254740993
