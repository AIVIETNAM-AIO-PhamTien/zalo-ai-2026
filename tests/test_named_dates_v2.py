import datetime as dt

import pytest

from datamate.foundation import date_value
from datamate.projections import infer_spec
from datamate.pipeline import prepare_business


@pytest.mark.parametrize("raw,expected", [
    ("1 thg 6 2025", dt.datetime(2025, 6, 1)),
    ("ngày 29 tháng 2 năm 2024", dt.datetime(2024, 2, 29)),
    (" 31 THG. 12 2026 23:59:58 ", dt.datetime(2026, 12, 31, 23, 59, 58)),
    ("1 thang 1 nam 2026 06:30", dt.datetime(2026, 1, 1, 6, 30)),
])
def test_named_month_conversion_keeps_raw_and_rule(raw, expected):
    converted = date_value(raw, format="vi_named")
    assert converted.value == expected
    assert converted.raw == raw
    assert converted.error is None
    assert "vi_named" in converted.rule


@pytest.mark.parametrize("raw", ["29 thg 2 2025", "1 thg 13 2025", "1 thg 6", "1 thg 6 25", "1 thg 6 2025 extra", "31 thg 4 2025", "1 thg 6 2025 25:00"])
def test_invalid_or_incomplete_date_never_defaults(raw):
    assert date_value(raw, format="vi_named").error


def test_mixed_unambiguous_formats_and_column_order_evidence():
    values = ["2026-01-31", "1 thg 2 2026", "28/02/2026", "01/03/2026"]
    spec = infer_spec(values, ["ngày hoàn tất"])
    assert spec["format"] == "dayfirst_mixed"
    assert date_value(values[-1], format=spec["format"]).value == dt.datetime(2026, 3, 1)
    assert infer_spec(["2026-01-31", "01/02/2026"], ["ngày hoàn tất"]) is None
    assert infer_spec(["1 thg 6 2025", "not-a-date"], ["ngày hoàn tất"]) is None


def test_source_date_projection_works_without_explicit_config(tmp_path):
    (tmp_path / "work.csv").write_text("Ref,Done\na,1 thg 6 2025\na,1 thg 6 2025\nb,2 thg 6 2025\n")
    (tmp_path / "README.md").write_text("## Columns in `work.csv`\n- **Ref**: mã phiếu\n- **Done**: ngày hoàn tất\n", encoding="utf-8")
    prepared = prepare_business(tmp_path, "")
    assert prepared.schemas["work.csv"]["Done"] == "date"
    assert prepared.frames["work.csv"]["Done"].tolist() == [dt.datetime(2025, 6, 1), dt.datetime(2025, 6, 1), dt.datetime(2025, 6, 2)]
    assert prepared.warehouse.catalog.foundation.tables["work.csv"].rows[1][1] == "1 thg 6 2025"
    assert prepared.conversions[0]["physical_rows"] == [2, 3, 4]
