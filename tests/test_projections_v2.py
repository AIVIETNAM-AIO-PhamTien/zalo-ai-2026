from decimal import Decimal
import sqlite3

import pytest

from datamate.pipeline import prepare_business, Checkpoint, run_questions
from datamate.projections import infer_spec


def test_auto_conversion_requires_meaning_and_agreement_across_whole_column():
    assert infer_spec(["1.234"], ["thành tiền sau giảm"]) is None
    assert infer_spec(["01/02/2026"], ["ngày hoàn tất"]) is None
    assert infer_spec(["1", "not-a-number"], ["số lượng"]) is None
    assert infer_spec(["1", "2"], []) is None
    assert infer_spec(["1", "2"], ["số lượng", "mã hàng"]) is None
    assert infer_spec(["001", "002"], ["mã đơn"]) is None
    assert infer_spec(["1.234,50", "2.000,25"], ["thành tiền sau giảm"])["locale"] == "vi"
    assert infer_spec(["31/01/2026", "01/02/2026"], ["ngày đặt hàng"])["format"] == "%d/%m/%Y"


def test_documented_csv_numeric_and_date_projection_keeps_raw(tmp_path):
    (tmp_path / "facts.csv").write_bytes(b"ID,Amount,Date\na,10,2026-02-01\na,10,2026-02-01\nb,20,2026-03-01\n")
    (tmp_path / "README.md").write_text("## Columns in `facts.csv`\n- **ID**: mã đơn\n- **Amount**: thành tiền sau giảm\n- **Date**: ngày đặt hàng\n", encoding="utf-8")
    prepared = prepare_business(tmp_path, "")
    assert prepared.schemas["facts.csv"] == {"ID": "text", "Amount": "number", "Date": "date"}
    assert prepared.frames["facts.csv"]["Amount"].tolist() == [Decimal(10), Decimal(10), Decimal(20)]
    raw = prepared.warehouse.catalog.foundation.tables["facts.csv"]
    assert raw.rows[1] == ("a", "10", "2026-02-01")
    assert all(c["physical_rows"] == [2, 3, 4] for c in prepared.conversions)


def test_unusable_column_does_not_remove_valid_columns_or_records(tmp_path):
    with sqlite3.connect(tmp_path / "facts.db") as con:
        con.execute("CREATE TABLE facts (id TEXT, amount, empty TEXT)")
        con.executemany("INSERT INTO facts VALUES (?,?,?)", [("a", 1, None), ("a", "not numeric", None)])
    prepared = prepare_business(tmp_path, "")
    frame = prepared.frames["facts.db::facts"]
    assert list(frame.columns) == ["id", "empty"]
    assert len(frame) == 2
    assert prepared.schemas["facts.db::facts"]["empty"] == "null"
    assert prepared.warehouse.catalog.foundation.tables["facts.db::facts"].rows[1][1] == "not numeric"
    checkpoint = Checkpoint(tmp_path / "out", ["q"], {})
    run_questions([{"id": "q", "question": "Đếm tất cả bản ghi trong bảng `facts.db::facts`"}], prepared, checkpoint)
    assert checkpoint.rows["q"]["answer"] == 2


def test_native_float_can_be_projected_exactly_without_dropping_table(tmp_path):
    with sqlite3.connect(tmp_path / "facts.db") as con:
        con.execute("CREATE TABLE facts (amount REAL)")
        con.executemany("INSERT INTO facts VALUES (?)", [(1.25,), (2.5,)])
    prepared = prepare_business(tmp_path, "")
    assert prepared.frames["facts.db::facts"]["amount"].tolist() == [Decimal("1.25"), Decimal("2.5")]
