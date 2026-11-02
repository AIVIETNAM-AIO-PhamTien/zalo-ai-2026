import datetime as dt
import sqlite3
from decimal import Decimal

import openpyxl
import pytest

from datamate.catalog import DataCatalog
from datamate.foundation import date_value, decimal_value, load_foundation


def test_csv_lossless_multiline_duplicates_empty_and_header(tmp_path):
    (tmp_path / "data.csv").write_bytes(b'Report\nID,Note,Value\na,"line1\nline2",0\na,"line1\nline2",0\n,,\n')
    foundation = load_foundation(tmp_path)
    raw = foundation.tables["data.csv"]
    assert len(raw.rows) == 5
    assert raw.rows[2] == ("a", "line1\nline2", "0")
    assert raw.rows[-1] == ("", "", "")
    assert len(raw.project()) == 5
    frame = raw.project(1)
    assert len(frame) == 3
    assert frame.attrs["physical_rows"] == [3, 5, 7]
    assert raw.cell(3, 1)["value"] == "line1\nline2"
    catalog = DataCatalog.build(foundation, {"data.csv": 1})
    assert catalog.profiles["data.csv"]["duplicate_rows"] == 1
    assert catalog.profiles["data.csv"]["columns"]["Value"]["nulls"] == 0


def test_excel_merged_header_formula_serial_and_raw_archive(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Report", None])
    ws.merge_cells("A1:B1")
    ws.append(["date", "amount"])
    ws.append([45000, "1.234,50"])
    ws.append([None, "=1+2"])
    wb.save(tmp_path / "book.xlsx")
    raw = next(iter(load_foundation(tmp_path).tables.values()))
    assert raw.rows[0] == ("Report", None)
    assert raw.rows[2] == (45000, "1.234,50")
    assert raw.rows[3][1] == "=1+2"
    assert raw.source.metadata["merged_cells"] == ["A1:B1"]
    assert date_value(45000, excel_epoch=dt.datetime(1899, 12, 30)).value == dt.datetime(2023, 3, 15)
    assert date_value(45000).error


def test_sqlite_null_empty_types_identifier_collision(tmp_path):
    for file in ("a.db", "b.db"):
        with sqlite3.connect(tmp_path / file) as con:
            con.execute('CREATE TABLE "odd""table" (id INTEGER, value TEXT)')
            con.executemany('INSERT INTO "odd""table" VALUES (?, ?)', [(1, None), (2, ""), (2, "")])
    foundation = load_foundation(tmp_path)
    assert len(foundation.tables) == 2
    raw = foundation.tables['a.db::odd"table']
    assert raw.rows == [(1, None), (2, ""), (2, "")]
    assert raw.project()["value"].tolist() == [None, "", ""]


@pytest.mark.parametrize("raw,locale,expected", [
    ("1.234,50", "vi", Decimal("1234.50")), ("1,234.50", "en", Decimal("1234.50")),
    (None, None, None), ("0", None, Decimal(0)), ("-12.5", "en", Decimal("-12.5")),
])
def test_decimal_conversion(raw, locale, expected):
    converted = decimal_value(raw, locale)
    assert converted.error is None
    assert converted.value == expected
    assert converted.raw == raw


@pytest.mark.parametrize("raw,locale", [("1.234", None), ("12,34.00", "en"),
    ("1 kg", "en"), ("", None), (True, None), ("1.0000000001", "en"), ("9" * 30, None)])
def test_ambiguous_invalid_and_inexact_numbers(raw, locale):
    assert decimal_value(raw, locale).error


def test_relations_require_semantics_and_partitions_do_not_autounion(tmp_path):
    (tmp_path / "a.csv").write_text("id,amount\na,1\na,2\nb,3\n")
    (tmp_path / "b.csv").write_text("id,amount\na,1\na,4\nc,5\n")
    foundation = load_foundation(tmp_path)
    cat = DataCatalog.build(foundation, {"a.csv": 0, "b.csv": 0})
    relation = next(r for r in cat.relations if r.left_column == r.right_column == "id")
    assert relation.cardinality == "many-to-many"
    assert not relation.verified
    assert len(cat.frames) == 2
    assert any("overlap" in p for p in cat.check_union(["a.csv", "b.csv"],
        units={"id": "unitless", "amount": "VND"}, grain="line", partition_evidence="month"))


def test_source_and_rulebook_cache_invalidation(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("id,value\na,1\n")
    key1 = load_foundation(tmp_path, "rule1").cache_key
    assert load_foundation(tmp_path, "rule2").cache_key != key1
    file.write_text("id,value\na,2\n")
    assert load_foundation(tmp_path, "rule1").cache_key != key1


def test_ragged_header_never_truncates_values(tmp_path):
    (tmp_path / "data.csv").write_text("id,id\na,b,extra\n")
    raw = load_foundation(tmp_path).tables["data.csv"]
    frame = raw.project(0)
    assert len(set(frame.columns)) == 3
    assert frame.iloc[0].tolist() == ["a", "b", "extra"]


def test_unresolved_header_is_visible(tmp_path):
    (tmp_path / "data.csv").write_text("An,Doctor\nBinh,Nurse\n")
    cat = DataCatalog.build(load_foundation(tmp_path))
    assert len(cat.frames["data.csv"]) == 2
    assert any("unresolved header" in a for a in cat.ambiguities)
