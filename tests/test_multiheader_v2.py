import openpyxl
import pytest

from datamate.foundation import load_foundation
from datamate.semantics import SemanticWarehouse


def workbook(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Visits"
    ws.append(["Ref", "Duration", None])
    ws.append([None, "Actual", "Planned"])
    ws.merge_cells("A1:A2")
    ws.merge_cells("B1:C1")
    ws.append(["V1", 12, 20])
    ws.append([None, None, None])
    ws.append(["V1", 12, 20])
    wb.save(path)


def test_multirow_merged_header_keeps_raw_body_and_header_lineage(tmp_path):
    workbook(tmp_path / "clinic.xlsx")
    raw = load_foundation(tmp_path).tables["clinic.xlsx::Visits"]
    frame = raw.project((0, 1))
    assert list(frame.columns) == ["Ref", "Duration / Actual", "Duration / Planned"]
    assert frame.values.tolist() == [["V1", 12, 20], [None, None, None], ["V1", 12, 20]]
    assert frame.attrs["header_physical_rows"] == [1, 2]
    assert frame.attrs["physical_rows"] == [3, 4, 5]
    assert raw.rows[0] == ("Ref", "Duration", None)
    assert raw.rows[1] == (None, "Actual", "Planned")
    assert raw.cell(1, 2)["value"] is None


def test_documented_composite_labels_ground_header_automatically(tmp_path):
    workbook(tmp_path / "clinic.xlsx")
    (tmp_path / "README.md").write_text(
        "## Columns in `clinic.xlsx`\n- **Ref**: visit identifier\n"
        "- **Duration / Actual**: minutes spent\n- **Duration / Planned**: scheduled minutes\n",
        encoding="utf-8")
    w = SemanticWarehouse.build(load_foundation(tmp_path))
    frame = w.catalog.frames["clinic.xlsx::Visits"]
    assert frame.attrs["header"] == (0, 1)
    assert len(frame) == 3
    assert len(w.bindings) == 3


def test_csv_explicit_multiheader_does_not_invent_merges(tmp_path):
    (tmp_path / "two.csv").write_bytes(b"Ref,Money,\n,Net,Tax\na,10,1\n")
    raw = load_foundation(tmp_path).tables["two.csv"]
    frame = raw.project((0, 1))
    assert list(frame.columns) == ["Ref", "Money / Net", "Tax"]
    assert frame.iloc[0].tolist() == ["a", "10", "1"]


@pytest.mark.parametrize("header", [(1, 0), (0, 2), (), (0, 0), (-1, 0), (0, 99), True])
def test_invalid_multiheader_selection_fails_before_projection(tmp_path, header):
    (tmp_path / "two.csv").write_bytes(b"x,y\na,b\n")
    raw = load_foundation(tmp_path).tables["two.csv"]
    with pytest.raises(ValueError):
        raw.project(header)


def test_data_body_merged_cells_are_never_filled(tmp_path):
    workbook(tmp_path / "clinic.xlsx")
    wb = openpyxl.load_workbook(tmp_path / "clinic.xlsx")
    ws = wb.active
    ws.merge_cells("A3:A5")
    wb.save(tmp_path / "clinic.xlsx")
    raw = load_foundation(tmp_path).tables["clinic.xlsx::Visits"]
    assert raw.project((0, 1))["Ref"].tolist() == ["V1", None, None]


def test_blank_preamble_does_not_create_duplicate_header_candidates(tmp_path):
    (tmp_path / "facts.csv").write_bytes(b"Export report\n\nRef,Measurement\na,10\n")
    (tmp_path / "README.md").write_text("## Columns in `facts.csv`\n- **Ref**: visit ID\n- **Measurement**: duration\n")
    w = SemanticWarehouse.build(load_foundation(tmp_path))
    frame = w.catalog.frames["facts.csv"]
    assert frame.attrs["header"] == 2
    assert frame.attrs["physical_rows"] == [4]
    assert len(w.catalog.foundation.tables["facts.csv"].rows) == 4
