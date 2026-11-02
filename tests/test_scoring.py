from datamate.scoring import normalize_string, score_item, score_file


def ref(type_, answer, alias=None):
    return {"id": "q", "type": type_, "answer": answer, "alias": alias or []}


def pred(type_, answer=None, **kw):
    d = {"id": "q", "type": type_}
    if answer is not None:
        d["answer"] = answer
    d.update(kw)
    return d


# --- chuẩn hoá chuỗi ---------------------------------------------------------

def test_normalize_string_keeps_diacritics_and_strips_punct():
    assert normalize_string("  Sting  DÂU-330ml! ") == "sting dâu 330ml"
    assert normalize_string("Sting dầu 330ml") != normalize_string("Sting dâu 330ml")


def test_normalize_string_nfc():
    decomposed = "Cà phê".replace("à", "à")
    assert normalize_string(decomposed) == normalize_string("Cà phê")


# --- number / percent --------------------------------------------------------

def test_number_exact():
    assert score_item(pred("number", 48250000), ref("number", 48250000)) == 1
    assert score_item(pred("number", 48250001), ref("number", 48250000)) == -1


def test_number_float_noise_ok():
    assert score_item(pred("number", 18.600000000000001), ref("number", 18.6)) == 1


def test_number_as_string_is_wrong():
    assert score_item(pred("number", "48250000"), ref("number", 48250000)) == -1


def test_bool_is_not_number():
    assert score_item(pred("number", True), ref("number", 1)) == -1


def test_percent_rounded_two_decimals():
    assert score_item(pred("percent", -12.4512), ref("percent", -12.45)) == 1
    assert score_item(pred("percent", -12.46), ref("percent", -12.45)) == -1
    assert score_item(pred("percent", 12.45), ref("percent", -12.45)) == -1


def test_type_mismatch_is_wrong():
    assert score_item(pred("number", 12.45), ref("percent", 12.45)) == -1


# --- string / date / boolean -------------------------------------------------

def test_string_alias_and_code():
    r = ref("string", "Xi măng Hà Tiên PCB40 bao 50kg", ["SP0001"])
    assert score_item(pred("string", "xi măng hà tiên pcb40 bao 50kg"), r) == 1
    assert score_item(pred("string", "sp0001"), r) == 1
    assert score_item(pred("string", "Xi mang Ha Tien PCB40 bao 50kg"), r) == -1


def test_string_in_list_is_wrong():
    assert score_item(pred("string", ["Sting dâu 330ml"]), ref("string", "Sting dâu 330ml")) == -1


def test_date_exact():
    assert score_item(pred("date", "2025-04-04"), ref("date", "2025-04-04")) == 1
    assert score_item(pred("date", "04/04/2025"), ref("date", "2025-04-04")) == -1


def test_boolean_exact():
    assert score_item(pred("boolean", False), ref("boolean", False)) == 1
    assert score_item(pred("boolean", 0), ref("boolean", False)) == -1


# --- list ----------------------------------------------------------------------

def test_list_positional_with_per_element_alias():
    r = ref("list", ["Cà phê sữa (M)", "Trà đào cam sả (M)"], [["SP0003"], ["SP0017"]])
    assert score_item(pred("list", ["SP0003", "trà đào cam sả (m)"]), r) == 1
    assert score_item(pred("list", ["Trà đào cam sả (M)", "Cà phê sữa (M)"]), r) == -1
    assert score_item(pred("list", ["Cà phê sữa (M)"]), r) == -1


def test_list_of_numbers():
    assert score_item(pred("list", [149738300, 6739400]), ref("list", [149738300, 6739400])) == 1
    assert score_item(pred("list", [6739400, 149738300]), ref("list", [149738300, 6739400])) == -1


# --- table ---------------------------------------------------------------------

TABLE_REF = ref(
    "table",
    {"columns": ["khach_hang", "so_tien_no"],
     "rows": [["Đỗ Đức An", 404500], ["Dương Tâm Bích", 158600]]},
    [["KH0124"], ["KH0233"]],
)


def test_table_rows_as_set_and_alias():
    p = pred("table", {"columns": ["khach_hang", "so_tien_no"],
                       "rows": [["KH0233", 158600], ["đỗ đức an", 404500]]})
    assert score_item(p, TABLE_REF) == 1


def test_table_columns_by_name_reordered():
    p = pred("table", {"columns": ["so_tien_no", "khach_hang"],
                       "rows": [[404500, "Đỗ Đức An"], [158600, "Dương Tâm Bích"]]})
    assert score_item(p, TABLE_REF) == 1


def test_table_columns_by_position_when_names_differ():
    p = pred("table", {"columns": ["khach", "no"],
                       "rows": [["Đỗ Đức An", 404500], ["Dương Tâm Bích", 158600]]})
    assert score_item(p, TABLE_REF) == 1


def test_table_wrong_value_or_missing_row():
    p = pred("table", {"columns": ["khach_hang", "so_tien_no"],
                       "rows": [["Đỗ Đức An", 404500], ["Dương Tâm Bích", 158601]]})
    assert score_item(p, TABLE_REF) == -1
    p = pred("table", {"columns": ["khach_hang", "so_tien_no"], "rows": [["Đỗ Đức An", 404500]]})
    assert score_item(p, TABLE_REF) == -1


# --- unanswerable / abstain ------------------------------------------------------

def test_unanswerable():
    assert score_item(pred("unanswerable", reason="thiếu dữ liệu"), ref("unanswerable", None)) == 1
    assert score_item(pred("unanswerable"), ref("number", 5)) == -1
    assert score_item(pred("number", 5), ref("unanswerable", None)) == -1


def test_abstain_and_missing():
    assert score_item(pred("abstain"), ref("number", 5)) == 0
    assert score_item(None, ref("number", 5)) == 0


def test_non_string_reason_line_is_blank():
    assert score_item(pred("unanswerable", reason=123), ref("unanswerable", None)) == 0


# --- cả file -------------------------------------------------------------------

def test_score_file(tmp_path):
    refs = tmp_path / "answers.jsonl"
    subs = tmp_path / "submission.jsonl"
    refs.write_text(
        '{"id":"a","type":"number","answer":1,"alias":[]}\n'
        '{"id":"b","type":"string","answer":"X","alias":[]}\n'
        '{"id":"c","type":"unanswerable","answer":null,"alias":[]}\n'
        '{"id":"d","type":"number","answer":2,"alias":[]}\n',
        encoding="utf-8",
    )
    subs.write_text(
        '{"id":"a","type":"number","answer":1}\n'
        '{"id":"b","type":"string","answer":"Y"}\n'
        '{"id":"c","type":"unanswerable"}\n'
        'not json\n',
        encoding="utf-8",
    )
    res = score_file(subs, refs)
    assert res["n"] == 4
    assert (res["correct"], res["wrong"], res["abstain"]) == (2, 1, 1)
    assert res["pa"] == 0.25
