from datamate.answer import answer_key, normalize_answer


def test_date_must_exist_in_calendar():
    for value in ["2026-02-30", "2026-13", "2026-00-01", "2026-01-01\nextra"]:
        assert normalize_answer({"type": "date", "answer": value}) is None
    assert normalize_answer({"type": "date", "answer": "2024-02-29"}) == {"type": "date", "answer": "2024-02-29"}


def test_number_normalization():
    assert normalize_answer({"type": "number", "answer": 48250000.0}) == {"type": "number", "answer": 48250000}
    assert normalize_answer({"type": "number", "answer": 18.6049}) == {"type": "number", "answer": 18.6}
    assert normalize_answer({"type": "number", "answer": "48250000"}) == {"type": "number", "answer": 48250000}
    assert normalize_answer({"type": "number", "answer": "4,82 triệu"}) is None
    assert normalize_answer({"type": "number", "answer": True}) is None


def test_percent_and_date_and_bool():
    assert normalize_answer({"type": "percent", "answer": -12.4512}) == {"type": "percent", "answer": -12.45}
    assert normalize_answer({"type": "date", "answer": "2025-09"}) == {"type": "date", "answer": "2025-09"}
    assert normalize_answer({"type": "date", "answer": "15/08/2026"}) is None
    assert normalize_answer({"type": "boolean", "answer": "true"}) == {"type": "boolean", "answer": True}
    assert normalize_answer({"type": "boolean", "answer": 1}) is None


def test_string_list_table():
    assert normalize_answer({"type": "string", "answer": " Sting dâu 330ml "}) == \
        {"type": "string", "answer": "Sting dâu 330ml"}
    assert normalize_answer({"type": "string", "answer": ["x"]}) is None
    assert normalize_answer({"type": "list", "answer": [3.0, "Bạc xỉu"]}) == {"type": "list", "answer": [3, "Bạc xỉu"]}
    t = normalize_answer({"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"],
                                                      "rows": [["Lan", 3200000.0]]}})
    assert t == {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Lan", 3200000]]}}
    assert normalize_answer({"type": "table", "answer": {"rows": []}}) is None


def test_unanswerable_and_bad_type():
    assert normalize_answer({"type": "unanswerable", "reason": "x"}) == {"type": "unanswerable", "reason": "x"}
    assert normalize_answer({"type": "unanswerable"}) == {"type": "unanswerable", "reason": ""}
    assert normalize_answer({"type": "foo", "answer": 1}) is None
    assert normalize_answer("abc") is None


def test_answer_key_equivalence():
    a = normalize_answer({"type": "string", "answer": "Cà phê sữa (M)"})
    b = normalize_answer({"type": "string", "answer": "cà phê sữa m"})
    assert answer_key(a) == answer_key(b)
    t1 = normalize_answer({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["X", 1], ["Y", 2]]}})
    t2 = normalize_answer({"type": "table", "answer": {"columns": ["a", "b"], "rows": [["Y", 2], ["X", 1]]}})
    assert answer_key(t1) == answer_key(t2)
    assert answer_key(normalize_answer({"type": "unanswerable", "reason": "a"})) == \
        answer_key(normalize_answer({"type": "unanswerable", "reason": "b"}))
    assert answer_key(normalize_answer({"type": "number", "answer": 1})) != \
        answer_key(normalize_answer({"type": "percent", "answer": 1}))
