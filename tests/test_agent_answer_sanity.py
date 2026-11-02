from datamate.agent import implausible
from datamate.answer import answer_key, normalize_answer


def test_money_like_numbers_round_to_dong_so_samples_agree():
    a = normalize_answer({"type": "number", "answer": 72270.25})
    b = normalize_answer({"type": "number", "answer": 72270})
    assert a == {"type": "number", "answer": 72270} and answer_key(a) == answer_key(b)
    assert normalize_answer({"type": "number", "answer": 445333.5})["answer"] == 445334  # nửa đồng lên
    assert normalize_answer({"type": "list", "answer": [21463.41, 3]})["answer"] == [21463, 3]


def test_small_fractions_are_not_money():
    assert normalize_answer({"type": "number", "answer": 2.5})["answer"] == 2.5
    assert normalize_answer({"type": "percent", "answer": 1234.567})["answer"] == 1234.57


def test_negative_only_where_question_allows_sign():
    neg = {"type": "number", "answer": -222727216}
    assert implausible("tính đến hết ngày 28/06/2026, khách còn nợ shop tổng cộng bn tiền", neg)
    assert not implausible("Xem giúp trừ hết chi phí rồi thì tháng 1/2025 còn lãi ròng bao nhiêu?", neg)
    assert not implausible("Tháng 3/2025, số nhập trừ số bán của Telfast là bao nhiêu viên?", neg)
    assert not implausible("doanh thu tháng 5 thay đổi bn % so vs tháng trước", {"type": "percent", "answer": -3})
    assert not implausible("doanh thu tháng 5", {"type": "number", "answer": 5})


def test_placeholder_numbers_are_implausible():
    assert implausible("dthu qua zalo và qua điện thoại", {"type": "list", "answer": [1234567, 890123]})
    assert implausible("tổng tiền thu máy cũ", {"type": "number", "answer": 12345678})
    assert not implausible("doanh thu tháng 5", {"type": "number", "answer": 100000})
    assert not implausible("doanh thu tháng 5", {"type": "number", "answer": 1234})
