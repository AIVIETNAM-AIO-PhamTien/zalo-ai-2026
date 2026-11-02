import pytest

from datamate.answer_type import expected_type


@pytest.mark.parametrize("q,t", [
    ("Tháng 4/2026 ngày nào cửa hàng bán đc nhiều tiền nhất?", "date"),
    ("Tháng nào trong năm 2025 doanh thu cao nhất?", "date"),
    ("Tháng 3/2026 top 5 hàng bán chạy nhất là gì?", "list"),
    ("Khách X tháng 1/2026 nợ thêm bao nhiêu, trả được bao nhiêu?", "list"),
    ("Cuối ngày hôm nay Bánh croissant còn mấy hộp chẵn, mấy cái lẻ?", "list"),
    ("Liệt kê 3 khách nợ nhiều nhất tính đến hết ngày 19/08/2026, kèm số tiền còn nợ.", "table"),
    ("dthu 3 thang 5/2025, 6/2025, 7/2025 co tang lien tuc khong", "boolean"),
    ("thang 9/2025 cua hang co bnhiu % don bi huy", "percent"),
    ("Tháng 7/2025 Otrivin 0,1% xịt mũi chai 10ml bán được bao nhiêu chai?", "number"),
    ("thang 6/2025 hang nao chay nhat", "string"),
    ("Tháng 6/2025 ai mua hàng nhiều tiền nhất?", "string"),
    ("Tháng 5/2025 cửa hàng bán được bao nhiêu tiền?", "number"),
    ("Trong tháng này, số ly size L bán được là bao nhiêu, tính mọi loại đồ uống?", "number"),
])
def test_expected_type(q, t):
    assert expected_type(q) == t


def test_unclear_question_has_no_guess():
    assert expected_type("Phân tích tình hình kinh doanh giúp mình") is None
def test_count_of_entities_in_ranked_set_is_not_a_list_or_percent():
    from datamate.answer_type import expected_type
    assert expected_type('Trong top 3 khách mua nhiều nhất có bao nhiêu khách nữ?') == 'number'
    assert expected_type('Bao nhiêu mặt hàng có mức giảm giá trên 10%?') == 'number'
    assert expected_type('Liệt kê top 3 khách mua nhiều nhất và số tiền của từng người') == 'table'
    assert expected_type('Cuối ngày còn bao nhiêu xe 4m3 chẵn, bao nhiêu m3 lẻ?') == 'list'
    assert expected_type('Món nào bán được nhiều mặt hàng nhất?') != 'number'
