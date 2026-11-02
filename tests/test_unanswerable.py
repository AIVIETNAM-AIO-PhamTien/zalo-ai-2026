import datetime as dt

import pytest

from datamate.unanswerable import ShopProfile, detect

D = dt.date
FULL = ShopProfile(first_day=D(2025, 4, 1), gender=True, address=True, stock=True, imports=True)
BARE = ShopProfile(first_day=D(2025, 4, 1), gender=False, address=False, stock=False, imports=False)


@pytest.mark.parametrize("q", [
    "Ai là người lành nghề nhất trong số nhân viên?",
    "thuốc có hoạt chất nhôm phosphat người lớn nên uống mỗi lần mấy viên",
    "lê phúc đạt có lễ phép, chu đáo với khách hông",
])
def test_subjective(q):
    assert detect(q, [], FULL)


@pytest.mark.parametrize("q", [
    "e thang 1/2026 khach nu mua tong cong bao nhieu tien",
    "Có bao nhiêu khách hàng ở đường Trần Não?",
    "tồn kho red bull 250ml cuối ngày hôm nay là bao nhiêu lon",
    "Cho hỏi tổng tiền nhập hàng tháng này là bao nhiêu?",
])
def test_missing_information(q):
    assert detect(q, [], BARE)
    assert detect(q, [], FULL) is None  # shop có thông tin đó thì để agent làm


def test_period_before_data():
    assert detect("dthu thang 2/2025 bnhiu", [("month", (D(2025, 2, 1), D(2025, 2, 28)))], FULL)
    assert detect("dthu thang 4/2025 bnhiu", [("month", (D(2025, 4, 1), D(2025, 4, 30)))], FULL) is None
    # '09/05/2026' đọc nhầm thành tháng 9/2005: năm vô lý → không kết luận
    assert detect("hôm 09/05/2026 bán giá bao nhiêu", [("month", (D(2005, 9, 1), D(2005, 9, 30)))], FULL) is None


def test_subjective_term_with_direct_column_is_not_declared_missing():
    import pandas as pd
    from datamate.loader import Shop
    profile = ShopProfile.from_shop(Shop('synthetic', '', {'staff': pd.DataFrame({'Điểm tay nghề': [4]})}, {}))
    assert detect('Thợ nào có tay nghề cao nhất?', [], profile) is None
    missing = ShopProfile.from_shop(Shop('synthetic', '', {'staff': pd.DataFrame({'Doanh thu': [4]})}, {}))
    assert detect('Thợ nào có tay nghề cao nhất?', [], missing)
