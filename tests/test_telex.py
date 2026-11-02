from datamate.telex import decode, telex_fix


def test_telex_words():
    assert telex_fix("soos đơn hoom nay")[0] == "số đơn hôm nay"
    assert telex_fix("cuwar hàng bán dduwowcj mấy chai")[0] == "cửa hàng bán được mấy chai"
    assert telex_fix("tir leej đơn huỷ thangs 6/2026")[0] == "tỉ lệ đơn huỷ tháng 6/2026"


def test_telex_leaves_brands_and_plain_words():
    assert telex_fix("bình gas namilux, plusssz max, D-nee 3000ml, doanh thu thang 5")[1] == []
    assert decode("ban") is None and decode("coca") is None
