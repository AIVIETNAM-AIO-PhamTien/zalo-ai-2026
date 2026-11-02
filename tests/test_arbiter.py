from datamate.arbiter import required, veto, violations

LOT_Q = "Thuốc nào có lô sắp hết hạn gần nhất?"


def test_required_guards_follow_question_meaning():
    assert "in_stock" in required(LOT_Q)
    assert "valid_order" in required("doanh thu tháng 3 bao nhiêu")
    assert "tie_break" in required("mặt hàng bán chạy nhất tuần rồi")
    assert "name_not_code" in required("khách nào mua nhiều nhất")
    assert "name_not_code" not in required("mã khách nào mua nhiều nhất")


def test_lot_answer_vetoed_only_when_every_sample_skips_stock_filter():
    ans = {"type": "string", "answer": "Panadol"}
    no_stock = "df = T['lo']; df.sort_values('han').iloc[0]['ten']"
    with_stock = "df = T['lo']; df = df[df['ton_con_lai'] > 0]; df.sort_values('han').iloc[0]['ten']"
    assert veto(LOT_Q, ans, [(ans, no_stock), (ans, no_stock)]) == {"in_stock"}
    assert veto(LOT_Q, ans, [(ans, no_stock), (ans, with_stock)]) == set()
    assert veto(LOT_Q, ans, [(ans, "SEM.lot_nearest_expiry('Panadol')")]) == set()


def test_non_veto_guards_do_not_block():
    ans = {"type": "number", "answer": 120}
    code = "df['tien'].sum()"
    assert "valid_order" in violations("doanh thu tháng 3", ans, code)
    assert veto("doanh thu tháng 3", ans, [(ans, code)]) == set()


def test_name_question_flags_code_like_answer():
    assert "name_not_code" in violations("khách nào mua nhiều nhất", {"type": "string", "answer": "KH012"}, "")
    assert "name_not_code" not in violations("khách nào mua nhiều nhất", {"type": "string", "answer": "Chị Lan"}, "")
