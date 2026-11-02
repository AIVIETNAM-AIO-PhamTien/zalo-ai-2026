from datamate.linking import hints, relevant_rules, rule_rows

RULEBOOK = """## 1. Doanh thu
| Luật | Mặc định |
|---|---|
| Ca sáng, ca chiều | theo giờ tạo đơn: ca sáng 06:00–13:59 |
| Doanh thu theo nhóm khách | nhóm theo cột nhóm khách |
| Tết năm Y, dịp Tết | từ 23 tháng Chạp đến mùng 10 |
"""


def test_relevant_rules_match_question_terms():
    rows = rule_rows(RULEBOOK)
    assert len(rows) == 3  # bỏ dòng tiêu đề và dòng kẻ
    assert relevant_rules("Doanh thu ca sáng tháng 5/2026 là bao nhiêu?", rows)[0].startswith("| Ca sáng")
    assert relevant_rules("Doanh thu dịp Tết năm nay?", rows)[0].startswith("| Tết")
    assert relevant_rules("Hôm qua bán được mấy đơn?", rows) == []


def test_hints_empty_without_matches():
    assert hints("Hôm qua bán được mấy đơn?", rule_rows(RULEBOOK), None) == ""


def test_installments_include_debt_dependencies_and_runtime_content():
    rb = """| Trả góp | chỉ dùng quy định mới của split |
| Đơn được tính vào nợ | chính sách nợ mới |
| Tuổi nợ | tính theo ngày đến hạn của split |
"""
    selected = relevant_rules("Tiền góp quá hạn hôm nay bao nhiêu?", rule_rows(rb), k=1)
    assert len(selected) == 3
    assert any("chỉ dùng quy định mới" in row for row in selected)
    assert any("ngày đến hạn của split" in row for row in selected)
    assert relevant_rules("Tiền góp quá hạn?", rule_rows("")) == []
    assert relevant_rules("Tiền góp quá hạn?", rule_rows(rb), k=0) == []


def test_warranty_preserves_both_industry_definitions():
    rb = """### Sửa xe
| Bảo hành | chia cho số phiếu có thu tiền |
### Cửa hàng điện thoại
| Bảo hành | chia cho số máy bán |
"""
    selected = relevant_rules("Tỉ lệ bảo hành quý này?", rule_rows(rb), k=1)
    assert len(selected) == 2
    assert any("số phiếu có thu tiền" in row and "Mục rulebook: Sửa xe" in row for row in selected)
    assert any("số máy bán" in row and "Mục rulebook: Cửa hàng điện thoại" in row for row in selected)


def test_public_rule_bundles_without_question_or_answer_data():
    from pathlib import Path
    import pytest

    path = Path("data/public_test/rulebook.md")
    if not path.exists():
        pytest.skip("Public rulebook is optional local data")
    rows = rule_rows(path.read_text(encoding="utf-8"))
    cases = [
        ("Khách mới đăng ký tháng này?", ["Khách thành viên mới trong tháng"]),
        ("Tiền góp quá hạn bao nhiêu?", ["Trả góp", "Tuổi nợ"]),
        ("Khách phải trả sau đổi máy?", ["Thu máy cũ", "Doanh thu một đơn"]),
        ("Vá xe có mấy lượt?", ["Việc ghi trong ô Làm gì", "Lượt xe, phiếu"]),
        ("Nhà cung cấp chính là ai?", ["Nhà cung cấp chính", "Tiền nhập hàng trong kỳ"]),
        ("Size bỉm chiếm mấy phần trăm?", ["Số sữa, size bỉm"]),
        ("Tổng tiền dùng điểm?", ["Tiền giảm voucher, tiền giảm nhờ dùng điểm, phí ship"]),
        ("Tiền mặt theo ca?", ["Báo cáo tiền mặt theo ca"]),
        ("Topping bán được bao nhiêu?", ["Topping ghi chung ô với món"]),
    ]
    for question, names in cases:
        selected = relevant_rules(question, rows)
        for name in names:
            assert any(row.startswith(f"| {name} |") for row in selected), (question, name)
