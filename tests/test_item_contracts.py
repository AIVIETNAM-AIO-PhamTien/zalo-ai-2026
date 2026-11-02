import datetime as dt

import pandas as pd

from datamate.semantic import OrderView, _items


def test_service_fee_is_revenue_but_not_an_item_and_ties_follow_code():
    raw = pd.DataFrame({"id": ["1", "1", "1"], "MaHang": ["Z", "A", "F"],
                        "name": ["Trà", "Bánh", "Phí phục vụ"], "qty": [1, 1, 100], "unit": ["ly", "cái", "lần"]})
    items = _items(raw, "id", {"name": "tên hàng", "qty": "số lượng", "unit": "đơn vị tính"}, {})
    # Bảng không có quy đổi vẫn phải cung cấp bqty/bunit theo loader/schema.
    if items is None:
        raw["baseqty"] = raw["qty"]
        items = _items(raw, "id", {"name": "tên hàng", "qty": "số lượng", "unit": "đơn vị tính",
                                    "baseqty": "số lượng đã đổi ra đơn vị cơ bản"}, {})
    assert set(items.item) == {"tra", "banh"}
    orders = pd.DataFrame({"order_id": ["1"], "date": pd.to_datetime(["2026-01-01"]), "cls": ["completed"]})
    lines = pd.DataFrame({"order_id": ["1", "1", "1"], "amount": [10, 20, 5]})
    v = OrderView(lines, orders, items=items)
    p, today = (dt.date(2026, 1, 1), dt.date(2026, 1, 1)), dt.date(2026, 1, 1)
    assert v.revenue(p, today) == 35
    assert v.top_items(p, today, 2) == ["Bánh", "Trà"]
    assert v.qty_sold(p, today, "lan") == 0


def test_unrelated_lower_rank_ties_do_not_block_unique_winner():
    orders = pd.DataFrame({"order_id": ["1"], "date": pd.to_datetime(["2026-01-01"]), "cls": ["completed"]})
    items = pd.DataFrame({"order_id": ["1"] * 4, "item": ["a", "a", "b", "c"],
                          "name": ["A", "A", "B", "C"], "bqty": [5, 5, 1, 1],
                          "rank_key": ["A1", "A2", "B", "C"]})
    v = OrderView(pd.DataFrame({"order_id": ["1"], "amount": [10]}), orders, items=items)
    p, today = (dt.date(2026, 1, 1), dt.date(2026, 1, 1)), dt.date(2026, 1, 1)
    assert v.top_items(p, today, 1) == ["A"]


def test_name_variants_of_one_item_code_are_one_item():
    # cùng mã hàng mà tên ghi nhiều cách (bỏ dấu / đổi chữ): một mặt hàng; tên trùng giữa hai mã thì không gộp
    raw = pd.DataFrame({"id": ["1", "2", "3", "4", "5"], "MaHang": ["SP1", "SP1", "SP1", "SP2", "SP3"],
                        "name": ["Mì Hảo Hảo", "Mì Hảo Hảo", "Mi Hao Hao", "Trà", "Tra"],
                        "qty": [1, 1, 1, 1, 1], "baseqty": [1, 1, 1, 1, 1], "unit": ["gói"] * 5})
    items = _items(raw, "id", {"name": "tên hàng", "qty": "số lượng", "unit": "đơn vị tính",
                               "baseqty": "số lượng đã đổi ra đơn vị cơ bản"}, {})
    assert items.item.tolist() == ["mi hao hao"] * 3 + ["tra", "tra"]
    raw["name"] = ["Mì Hảo Hảo", "Mì Hảo Hảo", "Mì Sìnhkh", "Trà", "Tra"]
    items = _items(raw, "id", {"name": "tên hàng", "qty": "số lượng", "unit": "đơn vị tính",
                               "baseqty": "số lượng đã đổi ra đơn vị cơ bản"}, {})
    assert items.item.tolist()[:3] == ["mi hao hao"] * 3
