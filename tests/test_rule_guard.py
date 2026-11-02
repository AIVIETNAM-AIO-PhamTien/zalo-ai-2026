from datamate.rule_guard import Guard, changed_rules, norm

RB = """| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**. Đơn đang giao vẫn tính |
| Khách sộp nhất | giá trị trung bình một đơn cao nhất |
| Giao nhiều đợt | mỗi đợt tính một đợt |
"""
KNOWN = {k: [norm(v)] for k, v in [("Đơn tính vào doanh thu", "đơn không huỷ và không trả hàng. Đơn đang giao vẫn tính"),
                                   ("Khách sộp nhất", "giá trị trung bình một đơn cao nhất"),
                                   ("Giao nhiều đợt", "mỗi đợt tính một đợt")]}


def test_same_rules_keep_everything():
    g = Guard(RB, KNOWN)
    assert g.changed == [] and g.plan and not g.view and not g.plan_blocked


def test_format_only_change_is_not_a_change():
    assert changed_rules(RB.replace("**", "").replace("huỷ", "hủy"), KNOWN) == []


def test_core_change_disables_view_and_plan():
    g = Guard(RB.replace("vẫn tính", "không tính"), KNOWN)
    assert g.core and g.view and not g.plan


def test_non_core_change_blocks_only_its_functions():
    g = Guard(RB.replace("cao nhất", "cao nhất, từ 5 đơn"), KNOWN)
    assert not g.core and g.plan and g.plan_blocked == {"big_spender"}
    g = Guard(RB.replace("mỗi đợt tính một đợt", "đợt của đơn trả không tính"), KNOWN)
    assert g.delivery and "delivery_count" in g.plan_blocked and not g.view


def test_new_rule_rows_are_ignored():
    assert changed_rules(RB + "| Luật mới | gì đó |\n", KNOWN) == []


def test_shipped_snapshot_matches_split_rulebooks():
    from pathlib import Path
    for p in ("data/train/rulebook.md", "data/public_test/rulebook.md"):
        if Path(p).exists():
            assert changed_rules(Path(p).read_text(encoding="utf-8")) == []


def test_changed_shared_rules_block_all_dependent_plan_functions():
    cases = [
        ("Xác định khách", {"customer_most_orders", "loyal_customer", "big_spender",
                            "fickle_customer", "top_spender", "customers_min_orders", "customer_count",
                            "customer_revenue", "customer_order_count", "customer_avg_order_value",
                                "customer_group_revenue", "top_customers"}),
        ("Bán chạy", {"qty_sold", "best_item", "top_items"}),
        ("Nhóm hàng", {"group_revenue", "group_order_count"}),
    ]
    for name, expected in cases:
        guard = Guard(f"| {name} | định nghĩa mới |", {name: [norm("định nghĩa cũ")]})
        assert guard.plan_blocked == expected
