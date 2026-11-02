"""Canh luật: rulebook của split (private có thể đổi luật) so với các dòng luật mà code đã mã hoá.

Tầng tất định và SEM cài luật train + public (datamate/known_rules.json, sinh bằng scripts/build_known_rules.py).
Dòng luật đã biết mà lời khác đi → phép tính dựa trên dòng đó bị tắt, câu hỏi về agent (agent đọc rulebook của split).
Dòng luật mới hoàn toàn không ảnh hưởng (code chưa cài thì vốn đã để agent). Ý tưởng: biên dịch chính sách thành
kiểm tra lúc build, chặn lúc chạy (Zwerdling et al., EMNLP 2025 industry); luật → test (Policy→Tests, 2025).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datamate.parsing import fold

_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$")

# nhóm luật → phần code dựa vào nó. 'core': định nghĩa đơn / kỳ / trạng thái — đổi thì cả bảng đơn chuẩn không tin được
GROUPS = {
    "core": ["Đơn tính vào doanh thu", "Bán được bao nhiêu tiền, thu về bao nhiêu tiền", "Đơn thuộc kỳ nào",
             "Đơn bị huỷ", "Số đơn", "Doanh thu một đơn", "Tỉ lệ đơn huỷ trong kỳ", "Phần trăm thay đổi, tăng trưởng",
             "Chênh lệch, hơn / kém, tăng / giảm bao nhiêu (số, không phải %)", "Ngày X, hôm nay, hôm qua",
             "Đến hết ngày X, tính đến hôm nay", "Từ X đến Y, nhóm X–Y", "N ngày gần nhất", "Tuần", "Tháng M/Y",
             "Tháng này", "Từ đầu năm, từ đầu năm tới nay", "Tháng M, quý Q không ghi năm", "Trạng thái đơn",
             "Ô để trống ở các dòng sau của cùng một nhóm dòng", "Sổ không có cột trạng thái",
             "Hai dòng của **cùng một đơn** giống nhau ở mọi ô", "Ngày không ghi năm", "Tết năm Y, dịp Tết, mùa Tết"],
    "debt": ["Đơn được tính vào nợ", "Nợ một đơn tại ngày X", "Nợ theo khách", "Đơn COD, đơn bán qua sàn TMĐT",
             "Tuổi nợ", "Nợ thêm, trả nợ, thu nợ trong kỳ"],
    "delivery": ["Giao nhiều đợt"],
    "lot": ["Hàng hết hạn trong N ngày tới", "Lô gần hết hạn nhất của một thuốc"],
}
# dòng luật → phép tính của tầng kế hoạch (datamate/plan.py)
PLAN_FUNCS = {
    "Doanh thu theo hình thức thanh toán": ["payment_revenue"],
    "Ca sáng, ca chiều": ["shift_revenue", "shift_order_count"],
    "Trung bình mỗi ngày thứ X của kỳ": ["weekday_avg"],
    "Bán chạy": ["qty_sold"], "Bán được bao nhiêu <đơn vị>": ["qty_sold"], "Đơn vị cơ bản, đơn vị lớn": ["qty_sold"],
    "Size M, size L": ["qty_sold"], "Nhóm hàng": ["group_revenue"], "Giá trị trung bình một đơn": ["avg_order_value"],
    "Khách trung thành nhất": ["loyal_customer"], "Khách sộp nhất": ["big_spender"],
    "Khách hay đổi ý nhất": ["fickle_customer"], "Mặt hàng được ưa chuộng nhất": ["popular_item"],
    "Ngày đắt khách nhất": ["busiest_day"], "Giờ cao điểm": ["peak_hour"], "Xác định khách": [
        "customer_most_orders", "loyal_customer", "big_spender", "fickle_customer",
        "top_spender", "customers_min_orders"],
    "Thuốc kê đơn": ["rx_revenue"], "Doanh thu theo hoạt chất": ["ingredient_revenue"],
    "Tiền nhập hàng trong kỳ": ["import_avg_price"], "Giá nhập bình quân một mặt hàng trong tháng": ["import_avg_price"],
}
# Các hàm dùng cùng định nghĩa phải cùng bị chặn nếu luật private thay đổi.
PLAN_FUNCS["Bán chạy"] += ["best_item", "top_items"]
PLAN_FUNCS["Nhóm hàng"] += ["group_order_count"]
PLAN_FUNCS["Xác định khách"] += ["customer_count", "customer_revenue", "customer_order_count",
                                  "customer_avg_order_value"]
PLAN_FUNCS["Xác định khách"] += ["top_customers"]
PLAN_FUNCS["Giá trị trung bình một đơn"] += ["customer_avg_order_value"]
PLAN_FUNCS["Doanh thu theo hình thức thanh toán"] += ["payment_order_count"]
PLAN_FUNCS["Mặt hàng"] = ["item_order_count"]
PLAN_FUNCS["Khách thành viên mới trong tháng"] = ["new_members"]
PLAN_FUNCS["Nhà cung cấp chính"] = ["main_supplier"]
PLAN_FUNCS["Tiền nhập hàng trong kỳ"] += ["main_supplier"]
PLAN_FUNCS["Tiền nhập hàng trong kỳ"] += ["import_total"]
PLAN_FUNCS["Lãi gộp"] = ["gross_profit", "net_profit"]
PLAN_FUNCS["Lãi ròng một tháng"] = ["expense_total", "net_profit"]
PLAN_FUNCS["Giá vốn"] = ["gross_profit", "net_profit"]
PLAN_FUNCS["Doanh thu theo nhóm khách"] = ["customer_group_revenue"]
PLAN_FUNCS["Nhân viên bán giỏi nhất"] = ["staff_best"]
PLAN_FUNCS["Nhân viên bán giỏi nhất"] += ["staff_revenue"]
PLAN_FUNCS["Xác định khách"] += ["customer_group_revenue"]
PLAN_FUNCS["Hàng lời nhất"] = ["best_profit_item"]
PLAN_FUNCS["Lãi gộp"] += ["best_profit_item"]
PLAN_FUNCS["Giá vốn"] += ["best_profit_item"]
PLAN_FUNCS["Mặt hàng không bán được trong kỳ (ế)"] = ["unsold_items", "unsold_count"]
PLAN_FUNCS["Tiền giảm voucher, tiền giảm nhờ dùng điểm, phí ship"] = ["order_adjustment_total"]
PLAN_FUNCS["Thu máy cũ"] = ["order_adjustment_total"]
PLAN_FUNCS["Hiệu ứng tăng giá"] = ["price_effect"]
_DEBT_FUNCS = ["debt_total", "debtor_count", "debt_added", "debt_repaid", "debt_by_age"]


def norm(text: str) -> str:
    return " ".join(re.sub(r"[*`_]", "", fold(text)).split())


def rule_table(rulebook: str) -> dict[str, str]:
    """Các dòng bảng luật '| Tên luật | Nội dung |' → {tên: nội dung}."""
    out = {}
    for line in rulebook.splitlines():
        m = _ROW.match(line.strip())
        if m and not set(m.group(1)) <= set("-: ") and m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    return out


def _known() -> dict[str, list[str]]:
    p = Path(__file__).with_name("known_rules.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def changed_rules(rulebook: str, known: dict | None = None) -> list[str]:
    """Dòng luật code đã cài mà rulebook này viết khác đi (so sau khi bỏ dấu, định dạng)."""
    known = _known() if known is None else known
    have = {norm(k): v for k, v in rule_table(rulebook).items()}
    out = []
    for k, texts in known.items():
        v = have.get(norm(k))
        if v is not None and norm(v) not in texts:
            out.append(k)
    return out


class Guard:
    """Phần nào của tầng tất định / kế hoạch còn dùng được với rulebook này."""

    def __init__(self, rulebook: str, known: dict | None = None, policy_aware: bool = False):
        """policy_aware: nơi gọi tính theo `rule_policy.Policy` (OrderView + Periods của tầng tất định) → dòng luật đổi
        mà Policy diễn đạt được không bị chặn. Nơi khác (SEM, kế hoạch, agent) vẫn chặn mọi dòng đổi."""
        from datamate.rule_policy import infer
        self.policy, resolved = infer(rulebook, known)
        if not policy_aware:
            resolved = set()
        self.changed = [k for k in changed_rules(rulebook, known) if k not in resolved]
        ch = set(self.changed)
        grp = {g for g, keys in GROUPS.items() if ch & set(keys)}
        self.core = "core" in grp
        self.debt = "debt" in grp
        self.delivery = "delivery" in grp
        self.lot = "lot" in grp
        # luật khác của bảng đơn (ca, nhóm, khách…) đổi: bộ giải regex không tách được theo luật → tắt cả tầng 1
        self.view = self.core or bool(ch & set(PLAN_FUNCS))
        blocked = {f for k in ch for f in PLAN_FUNCS.get(k, [])}
        if self.debt:
            blocked |= set(_DEBT_FUNCS)
        if self.delivery:
            blocked.add("delivery_count")
        self.plan_blocked = blocked
        self.plan = not self.core

    def summary(self) -> str:
        if not self.changed:
            return "rulebook khớp các luật đã cài"
        return f"{len(self.changed)} luật đổi so với bản đã cài: " + "; ".join(self.changed)
