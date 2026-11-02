"""Luật rulebook → tham số của phép tính tất định (Bước 4, docs/KE_HOACH_PA_V1.md).

Một số dòng luật private có thể đổi theo vài phương án rời rạc (đơn đang giao có tính doanh thu không, "N ngày gần
nhất" có gồm hôm nay không). `infer` đọc dòng luật đã đổi, nhận ra phương án thì trả tham số tương ứng; dòng đổi mà
không khớp trọn vẹn một phương án thì để `rule_guard` chặn như cũ. Nhận diện thận trọng: thiếu một mệnh đề là bỏ.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datamate.rule_guard import changed_rules, norm, rule_table


@dataclass(frozen=True)
class Policy:
    delivering_valid: bool = True  # đơn đang giao tính vào doanh thu / số đơn (luật train)
    last_n_include_today: bool = True  # "N ngày gần nhất" kể cả hôm nay (luật train)
    order_discount: bool = False  # doanh thu một đơn trừ giảm giá trên cả đơn
    ship_fee: bool = False  # doanh thu một đơn cộng phí giao hàng
    aov_floor: bool = False  # giá trị trung bình một đơn làm tròn xuống (train: nửa đồng làm tròn lên)
    dedup_same_order: bool = True  # §8: hai dòng cùng đơn giống nhau ở mọi ô tính một lần (train); False: tính cả hai
    merge_name_variants: bool = True  # §3: tên khách chỉ khác cách viết là một khách (train); False: mỗi cách viết một khách


def _valid_orders(text: str) -> dict | None:
    """'Đơn tính vào doanh thu': vẫn bỏ đơn huỷ và cả đơn có trả hàng (kể cả trả một phần); chỉ đọc đơn đang giao."""
    t = norm(text)
    if not (re.search(r"\bhuy\b", t) and re.search(r"\btra hang\b", t) and re.search(r"\bmot phan\b", t)):
        return None
    m = re.search(r"dang giao[^.;]*", t)
    if m is None:
        return None
    # mệnh đề chứa 'đang giao' và phần còn lại của câu: 'đơn huỷ, đơn trả … và đơn đang giao đều không tính'
    clause = t[max(t.rfind(".", 0, m.start()), t.rfind(";", 0, m.start())) + 1:m.end()]
    if re.search(r"\b(khong|chua) (duoc )?tinh\b|\bloai\b|\bbo\b", clause):
        return {"delivering_valid": False}
    if re.search(r"\bvan tinh\b|\b(duoc|co) tinh\b", clause):
        return {"delivering_valid": True}
    return None


def _last_n(text: str) -> dict | None:
    t = norm(text)
    excl = re.search(r"\bkhong (tinh|gom|ke) hom nay\b|\btruoc hom nay\b|\bngay cuoi la hom qua\b", t)
    incl = re.search(r"\bke ca hom nay\b|\bngay cuoi la hom nay\b|\bgom ca hom nay\b", t)
    if excl and not incl:
        return {"last_n_include_today": False}
    if incl and not excl:
        return {"last_n_include_today": True}
    return None


def _order_revenue(text: str) -> dict | None:
    """'Doanh thu một đơn': vẫn là thành tiền các dòng (đã trừ giảm giá dòng), không cộng VAT; đọc hai khoản cấp đơn."""
    t = norm(text)
    if not (re.search(r"thanh tien (cac|cua) dong", t) and re.search(r"khong cong thue vat|khong gom (thue )?vat", t)):
        return None
    got = {}
    for key, pat in (("order_discount", r"giam gia tren ca don"), ("ship_fee", r"phi (giao hang|giao|ship)")):
        hits = list(re.finditer(r"(\w+) (\w+) (?:" + pat + r")\b", t))
        if len(hits) != 1:
            return None  # không nhắc (hoặc nhắc nhiều lần): không đoán
        before, verb = hits[0].group(1), hits[0].group(2)
        if verb not in ("tru", "cong", "gom"):
            return None
        got[key] = before != "khong"
    return got


def _aov(text: str) -> dict | None:
    t = norm(text)
    if re.search(r"\blam tron xuong\b", t) and not re.search(r"\blen\b", t):
        return {"aov_floor": True}
    if re.search(r"\bnua dong lam tron len\b", t):
        return {"aov_floor": False}
    return None


def _dedup(text: str) -> dict | None:
    # chỉ đọc vế về CÙNG một đơn: vế 'hai dòng giống nhau của hai đơn khác nhau … tính cả hai' là luật khác
    t = re.sub(r"\bhai dong giong nhau cua hai don khac nhau\b.*", "", norm(text))
    once = re.search(r"\btinh mot lan\b|\bchi tinh mot\b|\bbo (dong )?trung\b", t)
    both = re.search(r"\btinh ca hai\b|\bmoi dong (la )?mot lan\b|\bkhong bo\b", t)
    if once and not both:
        return {"dedup_same_order": True}
    if both and not once:
        return {"dedup_same_order": False}
    return None


def _customer_identity(text: str) -> dict | None:
    t = norm(text)
    same = re.search(r"chi khac cach viet (la )?(cung|mot)", t)
    diff = re.search(r"khac cach viet (cung )?(la )?hai|moi cach viet (la )?(mot|rieng)", t)
    if same and not diff:
        return {"merge_name_variants": True}
    if diff and not same:
        return {"merge_name_variants": False}
    return None


READERS = {"Đơn tính vào doanh thu": _valid_orders, "N ngày gần nhất": _last_n,
           "Doanh thu một đơn": _order_revenue, "Giá trị trung bình một đơn": _aov,
           "Hai dòng của **cùng một đơn** giống nhau ở mọi ô": _dedup, "Xác định khách": _customer_identity}


def infer(rulebook: str, known: dict | None = None) -> tuple[Policy, set[str]]:
    """(tham số, các dòng luật đã đổi mà tham số này diễn đạt được)."""
    rows = {norm(k): v for k, v in rule_table(rulebook).items()}
    kwargs, resolved = {}, set()
    for name in changed_rules(rulebook, known):
        reader = READERS.get(name)
        got = reader(rows[norm(name)]) if reader else None
        if got is not None:
            kwargs.update(got)
            resolved.add(name)
    return Policy(**kwargs), resolved


_DESCRIBE = {
    "delivering_valid": ("đơn đang giao", "tính vào doanh thu / số đơn", "KHÔNG tính vào doanh thu / số đơn"),
    "last_n_include_today": ("'N ngày gần nhất'", "gồm hôm nay", "KHÔNG gồm hôm nay (kết thúc hôm qua)"),
    "order_discount": ("giảm giá trên cả đơn", "TRỪ vào doanh thu đơn", "không trừ"),
    "ship_fee": ("phí giao hàng", "CỘNG vào doanh thu đơn", "không cộng"),
    "aov_floor": ("giá trị trung bình một đơn", "làm tròn XUỐNG về đồng", "nửa đồng làm tròn lên"),
    "dedup_same_order": ("hai dòng cùng đơn giống nhau ở mọi ô", "tính một lần", "tính CẢ HAI"),
    "merge_name_variants": ("tên khách chỉ khác cách viết", "là một khách", "là HAI khách khác nhau"),
}


def policy_hint(rulebook: str) -> str:
    """Các luật của split khác bản train mà Policy đã đọc được, viết thành câu cho prompt (H3: phương án đã chọn).
    Rỗng khi giống train. G / SEM đã áp sẵn các phương án này."""
    pol, resolved = infer(rulebook)
    base = Policy()
    out = []
    for f, (what, yes, no) in _DESCRIBE.items():
        v = getattr(pol, f)
        if v != getattr(base, f):
            out.append(f"- {what}: {yes if v else no}")
    if not out:
        return ""
    return ("Rulebook của cửa hàng này KHÁC luật thường gặp ở các điểm sau (G / SEM đã áp sẵn; tự tính thì theo đúng):\n"
            + "\n".join(out))
