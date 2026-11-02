"""Bộ phát hiện câu không trả lời được, tất định (kiểu TrustSQL: thành phần riêng, không để LLM tự quyết).

Ba nhóm lấy từ 70 câu unanswerable của train:
1. kỳ hỏi nằm hẳn trước ngày đầu tiên có dữ liệu bán hàng;
2. câu chủ quan / cần hiểu biết bên ngoài (tay nghề, lễ phép, liều dùng…);
3. hỏi thông tin shop không ghi: giới tính khách, địa chỉ khách, tồn kho, nhập hàng.
Chỉ kết luận khi chắc; không chắc trả None để semantic / agent xử lý. Kiểm trên 3000 câu train: sai phải = 0.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold

_SUBJECTIVE = re.compile(r"\b(lanh nghe|tay nghe|le phep|chu dao|tan tam|tan tinh|quy men|khen|than thien|nhiet tinh"
                         r"|de thuong|dep trai|xinh|nen uong|lieu dung|moi lan may vien|co tot khong|uy tin|kho tinh nhat"
                         r"|chat luong tot|ba bau|co thai|cho con bu)\b")
# hỏi lý do / nguyên nhân (log Colab B9 v4 public: 'huỷ đơn chủ yếu vì sao', 'huỷ vì giao trễ', 'trả món vì món lỗi
# hay vì gọi nhầm'); chỉ khi shop không có cột lý do / ghi chú
_REASON_Q = re.compile(r"\b(vi sao|tai sao|ly do|nguyen nhan|huy vi|tra (mon|hang|lai) vi|vi (giao tre|mon bi loi|goi nham))\b")
_GENDER_Q = re.compile(r"\bkhach (nu|nam)\b")
_ADDRESS_Q = re.compile(r"\b(o|nha o|tai) duong\b|\bkhach (o )?duong\b")
_STOCK_Q = re.compile(r"\b(ton kho|con lai .* trong kho|trong kho con|kho con|con ton|gia tri hang con trong kho)\b")
_IMPORT_Q = re.compile(r"\b(nhap hang|nhap vao|gia nhap|tien nhap|nhap .* binh quan)\b")


def _has(names: list[str], pattern: str) -> bool:
    return any(re.search(pattern, n) for n in names)


@dataclass
class ShopProfile:
    first_day: dt.date | None
    gender: bool
    address: bool
    stock: bool
    imports: bool
    reason: bool = False
    names: frozenset = frozenset()  # từ trong tên hàng / nhóm / khách: 'xinh', 'khen' là tên thì không phải câu chủ quan
    columns: frozenset = frozenset()

    @classmethod
    def from_shop(cls, shop: Shop, first_day: dt.date | None = None) -> "ShopProfile":
        tables = [fold(t) for t in shop.tables]
        cols = [fold(str(c)) for df in shop.tables.values() for c in df.columns]
        readme = fold(shop.readme)
        values, names = set(), set()
        for df in shop.tables.values():
            for c in df.columns:
                s = df[c].dropna()
                if not (pd.api.types.is_string_dtype(s) or s.dtype == object):
                    continue
                uniq = s.unique()
                if 0 < len(uniq) <= 4:
                    values.update(fold(str(v)) for v in uniq)
                for v in uniq:
                    names.update(fold(str(v)).split())
        return cls(
            first_day=first_day,
            gender=_has(cols, r"gioi ?tinh|gender|nam ?nu|\bphai\b|\bsex\b") or {"nam", "nu"} <= values,
            address=_has(cols, r"dia ?chi|duong|address|phuong|quan huyen|\bkhu vuc\b"),
            stock=_has(tables + cols, r"ton|kho|stock|inventory"),
            imports=_has(tables + cols, r"nhap|import|purchase|ncc|nha cung cap"),
            reason=_has(cols, r"ly do|nguyen nhan|ghi chu|note|reason|mo ta|dien giai"),
            names=frozenset(names),
            columns=frozenset(cols),
        )


def check(question: str, as_of: dt.date, prof: ShopProfile, calendar_csv=None) -> dict | None:
    """Đáp án unanswerable nếu luật chắc chắn, None nếu không."""
    from datamate.baseline import extract_periods, norm  # tránh import vòng
    from datamate.helpers import Periods
    periods, _ = extract_periods(norm(question), Periods(as_of, calendar_csv))
    reason = detect(question, periods, prof)
    return {"type": "unanswerable", "reason": reason} if reason else None


def detect(question: str, periods: list, prof: ShopProfile) -> str | None:
    """Lý do không trả lời được, hoặc None. `periods`: [(nhãn, (đầu, cuối))] đã đọc từ câu hỏi."""
    s = " ".join(re.sub(r"[^a-z0-9%\s]", " ", fold(question).replace("đ", "d")).split())
    def has_direct_column(phrase):
        key = re.sub(r'[^a-z0-9]', '', phrase)
        return any(key in re.sub(r'[^a-z0-9]', '', col) for col in prof.columns)

    if any(m.group(1) not in prof.names and not has_direct_column(m.group(1)) for m in _SUBJECTIVE.finditer(s)):
        return "câu hỏi chủ quan / cần hiểu biết ngoài dữ liệu"
    if _REASON_Q.search(s) and not prof.reason:
        return "dữ liệu không ghi lý do"
    if _GENDER_Q.search(s) and not prof.gender:
        return "dữ liệu không ghi giới tính khách"
    if _ADDRESS_Q.search(s) and not prof.address:
        return "dữ liệu không ghi địa chỉ khách"
    if _STOCK_Q.search(s) and not prof.stock:
        return "dữ liệu không có sổ kho / tồn kho"
    if _IMPORT_Q.search(s) and not prof.imports:
        return "dữ liệu không có sổ nhập hàng"
    # kỳ đọc sai từ ngày đầy đủ ('09/05/2026' bị đọc thành tháng 9/2005) → năm vô lý → không dùng luật này
    if prof.first_day and periods and all(p[0] != "BAD" and p[1][0].year >= 2015 and p[1][1] < prof.first_day
                                          for p in periods):
        return "kỳ hỏi nằm trước ngày đầu tiên có dữ liệu"
    return None
