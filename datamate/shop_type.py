"""Ngành của shop suy từ chữ trong dữ liệu (tên hàng, nhóm hàng, README), không theo mã shop.

Dùng để chia fold "bỏ một ngành" khi đo và để few-shot không lấy ví dụ cùng ngành với shop đang hỏi (shop private
có thể thuộc ngành chưa gặp). Ngành không nhận ra → "khac".
"""
from __future__ import annotations

import re

import pandas as pd

from datamate.parsing import fold

KEYWORDS = {
    "nha_thuoc": r"\b(thuoc|hoat chat|siro|vien sui|vi x|ke don|vitamin|kem boi|xit mui|nho mat|men vi sinh)\b",
    "vat_lieu": r"\b(xi mang|thep|son lot|son nuoc|gach|cat (xay|be tong|san lap)|da 1x2|ong pvc|cadivi|keo dan|"
                r"chong tham|ngoi|luoi b40|cong tac|bong den|day dien)\b",
    "ca_phe": r"\b(ca phe|tra sua|espresso|latte|cappuccino|cold brew|croissant|tra dao|sinh to|nuoc ep|tiramisu|"
              r"bac xiu|matcha)\b",
    "quan_an": r"\b(com chien|goi|nuong|lau|canh chua|ga chien|chien nuoc mam|com trang|hai san|xao|bo la lot)\b",
    "tap_hoa": r"\b(bia|dau an|bot ngot|nep|thuoc la|banh oreo|mi hao hao|nuoc suoi|dau goi|bot giat|nuoc mam|"
               r"sua tuoi|coca|pepsi|gao)\b",
}


def _texts(shop, limit: int = 3000) -> list[str]:
    out = [shop.readme or ""]
    for df in shop.tables.values():
        for c in df.columns:
            s = df[c].dropna()
            if s.empty or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                continue
            uniq = s.astype(str).unique()
            if len(uniq) <= limit:
                out.extend(uniq)
    return out


def shop_type(shop) -> str:
    """Ngành có nhiều giá trị chữ khớp từ khoá nhất; hoà hoặc quá ít bằng chứng → 'khac'."""
    score = {k: 0 for k in KEYWORDS}
    for t in _texts(shop):
        f = fold(t)
        for k, pat in KEYWORDS.items():
            if re.search(pat, f):
                score[k] += 1
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    if ranked[0][1] < 5 or ranked[0][1] == ranked[1][1]:
        return "khac"
    return ranked[0][0]
