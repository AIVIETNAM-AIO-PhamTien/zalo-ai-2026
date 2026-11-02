"""Đoán trước kiểu đáp án từ câu hỏi (kiểu SemEval-2025 Task 8: dự đoán kiểu rồi mới sinh code).

Luật trên câu đã bỏ dấu; không chắc thì trả None. Dùng để nhắc model trong prompt và để gom các đáp án
cùng nghĩa khác kiểu khi bỏ phiếu. Độ chính xác đo trên nhãn kiểu của 3000 câu train (scripts/eval_answer_type.py).
"""
from __future__ import annotations

import re
import unicodedata

_N = r"(\d{1,2}|hai|ba|bon|nam|muoi)"  # 1–2 chữ số: '2025 hàng nào' là năm, không phải 2025 mặt hàng
_Q = r"(bao nhieu|bao nhiu|bnhiu|bnhieu|bn|may|nhieu|nhiu)"
_WHO = r"(hang|mon|mat hang|san pham|sp|khach|khach hang|ai|nha cung cap|ncc|nhan vien|nv|tho|chi nhanh|ca|kenh|nhom|loai|thuoc|hoat chat|dong may|hang xe|dong xe|xe|thu|hinh thuc)"


def _norm(q: str) -> str:
    """Bỏ dấu, lowercase, giữ % (fold() bỏ mất % nên không dùng)."""
    s = unicodedata.normalize("NFD", q.lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d")
    s = re.sub(r"[^a-z0-9%\s]", " ", s)
    return " ".join(s.split())


def expected_type(question: str) -> str | None:
    """number / percent / date / string / list / table / boolean, hoặc None khi không chắc."""
    s = _norm(question)
    # A count of entities remains a number even when the qualifying set is a
    # top-k or contains a percentage threshold. Do not override compound asks.
    count_q = r'(?:bao nhieu|bao nhiu|bnhiu|bnhieu|bn|may)'
    count = re.search(rf'\b{count_q}\s+{_WHO}\b', s)
    if (count and len(re.findall(rf'\b{count_q}\s+{_WHO}\b', s)) == 1
            and not re.search(r'\b(lan luot|moi ben|moi nhom|liet ke|bang|ty le|ti le|ti trong|ty trong|chiem)\b|\bchan\b.*\ble\b', s)):
        return 'number'
    many = re.search(rf"\btop {_N}\b|\b{_N} {_WHO}\b.*\b(nhat|nao)\b|\bnhung {_WHO}\b|\bcac {_WHO} nao\b|\b{_WHO} nao\b.*\bcon ton\b", s)
    # bảng: nhiều đối tượng kèm số đo của từng đối tượng
    if many and re.search(r"\b(kem|moi nguoi|tung nguoi|cua tung|moi khach|moi mat hang|moi mon)\b|\bva so\b", s):
        return "table"
    # nhiều số trong một câu: 'ghi hai số', 'mỗi tháng / mỗi bên / mỗi loại', 'nợ thêm … trả được …',
    # 'mấy hộp chẵn, mấy viên lẻ', 'tuổi nợ 0–7, 8–15…', 'X là bao nhiêu, Y là bao nhiêu'
    if re.search(rf"\b(ghi|tra loi) ({_N}|cac) so\b|\blan luot\b|\btheo thu tu\b|\bmoi (thang|ben|nhom)\b|\btuoi no\b"
                 r"|\bmoi loai\b(?! do uong)|\bno them\b.*\btra\b|\bchan\b.*\ble\b", s) \
            or len(re.findall(rf"\bla {_Q}\b", s)) >= 2:
        return "list"
    if many:
        return "list"
    if re.search(r"\b(tang lien tuc|giam lien tuc|luon cao hon|luon thap hon|co phai|dung khong|phai khong)\b", s):
        return "boolean"
    # '%' dính sau chữ số là một phần tên hàng ('Otrivin 0,1%'), không phải hỏi phần trăm
    if re.search(r"(?<![0-9])%|\bphan tram\b|\b(ti|ty) (le|trong)\b", s):
        return "percent"
    if re.search(r"\b(ngay|thang) nao\b", s):
        return "date"
    if re.search(rf"\b{_WHO} nao\b|\bla ai\b|\bai la\b|\bai\b.*\bnhat\b|\bthu may\b|\bla (gi|chi)\b", s):
        return "string"
    if re.search(rf"\b{_Q}\b", s):
        return "number"
    return None
