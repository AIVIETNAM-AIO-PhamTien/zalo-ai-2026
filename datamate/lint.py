"""Bắt lỗi code hay gặp ngay khi agent chạy (không phải thêm luật vào prompt): soi đoạn code vừa chạy và dữ liệu
thật của shop, có dấu hiệu sai chắc chắn thì nối một dòng nhắc vào kết quả chạy.

Lỗi lấy từ log agent (train + phân tích lỗi public): tự liệt kê giá trị trạng thái thay vì `valid_order`,
so `==` một ngày với cột ngày kèm giờ, so tên bằng `==` / `str.contains` thay vì `name_mask`.
Chỉ nhắc khi chắc (cột thật sự là trạng thái / thật sự có giờ / thật sự là cột tên) để tránh làm model sửa code đúng.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold


@dataclass
class LintHints:
    status_cols: set[str] = field(default_factory=set)     # cột có giá trị huỷ / trả
    timed_date_cols: set[str] = field(default_factory=set)  # cột ngày có giờ khác 00:00
    name_cols: set[str] = field(default_factory=set)        # cột tên hàng / khách / nhân viên (nhiều giá trị chữ)
    gold_tables: set[str] = field(default_factory=set)      # bảng G dựng được cho shop (run.py điền sau khi dựng sẵn G)


def build_hints(shop: Shop) -> LintHints:
    from datamate.semantic import classify_status  # tránh import vòng
    h = LintHints()
    for df in shop.tables.values():
        for c in df.columns:
            s = df[c].dropna()
            if s.empty:
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                if ((s.dt.hour != 0) | (s.dt.minute != 0)).any():
                    h.timed_date_cols.add(c)
                continue
            if not (pd.api.types.is_string_dtype(s) or s.dtype == object):
                continue
            vals = s.astype(str).unique()
            if len(vals) <= 50 and any(classify_status(v) in ("cancelled", "returned") for v in vals):
                h.status_cols.add(c)
            elif len(vals) > 50 and re.search(r"\b(ten|hang|mon|san pham|khach|nhan vien|nv|tho)\b", fold(c)) \
                    and not re.match(r"(ma|so|id)\b|ma[A-Z]", fold(c) if not str(c).startswith("Ma") else "ma"):
                h.name_cols.add(c)
    return h


def _cols_in(line: str, cols: set[str]) -> list[str]:
    return [c for c in cols if f"'{c}'" in line or f'"{c}"' in line]


def lint(code: str, hints: LintHints) -> list[str]:
    """Các dòng nhắc (mỗi loại tối đa một), rỗng nếu không thấy gì chắc chắn sai."""
    out: dict[str, str] = {}
    for line in code.splitlines():
        if line.lstrip().startswith("#"):
            continue
        status = _cols_in(line, hints.status_cols)
        # `== 'Đã huỷ'` để đếm đơn huỷ là hợp lệ; chỉ nhắc khi tự liệt kê ≥ 2 giá trị (tập đơn hợp lệ) hoặc `!=`
        listed = re.search(r"\.isin\(\s*\[\s*['\"][^\]]*,\s*['\"]", line) or re.search(r"!=\s*['\"]", line)
        if status and listed and "valid_order" not in line and "order_status" not in line:
            out["status"] = (f"Lưu ý: cột '{status[0]}' là cột trạng thái đơn — lọc đơn hợp lệ bằng "
                             f"`valid_order(df['{status[0]}'])` (hoặc `order_status(...)`), không tự liệt kê giá trị "
                             "trạng thái (dễ sót cách ghi khác, ô trống).")
        timed = _cols_in(line, hints.timed_date_cols)
        if timed and re.search(r"==\s*(TODAY|pd\.Timestamp|P\.|['\"]\d{4}-\d{2}-\d{2})", line) and ".dt." not in line:
            out["date"] = (f"Lưu ý: cột '{timed[0]}' có kèm giờ nên so `==` với một ngày gần như luôn ra rỗng — "
                           f"dùng `P.mask(df['{timed[0]}'], P.day(...))` / `P.mask(..., P.today())`.")
        names = _cols_in(line, hints.name_cols)
        if names and "name_mask" not in line and re.search(r"[!=]=\s*['\"][^'\"]{3,}['\"]|\.str\.contains\(\s*['\"]", line):
            out["name"] = (f"Lưu ý: so tên trên cột '{names[0]}' bằng `==` / `str.contains` dễ sót cách ghi khác — "
                           f"dùng `find_values(...)` rồi `name_mask(df['{names[0]}'], ten)`.")
        # rulebook 'nhiều nhất / top k': số đo giảm, hoà thì mã tăng — idxmax / nlargest / sort một khoá rồi lấy đầu
        # không phá hoà theo mã (log Colab: lỗi tie_break)
        if "top_k" not in line and (re.search(r"\.(idxmax|idxmin|nlargest|nsmallest)\(", line) or (
                re.search(r"\.sort_values\(\s*(by\s*=\s*)?['\"]", line)
                and re.search(r"\.(iloc\[0\]|head\(|index\[0\])", line))):
            out["tie"] = ("Lưu ý: xếp hạng / 'nhiều nhất' phải phá hoà theo rulebook (số đo giảm, bằng nhau thì mã tăng): "
                          "dùng `top_k(df, so_do, ma, k)` thay cho idxmax / nlargest / sort một khoá rồi lấy đầu.")
        # cộng cột gold có NaN (thiếu tiền / thiếu hệ số) bằng .sum() trần coi ô thiếu là 0
        if re.search(r"G\[['\"](don|dong|tien_\w+)['\"]\]", line) and re.search(r"\[['\"](revenue|bqty|amount)['\"]\]\.sum\(\s*\)", line):
            out["nan"] = ("Lưu ý: cột revenue / bqty / amount của G có thể NaN (thiếu dữ liệu) — kiểm `.isna().any()` "
                          "trước khi cộng; có NaN thì không trả số (không coi là 0).")
    # log Colab B9: 63/67 câu tính doanh thu / số lượng trên bảng thô T dù G có sẵn, và mọi câu sai đều lệch lên (đếm cả
    # đơn huỷ / dòng lặp §8). Nhắc khi cả đoạn code cộng / đếm trên T mà không đọc G, và G có bảng đơn
    if "don" in hints.gold_tables and "G[" not in code and re.search(r"\bT\[", code) \
            and re.search(r"\.(sum|nunique|count)\(", code):
        out["gold"] = ("Lưu ý: G['don'] / G['dong'] / G['tien_*'] đã áp sẵn rulebook (đơn huỷ / trả, dòng lặp, quy đổi, "
                       "đơn sau hôm nay). Doanh thu, số lượng, số đơn, tiền theo mặt hàng / nhóm: tính lại trên G và so với "
                       "kết quả từ T; lệch thì tin G. Chỉ dùng T cho thông tin G không có.")
    return list(out.values())
