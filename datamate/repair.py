"""Sửa tất định sau khi agent chạy: code đọc bảng thô T cho câu bán hàng mà quên lọc đơn hợp lệ (huỷ / trả hàng)
thì chạy lại đúng code đó trên T đã lọc. Lỗi thật trên log Colab v4 strange2: biz_101 tương ớt Conbo, agent cộng cả
đơn huỷ và đơn trả (1,654,000 thay vì 1,546,000). Khác phủ quyết của arbiter: không bỏ đáp án, chỉ sửa đầu vào."""
from __future__ import annotations

from pandas.api.types import is_object_dtype, is_string_dtype

from datamate import helpers
from datamate.arbiter import _VALID, required
from datamate.parsing import fold

# câu hỏi chính nó cần đơn huỷ / trả / nợ / trạng thái → không được lọc
_KEEP = ("huy", "tra hang", "hoan", "no ", " no", "cong no", "tra no", "dang giao", "trang thai", "tat ca don")


def needs_repair(question: str, code: str) -> bool:
    if "valid_order" not in required(question) or _VALID.search(code or ""):
        return False
    s = " " + fold(question) + " "
    return not any(k in s for k in _KEEP)


def status_columns(df) -> list[str]:
    """Cột trạng thái đơn: ít giá trị khác nhau và có giá trị huỷ / trả."""
    out = []
    for c in df.columns:
        s = df[c]
        if not (is_string_dtype(s) or is_object_dtype(s)) or s.nunique(dropna=True) > 12:
            continue
        try:
            st = set(helpers.order_status(s.astype(str)).unique())
        except Exception:
            continue
        if st & {"cancelled", "returned"}:
            out.append(c)
    return out


def filter_tables(tables: dict) -> dict:
    """Bản sao T chỉ giữ dòng của đơn hợp lệ; bảng không có cột trạng thái giữ nguyên."""
    res = {}
    for name, df in tables.items():
        cols = status_columns(df)
        if cols:
            mask = helpers.valid_order(df[cols[0]].astype(str))
            df = df[mask].copy()
        res[name] = df
    return res
