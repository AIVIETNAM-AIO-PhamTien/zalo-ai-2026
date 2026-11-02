"""Shop brief: mô tả ngắn gọn từng bảng đã nạp (cột, kiểu, khoảng giá trị, giá trị phân loại)
để LLM viết code đúng cột ngay từ đầu (kiểu "schema scouting" của các data agent)."""
from __future__ import annotations

import re

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold

MAX_CATEGORIES = 25


def _fmt_num(x) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    return f"{int(x):,}" if x.is_integer() else f"{x:,.2f}"


def _sources(srcs: list[str]) -> str:
    if len(srcs) <= 3:
        return ", ".join(srcs)
    return f"{srcs[0]} … {srcs[-1]} ({len(srcs)} sheet/file gộp lại, cột `_sheet` ghi nguồn)"


def describe_column(s: pd.Series, kind: str) -> str:
    n = len(s)
    nn = int(s.notna().sum())
    head = f"`{s.name}` ({kind}"
    if nn < n:
        head += f", trống {n - nn}"
    head += ")"
    if nn == 0:
        return head
    v = s.dropna()
    if kind.startswith(("int", "float")):
        return f"{head}: {_fmt_num(v.min())} → {_fmt_num(v.max())}"
    if kind == "date":
        has_time = (v.dt.hour != 0).any() or (v.dt.minute != 0).any()
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        return f"{head}: {v.min():{fmt}} → {v.max():{fmt}}"
    vc = v.astype(str).value_counts()
    if len(vc) <= MAX_CATEGORIES:
        vals = "; ".join(f"{k} ({c})" for k, c in vc.items())
        return f"{head}: {len(vc)} giá trị: {vals}"
    ex = "; ".join(map(str, vc.index[:4]))
    return f"{head}: {len(vc)} giá trị khác nhau, ví dụ: {ex}"


def build_brief(shop: Shop) -> str:
    lines = [f"# Dữ liệu đã nạp của {shop.biz_id}",
             "Các bảng nằm trong dict `T` (pandas DataFrame, đã đổi số/ngày bẩn sang kiểu chuẩn)."]
    from datamate.semantic import order_id_columns  # tránh import vòng
    oid = order_id_columns(shop)
    for name, df in shop.tables.items():
        m = shop.meta[name]
        lines.append(f"\n## T[{name!r}] — {m['rows']:,} dòng — nguồn: {_sources(m['sources'])}")
        notes = []
        if m["subtotal_rows_dropped"]:
            notes.append(f"đã bỏ {m['subtotal_rows_dropped']} dòng tổng phụ (Tổng/Cộng ngày …)")
        if m.get("note_rows_dropped"):
            notes.append(f"đã bỏ {m['note_rows_dropped']} dòng ghi chú một ô")
        if m["continuation_rows"]:
            cols = ", ".join(f"`{c}`" for c in m.get("continuation_cols", []))
            notes.append(f"{m['continuation_rows']} dòng nối tiếp của nhóm phía trên: trong `T` các cột {cols} "
                         "ĐÃ được điền theo dòng đầu nhóm (rulebook §8), không cần ffill lại")
        if m["unparsed"]:
            notes.append("ô không đọc được (thành NA): " + ", ".join(f"{c}: {k}" for c, k in m["unparsed"].items()))
        grain = row_grain(df, oid.get(name))
        if grain:
            notes.insert(0, grain)
        if notes:
            lines.append("Ghi chú: " + "; ".join(notes))
        for c in df.columns:
            kind = m["types"].get(c, "text")
            lines.append("- " + describe_column(df[c], kind))
    keys = join_keys(shop)
    if keys:
        lines.append("\n## Cột nối giữa các bảng (cùng giá trị mã)")
        lines += [f"- {k}" for k in keys]
    return "\n".join(lines)


_KEY_NAME = re.compile(r"\b(ma|so|id|code)\b|^ma|id$")


def row_grain(df: pd.DataFrame, order_col: str | None) -> str | None:
    """'Mỗi dòng là gì' (DS-STAR / KGMON: biết hạt dữ liệu trước khi cộng): một đơn hay một dòng hàng của đơn."""
    if order_col is None or order_col not in df.columns:
        return None
    ids = df[order_col].dropna()
    if ids.empty:
        return None
    per = len(ids) / ids.nunique()
    if per <= 1.0001:
        return f"mỗi dòng một `{order_col}` khác nhau (một dòng = một đơn / phiếu)"
    return (f"một `{order_col}` có thể nhiều dòng (trung bình {per:.2f} dòng): đếm đơn bằng `nunique()`; "
            "cột tiền của cả đơn lặp trên từng dòng thì không cộng theo dòng")


def join_keys(shop: Shop, limit: int = 12) -> list[str]:
    """Nhóm cột mã ở các bảng có tập giá trị gần trùng nhau (≥ 80% giá trị bảng nhỏ có trong bảng lớn)."""
    cand = []
    for t, df in shop.tables.items():
        for c in df.columns:
            if not _KEY_NAME.search(fold(str(c)).replace("_", " ")) and not str(c).lower().endswith("id"):
                continue
            s = df[c].dropna()
            if len(s) == 0 or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_float_dtype(s):
                continue
            vals = set(s.astype(str).unique())
            if 2 < len(vals) <= 300_000:
                cand.append((t, c, vals))
    parent = list(range(len(cand)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (t1, c1, v1) in enumerate(cand):
        for j in range(i + 1, len(cand)):
            t2, c2, v2 = cand[j]
            if t1 == t2:
                continue
            small, big = (v1, v2) if len(v1) <= len(v2) else (v2, v1)
            common = len(small & big)
            # cột gộp nhiều loại mã (thẻ kho: mã đơn + mã phiếu nhập) không được nối hai loại mã với nhau:
            # khác tên thì phải trùng cả phần lớn bảng lớn
            if common >= 0.8 * len(small) and (fold(str(c1)) == fold(str(c2)) or common >= 0.5 * len(big)):
                parent[root(j)] = root(i)
    groups: dict[int, list[str]] = {}
    for i, (t, c, _) in enumerate(cand):
        groups.setdefault(root(i), []).append(f"T[{t!r}][{c!r}]")
    return [" ↔ ".join(g) for g in groups.values() if len(g) > 1][:limit]
