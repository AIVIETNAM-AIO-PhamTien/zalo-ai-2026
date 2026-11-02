"""Tầng làm sạch không phụ thuộc dữ liệu (chạy ngay khi nạp, cho mọi shop: train, public, private).

Chỉ dùng bất biến của bản thân dữ liệu, không dùng từ vựng / tên cột đã thấy:
- entity_groups: gom các cách viết của cùng một giá trị chữ (hoa / thường, dấu, khoảng trắng) về một dạng chuẩn;
  near_duplicates chỉ báo cặp gần giống (lỗi gõ?), không gộp. Không sửa bảng: trả ánh xạ {biến thể: dạng chuẩn} để tầng tra cứu dùng.
- infer_contracts: tự suy ràng buộc từ phân bố của cột (cột gần như duy nhất → khoá, cột số gần như không âm → không
  âm, ngày nằm trong khoảng chính), rồi gắn cờ dòng vi phạm. Chỉ chẩn đoán, không xoá dòng.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


def fold_key(v: str) -> str:
    """Khoá so khớp: bỏ dấu, lowercase, đ→d, chỉ giữ chữ / số, gộp khoảng trắng."""
    s = unicodedata.normalize("NFD", str(v).lower().replace("đ", "d"))
    s = "".join(ch if ch.isalnum() else " " for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())


def entity_groups(values) -> dict:
    """{biến thể: dạng chuẩn} cho các giá trị chỉ khác cách viết (hoa / thường, dấu, khoảng trắng, dấu câu) — đúng
    rulebook §3/§8: 'hai tên chỉ khác cách viết là một; khác chữ hoặc khác số là hai'. Dạng chuẩn = cách viết hay gặp
    nhất. Cùng khoá với semantic.item_key nên tra tên ở tầng ngữ nghĩa gộp đúng như ánh xạ này."""
    cnt = Counter(str(v) for v in values if v is not None and str(v).strip())
    by_key = defaultdict(list)
    for v in cnt:
        by_key[fold_key(v)].append(v)
    out = {}
    for members in by_key.values():
        if len(members) > 1:
            canon = max(members, key=lambda v: (cnt[v], v))
            out.update({v: canon for v in members if v != canon})
    return out


def near_duplicates(values, cutoff: int = 94, max_distinct: int = 3000) -> list:
    """Cặp khoá gần giống (lỗi gõ một ký tự?) — CHỈ báo để kiểm, không gộp: rulebook coi khác chữ là hai thứ khác."""
    keys = sorted({fold_key(v) for v in values if v is not None and str(v).strip()})
    if fuzz is None or len(keys) > max_distinct:
        return []
    digits = {k: "".join(ch for ch in k if ch.isdigit()) for k in keys}
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if len(a) >= 5 and abs(len(a) - len(b)) <= 1 and digits[a] == digits[b] and fuzz.ratio(a, b) >= cutoff:
                out.append((a, b))
    return out[:50]


def infer_contracts(df: pd.DataFrame, types: dict, min_rows: int = 20) -> dict:
    """Ràng buộc tự suy + dòng vi phạm. Một ràng buộc chỉ được đặt khi ≥ 98% dòng đã thoả (dữ liệu tự nói), nên
    dòng vi phạm là dòng lạ đáng gắn cờ chứ không phải luật áp từ ngoài."""
    n = len(df)
    out = {}
    if n < min_rows:
        return out
    for c, kind in types.items():
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if len(s) < min_rows:
            continue
        rules = {}
        if kind.startswith(("int", "float")):
            x = pd.to_numeric(s, errors="coerce").dropna()
            neg = x[x < 0]
            if len(x) and 0 < len(neg) <= 0.02 * len(x):
                rules["non_negative"] = [int(i) for i in neg.index[:50]]
        elif kind.startswith("date"):
            d = pd.to_datetime(s, errors="coerce").dropna()
            if len(d) >= min_rows:
                lo, hi = d.quantile(0.01), d.quantile(0.99)
                span = max(hi - lo, pd.Timedelta(days=30))
                bad = d[(d < lo - span) | (d > hi + span)]
                if 0 < len(bad) <= 0.02 * len(d):
                    rules["date_in_range"] = [int(i) for i in bad.index[:50]]
        elif kind == "text":
            dup = s[s.duplicated(keep=False)]
            if 0 < s.duplicated().sum() <= 0.02 * len(s) and s.nunique() >= 0.98 * len(s):
                # mã lặp mà các ô khác khác nhau là quan hệ một-nhiều (một hoá đơn nhiều lần trả): không phải lỗi
                rows = df.loc[dup.index].astype(str)
                bad = rows[rows.duplicated(keep=False)]
                if len(bad):
                    rules["unique_key"] = [int(i) for i in bad.index[:50]]
                else:
                    rules["explained"] = {"unique_key": {"by": "one_to_many", "value": str(len(dup))}}
        if rules:
            out[c] = rules
    return _explain(df, types, out)


def _explain(df: pd.DataFrame, types: dict, contracts: dict) -> dict:
    """Vi phạm 'có lý do': ≥ 95% dòng vi phạm cùng một giá trị ở một cột chữ ít giá trị (≤ 30), mà giá trị đó hiếm
    ở phần còn lại (dòng trả hàng có số lượng âm). Đó là nghiệp vụ, không phải lỗi: chuyển sang 'explained'."""
    from datamate.semantic import classify_status  # tránh vòng import loader → clean → semantic → loader
    cats = [c for c, k in types.items() if k == "text" and c in df.columns and df[c].nunique() <= 30]
    for c, rules in contracts.items():
        for k in [k for k in rules if k != "explained"]:
            idx = rules[k]
            for cat in cats:
                col = df[cat].ffill()  # ô trạng thái chỉ ghi ở dòng đầu nhóm (§8)
                lab = col.map(classify_status)
                if lab.loc[idx].isin(["returned", "cancelled"]).mean() >= 0.95:
                    col = lab  # nhiều cách viết của cùng một trạng thái rulebook (§8): 'Trả hàng', 'trả lại', 'Đổi trả'
                v = col.loc[idx].value_counts()
                rest = col.drop(index=idx)
                if len(v) and v.iloc[0] >= 0.95 * len(idx) and (rest == v.index[0]).mean() < 0.2:
                    val = v.index[0]
                    rules.setdefault("explained", {})[k] = {"by": cat, "value": str(val)}
                    rules.pop(k)
                    break
    return {c: r for c, r in contracts.items() if r}


def clean_profile(df: pd.DataFrame, types: dict) -> dict:
    """Hồ sơ làm sạch của một bảng: ánh xạ thực thể cho mọi cột chữ + ràng buộc tự suy. Lỗi ở đây (dữ liệu lạ của
    bộ private) không được làm hỏng việc nạp: trả hồ sơ rỗng kèm lý do."""
    try:
        return _clean_profile(df, types)
    except Exception as e:  # noqa: BLE001
        return {"entities": {}, "contracts": {}, "error": f"{type(e).__name__}: {e}"}


def _clean_profile(df: pd.DataFrame, types: dict) -> dict:
    data = df.drop(columns=["_sheet"], errors="ignore")
    ent, near = {}, {}
    for c, kind in types.items():
        if kind == "text" and c in data.columns:
            vals = data[c].dropna().tolist()
            m = entity_groups(vals)
            if m:
                ent[c] = m
            nd = near_duplicates(vals)
            if nd:
                near[c] = nd
    return {"entities": ent, "near_duplicates": near, "contracts": infer_contracts(data, types)}


def canon(shop, table: str, col: str, value):
    """Dạng chuẩn của một giá trị theo hồ sơ đã nạp (không có thì trả nguyên)."""
    m = shop.meta.get(table, {}).get("clean", {}).get("entities", {}).get(col, {})
    return m.get(value, value)


_README_FACTOR = re.compile(r"^\s*[-*]\s*(.+?)\s*:\s*1\s+(\S+(?:\s\S+)?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(\S+)", re.M)


def readme_checks(shop) -> dict:
    """A4 — luật kiểm độc lập lấy từ README (không suy từ chính dữ liệu cần kiểm):
    - hệ số quy đổi README nêu ('- Coca 330ml: 1 thùng = 24 lon') so với bảng quy đổi trong dữ liệu: khác hệ số = xung đột;
    - mặt hàng README nêu mà dữ liệu không có tên nào cùng khoá.
    Chỉ chẩn đoán, không sửa: README và dữ liệu lệch nhau thì công thức quy đổi phải bỏ trống, không chọn một bên."""
    from datamate.semantic import _conversions, item_key, shop_columns
    facts = [(item_key(n), item_key(big), float(f.replace(",", ".")), item_key(base))
             for n, big, f, base in _README_FACTOR.findall(shop.readme or "")]
    if not facts:
        return {"readme_factors": 0}
    try:
        data = _conversions(shop, shop_columns(shop))
    except Exception:
        data = {}
    names = set()
    for df in shop.tables.values():
        for c in df.columns:
            if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])):
                names.update(item_key(v) for v in df[c].dropna().astype(str).unique()[:5000])
    conflicts, unknown = [], []
    for k, big, f, base in facts:
        if k not in names:
            unknown.append(k)
        have = data.get(k, (None, {}))[1].get(big)
        if have is not None and abs(have - f) > 1e-9:
            conflicts.append((k, big, f, have))
    return {"readme_factors": len(facts), "factor_conflicts": conflicts, "readme_items_not_in_data": unknown[:10],
            "n_readme_items_not_in_data": len(unknown)}
