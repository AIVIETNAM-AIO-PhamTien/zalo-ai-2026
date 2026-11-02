"""Semantic layer (bản đầu): bảng đơn hàng chuẩn dựng từ README + metric tất định theo rulebook §1.

Ánh xạ cột không dùng LLM: README mô tả cột bằng bộ từ vựng cố định ("mã đơn", "ngày bán",
"trạng thái đơn", "thành tiền sau giảm"…). Thiếu thông tin cần thiết → None (để agent / abstain lo).
"""
from __future__ import annotations

import datetime as dt
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from numbers import Integral
import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from datamate.helpers import round2, round_money
from datamate.loader import Shop
from datamate.parsing import fold


def _decimal_or_nan(value):
    if pd.isna(value):
        return Decimal('NaN')
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal('NaN')


def _sum_money(values: pd.Series) -> int | None:
    """Sum money without integer overflow or silent missing-value substitution."""
    if values.isna().any():
        return None
    if values.empty:
        return 0
    if pd.api.types.is_integer_dtype(values.dtype):
        bound = max(abs(int(values.min())), abs(int(values.max()))) * len(values)
        if bound <= 2**63 - 1:
            return int(values.sum())
        return sum(int(value) for value in values)
    amounts = [_decimal_or_nan(value) for value in values]
    if not all(value.is_finite() for value in amounts):
        return None
    return round_money(sum(amounts, Decimal(0)))

_COLS_HEADING = re.compile(r"^##\s+Cột trong\s+`([^`]+)`(?:,\s*sheet\s+`([^`]+)`)?", re.M)
_COL_LINE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+?)\s*$")

M_ORDER_ID = ("mã đơn", "số hoá đơn", "mã hoá đơn", "số đơn hàng", "số thứ tự đơn")  # so sau fold: hoá = hóa
M_DATE = ("ngày bán", "thời điểm đặt đơn (ngày giờ)")
M_STATUS = "trạng thái đơn"
M_LINE_AMOUNT = "thành tiền sau giảm"
M_ORDER_GOODS = "tiền hàng của đơn"  # 'tiền hàng của đơn: cộng thành tiền các mặt hàng, trước giảm giá của đơn…'
M_RETURN_QTY = "số lượng khách trả lại"


def parse_readme_columns(readme: str) -> dict[str, dict[str, str]]:
    """{'file' hoặc 'file:sheet': {tên cột: nghĩa}} từ các mục '## Cột trong `file`'."""
    out: dict[str, dict[str, str]] = {}
    heads = list(_COLS_HEADING.finditer(readme))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(readme)
        body = readme[h.end():end].split("\n## ")[0]
        key = h.group(1) + (f":{h.group(2)}" if h.group(2) else "")
        cols = {}
        for line in body.splitlines():
            m = _COL_LINE.match(line.strip())
            if m:
                cols[m.group(1).strip()] = m.group(2).strip()
        out[key] = cols
    return out


def classify_status(s) -> str | None:
    """Bốn trạng thái của rulebook §8: completed / cancelled / returned / delivering."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    f = fold(s)
    # so khớp theo từ: 'van chuyen' chứa chuỗi con 'huy' nhưng không phải huỷ
    if re.search(r"\b(huy|cancel\w*|khong giao duoc|giao that bai)\b", f):
        return "cancelled"
    if re.search(r"\b(tra hang|tra lai|hoan hang|hoan tra|doi tra|return\w*|hang tra ve|khach tra|tra ve|mang tra)\b", f):
        return "returned"
    if re.search(r"\b(dang giao|dang van chuyen|shipping|dang ship|dang xu ly|dang giao dich|cho giao)\b", f):
        return "delivering"
    return "completed"


def _file_pattern(fname: str) -> re.Pattern:
    """Tên file trong README có thể là mẫu: `BanHang_T<tháng>_<năm>.xlsx` → mỗi `<…>` khớp một đoạn bất kỳ."""
    parts = re.split(r"<[^<>]*>", fname)
    return re.compile("^" + ".+?".join(re.escape(p) for p in parts) + "$")


def _tables_for(shop: Shop, file_key: str) -> list[str]:
    if file_key.startswith("@table:"):  # nghĩa cột suy ra từ dữ liệu (infer_meanings), khoá theo tên bảng
        name = file_key[len("@table:"):]
        return [name] if name in shop.tables else []
    fname, _, sheet = file_key.partition(":")
    pat = _file_pattern(fname)
    hits = []
    for name, m in shop.meta.items():
        for src in m["sources"]:
            sf, _, ss = src.partition(":")
            if pat.match(sf) and (not sheet or ss == sheet):
                hits.append(name)
                break
    return hits


def _find(colmap: dict[str, str], meaning) -> str | None:
    meanings = (meaning,) if isinstance(meaning, str) else meaning
    for col, mean in colmap.items():
        # khớp tiền tố trọn từ: 'số thứ tự đơn' không khớp 'số thứ tự dòng trong đơn' ('dong' bắt đầu bằng 'don')
        if any(re.match(re.escape(fold(m)) + r"(?![a-z0-9])", fold(mean)) for m in meanings):
            return col
    return None


# so trên tên cột đã bỏ dấu và khoảng trắng; các dạng gặp ở train: 'Mã hóa đơn', 'MaHoaDon', 'Số HĐ', 'Mã bill',
# 'Mã chứng từ', 'Số CT', 'Mã giao dịch', 'Order ID'
_ORDER_ID_NAME = re.compile(r"^(ma|so)(hoadon|hd|donhang|don|chungtu|ct|bill|giaodich|gd)$|^orderid$|^donso$|^hd$")


def order_id_columns(shop: Shop) -> dict[str, str]:
    """Bảng ghi theo đơn → cột mã đơn: theo nghĩa cột trong README, không có thì theo tên cột."""
    out = {}
    for key, cols in parse_readme_columns(shop.readme).items():
        for t in _tables_for(shop, key):
            oid = _find({c: m for c, m in cols.items() if c in shop.tables[t].columns}, M_ORDER_ID)
            if oid:
                out.setdefault(t, oid)
    for t, df in shop.tables.items():
        if t not in out:
            oid = next((c for c in df.columns if _ORDER_ID_NAME.match(re.sub(r"[\s_]", "", fold(c)))), None)
            if oid:
                out[t] = oid
    # tên cột tiếng Anh / lạ ('OrderRef', 'Invoice No'…): lấy vai trò mã đơn đã suy cho bảng (shop_columns)
    missing = [t for t in shop.tables if t not in out]
    if missing:
        roles = shop_columns(shop)
        for t in missing:
            oid = _find(roles.get(f"@table:{t}", {}), M_ORDER_ID)
            if oid in shop.tables[t].columns:
                out[t] = oid
    return out


def clean_order_rows(shop: Shop, table: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Nơi duy nhất làm sạch bảng ghi theo đơn, dùng chung cho OrderView, finance_sem và namespace agent:
    dòng tiếp nối (mã đơn / ngày / trạng thái chỉ ghi ở dòng đầu) kế thừa dòng đầu nhóm, rồi rulebook §8: hai dòng
    của cùng một đơn giống nhau ở mọi ô là lỗi xuất file → tính một lần. Bảng không ghi theo đơn không gọi hàm này."""
    df = (shop.tables[table] if df is None else df).copy()
    meta = shop.meta.get(table, {})
    if meta.get("continuation_rows"):
        cc = [c for c in meta.get("continuation_cols", []) if c in df.columns]
        df[cc] = df[cc].ffill()
    policy = getattr(shop, "rule_policy", None)  # rulebook của split đổi §8 (datamate/rule_policy.py)
    if policy is not None and not policy.dedup_same_order:
        return df
    return df.drop_duplicates(subset=[c for c in df.columns if c != "_sheet"])


def _fname(c) -> str:
    return " ".join(re.sub(r"[^a-z0-9]", " ", fold(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(c)))).split())


def _num(df: pd.DataFrame, c: str | None) -> pd.Series | None:
    return None if c is None else pd.to_numeric(df[c], errors="coerce")


def infer_meanings(df: pd.DataFrame) -> dict[str, str]:
    """Nghĩa cột suy từ tên + giá trị khi README không mô tả bảng (đề: README có khi gần như trống).
    Chỉ gán khi chắc; cột tiền dòng phải khớp số học số lượng × đơn giá (− giảm giá dòng) trên ≥ 98% dòng."""
    names = {c: _fname(c) for c in df.columns if c != "_sheet"}
    from datamate.schema_roles import business_roles
    out: dict[str, str] = business_roles(df)

    def pick(pattern, cols=None):
        hits = [c for c in (cols or names) if re.search(pattern, names[c])]
        return hits[0] if len(hits) == 1 else None

    oid = pick(r"^(ma|so) ?(hoa don|hd|don hang|don|chung tu|ct|bill|giao dich|gd)$|^(order|invoice) ?(id|number|no)$"
               r"|^(id|no) ?(order|invoice|hoa don|hd|don hang|don|bill|chung tu|ct|giao dich|gd)$"
               r"|^(hd|don so|so don|hoa don)$")  # viết tắt hay gặp: 'HĐ', 'Đơn số', 'Số đơn' 
    if oid is None:
        # 'STT' là số thứ tự ĐƠN (không phải số dòng) khi giá trị lặp ở các dòng liền nhau (đơn nhiều dòng) và không
        # bao giờ giảm; số thứ tự dòng không lặp → không gán
        stt = pick(r"^(stt|so thu tu|so tt)( don| hd| hoa don)?$")
        if stt is not None:
            q = pd.to_numeric(df[stt], errors="coerce").ffill()
            d = q.diff().dropna()
            if q.notna().mean() >= 0.95 and len(d) and (d >= 0).all() and (d == 0).mean() >= 0.05:
                oid = stt
    if oid:
        out[oid] = "mã đơn"
    dates = [c for c in names if pd.api.types.is_datetime64_any_dtype(df[c])]
    date_pattern = (r"^(ngay|date|thoi gian|time)( (ban|dat|lap|hd|hoa don|xuat|mua|order))?$"
                    r"|^(order|invoice|sale|sales) (date|time)$")
    date = pick(date_pattern, dates) or (dates[0] if len(dates) == 1 else None)
    if date is None:
        candidates = [c for c in dates if re.search(date_pattern, names[c])]
        if len(candidates) > 1 and all(df[c].dt.normalize().equals(df[candidates[0]].dt.normalize()) for c in candidates[1:]):
            # Header exports often include both timestamp and its date-only copy.
            # Keep timestamp for shift/hour queries, but only when dates agree.
            date = max(candidates, key=lambda c: int(((df[c].dt.hour != 0) | (df[c].dt.minute != 0)).sum()))
    if date:
        out[date] = "ngày bán"
    from_status = [c for c in names if (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object)
                   and 0 < df[c].nunique() <= 50
                   and any(classify_status(v) in ("cancelled", "returned") for v in df[c].dropna().astype(str).unique())]
    if len(from_status) == 1:
        out[from_status[0]] = "trạng thái đơn"
    item = pick(r"^(ten )?(hang|hang hoa|mat hang|san pham|sp|mon)$|^ten (hang|hang hoa|san pham|sp|mon|mat hang)$")
    if item:
        out[item] = "tên hàng"
    qty = pick(r"^(so luong|sl|qty|quantity)( (san pham|hang|ban))?$")
    if qty:
        out[qty] = "số lượng"
    bq = pick(r"^(so luong|sl) .*(quy doi|co ban|le)$|^(sl|so luong) quy doi$|quy ra")
    if bq:
        out[bq] = "số lượng đã đổi ra đơn vị cơ bản"
    unit = pick(r"^(dvt|don vi|don vi tinh)$")
    if unit:
        out[unit] = "đơn vị tính"
    amt = pick(r"^(tt|thanh tien|tri gia|tong tien|thanh tien sau (giam|ck|chiet khau)|line total|amount)$")
    price = pick(r"^(dg|don gia|gia ban|gia|gia (san pham|ban|don vi)|unit price|price)$")
    disc = pick(r"^(giam gia|chiet khau|ck|giam|line discount|discount)$")
    if amt and qty and price:
        a, q, p = _num(df, amt), _num(df, qty), _num(df, price)
        both = a.notna() & q.notna() & p.notna()
        if disc:  # có cột giảm giá dòng: tiền dòng phải là sau giảm (= SL × giá − giảm)
            ok = (a - (q * p - _num(df, disc).fillna(0))).abs() <= 1
        else:  # giảm giá (nếu có) đã nằm trong tiền dòng: không vượt SL × giá, không lệch quá nửa
            ok = (a <= q * p + 1) & (a >= 0.5 * q * p)
        if both.any() and ok[both].mean() >= 0.98:
            out[amt] = "thành tiền sau giảm"
    elif amt and "sau" in names[amt]:
        out[amt] = "thành tiền sau giảm"
    if "thành tiền sau giảm" not in out.values():
        out.update(_money_by_invariant(df, names, out))
    return out


def _money_by_invariant(df: pd.DataFrame, names: dict, known: dict) -> dict[str, str]:
    """Tên cột tiền lạ (amount, t_money, order_price…): thử mọi bộ (SL, giá, giảm?, tiền) trong các cột số, nhận khi
    đúng một bộ thỏa tiền = SL × giá − giảm trên ≥ 98% dòng. SL là cột nguyên có trung vị nhỏ hơn giá."""
    if len(df) < 20:
        return {}
    nums = {}
    for c in names:
        if c in known and known[c] not in ("số lượng", "đơn giá"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if pd.api.types.is_bool_dtype(df[c]) or s.notna().mean() < 0.9 or s.nunique() < 3:
            continue
        nums[c] = s
    if len(nums) > 12:
        return {}
    full = nums
    if len(df) > 2000:  # dò trên mẫu, kiểm lại bộ thắng trên cả bảng
        idx = df.sample(2000, random_state=0).index
        nums = {c: s.loc[idx] for c, s in nums.items()}
    found = {}
    for a in nums:
        for q in nums:
            for p in nums:
                if len({a, q, p}) < 3 or not (nums[q] % 1 == 0).mean() >= 0.99:
                    continue
                if nums[q].median() >= nums[p].median():
                    continue
                prod = nums[q] * nums[p]
                both = nums[a].notna() & prod.notna() & (prod != 0)
                if both.mean() < 0.9:
                    continue
                for d in [None] + [c for c in nums if c not in (a, q, p) and nums[c].median() < nums[a].median()]:
                    want = prod - (nums[d].fillna(0) if d else 0)
                    rate = ((nums[a] - want).abs() <= 1)[both].mean()
                    if rate >= 0.98:
                        found[(q, p, a)] = max(rate, found.get((q, p, a), 0))
    named_qty = [c for c, m in known.items() if m == "số lượng"]
    if named_qty and any(f[0] == named_qty[0] for f in found):  # SL quy đổi / STT hay trùng SL: tin cột tên SL
        found = {f: r for f, r in found.items() if f[0] == named_qty[0]}
    # SL phần lớn = 1 thì đổi vai giá ↔ tiền cũng gần khớp: lấy bộ khớp nhiều dòng nhất, phải duy nhất
    best = [f for f, r in found.items() if r == max(found.values(), default=0)]
    if len(best) != 1:
        return {}
    q, p, a = best[0]
    prod = full[q] * full[p]
    both = full[a].notna() & prod.notna() & (prod != 0)
    if not any(((full[a] - prod + (full[d].fillna(0) if d else 0)).abs() <= 1)[both].mean() >= 0.98
               for d in [None] + [c for c in full if c not in (a, q, p)]):
        return {}
    return {q: "số lượng", p: "đơn giá", a: "thành tiền sau giảm"}


def gap_filled_columns(shop: Shop, readme_cols: dict) -> dict:
    """Phương án dự phòng khi README không cho bảng dòng bán nào: README đúng tên cột nhưng mô tả bằng lời khác
    ('số HĐ', 'thành tiền của dòng', 'số thứ tự đơn trong sổ') → lấp vai trò lõi (mã đơn, tiền dòng, ngày, trạng thái)
    còn thiếu bằng suy luận trên dữ liệu; cột README đã mô tả khớp vẫn giữ nghĩa README."""
    from datamate.schema_roles import align_declared_columns
    out = dict(readme_cols)
    core = {fold(m) for m in (*M_ORDER_ID, M_LINE_AMOUNT, *M_DATE, M_STATUS)}
    keys = {fold(m) for m in (*M_ORDER_ID, M_LINE_AMOUNT)}
    for key, declared in list(readme_cols.items()):
        if key.startswith("@table:"):
            continue
        for table in _tables_for(shop, key):
            bound = align_declared_columns(shop.tables[table], declared)
            if _find(bound, M_ORDER_ID) is not None and _find(bound, M_LINE_AMOUNT) is not None:
                continue
            have = {fold(m) for m in bound.values()}
            guess = {c: m for c, m in infer_meanings(shop.tables[table]).items()
                     if fold(m) in core and fold(m) not in have}
            if any(fold(m) in keys for m in guess.values()):
                out[f"@table:{table}"] = {**out.get(f"@table:{table}", {}), **bound, **guess}
    return out


def shop_columns(shop: Shop) -> dict[str, dict[str, str]]:
    """Nghĩa cột theo README; bảng README không mô tả thì suy từ dữ liệu (khoá '@table:<bảng>')."""
    out = parse_readme_columns(shop.readme)
    from datamate.schema_roles import align_declared_columns
    for key, declared in list(out.items()):
        for table in _tables_for(shop, key):
            bound = align_declared_columns(shop.tables[table], declared)
            if any(c not in declared for c in bound):
                # README viết theo tên cũ: vai trò README không còn khớp được thì lấy từ suy luận trên dữ liệu
                have = {fold(m) for m in bound.values()}
                guess = {c: m for c, m in infer_meanings(shop.tables[table]).items()
                         if c not in bound and fold(m) not in have}
                out[f'@table:{table}'] = {**out.get(f'@table:{table}', {}), **guess, **bound}
    card = getattr(shop, "card", None) or {}  # datamate/schema_card.py: vai trò cột do LLM gán, chỉ lấp chỗ thiếu
    covered = {t for key in out for t in _tables_for(shop, key)}
    # bảng README đã mô tả: chỉ theo README (thẻ từng gán 'STT' làm tiền dòng, 'Đơn giá' làm SL quy đổi ở bảng có
    # README → doanh thu sai); bảng README bỏ trống: luật suy từ tên cột trước, thẻ chỉ lấp vai trò còn thiếu
    extras, card_added = {}, set()
    for t, df in shop.tables.items():
        if t not in covered:
            inferred = infer_meanings(df)
            if inferred:
                out[f"@table:{t}"] = inferred
            extras[t] = _card_ok(df, _missing_roles(inferred, card.get(t, {})))
        elif card.get(t):
            # README mô tả cột bằng lời riêng ('thành tiền của dòng; Voucher ghi số âm…', 'số thứ tự đơn trong sổ'):
            # Chỉ lấp vai trò lõi còn thiếu; tên hàng là ngoại lệ phụ được kiểm nghĩa
            # riêng bên dưới. Không ghi đè SL/giá đã được README giải thích.
            desc = {c: m for key in out for tt in _tables_for(shop, key) if tt == t
                    for c, m in out[key].items() if c in df.columns}
            core = [M_ORDER_ID, M_DATE, M_STATUS, M_LINE_AMOUNT, M_ORDER_GOODS, M_QTY]
            have = {fold(m if isinstance(m, str) else m[0]) for m in core if _find(desc, m)}
            free = {c for c, m in desc.items() if not any(_find({c: m}, r) for r in core)}
            pick = {c: m for c, m in card[t].items() if c in free and fold(m) not in have
                    and fold(m) in {fold(r if isinstance(r, str) else r[0]) for r in core}}
            pick = _card_ok(df, _price_not_amount(df, desc, pick))
            if _find(desc, "tên hàng") is None:
                for c, meaning in card[t].items():
                    description = fold(f"{_fname(c)} {desc.get(c, '')}")
                    if (c in free and fold(meaning) == "ten hang" and df[c].notna().any()
                            and re.search(r"ten hang|ten san pham|ten mat hang|ten goi|mat hang|hang ban|noi dung|chi tiet hang|\bmon\b", description)
                            and not re.search(r"khach|nhan vien|nha cung cap|nhom|danh muc|phan loai", description)):
                        pick[c] = meaning
            if pick:
                out[f"@table:{t}"] = {**desc, **pick}
                extras[t] = {}
                card_added.update((t, c) for c in pick)
    # thẻ chỉ lấp chỗ thiếu: shop đã có bảng dòng bán (README / luật suy) thì thẻ không dựng thêm nguồn tiền dòng thứ
    # hai (từng gán 'Khách đã trả' của bảng hoá đơn làm tiền dòng → doanh thu cao hơn ~2%)
    has_line = any(fold(m) == fold(M_LINE_AMOUNT) for cols in out.values() for m in cols.values())
    for t, extra in extras.items():
        if has_line:
            extra = {c: m for c, m in extra.items() if fold(m) != fold(M_LINE_AMOUNT)}
        if extra:
            out[f"@table:{t}"] = {**out.get(f"@table:{t}", {}), **extra}
            card_added.update((t, c) for c in extra)
    shop.card_added = card_added
    return out


def _price_not_amount(df: pd.DataFrame, desc: dict[str, str], pick: dict[str, str]) -> dict[str, str]:
    """Thẻ gán đơn giá làm tiền dòng: nếu có cột khác ≈ SL × cột đó ở ≥ 90% dòng thì tiền dòng là cột kia."""
    L = next((c for c, m in pick.items() if fold(m) == fold(M_LINE_AMOUNT)), None)
    Q = _qty_col({**desc, **pick})
    if not L or not Q:
        return pick
    prod = (pd.to_numeric(df[Q], errors="coerce") * pd.to_numeric(df[L], errors="coerce")).astype(float)
    ok = prod.notna() & (prod != 0)
    for c in df.columns:
        if c in (L, Q) or not ok.any():
            continue
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        if ((v[ok] - prod[ok]).abs() <= 0.01 * prod[ok].abs() + 1).fillna(False).mean() >= 0.9:
            out = {k: m for k, m in pick.items() if k != L}
            out[c] = M_LINE_AMOUNT
            return out
    return pick


_STATUS_WORDS = re.compile(r"huy|tra hang|hoan|cancel|return|thanh cong|hoan thanh|da giao|dang giao")
_RETURN_NAME = re.compile(r"\btra\b|tra lai|return|hoan")


def _card_ok(df: pd.DataFrame, extra: dict[str, str]) -> dict[str, str]:
    """Kiểm vai trò thẻ cần đúng nghĩa chứ không chỉ đúng kiểu số:
    - trạng thái đơn: giá trị phải có chữ kiểu huỷ / trả / hoàn thành (không phải 'Phương thức thanh toán');
    - SL khách trả lại: tên cột phải nói 'trả' / 'hoàn' ('Khuyến mãi' cũng ≤ SL và đa số bằng 0)."""
    out = {}
    for c, m in extra.items():
        if c not in df.columns:
            continue  # thẻ cũ nhắc cột bảng này không có (tên cột đã đổi): bỏ, không làm hỏng cả view
        f = fold(m)
        if f in {fold("số lượng"), fold("số lượng đã đổi ra đơn vị cơ bản")}:
            if re.search(r"\bgia\b|\btien\b|\bprice\b|\bcost\b", _fname(c)):
                continue  # vai trò SL do thẻ đề xuất không thể lấy từ cột đơn giá/tiền
        if f == fold(M_STATUS):
            vals = " | ".join(fold(v) for v in df[c].dropna().astype(str).unique()[:200])
            if not _STATUS_WORDS.search(vals):
                continue
        if f == fold(M_RETURN_QTY) and not _RETURN_NAME.search(fold(c)):
            continue
        if f == fold("nhóm hàng"):
            # A schema card's free-form label is insufficient to turn a
            # price, customer name, or staff name into the product-category
            # domain used by downstream filters and rankings.
            key = fold(c)
            numeric = pd.to_numeric(df[c], errors="coerce")
            category_name = re.search(r"nhom|loai|danh muc|phan loai|category", key)
            if (numeric.notna().mean() >= 0.9 * df[c].notna().mean()
                    or not category_name or re.search(r"gia|price|tien|cost", key)):
                continue
        out[c] = m
    return out


def _missing_roles(have: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """Cột của `extra` chưa có trong `have` và mang nghĩa mà `have` chưa có cột nào mang."""
    known = {fold(m) for m in have.values()}
    return {c: m for c, m in extra.items() if c not in have and fold(m) not in known}


M_ITEM_NAME = "tên hàng"
M_QTY = "số lượng"
M_BASE_QTY = "số lượng đã đổi ra đơn vị cơ bản"
M_UNIT = "đơn vị tính"
M_BASE_UNIT = "đơn vị cơ bản"
M_BIG_UNIT = "đơn vị lớn"
M_FACTOR = "hệ số"


def item_key(name) -> str:
    """Khoá mặt hàng: hai tên chỉ khác hoa/thường, dấu, khoảng trắng, dấu câu là một mặt hàng."""
    return " ".join(re.sub(r"[^a-z0-9%]", " ", fold(str(name))).split())


def _qty_col(present: dict[str, str]) -> str | None:
    """Cột 'số lượng' ghi theo đơn vị của dòng (không phải cột đã quy đổi, không phải số lượng trả)."""
    for c, m in present.items():
        f = fold(m)
        if f.startswith(fold(M_QTY)) and not f.startswith(fold(M_BASE_QTY)) and not f.startswith(fold(M_RETURN_QTY)):
            return c
    return None


def _conversions(shop: Shop, readme_cols: dict) -> dict[str, tuple[str, dict[str, float]]]:
    """{khoá mặt hàng: (đơn vị cơ bản, {đơn vị lớn: hệ số})} từ bảng quy đổi (tên hàng, đơn vị lớn, hệ số, đơn vị cơ bản)."""
    out: dict[str, tuple[str, dict[str, float]]] = {}
    for key, cols in readme_cols.items():
        for t in _tables_for(shop, key):
            df = shop.tables[t]
            present = {c: m for c, m in cols.items() if c in df.columns}
            name, big = _find(present, M_ITEM_NAME), _find(present, M_BIG_UNIT)
            fac, base = _find(present, M_FACTOR), _find(present, M_BASE_UNIT)
            text_conv = _find(present, 'quy đổi')
            if name and base and text_conv and not (big and fac):
                for n, u, declaration in zip(df[name], df[base], df[text_conv]):
                    if pd.isna(n) or pd.isna(u):
                        continue
                    k, basic = item_key(n), item_key(u)
                    previous, factors = out.setdefault(k, (basic, {}))
                    if previous != basic:
                        out[k] = ('', factors)
                        continue
                    if pd.isna(declaration) or not str(declaration).strip():
                        continue
                    valid = True
                    # không dùng fold() ở đây: fold bỏ cả dấu '=' ('1 thùng = 4 bịch' → '1 thung 4 bich'); đơn vị so
                    # bằng item_key nên chữ có dấu vẫn khớp
                    for part in re.split(r';|\n|,\s*(?=1\s)', str(declaration).lower()):
                        m = re.fullmatch(r'\s*1\s+(.+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*', part)
                        if not m or item_key(m[3]) != basic:
                            valid = False
                            break
                        factor = float(m[2].replace(',', '.'))
                        large = item_key(m[1])
                        if factor <= 0 or (large in factors and factors[large] != factor):
                            valid = False
                            break
                        factors[large] = factor
                    if not valid:
                        out[k] = ('', factors)
            if not (name and big and fac and base):
                continue
            for n, b, f, u in zip(df[name], df[big], pd.to_numeric(df[fac], errors="coerce"), df[base]):
                if pd.isna(n) or pd.isna(b) or pd.isna(f) or pd.isna(u):
                    continue
                k = item_key(n)
                unit, facs = out.setdefault(k, (item_key(u), {}))
                if unit != item_key(u):
                    out[k] = ("", facs)  # hai đơn vị cơ bản khác nhau: không dùng
                facs[item_key(b)] = float(f)
    return out


def _merge_by_code(item: pd.Series, code: pd.Series) -> pd.Series:
    """Một mã hàng ghi nhiều cách viết tên ('Mì Hảo Hảo' / 'Mi Hao Hao'): một mặt hàng, gộp về khoá hay gặp nhất
    của mã (rulebook: nhận mọi cách viết tên có trong file). Chỉ gộp khi mỗi khoá tên chỉ thuộc đúng mã đó — tên
    trùng giữa hai mã thì giữ nguyên, không đoán."""
    ok = code.notna() & item.notna()
    codes_per_key = code[ok].groupby(item[ok]).nunique()
    single = ok & item.map(codes_per_key).eq(1)
    canon = item[single].groupby(code[single]).agg(lambda s: s.value_counts().index[0])
    return item.where(~single, code.map(canon))


def _items(df: pd.DataFrame, oid: str, present: dict[str, str], conv: dict, base_units=None) -> pd.DataFrame | None:
    """Dòng hàng → order_id, item (khoá tên), name (tên gốc), bqty (số lượng đơn vị cơ bản), bunit (đơn vị cơ bản
    của mặt hàng). bqty / bunit NaN khi không quy đổi chắc chắn được → câu hỏi đụng mặt hàng đó không trả lời."""
    name = _find(present, M_ITEM_NAME)
    if name is None:
        return None
    unit, qty, bq = _find(present, M_UNIT), _qty_col(present), _find(present, M_BASE_QTY)
    items = pd.DataFrame({"order_id": df[oid].astype(str), "name": df[name], "item": df[name].map(item_key)})
    line_unit = df[unit].map(lambda u: item_key(u) if pd.notna(u) else None) if unit else None
    if bq is not None:
        items["bqty"] = pd.to_numeric(df[bq], errors="coerce")
        declared = items['item'].map(base_units or {})
        if line_unit is None or qty is None:
            if declared.isna().all():
                return None
            items['bunit'] = declared
        else:
        # đơn vị cơ bản của mặt hàng = đơn vị của các dòng có số lượng đã đổi bằng số lượng dòng (hệ số 1)
            q = pd.to_numeric(df[qty], errors="coerce")
            same = (q == items["bqty"]) & line_unit.notna()
            units = line_unit[same].groupby(items.loc[same, "item"]).agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
            inferred = items['item'].map(units)
            items['bunit'] = inferred.fillna(declared)
            items.loc[inferred.notna() & declared.notna() & inferred.ne(declared), 'bunit'] = None
    elif qty is not None and line_unit is not None and (conv or base_units):
        q = pd.to_numeric(df[qty], errors="coerce")

        def convert(k, u, n):
            base, facs = conv.get(k, ((base_units or {}).get(k), {}))
            if base is None:  # mặt hàng không có trong bảng quy đổi: đơn vị của dòng là đơn vị cơ bản
                return n, u
            if not base:
                return float("nan"), None
            if u == base:
                return n, base
            if u in facs:
                return n * facs[u], base
            return float("nan"), None

        conv_rows = [convert(k, u, n) for k, u, n in zip(items["item"], line_unit, q)]
        items["bqty"] = [r[0] for r in conv_rows]
        items["bunit"] = [r[1] for r in conv_rows]
        # mặt hàng ngoài bảng quy đổi mà dòng ghi nhiều đơn vị khác nhau: không chắc đơn vị cơ bản
        per_item = items.dropna(subset=["bunit"]).groupby("item")["bunit"].nunique()
        items.loc[items["item"].isin(per_item[per_item > 1].index), ["bqty", "bunit"]] = [float("nan"), None]
    else:
        return None
    price = _find(present, "đơn giá")
    if price is not None and qty is not None:
        # A price is recorded per selling unit; normalize by the same conversion as quantity.
        q = pd.to_numeric(df[qty], errors="coerce")
        p = pd.to_numeric(df[price], errors="coerce")
        items["base_price"] = (p * q / items["bqty"]).where((q > 0) & (items["bqty"] > 0))
    known_code = _find(present, M_ITEM_CODE)
    code_cols = [known_code] if known_code else [c for c in df if re.sub(r"[^a-z0-9]", "", fold(c)) in
                 {"mahang", "mahanghoa", "masanpham", "masp", "mamon", "mathuoc"}]
    if len(code_cols) == 1:
        items["rank_key"] = df[code_cols[0]].map(lambda x: str(x).strip() if pd.notna(x) else None)
    elif not code_cols:
        items["rank_key"] = items["name"].astype(str)
    items = items.dropna(subset=["item"])
    # cùng mặt hàng ghi lệch khoảng trắng ('Red Bull 250ml' / 'Red Bull 250 ml'): gộp về cách viết hay gặp nhất
    tight = items["item"].str.replace(" ", "", regex=False)
    canon = items.groupby(tight)["item"].agg(lambda s: s.value_counts().index[0])
    items["item"] = tight.map(canon)
    if len(code_cols) == 1:
        items["item"] = _merge_by_code(items["item"], items["rank_key"])
    per_item = items.dropna(subset=["bunit"]).groupby("item")["bunit"].nunique()
    items.loc[items["item"].isin(per_item[per_item > 1].index), ["bqty", "bunit"]] = [float("nan"), None]
    # Phí phục vụ vẫn nằm trong lines để tính doanh thu, nhưng không là mặt hàng.
    return items[items["item"] != "phi phuc vu"]


def catalog_base_units(shop, columns):
    """Explicit catalogue base units, retaining conflicts as unknown; no unit from names."""
    values = {}
    for key, cols in columns.items():
        for table in _tables_for(shop, key):
            frame = shop.tables[table]
            present = {c: m for c, m in cols.items() if c in frame}
            base = _find(present, M_BASE_UNIT)
            name = _find(present, M_ITEM_NAME)
            if name is None:
                names = [c for c, m in present.items() if fold(m).strip() == 'ten']
                name = names[0] if len(names) == 1 else None
            if not base or not name:
                continue
            for n, u in frame[[name, base]].dropna().itertuples(index=False, name=None):
                values.setdefault(item_key(n), set()).add(item_key(u))
    return {k: next(iter(v)) for k, v in values.items() if len(v) == 1 and next(iter(v))}


def _catalog_item_names(shop: Shop, columns: dict, df: pd.DataFrame, present: dict):
    """Tra tên hàng từ danh mục qua mã có nghĩa đã biết; không join làm nhân dòng.

    Chỉ lấp tên còn thiếu. Mọi mã bán phải khớp; nhiều danh mục mâu thuẫn hoặc
    một mã có hai tên khác nghĩa thì bỏ tra cứu thay vì lấy dòng đầu tiên.
    """
    if _find(present, M_ITEM_NAME):
        return df, present, None
    code = _find(present, M_ITEM_CODE)
    if code is None or df[code].isna().any():
        return df, present, None
    keys = df[code].astype(str).str.strip()
    if keys.eq("").any():
        return df, present, None
    wanted = set(keys)
    candidates = []
    for key, cols in columns.items():
        for table in _tables_for(shop, key):
            cat = shop.tables[table]
            cc = _find(cols, M_ITEM_CODE)
            nc = _find(cols, (M_ITEM_NAME, "tên sản phẩm", "tên mặt hàng"))
            if nc is None:
                names = [c for c, m in cols.items() if fold(m).strip() == "ten"]
                nc = names[0] if len(names) == 1 else None
            if cc is None or nc is None or cc not in cat or nc not in cat:
                continue
            pairs = cat[[cc, nc]].dropna().copy()
            pairs[cc] = pairs[cc].astype(str).str.strip()
            pairs = pairs[pairs[cc].isin(wanted)]
            if pairs.empty or wanted - set(pairs[cc]):
                continue
            if pairs[nc].astype(str).str.strip().eq("").any():
                continue
            canonical = pairs[nc].map(item_key)
            if canonical.groupby(pairs[cc]).nunique().gt(1).any():
                return df, present, None
            lookup = pairs.drop_duplicates(cc).set_index(cc)[nc]
            candidates.append((table, lookup))
    if not candidates:
        return df, present, None
    table, lookup = candidates[0]
    expected = keys.map(lookup).map(item_key)
    if any(not keys.map(other).map(item_key).equals(expected) for _, other in candidates[1:]):
        return df, present, None
    name = "__catalog_item_name__"
    while name in df:
        name += "_"
    enriched = df.copy()
    enriched[name] = keys.map(lookup)
    return enriched, {**present, name: M_ITEM_NAME}, table


# kênh bán / hình thức thanh toán ghi trên đơn: (loại, nhãn) ← giá trị trong dữ liệu (mã tiếng Anh hoặc tiếng Việt)
ORDER_ATTRS = [
    ("pay", "cod", r"\bcod\b|thu ho"), ("pay", "cash", r"\bcash\b|tien mat"),
    ("pay", "transfer", r"transfer|chuyen khoan|\bbank\b"), ("pay", "ewallet", r"ewallet|e wallet|vi dien tu|momo"),
    ("pay", "debt", r"\bdebt\b|ghi no"),
    ("channel", "phone", r"\bphone\b|dien thoai"), ("channel", "zalo", r"zalo"),
    ("channel", "store", r"\bstore\b|tai quay|tai cua hang|tai tiem|truc tiep"),
    ("channel", "market", r"\bsan\b|tmdt|shopee|lazada|tiki"),
]


def _attr_of(v) -> tuple[str, str] | None:
    f = fold(str(v)).replace("_", " ")
    hits = [(k, lab) for k, lab, pat in ORDER_ATTRS if re.search(pat, f)]
    return hits[0] if len(hits) == 1 else None


_EXTRA_MEANING = {"disc": ("giảm giá trên cả đơn", "giảm giá của đơn", "giảm giá đơn", "chiết khấu đơn"),
                  "ship": ("phí giao", "phí ship", "phí vận chuyển")}
_EXTRA_NAME = {"disc": r"^(giamgia|chietkhau|ck)(don|hoadon)$|^orderdiscount$",
               "ship": r"^phi(ship|giao|giaohang|vanchuyen)$|^shippingfee$"}


def _order_extras(readme_cols: dict, frames: list[tuple[pd.DataFrame, str]]) -> pd.DataFrame | None:
    """Khoản cấp đơn ngoài tiền dòng hàng: order_id → disc (giảm giá trên cả đơn), ship (phí giao); cột không thấy thì
    không có trong kết quả. Chỉ dùng khi rulebook split tính chúng vào doanh thu đơn (datamate/rule_policy.py)."""
    meanings = {c: m for cols in readme_cols.values() for c, m in cols.items()}
    out = None
    for df, oid in frames:
        got = {}
        for c in df.columns:
            key = re.sub(r"[^a-z0-9]", "", fold(str(c)))
            for role in ("disc", "ship"):
                by_meaning = c in meanings and any(re.match(re.escape(fold(m)) + r"\b", fold(meanings[c]))
                                                   for m in _EXTRA_MEANING[role])  # 'giảm giá dòng' ≠ 'giảm giá đơn' 
                if role not in got and (by_meaning or re.match(_EXTRA_NAME[role], key)):
                    got[role] = c
        if not got:
            continue
        part = pd.DataFrame({"order_id": df[oid].astype(str)})
        for role, c in got.items():
            part[role] = pd.to_numeric(df[c], errors="coerce")
        part = part.dropna(subset=["order_id"]).groupby("order_id", as_index=False).first()
        out = part if out is None else out.merge(part[["order_id"] + [r for r in got if r not in out]], how="outer",
                                                 on="order_id")
    return out


def _order_attrs(frames: list[tuple[pd.DataFrame, str]]) -> pd.DataFrame | None:
    """order_id → pay / channel (nhãn chuẩn) từ cột chữ ít giá trị mà ≥ 95% giá trị nhận ra cùng một loại."""
    out = None
    for df, oid in frames:
        for c in df.columns:
            s = df[c].dropna()
            if s.empty or not (pd.api.types.is_string_dtype(s) or s.dtype == object):
                continue
            vals = s.astype(str).unique()
            if len(vals) > 30:
                continue
            m = {v: _attr_of(v) for v in vals}
            kinds = {a[0] for a in m.values() if a}
            labels = {a[1] for a in m.values() if a}
            share = s.astype(str).map(lambda v: m[v] is not None).mean()
            if len(kinds) != 1 or len(labels) < 2 or share < 0.95:
                continue
            kind = kinds.pop()
            col = pd.DataFrame({"order_id": df[oid].astype(str),
                                kind: df[c].map(lambda v: m.get(str(v), (None, None))[1] if pd.notna(v) and m.get(str(v)) else None)})
            col = col.dropna().drop_duplicates("order_id")
            if out is None:
                out = col
            elif kind not in out.columns:
                out = out.merge(col, on="order_id", how="outer")
    return out


# rulebook 'Xác định khách': có mã khách thì mỗi mã là một khách; không có thì theo tên ghi trên đơn (hai tên chỉ khác
# cách viết là một khách); đơn khách lẻ không phải khách
CUSTOMER_CODE = re.compile(r"^(ma ?(kh|khach|khach ?hang)( mua)?|makhachhang|makh|customer ?(id|code))$")
CUSTOMER_NAME = re.compile(r"^(ten ?(kh|khach|khach ?hang)|khach ?hang|khach)$")


def _cust_keys(*names):
    from datamate.schema_roles import bilingual_key
    return {bilingual_key(x) for x in names}


_CODE_KEYS = _cust_keys("Mã KH", "Mã khách", "Mã khách hàng", "Mã KH mua", "Mã khách hàng mua")
_NAME_KEYS = _cust_keys("Tên KH", "Tên khách", "Tên khách hàng", "Khách hàng", "Khách")


def is_cust_code(c) -> bool:
    """Cột mã khách, kể cả tên đổi cách viết / dịch ('id_kh', 'CustomerId')."""
    from datamate.schema_roles import bilingual_key
    return bool(CUSTOMER_CODE.match(_fname(c))) or bilingual_key(c) in _CODE_KEYS


def is_cust_name(c) -> bool:
    from datamate.schema_roles import bilingual_key
    return bool(CUSTOMER_NAME.match(_fname(c))) or bilingual_key(c) in _NAME_KEYS


def _name_key(v) -> str:
    """Tên khách chỉ khác hoa / thường, khoảng trắng, dấu câu là một khách; khác chữ (kể cả dấu thanh) là hai khách."""
    t = unicodedata.normalize("NFC", str(v)).lower()
    return " ".join(re.sub(r"[^\w]", " ", t).split())


def _order_customers(frames: list[tuple[pd.DataFrame, str]], merge_variants: bool = True) -> pd.DataFrame | None:
    """order_id → customer (khoá khách) theo cột mã khách, không có thì cột tên khách; khách lẻ bỏ.
    merge_variants=False (rulebook đổi §3): mỗi cách viết tên là một khách."""
    from datamate.parsing import is_walkin
    name_key = _name_key if merge_variants else (lambda v: str(v).strip())
    for is_col, key in ((is_cust_code, lambda v: str(v).strip()), (is_cust_name, name_key)):
        for df, oid in frames:
            cols = [c for c in df.columns if is_col(c)]
            if len(cols) != 1:
                continue
            c = cols[0]
            out = pd.DataFrame({"order_id": df[oid].astype(str),
                                "customer": df[c].map(lambda v: None if pd.isna(v) or is_walkin(v) else key(v))})
            # tên hiển thị cho câu 'khách nào…' (đáp án là tên, mã khách nằm trong alias): cột tên khách nếu có
            names = [n for n in df.columns if is_cust_name(n)]
            out["display"] = (df[names[0]].astype(str).str.strip() if len(names) == 1 else df[c].astype(str).str.strip())
            # mỗi đơn một khách: đơn ghi hai khách khác nhau là nối sai cột
            per = out.dropna().groupby("order_id")["customer"].nunique()
            if len(per) and (per > 1).any():
                return None
            return out.dropna().drop_duplicates("order_id")
    return None


def _has_time(s: pd.Series) -> bool:
    s = s.dropna()
    return not s.empty and ((s.dt.hour != 0) | (s.dt.minute != 0)).mean() > 0.5


def _order_time(df: pd.DataFrame, date_col: str) -> pd.Series | None:
    """Giờ tạo đơn (rulebook: ca theo giờ tạo đơn): cột ngày nếu có giờ, không thì một cột ngày giờ khác cùng bảng
    có tên giờ / thời gian / tạo. None nếu không có."""
    if _has_time(df[date_col]):
        return df[date_col]
    cand = [c for c in df.columns if c != date_col and pd.api.types.is_datetime64_any_dtype(df[c]) and _has_time(df[c])
            and re.search(r"\b(gio|thoi gian|tao|time)\b", _fname(c))]
    return df[cand[0]] if len(cand) == 1 else None


def parse_shifts(rulebook: str) -> dict[str, tuple[int, int]]:
    """Khung giờ ca từ rulebook của split ('ca sáng 06:00–13:59') → {'sang': (phút đầu, phút cuối)}."""
    out = {}
    for m in re.finditer(r"ca (\w+) (\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})", rulebook, re.I):
        out[fold(m.group(1))] = (int(m.group(2)) * 60 + int(m.group(3)), int(m.group(4)) * 60 + int(m.group(5)))
    base = re.search(r"mốc là \**ca (\w+)", rulebook, re.I)  # 'So ca chiều với ca sáng: mốc là **ca sáng**'
    if out and base:
        out["_base"] = fold(base.group(1))
    return out


M_ITEM_CODE = "mã hàng"
M_GROUP = "nhóm hàng"
M_INGREDIENT = "hoạt chất"
M_RX = "thuốc kê đơn"


def _name_to_code(shop: Shop, readme_cols: dict, code_col: str) -> dict | None:
    """Tên hàng (item_key) → mã hàng, tra từ bảng danh mục có cả cột tên và cột mã cùng tên cột `code_col`;
    None nếu không có hoặc một tên ứng với nhiều mã."""
    for key, cols in readme_cols.items():
        for t in _tables_for(shop, key):
            m = shop.tables[t]
            if code_col not in m.columns:
                continue
            mp = {c: v for c, v in cols.items() if c in m.columns}
            name = next((c for c, v in mp.items() if fold(v).strip() in ("ten", "ten hang", "ten san pham")), None)
            name = name or next((c for c in m.columns if re.sub(r"[\s_]", "", fold(c)) in ("tensanpham", "tenhang")), None)
            if not name:
                continue
            pairs = m.dropna(subset=[name, code_col]).assign(_k=lambda x: x[name].map(item_key))
            if (pairs.groupby("_k")[code_col].nunique() > 1).any():
                return None
            return dict(zip(pairs["_k"], pairs[code_col].astype(str)))
    return None


def _truthy(v) -> bool:
    """Ô cờ kê đơn: TRUE / 1 / có / x."""
    if pd.isna(v):
        return False
    if isinstance(v, (bool, int, float)):
        return bool(v)
    return fold(str(v)).strip() in ("true", "1", "co", "x", "yes", "y", "ke don")


def _line_groups(shop: Shop, readme_cols: dict, df: pd.DataFrame, present: dict[str, str],
                 meaning: str = M_GROUP) -> pd.Series | None:
    """Nhóm hàng của từng dòng bán (rulebook: theo cột nhóm hàng): cột nhóm ngay trên dòng, hoặc tra danh mục
    hàng (mã hàng / tên + nhóm hàng) theo mã hàng, không có mã thì theo tên. None nếu không chắc."""
    own = _find(present, meaning)
    if own:
        return df[own]
    line_code, line_name = _find(present, M_ITEM_CODE), _find(present, M_ITEM_NAME)
    for key, cols in readme_cols.items():
        for t in _tables_for(shop, key):
            m = shop.tables[t]
            mp = {c: v for c, v in cols.items() if c in m.columns}
            grp, code = _find(mp, meaning), _find(mp, M_ITEM_CODE)
            name = next((c for c, v in mp.items() if fold(v).strip() in ("ten", "ten hang")), None)
            if not grp:
                continue
            bridge = None
            if not line_code and line_name and code and not name:
                bridge = _name_to_code(shop, readme_cols, code)  # dòng bán chỉ có tên, danh mục thuốc chỉ có mã
            if line_code and code:
                values = m.dropna(subset=[code]).assign(_key=lambda x: x[code].astype(str))
                if values.groupby('_key')[grp].nunique(dropna=False).gt(1).any():
                    return None
                lut = m.dropna(subset=[code]).drop_duplicates(code).set_index(code)[grp]
                lut.index = lut.index.astype(str)
                keys = df[line_code].astype(str)
            elif bridge is not None:
                values = m.dropna(subset=[code]).assign(_key=lambda x: x[code].astype(str))
                if values.groupby('_key')[grp].nunique(dropna=False).gt(1).any():
                    return None
                lut = m.dropna(subset=[code]).drop_duplicates(code).set_index(code)[grp]
                lut.index = lut.index.astype(str)
                keys = df[line_name].map(lambda v: bridge.get(item_key(v)) if pd.notna(v) else None)
            elif line_name and name:
                values = m.dropna(subset=[name]).assign(_k=lambda x: x[name].map(item_key))
                if values.groupby('_k')[grp].nunique(dropna=False).gt(1).any():
                    return None
                lut = m.dropna(subset=[name]).assign(_k=lambda x: x[name].map(item_key)).drop_duplicates("_k")
                lut = lut.set_index("_k")[grp]
                keys = df[line_name].map(lambda v: item_key(v) if pd.notna(v) else None)
            else:
                continue
            found = keys.isin(lut.index).mean()
            # nhóm hàng: mọi dòng phải tra ra; hoạt chất: danh mục thuốc có thể chỉ liệt kê thuốc (hàng khác = không
            # có hoạt chất), nhưng tra ra quá ít dòng nghĩa là nối sai cột
            if found < (0.95 if meaning == M_GROUP else 0.3):
                return None
            return keys.map(lut)
    return None


M_GOODS_TOTAL = ("tổng tiền hàng của đơn", "tiền hàng của đơn")


def goods_agreement(shop: Shop, readme_cols: dict, lines: pd.DataFrame, line_table: str) -> float | None:
    """Đối chiếu chéo: tỉ lệ đơn mà tổng thành tiền các dòng bằng cột 'tiền hàng của đơn' ở bảng đơn (lệch ≤ 1đ).
    None khi shop không có cột đó để so."""
    for key, cols in readme_cols.items():
        for t in _tables_for(shop, key):
            if t == line_table:
                continue
            df = shop.tables[t]
            present = {c: m for c, m in cols.items() if c in df.columns}
            oid, goods = _find(present, M_ORDER_ID), _find(present, M_GOODS_TOTAL)
            if oid is None or goods is None:
                continue
            ref = pd.Series(pd.to_numeric(df[goods], errors="coerce").values, index=df[oid].astype(str))
            ref = ref[~ref.index.duplicated()].dropna()
            got = lines.groupby("order_id")["amount"].sum()
            both = got.index.intersection(ref.index)
            if len(both) < 20:
                return None
            agree = float(((got[both] - ref[both]).abs() <= 1).mean())
            # cột tiền đơn do thẻ đề xuất mà không khớp: thẻ đoán sai cột (vd 'Khách trả' đã trừ bớt), bỏ cột đó
            if agree < 0.95 and (t, goods) in getattr(shop, "card_added", set()):
                continue
            return agree
    return None


def view_contract(shop: Shop, table: str, roles: dict[str, str | None], agree: float | None) -> dict:
    """Contract của bảng đơn chuẩn: nguồn, grain, đơn vị và bằng chứng từng vai trò lõi.
    readme: README khai (kể cả tên đổi chính tả / dịch); card: thẻ LLM đã qua kiểm số liệu; inferred: chỉ suy từ tên cột."""
    from datamate.schema_roles import align_declared_columns
    declared = set()
    for key, cols in parse_readme_columns(shop.readme).items():
        for t in _tables_for(shop, key):
            declared.update((t, c) for c in align_declared_columns(shop.tables[t], cols))
    card = getattr(shop, "card_added", set())
    evidence = {r: None if c is None else "readme" if (table, c) in declared else "card" if (table, c) in card
                else "inferred" for r, c in roles.items()}
    gaps = [r for r in ("order_id", "amount") if evidence[r] == "inferred"]
    return {"table": table, "grain": "dòng hàng", "units": "VND", "roles": roles, "evidence": evidence,
            "crosscheck": agree, "gaps": gaps if agree is None else []}


def view_report(view: "OrderView | None") -> str:
    """Báo cáo ngắn cho prompt agent (DS-STAR: phân tích file trước; AutoDCWorkflow: báo cáo chất lượng dữ liệu)."""
    if view is None:
        why = OrderView.why or "không rõ"
        return ("## Bảng đơn chuẩn\nChưa dựng được bảng đơn chuẩn đã kiểm (" + why + "): SEM.* về đơn có thể "
                "trả None; tự tìm cột mã đơn / ngày / trạng thái / tiền dòng, đọc kỹ README và rulebook.")
    s = view.source or {}
    lines = ["## Bảng đơn chuẩn (đã kiểm chéo số liệu; SEM.* tính theo đúng các cột này)",
             f"- dòng bán: T[{s.get('table')!r}], mã đơn `{s.get('order_id')}`, tiền dòng `{s.get('amount')}`",
             f"- ngày đặt `{s.get('date')}`, trạng thái `{s.get('status')}` ở T[{s.get('status_table')!r}]"]
    if s.get("nongoods_rows"):
        lines.append(f"- {s['nongoods_rows']} dòng không phải hàng (ô số lượng trống: voucher, phí ship…) KHÔNG tính "
                     "vào doanh thu hàng")
    if s.get("duplicate_rows"):
        lines.append(f"- {s['duplicate_rows']} dòng trùng hệt nhau trong cùng đơn: tính một lần")
    if s.get("card_cols"):
        lines.append("- vai trò cột do model đề xuất và đã qua kiểm số liệu: " + ", ".join(s["card_cols"]))
    return "\n".join(lines)


def _notes_status(df: pd.DataFrame) -> str | None:
    """Sổ tự ghi không có cột trạng thái (rulebook 'Sổ không có cột trạng thái': trạng thái ghi ở cột Ghi chú, ghi chú
    không nhắc trạng thái là đơn hoàn thành) → cột ghi chú làm trạng thái. Cột tên kiểu trạng thái / tình trạng thì
    phần lớn ô có ghi phải phân loại ra huỷ / trả (tránh 'tình trạng máy': trầy, móp…)."""
    for c in df.columns:
        f = fold(c)
        if re.match(r"ghi ?chu|note", f):
            return c
        if re.match(r"trang thai|tinh trang", f):
            v = df[c].dropna().astype(str)
            if len(v) and v.map(classify_status).isin(["cancelled", "returned"]).mean() >= 0.8:
                return c
    return None


@dataclass
class OrderView:
    """Mỗi dòng: một dòng hàng (hoặc một đơn) với order_id, date, status, amount."""
    lines: pd.DataFrame
    orders: pd.DataFrame  # order_id, date, status (một dòng một đơn)
    items: pd.DataFrame | None = None  # order_id, item, name, bqty, bunit (xem _items)
    debt: object | None = None  # datamate.debt.DebtView
    groups: pd.DataFrame | None = None  # order_id, amount, group (nhóm hàng của từng dòng bán)
    attrs: pd.DataFrame | None = None  # order_id, pay / channel (nhãn chuẩn ORDER_ATTRS)
    ingredients: pd.DataFrame | None = None  # order_id, amount, group (hoạt chất ghi trong danh mục thuốc)
    shifts: dict | None = None  # khung giờ ca đọc từ rulebook (parse_shifts), gán sau khi dựng
    imports: object | None = None  # datamate.imports.ImportView
    customers: pd.DataFrame | None = None  # order_id, customer (khách lẻ không có dòng)
    rx: pd.DataFrame | None = None  # order_id, amount, group ('rx' thuốc kê đơn / 'otc' mọi hàng còn lại)
    item_lines: pd.DataFrame | None = None  # order_id, amount, group (khoá tên hàng), gname — doanh thu một mặt hàng
    source: dict | None = None  # bảng / cột đã dùng và phần dữ liệu đã bỏ (báo cáo cho agent, soi lỗi)

    why: ClassVar[str | None] = None  # lý do lần dựng view gần nhất thất bại (soi lỗi shop lạ)

    @classmethod
    def _fail(cls, why: str):
        cls.why = why
        return None

    @classmethod
    def from_shop(cls, shop: Shop) -> "OrderView | None":
        readme_cols = shop_columns(shop)
        line_src = order_src = None
        for key, cols in readme_cols.items():
            for t in _tables_for(shop, key):
                df = shop.tables[t]
                present = {c: m for c, m in cols.items() if c in df.columns}
                oid = _find(present, M_ORDER_ID)
                if oid is None:
                    continue
                amt = _find(present, M_LINE_AMOUNT) or _find(present, M_ORDER_GOODS)
                date, status = _find(present, M_DATE), _find(present, M_STATUS)
                # A header total is a fallback, not a substitute for sales lines.
                # Prefer an explicit line amount regardless of file/table ordering.
                better_lines = (line_src is not None and _find(present, M_LINE_AMOUNT) == amt
                                and t != line_src[0]
                                and (t, amt) not in getattr(shop, 'card_added', set())
                                and _find(line_present, M_LINE_AMOUNT) != line_src[2])
                if amt and (line_src is None or better_lines):
                    line_src = (t, oid, amt, date, status, _find(present, M_RETURN_QTY))
                    line_present = present
                elif amt and line_src[:3] == (t, oid, amt):
                    # Chỉ lấp tên còn thiếu. Không ghi đè vai trò SL/giá đã có
                    # trong README bằng ánh xạ phụ từ một thẻ cũ.
                    if _find(line_present, M_ITEM_NAME) is None:
                        extra_name = _find(present, M_ITEM_NAME)
                        if extra_name:
                            line_present[extra_name] = present[extra_name]
                if date and status and order_src is None:
                    order_src = (t, oid, date, status)
        if line_src is None and not getattr(shop, "_gap_fill_tried", False):
            filled = gap_filled_columns(shop, readme_cols)
            if filled != readme_cols:
                shop._gap_fill_tried = True
                orig = shop_columns
                try:
                    globals()["shop_columns"] = lambda _s: filled
                    return cls.from_shop(shop)
                finally:
                    globals()["shop_columns"] = orig
                    shop._gap_fill_tried = False
        if line_src is None:
            return cls._fail('không có bảng dòng bán (mã đơn + tiền)')
        t, oid, amt, date, status, ret = line_src
        n0 = len(shop.tables[t])
        df = clean_order_rows(shop, t)
        # bảng đơn (mỗi mã đơn một dòng) có cột tiền hàng mà 'tiền dòng' chỉ do thẻ đề xuất: thẻ hay gán nhầm tổng phải
        # trả (gồm phí giao, đã trừ giảm giá đơn) làm tiền dòng; doanh thu là tiền hàng (rulebook §1)
        goods = _find(line_present, M_ORDER_GOODS)
        if goods and goods != amt and (t, amt) in getattr(shop, "card_added", set()) and df[oid].dropna().is_unique:
            amt = goods
        notes = False
        if date and not status and order_src is None:
            status = _notes_status(df)
            notes = status is not None
        dups, nongoods = n0 - len(df), 0
        # dòng không phải hàng ghi chung bảng (Voucher số âm, Phí ship…: ô SL trống) → không phải doanh thu hàng
        q = _qty_col(line_present)
        if q is not None:
            qn = pd.to_numeric(df[q], errors="coerce")
            # dòng hàng thật luôn có SL; ngưỡng 0.5 chỉ để không xoá sạch bảng khi nhận nhầm cột SL (dev: Voucher,
            # Dùng điểm, Phí ship chiếm ~35% dòng)
            if 0 < qn.isna().mean() < 0.5:
                nongoods = int(qn.isna().sum())
                df = df[qn.notna()]
        lines = pd.DataFrame({"order_id": df[oid].astype(str), "amount": pd.to_numeric(df[amt], errors="coerce")})
        if ret:
            lines["ret_qty"] = pd.to_numeric(df[ret], errors="coerce").fillna(0)
        if date and status:
            src, src_date = df, date
            orders = pd.DataFrame({"order_id": df[oid].astype(str), "date": df[date], "status": df[status]})
        elif order_src is not None:
            ot, ooid, odate, ostatus = order_src
            o = shop.tables[ot]
            src, src_date = o, odate
            orders = pd.DataFrame({"order_id": o[ooid].astype(str), "date": o[odate], "status": o[ostatus]})
        else:
            return cls._fail('không có ngày + trạng thái đơn')
        if not pd.api.types.is_datetime64_any_dtype(orders["date"]):
            return cls._fail('cột ngày không phải kiểu ngày')
        orders["ts"] = _order_time(src, src_date)
        orders = orders.dropna(subset=["order_id"]).groupby("order_id", as_index=False).first()
        orders["date"] = orders["date"].dt.normalize()
        # cột kiểu ghi chú (chỉ ghi khi đơn huỷ / trả, phần lớn trống): ô trống = đơn bình thường
        given = orders["status"].dropna().astype(str)
        if notes or orders["status"].notna().mean() < 0.5 or (
                len(given) and given.map(classify_status).isin(["cancelled", "returned"]).mean() >= 0.8):
            orders["status"] = orders["status"].fillna("")
        orders["cls"] = orders["status"].map(classify_status)
        if orders["cls"].isna().mean() > 0.01:
            return cls._fail('trạng thái không phân loại được')
        # cột tiền là tổng cả đơn lặp lại trên mỗi dòng hàng (fan-out): cộng theo dòng sẽ nhân doanh thu lên
        # → không dùng bộ giải tất định cho shop này. Đơn thật có mọi dòng cùng giá chỉ chiếm vài phần trăm.
        sizes = lines.groupby("order_id")["amount"].agg(["size", "nunique"])
        multi = sizes[sizes["size"] > 1]
        if len(multi) and (multi["nunique"] == 1).mean() > 0.5:
            return cls._fail('tiền dòng là tổng đơn lặp lại (fan-out)')
        # đối chiếu chéo với tiền hàng ghi ở bảng đơn: lệch nhiều nghĩa là đọc sai cột tiền / sai cách nối → để agent
        agree = goods_agreement(shop, readme_cols, lines, t) if amt != _find(line_present, M_ORDER_GOODS) else None
        if agree is not None and agree < 0.95:
            return cls._fail('tổng dòng lệch tiền hàng của đơn')
        contract = view_contract(shop, t, {"order_id": oid, "amount": amt, "date": date, "status": status}, agree)
        if "ret_qty" in lines:
            returned = set(lines.loc[lines["ret_qty"] > 0, "order_id"])
            orders.loc[orders["order_id"].isin(returned), "cls"] = "returned"
        item_df, item_present, item_catalog = _catalog_item_names(shop, readme_cols, df, line_present)
        items = _items(item_df, oid, item_present, _conversions(shop, readme_cols), catalog_base_units(shop, readme_cols))
        from datamate.debt import DebtView  # tránh import vòng
        g = _line_groups(shop, readme_cols, df, line_present)
        groups = None if g is None else pd.DataFrame({"order_id": lines["order_id"], "amount": lines["amount"],
                                                      "group": g.map(lambda v: item_key(v) if pd.notna(v) else None),
                                                      "gname": g})
        frames = [(df, oid)]
        if order_src is not None:
            frames.append((shop.tables[order_src[0]], order_src[1]))
        attrs = _order_attrs(frames)
        customers = _order_customers(frames, getattr(getattr(shop, 'rule_policy', None), 'merge_name_variants', True))
        if customers is None:
            # Unusual customer column names are safe when README declares their role.
            declared = {}
            for cols in readme_cols.values():
                for col, meaning in cols.items():
                    if fold(meaning).startswith('ma khach'):
                        declared[col] = 'Mã khách'
            adapted = [(frame.rename(columns={c: target for c, target in declared.items()
                                             if c in frame and target not in frame}), ident)
                       for frame, ident in frames]
            customers = _order_customers(adapted, getattr(getattr(shop, 'rule_policy', None), 'merge_name_variants', True))
        ing = _line_groups(shop, readme_cols, df, line_present, M_INGREDIENT)
        ingredients = None if ing is None else pd.DataFrame({
            "order_id": lines["order_id"], "amount": lines["amount"],
            "group": ing.map(lambda v: item_key(v) if pd.notna(v) and str(v).strip() else None), "gname": ing})
        rxv = _line_groups(shop, readme_cols, df, line_present, M_RX)
        rx = None if rxv is None else pd.DataFrame({
            "order_id": lines["order_id"], "amount": lines["amount"],
            "group": rxv.map(lambda v: "rx" if _truthy(v) else "otc"), "gname": ""})
        from datamate.imports import ImportView
        try:
            imports = ImportView.from_shop(shop)
        except Exception:  # bảng nhập lạ: bỏ, để agent làm
            imports = None
        nm = _find(item_present, M_ITEM_NAME)
        # khoá tên của từng cách viết → khoá mặt hàng đã gộp (_items): doanh thu một mặt hàng cộng đủ mọi cách viết
        alias = {} if items is None else {k: v for k, v in zip(items["name"].map(lambda n: item_key(n) if pd.notna(n)
                                                                                 else None), items["item"]) if k}
        item_lines = None if nm is None else pd.DataFrame({
            "order_id": lines["order_id"], "amount": lines["amount"],
            "group": item_df[nm].map(lambda v: alias.get(item_key(v), item_key(v)) if pd.notna(v) else None).values,
            "gname": item_df[nm].values})
        source = {"table": t, "order_id": oid, "amount": amt, "date": date or (order_src or [None] * 3)[2],
                  "status": status or (order_src or [None] * 4)[3], "status_table": t if status else
                  (order_src or [None])[0], "duplicate_rows": dups, "nongoods_rows": nongoods,
                  "card_cols": sorted(c for tt, c in getattr(shop, "card_added", set()) if tt == t), "contract": contract}
        if item_catalog:
            source["item_name_catalog"] = item_catalog
        view = cls(lines[["order_id", "amount"]], orders, items, DebtView.from_shop(shop), groups, attrs, ingredients,
                   imports=imports, customers=customers, rx=rx, source=source,
                   item_lines=item_lines)
        view.order_extras = _order_extras(readme_cols, frames)
        view.item_alias = {k: v for k, v in alias.items() if k != v}
        return view

    # --- truy vấn ---------------------------------------------------------------
    def date_range(self) -> tuple[dt.date, dt.date]:
        d = self.orders["date"].dropna()
        return d.min().date(), d.max().date()

    def covers(self, period) -> bool:
        lo, hi = self.date_range()
        return not (period[1] < lo or period[0] > hi)

    def _orders_in(self, period, as_of) -> pd.DataFrame:
        end = min(period[1], as_of)
        d = self.orders["date"]
        return self.orders[(d >= pd.Timestamp(period[0])) & (d <= pd.Timestamp(end))]

    def _valid(self, o: pd.DataFrame) -> pd.DataFrame:
        # luật 'Đơn tính vào doanh thu' của split (datamate/rule_policy.py)
        policy = getattr(self, "policy", None)
        ok = ["completed", "delivering"] if policy is None or policy.delivering_valid else ["completed"]
        return o[o["cls"].isin(ok)]

    def revenue(self, period, as_of) -> int | None:
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        return _sum_money(self.lines.loc[self.lines["order_id"].isin(ids), "amount"])

    def order_count(self, period, as_of) -> int:
        return int(self._valid(self._orders_in(period, as_of))["order_id"].nunique())

    def cancelled_count(self, period, as_of) -> int:
        o = self._orders_in(period, as_of)
        return int(o.loc[o["cls"] == "cancelled", "order_id"].nunique())

    def avg_order_value(self, period, as_of) -> int | None:
        """Giá trị trung bình một đơn: doanh thu / số đơn hợp lệ, làm tròn về đồng (rulebook §4, §9)."""
        n = self.order_count(period, as_of)
        total = self.revenue(period, as_of) if n else None
        return self._round_aov(Decimal(total) / n) if total is not None else None

    def _round_aov(self, x) -> int:
        """Luật 'Giá trị trung bình một đơn' của split: nửa đồng làm tròn lên (train) hoặc làm tròn xuống."""
        policy = getattr(self, "policy", None)
        if policy is not None and policy.aov_floor:
            return int(Decimal(str(x)).to_integral_value(rounding=ROUND_FLOOR))
        return round_money(x)

    def apply_policy(self, policy) -> bool:
        """Áp tham số luật lên view (một lần). Doanh thu đơn gồm khoản cấp đơn → thêm một dòng phụ / đơn (không phải
        mặt hàng, nên doanh thu theo nhóm / mặt hàng vẫn chỉ cộng dòng hàng). False: shop thiếu cột luật cần."""
        if getattr(self, "_policy_applied", None) == policy:
            return True
        self.policy = policy
        base = getattr(self, "_goods_lines", None)
        if base is None:
            base = self._goods_lines = self.lines
        self.lines = base
        self._policy_applied = policy
        need = [r for r, on in (("disc", policy.order_discount), ("ship", policy.ship_fee)) if on]
        if not need:
            return True
        ex = getattr(self, "order_extras", None)
        if ex is None or any(r not in ex for r in need):
            # khoản cấp đơn không biết → tiền mọi đơn không biết (NaN); số đơn, doanh thu nhóm / mặt hàng vẫn đúng
            unknown = pd.DataFrame({"order_id": self.orders["order_id"].dropna().unique(), "amount": float("nan")})
            self.lines = pd.concat([base, unknown], ignore_index=True)
            return False
        adj = sum(ex[r].fillna(0) * (-1 if r == "disc" else 1) for r in need)
        extra = pd.DataFrame({"order_id": ex["order_id"], "amount": adj})
        self.lines = pd.concat([base, extra[extra["amount"] != 0]], ignore_index=True)
        return True

    def order_money(self) -> pd.Series:
        """Tiền từng đơn (tổng dòng); đơn có dòng chưa biết tiền (NaN) → NaN, không cộng bỏ qua."""
        amount = self.lines["amount"]
        per = amount.groupby(self.lines["order_id"]).sum()
        return per.mask(amount.isna().groupby(self.lines["order_id"]).any())

    def best_revenue_day(self, period, as_of) -> str | None:
        """Ngày doanh thu cao nhất trong kỳ; bằng nhau lấy ngày sớm hơn (rulebook §9: xếp tăng dần)."""
        o = self._valid(self._orders_in(period, as_of))
        if o.empty:
            return None
        per = self.order_money().reindex(o["order_id"], fill_value=0)
        if per.isna().any():
            return None
        daily = per.groupby(o["date"].values).sum()
        top = daily.max()
        return min(d for d, v in daily.items() if v == top).strftime("%Y-%m-%d")

    def item_keys(self) -> set[str]:
        return set() if self.items is None else set(self.items["item"])

    def units(self) -> set[str]:
        return set() if self.items is None else set(self.items["bunit"].dropna())

    def qty_sold(self, period, as_of, unit: str, item: str | None = None, size: str | None = None) -> float | None:
        """Số lượng bán quy về đơn vị cơ bản (rulebook §5 'Bán được bao nhiêu <đơn vị>'): đơn hợp lệ trong kỳ,
        chỉ mặt hàng có đơn vị cơ bản là `unit`; `item` = một mặt hàng (khoá tên), `size` = tên kết thúc '(M)'…
        None khi mặt hàng hỏi tới không quy đổi chắc chắn được."""
        if self.items is None:
            return None
        it = self.items
        if item is not None:
            it = it[it["item"] == item]
            if it.empty or it["bunit"].isna().any() or set(it["bunit"]) != {unit}:
                return None
        if size is not None:
            it = it[it["name"].astype(str).str.strip().str.lower().str.endswith(f"({size})")]
        if item is None and it["bunit"].isna().any():
            return None  # có dòng không rõ đơn vị cơ bản: tổng theo đơn vị không chắc đúng
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        sel = it[it["order_id"].isin(ids) & (it["bunit"] == unit)]
        if sel["bqty"].isna().any():
            return None
        quantities = sel["bqty"]
        if all(isinstance(value, Integral) for value in quantities):
            return sum(int(value) for value in quantities)
        total = quantities.sum()
        v = float(total)
        return int(v) if v.is_integer() else round2(v)

    def attr_labels(self) -> dict[str, set[str]]:
        """{'pay' | 'channel': các nhãn có trong dữ liệu}."""
        if self.attrs is None:
            return {}
        return {k: set(self.attrs[k].dropna()) for k in ("pay", "channel") if k in self.attrs.columns}

    def attr_revenue(self, period, as_of, kind: str, label: str) -> int | None:
        """Doanh thu các đơn có hình thức thanh toán / kênh ghi trên đơn = label (rulebook §1 'theo hình thức ghi
        trên đơn': đơn ghi nợ không tính vào tiền mặt hay chuyển khoản, COD là hình thức riêng)."""
        if self.attrs is None or kind not in self.attrs.columns:
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        sel = set(self.attrs.loc[self.attrs[kind] == label, "order_id"]) & ids
        return _sum_money(self.lines.loc[self.lines["order_id"].isin(sel), "amount"])

    def item_of(self, name: str) -> str:
        """Khoá mặt hàng của một tên (mọi cách viết của cùng mã hàng về một khoá, xem _merge_by_code)."""
        k = item_key(name)
        return (getattr(self, "item_alias", None) or {}).get(k, k)

    def item_name_pairs(self) -> list[tuple[str, str]]:
        """(khoá mặt hàng, tên) cho mọi cách viết tên trong dòng bán — để khớp tên trong câu hỏi."""
        if self.item_lines is None:
            return []
        g = self.item_lines.dropna(subset=["group"]).drop_duplicates(["group", "gname"])
        return list(zip(g["group"], g["gname"].astype(str)))

    @staticmethod
    def _names(frame: pd.DataFrame | None) -> dict[str, str]:
        if frame is None:
            return {}
        g = frame.dropna(subset=["group"]).drop_duplicates("group")
        return dict(zip(g["group"], g["gname"].astype(str)))

    def _frame_revenue(self, frame: pd.DataFrame | None, period, as_of, key: str) -> int | None:
        if frame is None:
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        g = frame[frame["order_id"].isin(ids) & (frame["group"] == key)]
        return _sum_money(g["amount"])

    def item_revenue(self, period, as_of, item: str) -> int | None:
        """Doanh thu một mặt hàng (khoá tên): tổng thành tiền các dòng của mặt hàng trên đơn hợp lệ trong kỳ."""
        if self.item_lines is None or item not in set(self.item_lines["group"].dropna()):
            return None
        return self._frame_revenue(self.item_lines, period, as_of, item)

    def group_order_count(self, period, as_of, group: str) -> int | None:
        """Số đơn hợp lệ trong kỳ có ít nhất một mặt hàng thuộc nhóm (rulebook: số đơn có một mặt hàng = đơn §1)."""
        if self.groups is None or group not in self.group_names():
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        g = self.groups[self.groups["order_id"].isin(ids) & (self.groups["group"] == group)]
        return int(g["order_id"].nunique())

    def days_revenue_min(self, period, as_of, x: float) -> int | None:
        """Số ngày trong kỳ có doanh thu (đơn hợp lệ) từ x trở lên."""
        o = self._valid(self._orders_in(period, as_of))
        lines = self.lines[self.lines["order_id"].isin(set(o["order_id"]))].copy()
        lines["amount"] = lines["amount"].map(_decimal_or_nan)
        if not _decimal_or_nan(x).is_finite() or not lines["amount"].map(lambda v: v.is_finite()).all():
            return None
        per = lines.groupby("order_id")["amount"].sum()
        day = o.set_index("order_id")["date"]
        daily = per.groupby(day.reindex(per.index).values).sum()
        return int((daily >= Decimal(str(x))).sum())

    def group_names(self) -> dict[str, str]:
        """{khoá nhóm: tên nhóm gốc}."""
        return self._names(self.groups)

    def group_revenue(self, period, as_of, group: str) -> int | None:
        """Doanh thu nhóm hàng (rulebook: đúng tên nhóm, đơn hợp lệ §1)."""
        return self._frame_revenue(self.groups, period, as_of, group)

    def rx_revenue(self, period, as_of, rx: bool) -> int | None:
        """Doanh thu thuốc kê đơn (rx=True) / hàng không kê đơn (mọi sản phẩm còn lại) — rulebook §6."""
        return self._frame_revenue(self.rx, period, as_of, "rx" if rx else "otc")

    def ingredient_names(self) -> dict[str, str]:
        return self._names(self.ingredients)

    def ingredient_revenue(self, period, as_of, ingredient: str) -> int | None:
        """Doanh thu các thuốc có hoạt chất ghi đúng như vậy (rulebook: thuốc phối hợp là loại riêng)."""
        return self._frame_revenue(self.ingredients, period, as_of, ingredient)

    def top_items(self, period, as_of, k: int) -> list[str] | None:
        """Mặt hàng bán chạy (rulebook §2 'Bán chạy'): xếp theo số lượng quy về đơn vị cơ bản trên đơn hợp lệ,
        khác đơn vị cơ bản vẫn xếp chung. Dòng phí / voucher không phải mặt hàng. Hoà ở ranh giới top k
        (đề loại những câu này) hoặc thiếu số lượng quy đổi → None."""
        if self.items is None:
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        it = self.items[self.items["order_id"].isin(ids)]
        it = it[~it["item"].str.contains(r"^(?:phi |voucher|dung \d* ?diem|giam gia|chiet khau)")]
        if it.empty or it["bqty"].isna().any():
            return None
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            return None
        qty = it.groupby("item")["bqty"].sum().sort_values(ascending=False)
        tied = qty[qty.duplicated(keep=False) & (qty >= qty.iloc[min(k, len(qty)) - 1])]
        if not tied.empty and "rank_key" in it and it["rank_key"].notna().all():
            rank = it.groupby("item")["rank_key"]
            if rank.nunique().reindex(tied.index).gt(1).any():
                return None
            keys = rank.first()
            qty = qty.reindex(sorted(qty.index, key=lambda key: (-qty[key], keys[key])))
        elif not tied.empty:
            return None  # thiếu mã khi hoà: không đoán theo thứ tự dòng
        names = it.groupby("item")["name"].agg(lambda s: s.astype(str).value_counts().index[0])
        return [names[i] for i in qty.index[:k]]

    def _in_shift(self, o: pd.DataFrame, shift: str) -> pd.DataFrame | None:
        win = (self.shifts or {}).get(shift)
        if win is None or "ts" not in o.columns or o["ts"].isna().mean() > 0.01:
            return None
        minute = o["ts"].dt.hour * 60 + o["ts"].dt.minute
        return o[(minute >= win[0]) & (minute <= win[1])]

    def shift_revenue(self, period, as_of, shift: str) -> int | None:
        """Doanh thu một ca (rulebook: theo giờ tạo đơn, khung giờ ca đọc từ rulebook của split)."""
        o = self._in_shift(self._valid(self._orders_in(period, as_of)), shift)
        if o is None:
            return None
        amount = self.lines.loc[self.lines["order_id"].isin(set(o["order_id"])), "amount"]
        return None if amount.isna().any() else int(round(amount.sum()))

    def shift_order_count(self, period, as_of, shift: str) -> int | None:
        o = self._in_shift(self._valid(self._orders_in(period, as_of)), shift)
        return None if o is None else int(o["order_id"].nunique())

    def customers_with_orders(self, period, as_of, n: int) -> tuple[int, int] | None:
        """(số khách có từ n đơn hợp lệ trở lên trong kỳ, số khách có ít nhất một đơn hợp lệ trong kỳ)."""
        if self.customers is None:
            return None
        o = self._valid(self._orders_in(period, as_of)).merge(self.customers, on="order_id")
        per = o.groupby("customer")["order_id"].nunique()
        return int((per >= n).sum()), int(len(per))

    # --- "nhất" theo rulebook public (khách trung thành / sộp / đổi ý, mặt hàng ưa chuộng, ngày đắt khách, giờ cao
    # điểm). Khách hoà xếp theo khoá; ngày/giờ hoà lấy giá trị sớm hơn (§9).
    def _cust(self, o: pd.DataFrame) -> pd.DataFrame | None:
        return None if self.customers is None else o.merge(self.customers, on="order_id")

    def _top_name(self, score: pd.Series, o: pd.DataFrame) -> str | None:
        if score.empty:
            return None
        best = min(score[score == score.max()].index, key=str)
        return o.loc[o["customer"] == best, "display"].value_counts().index[0]

    def customer_most_orders(self, period, as_of) -> str | None:
        """Khách có nhiều đơn hợp lệ nhất trong kỳ."""
        o = self._cust(self._valid(self._orders_in(period, as_of)))
        return None if o is None else self._top_name(o.groupby("customer")["order_id"].nunique(), o)

    def loyal_customer(self, period, as_of) -> str | None:
        """Khách trung thành nhất: nhiều ngày có mua nhất trong kỳ (đơn hợp lệ)."""
        o = self._cust(self._valid(self._orders_in(period, as_of)))
        return None if o is None else self._top_name(o.groupby("customer")["date"].nunique(), o)

    def top_spender(self, period, as_of) -> str | None:
        """Khách mua nhiều tiền nhất: tổng doanh thu các đơn hợp lệ của khách trong kỳ (khác 'sộp' = TB một đơn)."""
        o = self._cust(self._valid(self._orders_in(period, as_of)))
        if o is None:
            return None
        o = o.assign(amount=self.order_money().reindex(o["order_id"], fill_value=0).values)
        if o["amount"].isna().any():
            return None
        return self._top_name(o.groupby("customer")["amount"].sum().round(), o)

    def top_customers(self, period, as_of, k: int) -> list[str] | None:
        """Top-k spending, valid orders only; ties follow customer identity keys."""
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            return None
        o = self._cust(self._valid(self._orders_in(period, as_of)))
        if o is None or o.empty or o['order_id'].duplicated().any():
            return None
        o = o.assign(amount=o['order_id'].map(self.order_money()))
        if o['amount'].isna().any():
            return None
        totals = o.groupby('customer')['amount'].sum()
        keys = sorted(totals.index, key=lambda key: (-totals[key], str(key)))[:k]
        names = o.groupby('customer')['display'].agg(lambda x: x.value_counts().index[0])
        return [str(names[key]) for key in keys]

    def big_spender(self, period, as_of) -> str | None:
        """Khách sộp nhất: giá trị trung bình một đơn cao nhất, chỉ xét khách có từ 3 đơn hợp lệ trong kỳ."""
        o = self._cust(self._valid(self._orders_in(period, as_of)))
        if o is None:
            return None
        o = o.assign(amount=self.order_money().reindex(o["order_id"], fill_value=0).values)
        if o["amount"].isna().any():
            return None
        g = o.groupby("customer").agg(n=("order_id", "nunique"), amt=("amount", "sum"))
        g = g[g["n"] >= 3]
        return self._top_name((g["amt"] / g["n"]).map(round_money), o)

    def fickle_customer(self, period, as_of) -> str | None:
        """Khách hay đổi ý nhất: nhiều đơn huỷ nhất trong kỳ."""
        o = self._orders_in(period, as_of)
        o = self._cust(o[o["cls"] == "cancelled"])
        return None if o is None else self._top_name(o.groupby("customer")["order_id"].nunique(), o)

    def popular_item(self, period, as_of) -> str | None:
        """Mặt hàng được ưa chuộng nhất: có mặt trong nhiều đơn (hợp lệ) nhất. Dòng phí / voucher không tính."""
        if self.items is None:
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        it = self.items[self.items["order_id"].isin(ids)]
        it = it[~it["item"].str.contains(r"^(?:phi |voucher|dung \d* ?diem|giam gia|chiet khau)")]
        n = it.groupby("item")["order_id"].nunique()
        if n.empty or (n == n.max()).sum() > 1:
            return None
        return it.loc[it["item"] == n.idxmax(), "name"].astype(str).value_counts().index[0]

    def busiest_day(self, period, as_of) -> str | None:
        """Ngày đắt khách nhất: hoà lấy ngày sớm nhất theo thứ tự tăng dần §9."""
        o = self._valid(self._orders_in(period, as_of))
        n = o.groupby("date")["order_id"].nunique()
        if n.empty:
            return None
        return min(n[n == n.max()].index).strftime("%Y-%m-%d")

    def peak_hour(self, period, as_of) -> int | None:
        """Giờ cao điểm: khung một giờ tròn theo giờ tạo đơn có nhiều đơn (hợp lệ) nhất → giờ bắt đầu khung."""
        o = self._valid(self._orders_in(period, as_of))
        if "ts" not in o.columns or o.empty or o["ts"].isna().mean() > 0.01:
            return None
        n = o.groupby(o["ts"].dt.hour)["order_id"].nunique()
        return int(min(n[n == n.max()].index))

    def weekday_avg(self, period, as_of, weekday: int, measure: str) -> int | float | None:
        """Trung bình mỗi ngày thứ X của kỳ (weekday 0 = thứ Hai): tổng / số ngày thứ X trong kỳ, kể cả ngày không
        có đơn. measure 'revenue' (làm tròn đồng) hoặc 'orders' (2 chữ số)."""
        end = min(period[1], as_of)
        days = [d for d in pd.date_range(period[0], end) if d.weekday() == weekday]
        if not days:
            return None
        o = self._valid(self._orders_in(period, as_of))
        o = o[o["date"].dt.day_of_week == weekday]
        if measure == "revenue":
            amount = self.lines.loc[self.lines["order_id"].isin(set(o["order_id"])), "amount"]
            return None if amount.isna().any() else round_money(amount.sum() / len(days))
        return round2(o["order_id"].nunique() / len(days))

    def order_count_min(self, period, as_of, x: float) -> int | None:
        """Số đơn hợp lệ trong kỳ có doanh thu (tổng thành tiền các dòng) từ x trở lên."""
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        lines = self.lines[self.lines["order_id"].isin(ids)].copy()
        lines["amount"] = lines["amount"].map(_decimal_or_nan)
        if not _decimal_or_nan(x).is_finite() or not lines["amount"].map(lambda v: v.is_finite()).all():
            return None
        per = lines.groupby("order_id")["amount"].sum()
        return int((per >= Decimal(str(x))).sum())

    def cancel_rate(self, period, as_of) -> float | None:
        o = self._orders_in(period, as_of)
        n = o["order_id"].nunique()
        return round2(self.cancelled_count(period, as_of) / n * 100) if n else None

    def returned_count(self, period, as_of) -> int:
        """Đơn trả hàng, theo kỳ đặt đơn; không phải số lần nhận hàng trả."""
        o = self._orders_in(period, as_of)
        return int(o.loc[o["cls"] == "returned", "order_id"].nunique())

    def item_order_count(self, period, as_of, item: str) -> int | None:
        """Một đơn chứa mặt hàng nhiều dòng vẫn chỉ đếm một lần."""
        if self.items is None or item not in self.item_keys():
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        return int(self.items.loc[self.items["order_id"].isin(ids) & (self.items["item"] == item),
                                  "order_id"].nunique())

    def customer_metric(self, period, as_of, customer: str, metric: str):
        """Mã khách chính xác hoặc tên duy nhất; trùng tên nhiều mã → None.

        Giải tên trên toàn dữ liệu, không chỉ kỳ hỏi, tránh nhận nhầm khách.
        """
        if self.customers is None:
            return None
        c = self.customers
        keys = set(c.loc[c["customer"].astype(str) == str(customer).strip(), "customer"])
        if not keys:
            keys = set(c.loc[c["display"].map(_name_key) == _name_key(customer), "customer"])
        if len(keys) != 1:
            return None
        ids = set(c.loc[c["customer"].isin(keys), "order_id"])
        o = self._valid(self._orders_in(period, as_of))
        ids &= set(o["order_id"])
        n = len(ids)
        if metric == "orders":
            return n
        amount = self.lines.loc[self.lines["order_id"].isin(ids), "amount"]
        if amount.isna().any():
            return None
        total = amount.sum()
        if metric == "revenue":
            return round_money(total)
        return self._round_aov(total / n) if n else None

    def active_days(self, period, as_of) -> int:
        """Ngày có ít nhất một đơn hợp lệ, không phải ngày mở cửa của shop."""
        return int(self._valid(self._orders_in(period, as_of))["date"].dt.normalize().nunique())

    def daily_average(self, period, as_of, metric: str):
        """Chia mọi ngày lịch trong kỳ, kể cả ngày không bán; cắt ở as_of."""
        n = (min(period[1], as_of) - period[0]).days + 1
        if n <= 0:
            return None
        if metric == "revenue":
            return round_money(self.revenue(period, as_of) / n)
        return round2(self.order_count(period, as_of) / n)

    def attr_order_count(self, period, as_of, kind: str, label: str) -> int | None:
        if self.attrs is None or kind not in self.attrs:
            return None
        ids = set(self._valid(self._orders_in(period, as_of))["order_id"])
        attrs = self.attrs[self.attrs["order_id"].isin(ids)]
        # Không biến thiếu thông tin kênh/thanh toán thành số 0.
        if ids - set(attrs["order_id"]) or attrs[kind].isna().any():
            return None
        return int(attrs.loc[attrs[kind] == label, "order_id"].nunique())
