"""Bộ giải tất định (không LLM) cho các dạng câu phổ biến nhất trên bảng đơn hàng chuẩn:
doanh thu / số đơn / số đơn huỷ / tỉ lệ huỷ trong một kỳ, % thay đổi doanh thu so với kỳ trước.

Nguyên tắc chính xác cao: chỉ trả lời khi nhận diện được TOÀN BỘ câu hỏi (kỳ + ý định + từ đệm quen);
còn từ lạ (tên món, nhóm hàng, khách, kênh…) thì trả None để agent xử lý hoặc abstain.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata

import pandas as pd

from datamate.helpers import Periods, pct_change
from datamate.helpers import share as share_pct
from datamate.parsing import fold
from datamate.telex import telex_fix
from datamate.semantic import OrderView


def norm(q: str) -> str:
    """Sửa lỗi gõ Telex, bỏ dấu, lowercase, giữ / - % làm dấu phân cách, chuẩn hoá teencode hay gặp."""
    s = unicodedata.normalize("NFD", telex_fix(q)[0].lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d")
    s = re.sub(r"[^a-z0-9/%\-\s]", " ", s)
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "/", s)
    rep = [
        (r"\bdthu\b|\bdoanh thu\b", " DOANHTHU "),
        (r"\b(bao nhieu|bao nhiu|bnhiu|bnhieu|bn|nhieu|nhiu|nhieu|bao nhieu tien|bao nhiu tien|may)\b", " BAONHIEU "),
        (r"\b(hom nay|hnay|ngay hom nay|bua nay)\b", " HOMNAY "),
        (r"\b(hom qua|hqua|bua qua)\b", " HOMQUA "),
        (r"\b(truoc|trc)\b", " TRUOC "),
        (r"\bth\s*(?=\d)", "thang "),
        (r"\bthang\s*(?=\d)", "thang "),
    ]
    for pat, sub in rep:
        s = re.sub(pat, sub, s)
    return " ".join(s.split())


_YEAR = r"(\d{4}|\d{2})"


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def extract_periods(s: str, P: Periods) -> tuple[list, str]:
    """Tìm các cụm chỉ kỳ, trả về [(nhãn, (đầu, cuối))] theo thứ tự xuất hiện và câu đã xoá các cụm đó."""
    found = []

    def take(pattern, fn):
        nonlocal s
        for m in list(re.finditer(pattern, s)):
            try:
                found.append((m.start(), fn(m)))
            except (ValueError, KeyError):
                found.append((m.start(), ("BAD", None)))
        s = re.sub(pattern, " ", s)

    take(r"\bngay (\d{1,2})/(\d{1,2})/(\d{4})\b",
         lambda m: ("day", P.day(dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1))))))
    # quý trước tháng: 'quy 2/2026' không được đọc thành tháng 2/2026
    take(r"\bquy (\d) nam (\d{4})\b|\bquy (\d)/(\d{4})\b",
         lambda m: ("quarter", P.quarter(int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4)))))
    take(r"\b(?:thang )?(\d{1,2})/" + _YEAR + r"\b",
         lambda m: ("month", P.month(int(m.group(1)), _year(m.group(2)))))
    take(r"\bthang (\d{1,2}) nam (\d{4})\b", lambda m: ("month", P.month(int(m.group(1)), int(m.group(2)))))
    take(r"\bthang nay\b", lambda m: ("this_month", P.this_month()))
    take(r"\bcung ky thang TRUOC\b", lambda m: ("prev_month_same_days", P.prev_month_same_days()))
    take(r"\btuan nay\b", lambda m: ("this_week", P.this_week()))
    take(r"\btuan (TRUOC|roi)\b", lambda m: ("last_week", P.last_week()))
    take(r"\b(\d{1,3}) ngay (gan nhat|gan day|qua|vua qua)\b", lambda m: ("last_n", P.last_n_days(int(m.group(1)))))
    take(r"\b(tu dau nam (toi|den) (nay|HOMNAY|het HOMNAY)|tu dau nam|tu ngay 1/1 den het HOMNAY)\b",
         lambda m: ("ytd", P.year_to_date()))
    take(r"\bHOMNAY\b", lambda m: ("today", P.today()))
    take(r"\bHOMQUA\b", lambda m: ("yesterday", P.yesterday()))
    take(r"\bnam (\d{4})\b", lambda m: ("year", P.year(int(m.group(1)))))
    take(r"\bthang (\d{1,2})\b", lambda m: ("month", P.month(int(m.group(1)))))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found], " ".join(s.split())


FILLER = set("""
cua hang quan tiem shop la bao BAONHIEU the nao vay z v nhi nhe ha do a oi cho toi minh em anh chi hoi xem giup gium
coi check biet duoc dc tong cong ca ban tat het trong ngay rieng thi luc tinh den toi tu va voi duoc co khong
roi nao nhieu it tien so luong cac mot nhu sao ne nha xiu chut bit cko mik dang hien tai tai thu ve thu
""".split())


def _clean(rest: str, words: str) -> bool:
    toks = [t for t in rest.split() if t not in FILLER]
    extra = [t for t in toks if t not in words.split()]
    return not extra


QTY_WORDS = "ban ra mat hang san pham sp mon thuoc loai size"


def _qty_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> tuple[bool, dict | None]:
    """'<kỳ> (mặt hàng X) bán được bao nhiêu <đơn vị>', 'bao nhiêu ly size M' (rulebook §5: số lượng quy về đơn vị
    cơ bản, chỉ mặt hàng có đơn vị cơ bản là <đơn vị>). Trả (đã nhận câu này, đáp án | None)."""
    units = view.units()
    if not units or not re.search(r"\bban\b", s):
        return False, None
    t = f" {s} "
    by_norm: dict[str, str] = {}
    for key, name in view.items.drop_duplicates(["item", "name"])[["item", "name"]].itertuples(index=False):
        by_norm.setdefault(norm(str(name)), key)
    hits = [n for n in by_norm if n and f" {n} " in t]
    item = None
    if hits:
        longest = max(len(n) for n in hits)
        top = {by_norm[n] for n in hits if len(n) == longest}
        if len(top) != 1:
            return True, None
        n = next(n for n in hits if len(n) == longest)
        item = by_norm[n]
        t = t.replace(f" {n} ", " ", 1)
    t = re.sub(r"\b(cong tat ca cac mon|tinh moi loai do uong|moi loai do uong|tat ca cac mon)\b", " ", t)
    size = None
    m = re.search(r"\bsize (m|l|s)\b", t)
    if m:
        size, t = m.group(1), t.replace(m.group(0), " ")
    periods, rest = extract_periods(" ".join(t.split()), P)
    found = [w for w in rest.split() if w in units]
    if len(set(found)) != 1 or "BAONHIEU" not in rest.split() and not re.search(r"\bso\b", rest):
        return item is not None, None
    unit = found[0]
    # 'bao nhiêu mặt hàng không bán được cái nào' là đếm mặt hàng, không phải số lượng bán
    counts_items = re.search(r"BAONHIEU (mat hang|san pham|sp|mon|loai|thuoc)\b|\b(khong|nao|chua)\b", rest)
    if counts_items or re.search(r"\b(tien|DOANHTHU)\b", rest) or not _clean(rest, QTY_WORDS + " " + unit):
        return item is not None, None
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of:
        return True, None
    period = periods[0][1]
    if not view.covers(period):
        return True, {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    v = view.qty_sold(period, as_of, unit, item=item, size=size)
    return True, None if v is None else {"type": "number", "answer": v}


DEBT_WORDS = "khach hang no cong con voi cua tong so tien thu hoi tu mua chiu cac don cho da tra ghi them lai bot"


def debt_answer(question: str, as_of: dt.date, dv, P: Periods) -> dict | None:
    """Công nợ (rulebook §4): tổng còn nợ / số khách còn nợ đến hết ngày D; nợ thêm, thu nợ trong kỳ.
    Câu nêu tên khách, tuổi nợ, 'trả hết nợ', top k… → None."""
    s = norm(question)
    s = re.sub(r'\bchua tra het no\b|\bcon thieu tien\b', 'con no', s)
    if not re.search(r"\bno\b", s) or re.search(r"\b(tuoi|qua|hon|het no|tra het|nhat|top|liet ke|moi nguoi|bang)\b", s):
        return None
    periods, rest = extract_periods(s, P)
    if len(periods) != 1 or periods[0][0] == "BAD":
        return None
    label, period = periods[0]
    if period[0] > as_of:
        return None
    if re.search(r"\bthu (duoc|dc|hoi|ve)\b|\btra (bot )?no\b", rest) and not re.search(r"\bno them\b", rest):
        kind = "repaid"
    elif re.search(r"\bno them\b|\bmua chiu\b", rest):
        kind = "added"
    elif re.search(r"\b(BAONHIEU|so) khach\b", rest) and label in {"day", "today", "yesterday"}:
        kind = "count"
    elif re.search(r"\bcon no\b|\bcong no\b", rest) and label in {"day", "today", "yesterday"}:
        kind = "total"
    else:
        return None
    if not _clean(rest, DEBT_WORDS):
        return None
    if not dv.covers(period):
        return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    day = min(period[1], as_of)
    v = {"repaid": lambda: dv.repaid(period, as_of), "added": lambda: dv.added(period, as_of),
         "count": lambda: dv.debtor_count(day), "total": lambda: dv.total_owed(day)}[kind]()
    return {"type": "number", "answer": v}


DEBT_TOP_WORDS = DEBT_WORDS + " top nao nhieu nhat lon nhat nhiu dang kem moi nguoi tung nguoi va liet ke xem giup la ai"
DEBT_AGE_WORDS = DEBT_WORDS + " tinh den het o nhom tuoi theo phan chia moi la BAONHIEU dat ma chua con lai giup minh xem"
DEBT_CUST_WORDS = DEBT_WORDS + " va duoc dc bot hai so theo thu tu ghi tra loi bn nhieu bao nhieu e cho hoi vay"


def debt_detail_answer(question: str, as_of: dt.date, dv, P: Periods) -> dict | None:
    """Công nợ chi tiết (rulebook §4, §3 bằng nhau): 'k khách nợ nhiều nhất đến hết ngày D, kèm số nợ' → table;
    'khách <tên> (trong kỳ) nợ thêm bao nhiêu và trả được bao nhiêu' → list theo thứ tự câu hỏi."""
    s = norm(question)
    if not re.search(r"\bno\b", s):
        return None
    periods, rest = extract_periods(s, P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of:
        return None
    label, period = periods[0]
    m = re.search(r"\b(?:top )?(\d{1,2}|hai|ba|bon|nam) khach(?: hang)?\b", rest)
    if m and label == "day" and re.search(r"\bno\b.*\b(nhieu|BAONHIEU|lon) nhat\b|\b(nhieu|BAONHIEU|lon) nhat\b.*\bno\b", rest):
        k = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]
        if not re.search(r"\b(so tien|so no|moi nguoi|tung nguoi|bao nhieu|BAONHIEU|kem)\b", rest):
            return None  # chỉ hỏi tên → list, chưa hỗ trợ
        if not _clean(rest.replace(m.group(0), " "), DEBT_TOP_WORDS):
            return None
        top = dv.top_debtors(min(period[1], as_of), k)
        if top is None:
            return None
        return {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [list(r) for r in top]}}
    if label == "day" and re.search(r"\bcon no\b|\bchua tra het\b", rest):
        day = min(period[1], as_of)
        b = re.search(r"\b0 7 ngay 8 15 ngay (?:va )?tren 15 ngay\b|\b0 7 8 15 (?:va )?tren 15 ngay\b", rest)
        if b and re.search(r"\btuoi no\b", rest):
            if not _clean(rest.replace(b.group(0), " "), DEBT_AGE_WORDS):
                return None
            return {"type": "list", "answer": [dv.owed_by_age(day, 0, 7), dv.owed_by_age(day, 8, 15),
                                               dv.owed_by_age(day, 16, None)]}
        q = re.search(r"\b(?:dat )?(?:da )?(?:qua|hon) (\d{1,3}) ngay\b", rest)
        if q and re.search(r"\bdon\b", rest):
            if not _clean(rest.replace(q.group(0), " "), DEBT_AGE_WORDS):
                return None
            return {"type": "number", "answer": dv.owed_by_age(day, int(q.group(1)) + 1, None)}
        return None
    i_add, i_pay = rest.find("no them"), max(rest.find("tra no"), rest.find("tra duoc"), rest.find("tra dc"),
                                             rest.find("tra bot"))
    if i_add < 0 or i_pay < 0 or not dv.covers(period):
        return None
    hits = [(n, c) for n, c in dv.customer_keys().items() if n and f" {n} " in f" {rest} "]
    if len({c for _, c in hits}) != 1:
        return None
    n = max((n for n, _ in hits), key=len)
    left = f" {rest} ".replace(f" {n} ", " ", 1)
    if not _clean(left, DEBT_CUST_WORDS):
        return None
    cust = hits[0][1]
    added, paid = dv.added_by(period, as_of, cust), dv.repaid(period, as_of, cust)
    return {"type": "list", "answer": [added, paid] if i_add < i_pay else [paid, added]}


DELIVERY_WORDS = ("xe da giao tong cong BAONHIEU dot hang cua cho khach dem so trong co total check gium xem giup "
                  "minh nhi duoc dc hoi la ban")


def delivery_answer(question: str, as_of: dt.date, dlv, P: Periods) -> dict | None:
    """'<kỳ> giao / chở được bao nhiêu đợt hàng' (rulebook: mỗi đợt của mỗi đơn tính một đợt, theo ngày giao)."""
    s = norm(question)
    if not re.search(r"\bdot\b", s):
        return None
    periods, rest = extract_periods(s, P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of:
        return None
    if not re.search(r"\b(BAONHIEU|so|dem)\b", rest) or not re.search(r"\b(giao|cho)\b", rest):
        return None
    if not _clean(rest, DELIVERY_WORDS) or not dlv.covers(periods[0][1]):
        return None
    return {"type": "number", "answer": dlv.count(periods[0][1], as_of)}


STOCK_WORDS = ("ton kho con lai trong cuoi ngay HOMNAY het luc la duoc dc tinh den xem giup check gium coi cho hoi "
               "toi biet va BAONHIEU chan nguyen le cua hang thuc te z vay ha nhi a")


def stock_answer(question: str, as_of: dt.date, sv) -> dict | None:
    """'Tồn kho <hàng> cuối ngày hôm nay là bao nhiêu <đơn vị cơ bản>' → number; '... mấy <đơn vị lớn> chẵn, mấy
    <đơn vị cơ bản> lẻ' → [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ] (rulebook §5). Chỉ khi hôm nay là ngày cuối có
    dữ liệu (cột tồn ghi tồn lúc đó)."""
    s = norm(question)
    if "HOMNAY" not in s or not re.search(r"\b(ton|con lai|kho)\b", s) or as_of != sv.last_day:
        return None
    names = {norm(n): k for k, (n, _) in sv.stock.items() if norm(n)}
    t = f" {s} "
    hits = [n for n in names if f" {n} " in t]
    if not hits:
        return None
    longest = max(len(n) for n in hits)
    if len({names[n] for n in hits if len(n) == longest}) != 1:
        return None
    n = next(n for n in hits if len(n) == longest)
    item, t = names[n], t.replace(f" {n} ", " ", 1)
    # Resolve names before rejecting metric words: a product can contain
    # "giá", "bán", or "lọ" without asking for valuation/sales/expiry.
    if re.search(r"\b(gia|tien|von|DOANHTHU|ban|nhap|het han|het hang|mat hang nao|hang nao)\b", t):
        return None
    qty = sv.stock[item][1]
    conv = sv.conv.get(item)
    if conv is None or not conv[0] or pd.isna(qty) or qty < 0 or qty != int(qty):
        return None
    base, facs = conv
    bigs = [b for b in facs if f" {b} " in t]
    split = re.search(r"\b(chan|nguyen)\b", t) and re.search(r"\ble\b", t)
    if split:
        if len(bigs) != 1 or f" {base} " not in t.replace(f" {bigs[0]} ", " ", 1) or facs[bigs[0]] != int(facs[bigs[0]]):
            return None
        f = int(facs[bigs[0]])
        rest = t.replace(f" {bigs[0]} ", " ", 1).replace(f" {base} ", " ", 1)
        ans = {"type": "list", "answer": [int(qty) // f, int(qty) % f]}
    else:
        if bigs and any(b != base for b in bigs) or f" {base} " not in t:
            return None
        rest = t.replace(f" {base} ", " ", 1)
        ans = {"type": "number", "answer": int(qty)}
    return ans if _clean(rest, STOCK_WORDS) else None


def debt_any(question: str, as_of: dt.date, dv, P: Periods) -> dict | None:
    """Mọi dạng câu công nợ bộ giải tất định nhận được (bảng công nợ dựng độc lập với bảng đơn chuẩn)."""
    r = debt_answer(question, as_of, dv, P)
    return r if r is not None else debt_detail_answer(question, as_of, dv, P)


_NUM_WORDS = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
TOP_WORDS = ("mat hang hang mon sp san pham nao ban chay nhat hot BAONHIEU duoc dc top liet ke la gi nhung "
             "cac cho xem giup mon an do uong")


def top_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """'<kỳ> (mặt hàng / món) nào bán chạy nhất', 'top k hàng bán chạy', 'k món hot nhất'."""
    if not re.search(r"\b(ban chay|chay nhat|hot nhat|ban (duoc|dc) BAONHIEU nhat)\b", s):
        return None
    if re.search(r"\b(tien|DOANHTHU|loi|lai|it nhat|cham|nhom|khach|nhan vien|loai|ca|kenh|thap)\b", s):
        return None
    periods, rest = extract_periods(s, P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of:
        return None
    m = re.search(r"\btop (\d{1,2}|hai|ba|bon|nam)\b|\b(\d{1,2}|hai|ba|bon|nam) (mat hang|hang|mon|sp|san pham)\b", rest)
    k = 1
    if m:
        w = m.group(1) or m.group(2)
        k = int(w) if w.isdigit() else _NUM_WORDS[w]
        rest = rest.replace(m.group(0), " ")
    if not _clean(rest, TOP_WORDS):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    top = view.top_items(period, as_of, k)
    if top is None:
        return None
    return {"type": "list", "answer": top} if m else {"type": "string", "answer": top[0]}


GROUP_WORDS = "nhom hang DOANHTHU tien ban thu ve chiem % phan tram ti ty trong le tong cac mat"


INGREDIENT_WORDS = "thuoc co hoat chat DOANHTHU tien ban thu ve cac chiem % phan tram ti ty trong le tong"


def group_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """'<kỳ> nhóm X bán được bao nhiêu tiền', 'nhóm X chiếm bao nhiêu % doanh thu' (rulebook: theo cột nhóm hàng,
    đúng tên nhóm; tỉ trọng chia doanh thu cả shop cùng kỳ)."""
    if not re.search(r"\bnhom\b", s) or re.search(r"\bnhom (khach|tuoi)\b", s):
        return None
    return _named_revenue(s, as_of, view, P, view.group_names(), view.group_revenue, GROUP_WORDS)


def ingredient_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """'Thuốc có hoạt chất X <kỳ> bán được bao nhiêu tiền' (rulebook: hoạt chất ghi đúng như vậy trong danh mục)."""
    if not re.search(r"\bhoat chat\b", s):
        return None
    return _named_revenue(s, as_of, view, P, view.ingredient_names(), view.ingredient_revenue, INGREDIENT_WORDS)


RX_WORDS = ("thuoc cac mat hang con lai ban duoc dc DOANHTHU tien thu ve chiem % phan tram ti ty trong le tong cua "
            "nha cua hang trong tu z vay ha nhi bang tren la xem giup cho hoi tong")


def rx_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """Thuốc kê đơn (rulebook §6: sản phẩm ghi kê đơn trong danh mục thuốc; mọi sản phẩm còn lại là không kê đơn):
    '<kỳ> thu bao nhiêu tiền từ thuốc kê đơn', 'trừ thuốc kê đơn ra … bán được bao nhiêu', 'kê đơn chiếm bao nhiêu %'."""
    if not re.search(r"\bke\s+don\b|\bdon bac si\b", s):
        return None
    otc = re.search(r"\btru (cac )?thuoc (phai )?ke don ra\b|\b(khong|ko|k|hong|hok) (can |phai )?(ke )?don\b( bac si\b)?", s)
    if otc:
        t, key = s.replace(otc.group(0), " ", 1), False
    else:
        t, key = re.sub(r"\b(cac )?(thuoc |hang )?(phai )?ke don\b", " ", s, count=1), True
    if re.search(r"\bke don\b|\bbac si\b|\bhoat chat\b|\bnhom", t):
        return None
    periods, rest = extract_periods(" ".join(t.split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of:
        return None
    share = re.search(r"%|\bphan tram\b|\b(ti|ty) (trong|le)\b|\bchiem\b", rest)
    is_rev = "DOANHTHU" in rest or re.search(r"\b(tien|ban (duoc|dc) BAONHIEU|thu ve)\b", rest)
    if not (share or is_rev) or not _clean(rest, RX_WORDS):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return None
    v = view.rx_revenue(period, as_of, key)
    if v is None:
        return None
    if share:
        total = view.revenue(period, as_of)
        return None if not total else {"type": "percent", "answer": share_pct(v, total)}
    return {"type": "number", "answer": v}


def _named_revenue(s, as_of, view, P, named: dict[str, str], revenue, words: str) -> dict | None:
    """Doanh thu / tỉ trọng của một nhóm có tên (nhóm hàng, hoạt chất): tên phải khớp trọn trong câu hỏi."""
    names = {norm(n): k for k, n in named.items() if norm(n)}
    t = f" {s} "
    hits = [n for n in names if f" {n} " in t]
    if not hits:
        return None
    longest = max(len(n) for n in hits)
    top = {names[n] for n in hits if len(n) == longest}
    if len(top) != 1:
        return None
    n = next(n for n in hits if len(n) == longest)
    periods, rest = extract_periods(" ".join(t.replace(f" {n} ", " ", 1).split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of:
        return None
    share = re.search(r"%|\bphan tram\b|\b(ti|ty) (trong|le)\b|\bchiem\b", rest)
    is_rev = "DOANHTHU" in rest or re.search(r"\b(tien|ban duoc BAONHIEU|thu ve)\b", rest)
    if not (share or is_rev) or not _clean(rest, words):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    v = revenue(period, as_of, top.pop())
    if v is None:
        return None
    if share:
        total = view.revenue(period, as_of)
        return None if not total else {"type": "percent", "answer": share_pct(v, total)}
    return {"type": "number", "answer": v}


ATTR_MENTIONS = [  # cách nói trong câu hỏi → (loại, nhãn chuẩn của semantic.ORDER_ATTRS)
    (r"\bcod\b|\bthu ho\b", ("pay", "cod")), (r"\btien mat\b|\btm\b", ("pay", "cash")),
    (r"\bchuyen khoan\b|\bck\b", ("pay", "transfer")), (r"\bvi dien tu\b|\bewallet\b", ("pay", "ewallet")),
    (r"\b(qua )?dien thoai\b", ("channel", "phone")), (r"\bzalo( oa)?\b", ("channel", "zalo")),
    (r"\btai (quay|cua hang|tiem)\b", ("channel", "store")), (r"\b(qua )?san( tmdt)?\b|\bshopee\b", ("channel", "market")),
]
ATTR_WORDS = ("DOANHTHU cac don ban tra bang qua va moi loai ben lan luot la so voi chiem % phan tram tong "
              "ti ty trong le tien thu ve oa hang")


def attr_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """Doanh thu các đơn trả bằng <hình thức> / bán qua <kênh> (rulebook §1: theo hình thức ghi trên đơn):
    một nhãn → số hoặc % tổng doanh thu; hai nhãn → list theo thứ tự câu hỏi."""
    if "DOANHTHU" not in s and not re.search(r"\btien\b", s):
        return None
    found, rest = [], s
    for pat, lab in ATTR_MENTIONS:
        m = re.search(pat, rest)
        if m:
            found.append((m.start(), lab))
            rest = rest[:m.start()] + " " + rest[m.end():]
    if not found or re.search(r"\b(nao|hon|nhieu nhat|khach|nhom|mat hang)\b", rest):
        return None
    labels = [lab for _, lab in sorted(found)]
    have = view.attr_labels()
    if any(lab[1] not in have.get(lab[0], set()) for lab in labels):
        return None
    periods, rest = extract_periods(" ".join(rest.split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of or not _clean(rest, ATTR_WORDS):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    vals = [view.attr_revenue(period, as_of, *lab) for lab in labels]
    if any(v is None for v in vals):
        return None
    if len(vals) >= 2:
        return {"type": "list", "answer": vals}
    if re.search(r"%|\bphan tram\b|\bchiem\b|\b(ti|ty) (trong|le)\b", rest):
        total = view.revenue(period, as_of)
        return None if not total else {"type": "percent", "answer": share_pct(vals[0], total)}
    return {"type": "number", "answer": vals[0]}


SHIFT_WORDS = ("DOANHTHU tien ban thu ve rieng don hang bill hoa so dem hon hay kem chenh lech % phan tram voi "
               "tang giam thay doi")
_SHIFT = r"\bca (sang|chieu|toi|dem)\b"


def shift_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """Doanh thu / số đơn một ca; % ca này hơn kém ca mốc (rulebook: theo giờ tạo đơn, khung giờ và ca mốc đọc
    từ rulebook của split)."""
    shifts = list(dict.fromkeys(re.findall(_SHIFT, s)))
    win = getattr(view, "shifts", None) or {}
    if not shifts or any(x not in win for x in shifts):
        return None
    periods, rest = extract_periods(" ".join(re.sub(_SHIFT, " ", s).split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][0] > as_of or not _clean(rest, SHIFT_WORDS):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
    is_rev = "DOANHTHU" in rest or re.search(r"\b(tien|thu ve)\b", rest)
    if len(shifts) == 2:
        base = win.get("_base")
        if base not in shifts or not re.search(r"%|\bphan tram\b", rest):
            return None
        other = next(x for x in shifts if x != base)
        metric = view.shift_revenue if is_rev else view.shift_order_count if re.search(r'\b(don|bill)\b', rest) else None
        if metric is None:
            return None
        v = pct_change(metric(period, as_of, other), metric(period, as_of, base))
        return None if v is None else {"type": "percent", "answer": v}
    if re.search(r"%|\bphan tram\b|\bhon\b|\bkem\b", rest):
        return None
    if is_rev:
        v = view.shift_revenue(period, as_of, shifts[0])
    elif re.search(r"\b(don|bill|hoa don)\b", rest):
        v = view.shift_order_count(period, as_of, shifts[0])
    else:
        return None
    return None if v is None else {"type": "number", "answer": v}


PRICE_WORDS = ("so ban ra trong ngay ke tu tang gia len voi truoc do hay giam BAONHIEU % phan tram thay doi hon "
               "lien tinh theo doi vao duoc d dong mat hang lan vs nhung cua sau TRUOC ta")


def _match_item(t: str, view) -> tuple[str | None, str, bool]:
    """Tên mặt hàng dài nhất khớp trọn trong câu (đã norm) → (khoá, câu đã bỏ tên, mơ hồ?)."""
    by_norm: dict[str, str] = {}
    for key, name in view.items.drop_duplicates(["item", "name"])[["item", "name"]].itertuples(index=False):
        by_norm.setdefault(norm(str(name)), key)
    t = f" {t} "
    hits = [n for n in by_norm if n and f" {n} " in t]
    if not hits:
        return None, t, False
    longest = max(len(n) for n in hits)
    top = {by_norm[n] for n in hits if len(n) == longest}
    n = next(n for n in hits if len(n) == longest)
    return by_norm[n], t.replace(f" {n} ", " ", 1), len(top) != 1


def price_effect_answer(question: str, s: str, as_of: dt.date, view: OrderView) -> dict | None:
    """Hiệu ứng tăng giá (rulebook §5): % thay đổi số lượng (đơn vị cơ bản) N ngày kể từ D (tính cả D) so với N ngày
    ngay trước D, bỏ đơn huỷ / trả. Chỉ nhận câu ghi rõ ngày D và một mặt hàng."""
    if not re.search(r"\b(tang gia|doi gia|len gia)\b", s) or not re.search(r"%|\bphan tram\b", s):
        return None
    dates = set(re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", question))
    ns = set(re.findall(r"\b(\d{1,3}) (?:ngày|ngay)\b", fold(question).replace("đ", "d")))
    if len(dates) != 1 or len(ns) != 1 or re.search(r"\b(tien|DOANHTHU)\b", s):
        return None
    item, rest, ambiguous = _match_item(s, view)
    if item is None or ambiguous:
        return None
    units = set(view.items.loc[view.items["item"] == item, "bunit"].dropna())
    if len(units) != 1:
        return None
    unit = units.pop()
    rest = re.sub(r"\b\d+(/\d+)*d?\b", " ", rest)  # ngày, số ngày, giá ('53 500d') đã đọc riêng
    words = [w for w in rest.split() if w not in FILLER and w not in PRICE_WORDS.split()]
    if any(w != unit for w in words):
        return None
    d_, m_, y_ = (int(x) for x in dates.pop())
    try:
        day = dt.date(y_, m_, d_)
    except ValueError:
        return None
    n = int(ns.pop())
    after = (day, day + dt.timedelta(days=n - 1))
    before = (day - dt.timedelta(days=n), day - dt.timedelta(days=1))
    if after[1] > as_of or not (view.covers(after) and view.covers(before)):
        return None
    new, old = view.qty_sold(after, as_of, unit, item=item), view.qty_sold(before, as_of, unit, item=item)
    v = pct_change(new, old)
    return None if v is None else {"type": "percent", "answer": v}


IMPORT_WORDS = "gia nhap binh quan vao mua don vi tinh theo"


def import_price_answer(s: str, as_of: dt.date, iv, P: Periods) -> dict | None:
    """'Giá nhập bình quân một <đơn vị> <mặt hàng> trong <kỳ>' (rulebook: bình quân theo số lượng nhập, theo đơn
    vị nhập của dòng, bỏ phiếu huỷ)."""
    by_norm: dict[str, str] = {}
    for key, name in iv.lines.drop_duplicates("item")[["item", "name"]].itertuples(index=False):
        by_norm.setdefault(norm(str(name)), key)
    t = f" {s} "
    hits = [n for n in by_norm if n and f" {n} " in t]
    if not hits:
        return None
    longest = max(len(n) for n in hits)
    if len({by_norm[n] for n in hits if len(n) == longest}) != 1:
        return None
    n = next(n for n in hits if len(n) == longest)
    item, t = by_norm[n], t.replace(f" {n} ", " ", 1)
    units = set(iv.lines.loc[iv.lines["item"] == item, "unit"].dropna())
    m = re.findall(r"\bmot (\w+)\b", t)
    if len(m) != 1 or m[0] not in units:
        return None
    unit, t = m[0], t.replace(f" mot {m[0]} ", " ", 1)
    periods, rest = extract_periods(" ".join(t.split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][1] > as_of:
        return None
    if not _clean(re.sub(r"\b\d+\b", " ", rest), IMPORT_WORDS + " " + unit):
        return None
    v = iv.avg_price(item, unit, periods[0][1])
    return None if v is None else {"type": "number", "answer": v}


CUST_WORDS = "khach khachs hang mua da co so chiem ti le ty phan tram % dem trong do"
_N_ORDERS = re.compile(r"\b(?:(?:tu|tuwf) (\d+) don tro (?:len|leen)|it nhat (\d+) don|(?:≥|>=) ?(\d+) don)\b")


def customer_orders_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """'Có bao nhiêu khách mua từ N đơn trở lên', '… chiếm bao nhiêu % số khách đã mua' (rulebook: từ N trở lên gồm
    N; xác định khách theo mã khách / tên, khách lẻ không tính; chỉ đơn hợp lệ)."""
    m = list(_N_ORDERS.finditer(s))
    if len(m) != 1:
        return None
    n = int(next(g for g in m[0].groups() if g))
    t = s.replace(m[0].group(0), " ", 1)
    periods, rest = extract_periods(" ".join(t.split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][1] > as_of:
        return None
    if not _clean(rest, CUST_WORDS):
        return None
    pct = bool(re.search(r"%|\bphan tram\b|\bti le\b|\bty le\b", rest))
    if not pct and not re.search(r"\b(BAONHIEU|may) khach\b|\bso khach\b|\bdem\b", rest):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return None
    r = view.customers_with_orders(period, as_of, n)
    if r is None or (pct and r[1] == 0):
        return None
    return {"type": "percent", "answer": share_pct(r[0], r[1])} if pct else {"type": "number", "answer": r[0]}


BIG_ORDER_WORDS = "don hang co DOANHTHU tien so dem may BAONHIEU"
_MIN_AMOUNT = re.compile(r"\b(?:tu|it nhat) (\d{1,3}(?: \d{3})*) (?:dong|d) (?:tro len|trowr leen|tro leen)\b")


def big_order_answer(s: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    """'Có bao nhiêu đơn có doanh thu / tiền hàng từ X đồng trở lên' (rulebook §1: doanh thu một đơn = tổng thành
    tiền các dòng; từ X trở lên gồm X; chỉ đơn hợp lệ)."""
    m = list(_MIN_AMOUNT.finditer(s))
    if len(m) != 1:
        return None
    x = int(m[0].group(1).replace(" ", ""))
    t = s.replace(m[0].group(0), " ", 1)
    periods, rest = extract_periods(" ".join(t.split()), P)
    if len(periods) != 1 or periods[0][0] == "BAD" or periods[0][1][1] > as_of:
        return None
    if not _clean(rest, BIG_ORDER_WORDS) or not re.search(r"\b(BAONHIEU|may|so) don\b", rest):
        return None
    if not re.search(r"\b(DOANHTHU|tien hang)\b", rest):
        return None
    period = periods[0][1]
    if not view.covers(period):
        return None
    v = view.order_count_min(period, as_of, x)
    return None if v is None else {"type": "number", "answer": v}


LOT_WORDS = ("lo hang mat hang hang nao thuoc con ton kho het han sap gan nhat co han dung ngay toi theo tinh den "
             "het tu nay tuc la may BAONHIEU dem hien van ma se tai trong cua hang shop dang den nhung list liet ke ke")
_DMY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def _dmy(t: str) -> list[dt.date]:
    out = []
    for d_, m_, y_ in _DMY.findall(t):
        try:
            out.append(dt.date(int(y_), int(m_), int(d_)))
        except ValueError:
            return []
    return out


def lot_answer(question: str, as_of: dt.date, lv) -> dict | None:
    """Hạn dùng theo lô (rulebook nhà thuốc): 'lô <hàng> gần hết hạn nhất có hạn dùng ngày nào', 'có mấy lô / mặt hàng
    còn tồn sẽ hết hạn từ nay đến hết ngày X (trong N ngày tới)', 'hàng nào hết hạn trong N ngày tới mà còn tồn'."""
    s = norm(question)
    if not re.search(r"\bhet han\b", s) or re.search(r"\b(tien|von|gia tri|DOANHTHU|so luong)\b", s):
        return None
    dates = _dmy(question)
    if any(d != as_of for d in dates[:-1]) or (dates and dates[-1] < as_of):
        return None  # chỉ nhận mốc 'hôm nay' = as_of
    t = _DMY.sub(" ", s)
    if re.search(r"\bgan het han nhat\b", t):
        by_norm: dict[str, str] = {}
        for key, name in lv.lots.drop_duplicates("item")[["item", "name"]].itertuples(index=False):
            by_norm.setdefault(norm(str(name)), key)
        hits = [n for n in by_norm if n and f" {n} " in f" {t} "]
        if not hits:
            return None
        longest = max(len(n) for n in hits)
        if len({by_norm[n] for n in hits if len(n) == longest}) != 1:
            return None
        n = next(n for n in hits if len(n) == longest)
        rest = f" {t} ".replace(f" {n} ", " ", 1)
        if not _clean(re.sub(r"\b\d+\b", " ", rest), LOT_WORDS):
            return None
        d = lv.nearest_expiry(by_norm[n], as_of)
        return None if d is None else {"type": "date", "answer": d.isoformat()}
    ns = re.findall(r"\b(\d{1,4}) ngay toi\b", t)
    until = dates[-1] if len(dates) and dates[-1] > as_of else None
    if ns:
        u2 = as_of + dt.timedelta(days=int(ns[0]))
        if len(ns) > 1 or (until is not None and until != u2):
            return None
        until = u2
    if until is None or not _clean(re.sub(r"\b\d+\b", " ", t), LOT_WORDS):
        return None
    e = lv.expiring(as_of, until)
    if re.search(r"\b(BAONHIEU|may) lo\b|\bdem so lo\b", t):
        return {"type": "number", "answer": int(len(e))}
    if re.search(r"\b(BAONHIEU|may) (mat hang|hang|thuoc|san pham)\b", t):
        return {"type": "number", "answer": int(e["item"].nunique())}
    if re.search(r"\b(hang|thuoc|mat hang|san pham) nao\b", t):
        first = e.groupby("name")["expiry"].min().sort_values(kind="stable")
        return {"type": "list", "answer": list(first.index)}
    return None


def prepare_sem(view: OrderView, shop, rulebook="", calendar=None, public_rules=True):
    """Chuẩn bị cột một lần/shop cho các luật SEM trong cùng bộ giải tất định."""
    from datamate.baseline_sem import SemRules
    view._sem_rules = SemRules(shop, view, rulebook, calendar, public_rules=public_rules)


def answer(question: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    rules = getattr(view, "_sem_rules", None)
    money_known = True
    if rules is not None:  # luật đơn / kỳ / tính tiền của split khác bản đã cài → mọi nhánh dưới đều sai, để agent
        from datamate.rule_guard import Guard
        if not hasattr(rules, "_guard"):
            rules._guard = Guard(rules.rulebook, policy_aware=True)
        if rules._guard.view:
            return None
        pol = rules._guard.policy
        # view dựng trước khi biết luật (§8 bỏ trùng, §3 gộp tên khách mặc định theo train): luật đổi → để nguồn khác
        if (not pol.dedup_same_order or not pol.merge_name_variants) and getattr(view, "_cleaning_policy", None) != pol:
            return None
        # luật cần cột shop không có (giảm giá cả đơn, phí giao) → tiền đơn NaN trong view, các nhánh đọc bảng gốc bỏ qua
        money_known = view.apply_policy(rules._guard.policy)
        P.last_n_include_today = view.policy.last_n_include_today
    result = _standard_answer(question, as_of, view, P)
    if result is not None:
        return result
    from datamate.baseline_series import answer as series_answer
    result = series_answer(question, as_of, view, P)
    if result is not None:
        return result
    from datamate.business_query import answer as business_answer, canonical, readable
    result = business_answer(question, as_of, view, P) if money_known else None
    if result is not None:
        return result
    normalized = readable(canonical(question))
    result = series_answer(normalized, as_of, view, P)
    if result is not None:
        return result
    result = _standard_answer(normalized, as_of, view, P)
    if result is not None:
        return result
    rules = getattr(view, "_sem_rules", None)
    if rules is not None and money_known:  # SEM đọc bảng gốc, không thấy tiền đơn NaN
        try:
            result = rules.answer(question, as_of)
            return result if result is not None else rules.answer(normalized, as_of)
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
    return None


def _standard_answer(question: str, as_of: dt.date, view: OrderView, P: Periods) -> dict | None:
    s = norm(question)
    if getattr(view, "imports", None) is not None and re.search(r"\bbinh quan\b", s) and re.search(r"\bnhap\b", s):
        return import_price_answer(s, as_of, view.imports, P)
    s_ge = norm(question.replace("≥", " ít nhất "))  # norm bỏ ký hiệu '≥'
    if getattr(view, "customers", None) is not None and _N_ORDERS.search(s_ge) and re.search(r"\bkhach", s_ge):
        return customer_orders_answer(s_ge, as_of, view, P)
    if _MIN_AMOUNT.search(s) and re.search(r"\bdon\b", s):
        return big_order_answer(s, as_of, view, P)
    if getattr(view, "items", None) is not None and re.search(r"\b(tang gia|doi gia|len gia)\b", s):
        return price_effect_answer(question, s, as_of, view)
    if re.search(_SHIFT, s):
        return shift_answer(s, as_of, view, P)
    if getattr(view, "attrs", None) is not None:
        res = attr_answer(s, as_of, view, P)
        if res is not None:
            return res
    if getattr(view, "groups", None) is not None:
        res = group_answer(s, as_of, view, P)
        if res is not None:
            return res
    if getattr(view, "rx", None) is not None:
        res = rx_answer(s, as_of, view, P)
        if res is not None:
            return res
    if getattr(view, "ingredients", None) is not None:
        res = ingredient_answer(s, as_of, view, P)
        if res is not None:
            return res
    if getattr(view, "items", None) is not None:
        res = top_answer(s, as_of, view, P)
        if res is not None:
            return res
    if getattr(view, "debt", None) is not None and re.search(r"\bno\b", s):
        return debt_answer(question, as_of, view.debt, P)
    if getattr(view, "items", None) is not None:
        handled, res = _qty_answer(s, as_of, view, P)
        if handled:
            return res
    periods, rest = extract_periods(s, P)
    if any(p[0] == "BAD" for p in periods) or not periods:
        return None
    labels = [p[0] for p in periods]
    # rulebook §1: 'bán được / thu về bao nhiêu tiền' là doanh thu
    is_rev = "DOANHTHU" in rest or re.search(
        r"\b(tien ban|(ban|thu) (duoc|dc)( tong cong)? BAONHIEU tien|thu ve BAONHIEU)\b", rest)
    # 'từ đầu tháng này đến hết hôm nay': hôm nay chỉ là mốc cuối của tháng này
    if "this_month" in labels and "today" in labels:
        periods = [p for p in periods if p[0] != "today"]
        labels = [p[0] for p in periods]

    # % thay đổi doanh thu so với kỳ trước
    if is_rev and re.search(r"%|phan tram", rest) and re.search(r"thay doi|tang|giam|hon hay kem|chenh", rest):
        if labels == ["month"] and re.search(r"thang (lien )?TRUOC|thang lien truoc|so voi thang TRUOC", rest):
            cur = periods[0][1]
            prev = P.prev_month(cur[0].month, cur[0].year)
            pair = (cur, prev)
            words = "DOANHTHU % phan tram thay doi tang giam hon hay kem so voi vs thang lien TRUOC truong"
        elif sorted(labels) == ["prev_month_same_days", "this_month"]:
            pair = (dict(periods)["this_month"], dict(periods)["prev_month_same_days"])
            words = "DOANHTHU % phan tram thay doi tang giam hon hay kem so voi vs toi HOMNAY truong dau"
        elif labels == ["month", "month"]:
            pair = (periods[0][1], periods[1][1])
            words = "DOANHTHU % phan tram thay doi tang giam hon hay kem so voi vs truong"
        else:
            return None
        if not _clean(rest, words):
            return None
        new, old = pair
        if not (view.covers(new) and view.covers(old)):
            return None
        v = pct_change(view.revenue(new, as_of), view.revenue(old, as_of))
        return None if v is None else {"type": "percent", "answer": v}

    if len(periods) != 1:
        return None
    period = periods[0][1]
    if period[0] > as_of:
        return None

    def done(t, v):
        if not view.covers(period):
            return {"type": "unanswerable", "reason": "kỳ hỏi nằm ngoài khoảng ngày có dữ liệu"}
        return None if v is None else {"type": t, "answer": v}

    # chữ thừa lại khi kỳ đã đọc đúng: 'doanh thu năm nay' (từ đầu năm), 'quý' (kỳ là quý)
    period_words = {"ytd": " nam nay", "quarter": " quy"}.get(labels[0], "")
    if re.search(r"\b(ti le|ty le)\b|%|phan tram", rest) and re.search(r"\bhuy\b", rest):
        if _clean(rest, "ti ty le don huy bi % phan tram la BAONHIEU" + period_words):
            return done("percent", view.cancel_rate(period, as_of))
        return None
    if re.search(r"\bhuy\b", rest) and re.search(r"\b(don|bill|hoa don)\b", rest):
        if _clean(rest, "don bill hoa bi huy so dem da o trang thai" + period_words):
            return done("number", view.cancelled_count(period, as_of))
        return None
    # 'so luong', 'it' nằm trong FILLER: chặn rõ để 'số lượng trung bình mỗi đơn', 'ngày ít nhất' không lọt
    other_measure = re.search(r"\b(luong|sl|san pham|mat hang|mon|sp|khach|ly|kg|it|thap|kem)\b", rest)
    if re.search(r"\b(trung binh|binh quan)\b", rest) and re.search(r"\b(don|bill|hoa don)\b", rest):
        if not other_measure and _clean(rest, "DOANHTHU trung binh binh quan moi tren mot don hang bill hoa gia tri mang ve dem lai"):
            return done("number", view.avg_order_value(period, as_of))
        return None
    if re.search(r"\bngay nao\b", rest) and re.search(r"\b(cao|BAONHIEU) nhat\b", rest) and is_rev:
        if not other_measure and _clean(rest, "DOANHTHU ngay nao cao nhat"):
            return done("date", view.best_revenue_day(period, as_of))
        return None
    if is_rev:
        if _clean(rest, "DOANHTHU tien ban thu ve" + period_words):
            return done("number", view.revenue(period, as_of))
        return None
    if re.search(r"\b(don|bill|hoa don)\b", rest) and re.search(r"BAONHIEU|so don", rest):
        if _clean(rest, "don bill hoa so da" + period_words):
            return done("number", view.order_count(period, as_of))
    return None
