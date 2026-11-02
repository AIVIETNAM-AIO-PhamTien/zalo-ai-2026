"""Sinh câu hỏi + đáp án cho một World. Đáp án tính bằng **oracle độc lập** trên dữ liệu chuẩn (không dùng
datamate.semantic), theo đúng rulebook train; câu nào có đáp án mơ hồ (hoà hạng, kỳ thiếu dữ liệu, mẫu số 0) thì bỏ.
Mỗi ý định có nhiều cách hỏi (trang trọng / nói chuyện / không dấu / viết tắt / gõ sai), theo kiểu câu của train."""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from datamate.synth.render import Style
from datamate.synth.simulate import World


def r2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def r0(x: float) -> int:
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def pct(new: float, old: float) -> float | None:
    return None if not old else r2((new - old) / old * 100)


def month_end(y: int, m: int) -> dt.date:
    return (dt.date(y + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1))


@dataclass(frozen=True)
class Policy:
    """Luật khác rulebook train, để đo khả năng chịu rulebook mới của split chưa gặp; mặc định = train."""
    delivering_valid: bool = True  # đơn đang giao tính vào doanh thu / số đơn
    revenue_order_level: bool = False  # doanh thu đơn trừ giảm giá cả đơn và cộng phí giao
    last_n_include_today: bool = True  # "N ngày gần nhất" kể cả hôm nay
    aov_round_down: bool = False  # giá trị trung bình một đơn làm tròn xuống


POLICIES = {
    "train": Policy(),
    "r1": Policy(delivering_valid=False, last_n_include_today=False),
    "r2": Policy(revenue_order_level=True, aov_round_down=True),
}

# dòng rulebook train (theo nhãn cột đầu) → nội dung mới khi policy đổi luật đó
_RULE_TEXT = {
    "delivering_valid": ("Đơn tính vào doanh thu",
                         "đơn **hoàn thành**. Đơn huỷ, đơn trả hàng (**kể cả chỉ trả một phần**) và đơn **đang giao** "
                         "đều **không** tính"),
    "revenue_order_level": ("Doanh thu một đơn",
                            "tổng **thành tiền các dòng hàng** (đã trừ giảm giá trên từng dòng), **trừ** giảm giá trên "
                            "cả đơn (khuyến mãi theo tổng tiền đơn, voucher shop) và **cộng** phí giao hàng; **không** "
                            "cộng thuế VAT. Doanh thu theo nhóm hàng chỉ cộng thành tiền các dòng của nhóm. "
                            "**Tiền hàng**, **tiền bán** của đơn bán cùng nghĩa với doanh thu"),
    "last_n_include_today": ("N ngày gần nhất", "N ngày liền nhau **trước hôm nay**, ngày cuối là **hôm qua**; "
                             "không tính hôm nay"),
    "aov_round_down": ("Giá trị trung bình một đơn", "làm tròn **xuống** về đồng"),
}


def rulebook_for(text: str, policy: Policy) -> str:
    """Viết lại các dòng rulebook train mà policy đổi; báo lỗi nếu không tìm đúng một dòng."""
    for field, default in Policy().__dict__.items():
        if getattr(policy, field) == default:
            continue
        label, body = _RULE_TEXT[field]
        pat = re.compile(rf"^\| {re.escape(label)} \|.*\|$", re.M)
        if len(pat.findall(text)) != 1:
            raise ValueError(f"rulebook không có đúng một dòng '{label}'")
        text = pat.sub(lambda _: f"| {label} | {body} |", text)
    return text


class Oracle:
    """Các phép đo theo rulebook (train hoặc biến thể Policy) trên dữ liệu chuẩn."""

    def __init__(self, w: World, st: Style, policy: Policy = Policy()):
        self.w, self.st, self.policy = w, st, policy
        o = w.orders.copy()
        o["date"] = pd.to_datetime(o["date"])
        self.o = o
        self.l = w.lines.merge(o[["order_id", "date", "status", "cust_id", "channel", "payment"]], on="order_id")
        names = dict(zip(w.customers["cust_id"], w.customers["name"]))
        if st.has_cust_code:  # §3: mỗi mã khách là một khách
            self.o["ckey"] = self.o["cust_id"]
        else:  # theo tên trên đơn, hai tên chỉ khác cách viết là một khách (tên trùng khác mã gộp làm một)
            self.o["ckey"] = self.o["cust_id"].map(lambda c: " ".join(names[c].lower().split()) if isinstance(c, str) else None)
        self.items = w.items.set_index("item_id")

    def _in(self, df, p, as_of):
        hi = min(p[1], as_of)
        return df[(df["date"] >= pd.Timestamp(p[0])) & (df["date"] <= pd.Timestamp(hi))]

    def _valid_status(self):
        return ["completed", "delivering"] if self.policy.delivering_valid else ["completed"]

    def valid_orders(self, p, as_of):
        o = self._in(self.o, p, as_of)
        return o[o["status"].isin(self._valid_status())]

    def valid_lines(self, p, as_of):
        ln = self._in(self.l, p, as_of)
        return ln[ln["status"].isin(self._valid_status())]

    def revenue(self, p, as_of, **flt) -> int:
        ln = self.valid_lines(p, as_of)
        for k, v in flt.items():
            ln = ln[ln[k] == v] if k != "group" else ln[ln["item_id"].map(self.items["group"]) == v]
        total = int(ln["amount"].sum())
        if self.policy.revenue_order_level and "group" not in flt:  # khoản cấp đơn của các đơn có trong ln
            o = self.o.set_index("order_id").loc[ln["order_id"].unique()]
            total += int(o["ship_fee"].sum() - o["order_discount"].sum())
        return total

    def order_count(self, p, as_of) -> int:
        return int(self.valid_orders(p, as_of)["order_id"].nunique())

    def cancelled(self, p, as_of) -> int:
        o = self._in(self.o, p, as_of)
        return int((o["status"] == "cancelled").sum())

    def cancel_rate(self, p, as_of) -> float | None:
        o = self._in(self.o, p, as_of)
        return r2((o["status"] == "cancelled").mean() * 100) if len(o) else None

    def aov(self, p, as_of) -> int | None:
        n = self.order_count(p, as_of)
        if not n:
            return None
        v = self.revenue(p, as_of) / n
        return int(Decimal(str(v)).to_integral_value(rounding=ROUND_FLOOR)) if self.policy.aov_round_down else r0(v)

    def order_revenue(self, p, as_of) -> pd.Series:
        """Doanh thu từng đơn hợp lệ trong kỳ theo luật 'Doanh thu một đơn' (cấp đơn khi policy đổi luật đó)."""
        ln = self.valid_lines(p, as_of).groupby("order_id")["amount"].sum()
        if self.policy.revenue_order_level:
            o = self.o.set_index("order_id").loc[ln.index]
            ln = ln + o["ship_fee"] - o["order_discount"]
        return ln

    def day_revenue(self, p, as_of) -> pd.Series:
        rev = self.order_revenue(p, as_of)
        return rev.groupby(self.o.set_index("order_id").loc[rev.index, "date"].values).sum()

    def qty_by_item(self, p, as_of) -> pd.Series:
        return self.valid_lines(p, as_of).groupby("item_id")["bqty"].sum()

    def customers_orders(self, p, as_of) -> pd.Series:
        o = self.valid_orders(p, as_of).dropna(subset=["ckey"])
        return o.groupby("ckey")["order_id"].nunique()

    def debt_owed(self, day: dt.date) -> tuple[int, int]:
        """§4: còn nợ hết ngày `day` = cộng phần chưa trả (>0) của đơn ghi nợ không huỷ; số khách còn nợ."""
        o = self.o[(self.o["payment"] == "Ghi nợ") & (self.o["status"] != "cancelled")
                   & (self.o["date"] <= pd.Timestamp(day))]
        goods = self.w.lines.groupby("order_id")["amount"].sum()
        due = o.set_index("order_id")
        due = goods.reindex(due.index) - due["order_discount"] + due["ship_fee"]
        pay = self.w.payments
        paid = pay[pd.to_datetime(pay["date"]) <= pd.Timestamp(day)].groupby("order_id")["amount"].sum()
        left = (due - paid.reindex(due.index).fillna(0))
        left = left[left > 0]
        keys = o.set_index("order_id").loc[left.index, "ckey"].dropna()
        return int(left.sum()), int(keys.nunique())

    def import_avg(self, item_id: str, p) -> tuple[str, int] | None:
        r = self.w.receipts
        ok = r[(r["status"] != "cancelled") & (pd.to_datetime(r["date"]) >= pd.Timestamp(p[0]))
               & (pd.to_datetime(r["date"]) <= pd.Timestamp(p[1]))]["receipt_id"]
        rl = self.w.receipt_lines
        rl = rl[rl["receipt_id"].isin(ok) & (rl["item_id"] == item_id)]
        if rl.empty or rl["unit"].nunique() != 1:
            return None
        return rl["unit"].iloc[0], r0((rl["qty"] * rl["unit_cost"]).sum() / rl["qty"].sum())


# ---------------------------------------------------------------- cách nói
def _vi_period_month(rng, y, m):
    return str(rng.choice([f"tháng {m}/{y}", f"tháng {m} năm {y}", f"tháng {m:02d}/{y}"]))


def noisy(rng, q: str) -> str:
    """Biến tấu cách gõ như câu train: viết tắt, bỏ dấu, thêm từ đệm."""
    if rng.random() < 0.3:
        q = q.replace("bao nhiêu", str(rng.choice(["bnhiu", "bn", "bao nhiu", "nhiêu", "bao nhiêu"])))
        q = q.replace("được", str(rng.choice(["đc", "được"]))).replace("doanh thu", str(rng.choice(["dthu", "doanh thu"])))
    if rng.random() < 0.2:
        q = "".join(c for c in unicodedata.normalize("NFD", q.replace("đ", "d").replace("Đ", "D"))
                    if unicodedata.category(c) != "Mn")
    if rng.random() < 0.25:
        q = str(rng.choice(["cho hỏi ", "xem giúp ", "ê ", "check giùm ", "Cho mình hỏi "])) + q[:1].lower() + q[1:]
    if rng.random() < 0.3:
        q = q.rstrip("?.") + str(rng.choice([" vậy", " nhỉ", " ha", "?", " z"]))
    return q


def _months(w: World, as_of: dt.date) -> list[tuple[int, int]]:
    out, d = [], w.start
    while month_end(d.year, d.month) <= as_of:
        out.append((d.year, d.month))
        d = month_end(d.year, d.month) + dt.timedelta(days=1)
    return out


def generate(w: World, st: Style, rng: np.random.Generator, n: int,
             policy: Policy = Policy()) -> list[tuple[str, dict, str]]:
    """→ [(câu hỏi, đáp án {type, answer}, ý định)]."""
    O, as_of = Oracle(w, st, policy), w.as_of
    months = _months(w, as_of)
    items = w.items
    out: list[tuple[str, dict, str]] = []

    def num(v):
        return None if v is None else {"type": "number", "answer": v}

    def per(v):
        return None if v is None else {"type": "percent", "answer": v}

    def pick_month():
        y, m = months[int(rng.integers(len(months)))]
        return (dt.date(y, m, 1), month_end(y, m)), _vi_period_month(rng, y, m), (y, m)

    def pick_period():
        k = rng.random()
        if k < 0.6:
            p, t, _ = pick_month()
            return p, t
        if k < 0.75:
            nd = int(rng.choice([7, 14, 30]))
            end = as_of if policy.last_n_include_today else as_of - dt.timedelta(days=1)
            return (end - dt.timedelta(days=nd - 1), end), str(rng.choice([f"{nd} ngày gần nhất", f"{nd} ngày qua"]))
        if k < 0.88:
            return (as_of.replace(day=1), as_of), "tháng này"
        d = w.start + dt.timedelta(days=int(rng.integers((as_of - w.start).days)))
        return (d, d), f"ngày {d.strftime('%d/%m/%Y')}"

    def q_revenue():
        p, t = pick_period()
        tpl = rng.choice(["Doanh thu {t} là bao nhiêu?", "{t} cửa hàng bán được bao nhiêu tiền?",
                          "Tổng doanh thu {t} là bao nhiêu vậy?", "{T} thu về bao nhiêu tiền"])
        return tpl.format(t=t, T=t[:1].upper() + t[1:]), num(O.revenue(p, as_of))

    def q_orders():
        p, t = pick_period()
        return str(rng.choice([f"{t[:1].upper() + t[1:]} có bao nhiêu đơn hàng?", f"Số đơn {t} là bao nhiêu?"])), \
            num(O.order_count(p, as_of))

    def q_cancel():
        p, t, _ = pick_month()
        if rng.random() < 0.5:
            return f"{t[:1].upper() + t[1:]} có bao nhiêu đơn bị hủy?", num(O.cancelled(p, as_of))
        return str(rng.choice([f"Tỉ lệ đơn hủy {t} là bao nhiêu %?", f"{t} hủy bao nhiêu phần trăm đơn"])), \
            per(O.cancel_rate(p, as_of))

    def q_aov():
        p, t = pick_period()
        return str(rng.choice([f"Giá trị trung bình một đơn {t} là bao nhiêu?",
                               f"{t} trung bình mỗi đơn bao nhiêu tiền"])), num(O.aov(p, as_of))

    def q_mom():
        if len(months) < 2:
            return None
        i = int(rng.integers(1, len(months)))
        (y, m), (py, pm) = months[i], months[i - 1]
        cur, prev = (dt.date(y, m, 1), month_end(y, m)), (dt.date(py, pm, 1), month_end(py, pm))
        if prev[0] < w.start:
            return None
        t = _vi_period_month(rng, y, m)
        return str(rng.choice([f"Doanh thu {t} tăng hay giảm bao nhiêu % so với tháng trước?",
                               f"{t} doanh thu thay đổi bao nhiêu phần trăm so với tháng liền trước"])), \
            per(pct(O.revenue(cur, as_of), O.revenue(prev, as_of)))

    def q_best_day():
        p, t, _ = pick_month()
        s = O.day_revenue(p, as_of)
        if s.empty or (s == s.max()).sum() > 1:
            return None
        return f"{t[:1].upper() + t[1:]} ngày nào có doanh thu cao nhất?", \
            {"type": "date", "answer": s.idxmax().strftime("%Y-%m-%d")}

    def q_qty_item():
        p, t, _ = pick_month()
        it = items.iloc[int(rng.integers(len(items)))]
        v = O.qty_by_item(p, as_of).get(it.item_id, 0)
        return str(rng.choice([f"{t[:1].upper() + t[1:]} bán được bao nhiêu {it.bunit} {it['name']}?",
                               f"Trong {t} cửa hàng bán ra mấy {it.bunit} {it['name']}"])), num(int(v))

    def q_top():
        p, t, _ = pick_month()
        s = O.qty_by_item(p, as_of).sort_values(ascending=False)
        k = int(rng.choice([1, 1, 3]))
        if len(s) <= k or s.iloc[k - 1] == s.iloc[k]:
            return None
        names = [items.set_index("item_id").loc[i, "name"] for i in s.index[:k]]
        if k == 1:
            return str(rng.choice([f"Mặt hàng nào bán chạy nhất {t}?", f"{t} món nào hot nhất"])), \
                {"type": "string", "answer": names[0]}
        return f"Top 3 mặt hàng bán chạy nhất {t} là gì?", {"type": "list", "answer": names}

    def q_group():
        p, t, _ = pick_month()
        g = str(rng.choice(sorted(items["group"].unique())))
        v = O.revenue(p, as_of, group=g)
        if rng.random() < 0.5:
            return f"Doanh thu nhóm {g} {t} là bao nhiêu?", num(v)
        tot = O.revenue(p, as_of)
        return f"Nhóm {g} chiếm bao nhiêu % doanh thu {t}?", per(None if not tot else r2(v / tot * 100))

    def q_payment():
        p, t, _ = pick_month()
        pay = str(rng.choice([x for x in w.arch.payments if x in ("Tiền mặt", "Chuyển khoản")]))
        return f"{t[:1].upper() + t[1:]} doanh thu thanh toán bằng {pay.lower()} là bao nhiêu?", \
            num(O.revenue(p, as_of, payment=pay))

    def q_channel():
        if len(w.arch.channels) < 2:
            return None
        p, t, _ = pick_month()
        ch = str(rng.choice(w.arch.channels[1:]))
        return f"Doanh thu từ kênh {ch} {t} là bao nhiêu?", num(O.revenue(p, as_of, channel=ch))

    def q_debt():
        if not w.arch.debt:
            return None
        total, cnt = O.debt_owed(as_of)
        if rng.random() < 0.5:
            return str(rng.choice(["Tính đến hôm nay khách còn nợ tổng cộng bao nhiêu tiền?",
                                   f"Đến hết ngày {as_of.strftime('%d/%m/%Y')}, tổng tiền khách còn nợ là bao nhiêu?"])), num(total)
        return "Hiện có bao nhiêu khách đang còn nợ?", num(cnt)

    def q_import():
        if not len(w.receipts):
            return None
        p, t, _ = pick_month()
        it = items.iloc[int(rng.integers(len(items)))]
        r = O.import_avg(it.item_id, p)
        if r is None:
            return None
        unit, v = r
        return str(rng.choice([f"Giá nhập bình quân một {unit} {it['name']} trong {t} là bao nhiêu?",
                               f"{t} cửa hàng nhập {it['name']} bình quân bao nhiêu tiền một {unit}"])), num(v)

    def q_price_effect():
        ch = w.prices[w.prices["date"] > w.start]
        if ch.empty:
            return None
        r = ch.iloc[int(rng.integers(len(ch)))]
        nd, d = int(rng.choice([7, 14, 30])), r["date"]
        after, before = (d, d + dt.timedelta(days=nd - 1)), (d - dt.timedelta(days=nd), d - dt.timedelta(days=1))
        if after[1] > as_of or before[0] < w.start:
            return None
        it = items.set_index("item_id").loc[r["item_id"]]
        a = O.qty_by_item(after, as_of).get(r["item_id"], 0)
        b = O.qty_by_item(before, as_of).get(r["item_id"], 0)
        price = f"{int(r['price']):,}".replace(",", ".")
        return (f"So số {it.bunit} {it['name']} bán ra trong {nd} ngày kể từ ngày {d.strftime('%d/%m/%Y')}, ngày tăng giá "
                f"lên {price}đ, với {nd} ngày ngay trước ngày đó: tăng hay giảm bao nhiêu %?"), per(pct(a, b))

    def q_customers():
        p, t, _ = pick_month()
        s = O.customers_orders(p, as_of)
        if s.empty:
            return None
        n_ = int(rng.integers(2, 5))
        k = int((s >= n_).sum())
        if rng.random() < 0.5:
            return str(rng.choice([f"{t[:1].upper() + t[1:]} có bao nhiêu khách mua từ {n_} đơn trở lên?",
                                   f"Trong {t}, số khách mua ít nhất {n_} đơn là bao nhiêu?"])), num(k)
        return f"{t[:1].upper() + t[1:]} khách mua từ {n_} đơn trở lên chiếm bao nhiêu % số khách đã mua?", \
            per(r2(k / len(s) * 100))

    def q_big_orders():
        p, t, _ = pick_month()
        ln = O.order_revenue(p, as_of)
        if ln.empty:
            return None
        x = int(round(float(ln.quantile(rng.uniform(0.5, 0.9))), -4))
        xs = f"{x:,}".replace(",", ".")
        return f"{t[:1].upper() + t[1:]} có bao nhiêu đơn có doanh thu từ {xs} đồng trở lên?", num(int((ln >= x).sum()))

    def q_unanswerable():
        if rng.random() < 0.5:
            y = w.start.year - 1
            return f"Doanh thu tháng {int(rng.integers(1, 13))}/{y} là bao nhiêu?", {"type": "unanswerable", "answer": None}
        p, t, _ = pick_month()
        return str(rng.choice([f"Nhân viên nào bán giỏi nhất {t}?", f"Lợi nhuận ròng sau thuế {t} là bao nhiêu?"])), \
            {"type": "unanswerable", "answer": None}

    def q_bool():
        p, t, _ = pick_month()
        v = O.revenue(p, as_of)
        thr = int(round(v * rng.uniform(0.8, 1.2), -6)) or 1_000_000
        if thr == v:
            return None
        return f"Doanh thu {t} có vượt {thr // 1_000_000} triệu không?", {"type": "boolean", "answer": bool(v > thr)}

    gens = {"revenue": (q_revenue, 5), "orders": (q_orders, 2), "cancel": (q_cancel, 2), "aov": (q_aov, 1.5),
            "mom": (q_mom, 2), "best_day": (q_best_day, 1), "qty_item": (q_qty_item, 2), "top": (q_top, 2),
            "group": (q_group, 2), "payment": (q_payment, 1), "channel": (q_channel, 1), "debt": (q_debt, 1.5),
            "import_avg": (q_import, 1.5), "price_effect": (q_price_effect, 1.5), "customers": (q_customers, 1.5),
            "big_orders": (q_big_orders, 1), "unanswerable": (q_unanswerable, 1), "boolean": (q_bool, 0.5)}
    keys = list(gens)
    wts = np.array([gens[k][1] for k in keys], float)
    seen = set()
    tries = 0
    while len(out) < n and tries < n * 20 and months:
        tries += 1
        k = keys[int(rng.choice(len(keys), p=wts / wts.sum()))]
        r = gens[k][0]()
        if r is None or r[1] is None:
            continue
        q = noisy(rng, re.sub(r"\s+", " ", r[0]).strip())
        if q in seen:
            continue
        seen.add(q)
        out.append((q, r[1], k))
    return out


def qid(biz: str, q: str) -> str:
    return "s_" + hashlib.sha1(f"{biz}|{q}".encode()).hexdigest()[:12]
