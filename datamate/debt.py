"""Công nợ tất định theo rulebook §4, dựa trên quy ước README:
'**Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **X** là số tiền trả lần đó; **Y** là số khách phải
trả cho cả đơn'. Bảng thanh toán = bảng có cột X + mã đơn + một cột ngày; bảng đơn = bảng có cột Y + mã đơn.
Các cột còn lại của bảng đơn (trạng thái, hình thức thanh toán, mã khách) nhận theo giá trị / tên cột,
vì README nhiều shop không mô tả chúng. Thiếu gì hoặc không chắc → None (để agent làm).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold, is_walkin
from datamate.schema_roles import bilingual_key
from datamate.semantic import is_cust_code as _is_cust_code, is_cust_name as _is_cust_name
from datamate.semantic import classify_status, order_id_columns, parse_readme_columns, _tables_for, _find, M_STATUS

_DEBT_NOTE = re.compile(r"\*\*Công nợ\*\*: .*?[*`]{1,2}([^*`]+)[*`]{1,2} là số tiền trả lần đó; "
                        r"[*`]{1,2}([^*`]+)[*`]{1,2} là số khách phải trả")


def _locate(shop: Shop, ref: str) -> tuple[str, str] | None:
    """'bang.cot' hoặc 'cot' → (bảng, cột) duy nhất chứa cột đó. Tên README cũ so với cột đã đổi cách viết / dịch
    (`bilingual_key`: 'Giá trị' ~ 'price_tri', 'Số tiền thu' ~ 'SoTienThu') chỉ khi có đúng một cột tương đương."""
    table, _, col = ref.rpartition(".")
    if table and table in shop.tables and col in shop.tables[table].columns:
        return table, col
    hits = [t for t, df in shop.tables.items() if ref in df.columns]
    if hits:
        return (hits[0], ref) if len(hits) == 1 else None
    want = bilingual_key(col if table in shop.tables else ref)
    scope = {table: shop.tables[table]} if table in shop.tables else shop.tables
    hits = [(t, c) for t, df in scope.items() for c in df.columns if bilingual_key(c) == want]
    return hits[0] if len(hits) == 1 else None


def _value_col(df: pd.DataFrame, pred) -> str | None:
    """Cột chữ đầu tiên có ít giá trị khác nhau thoả `pred(tập giá trị đã fold)`."""
    for c in df.columns:
        s = df[c].dropna()
        if s.empty or not (pd.api.types.is_string_dtype(s) or s.dtype == object):
            continue
        vals = s.astype(str).unique()
        if len(vals) <= 30 and pred({fold(v) for v in vals}):
            return c
    return None


def _is_method(vals: set[str]) -> bool:
    return any(re.search(r"\b(debt|ghi no)\b", v) for v in vals) and any(re.search(r"\bcod\b", v) for v in vals)


def _is_marketplace(v: str) -> bool:
    return bool(re.search(r"\b(san|tmdt|shopee|lazada|tiki|tiktok)\b", v))


@dataclass
class DebtView:
    """orders: order_id, date, cust (khoá khách, None = khách lẻ), total, debt (đơn ghi nợ), counted (vào công nợ);
    pays: order_id, date, amount."""
    orders: pd.DataFrame
    pays: pd.DataFrame

    @classmethod
    def from_shop(cls, shop: Shop) -> "DebtView | None":
        m = _DEBT_NOTE.search(shop.readme)
        if not m:
            return None
        pay_at, tot_at = _locate(shop, m.group(1).strip()), _locate(shop, m.group(2).strip())
        if not pay_at or not tot_at or pay_at[0] == tot_at[0]:
            return None
        oid = order_id_columns(shop)
        if pay_at[0] not in oid or tot_at[0] not in oid:
            return None
        p, o = shop.tables[pay_at[0]], shop.tables[tot_at[0]]
        p_dates = [c for c in p.columns if pd.api.types.is_datetime64_any_dtype(p[c])]
        o_dates = [c for c in o.columns if pd.api.types.is_datetime64_any_dtype(o[c])]
        if len(p_dates) != 1 or not o_dates:
            return None
        # ngày đặt: cột ngày không kèm giờ nếu có (cột kia là giờ tạo đơn)
        day_cols = [c for c in o_dates if (o[c].dropna().dt.normalize() == o[c].dropna()).all()]
        o_date = day_cols[0] if len(day_cols) == 1 else (o_dates[0] if len(o_dates) == 1 else None)
        # Prefer an explicit status header.  A shop can have no cancellations
        # in the observed period, so looking for a cancellation value alone
        # would make an otherwise well-defined ledger disappear.
        status = next((c for c in o.columns
                       if re.match(r'^(trangthai|tinhtrang|status|state)$',
                                   re.sub(r'[^a-z0-9]', '', fold(c)))), None)
        if status is None:
            status = _value_col(o, lambda vs: any(classify_status(v) in {"cancelled", "returned", "delivering"}
                                                   for v in vs))
        method = _value_col(o, _is_method)
        if o_date is None or status is None or method is None:
            return None
        # Free-form notes use blank / ordinary text for completed orders only
        # when the column itself is explicitly a note. A blank or unknown enum
        # status and a missing payment method cannot establish debt eligibility.
        status_name = re.sub(r'[^a-z0-9]', '', fold(status))
        is_note = bool(re.match(r'^(ghichu|note|diengiai)', status_name))
        if not is_note and not re.match(r'^(trangthai|tinhtrang|status|state)', status_name):
            declared = any(tot_at[0] in _tables_for(shop, key) and _find({status: cols[status]}, M_STATUS) == status
                           for key, cols in parse_readme_columns(shop.readme).items() if status in cols)
            if not declared:
                return None
        if o[method].isna().any() or o[method].astype(str).str.strip().eq('').any():
            return None
        if not is_note:
            status_values = o[status].dropna().map(fold)
            completed = re.compile(r'\b(hoan thanh|hoan tat|thanh cong|da giao|da ban|da lay|xong|complete\w*|done)\b')
            def known_status(value):
                return classify_status(value) in {'cancelled', 'returned', 'delivering'} or bool(completed.search(value))
            if o[status].isna().any() or status_values.str.strip().eq('').any() or not status_values.map(known_status).all():
                return None
        code = next((c for c in o.columns if _is_cust_code(c)), None)
        name = next((c for c in o.columns if _is_cust_name(c)), None)
        cust_col = code or name
        if cust_col is None:
            return None
        channel = _value_col(o, lambda vs: any(_is_marketplace(v) for v in vs))
        meth = o[method].map(lambda v: fold(v) if pd.notna(v) else "")
        cancelled = o[status].map(lambda v: classify_status(v) == "cancelled" if pd.notna(v) else False)
        cod = meth.str.contains(r"\bcod\b") | (o[channel].map(lambda v: pd.notna(v) and _is_marketplace(fold(v)))
                                               if channel else False)
        cust = o[cust_col].map(lambda v: None if pd.isna(v) or is_walkin(str(v)) or not str(v).strip()
                               else " ".join(fold(str(v)).split()))
        # tên hiển thị (đáp án chép nguyên tên) và khoá xếp hạng khi bằng nhau (mã tăng dần, không có mã thì tên)
        shown = o[name] if name else o[cust_col]
        if name is None and code is not None:  # tên khách nằm ở bảng danh mục khách (mã → tên)
            for t, df in shop.tables.items():
                c_ = next((c for c in df.columns if _is_cust_code(c)), None)
                n_ = next((c for c in df.columns if _is_cust_name(c)
                           or re.sub(r"[\s_]", "", fold(c)) in ("tengoi", "ten", "hoten", "hovaten", "name", "fullname")),
                          None)
                if t != tot_at[0] and c_ and n_ and df[c_].is_unique:
                    names = df.set_index(df[c_].astype(str).str.strip())[n_]
                    shown = o[code].map(lambda v: names.get(str(v).strip()) if pd.notna(v) else None)
                    break
        orders = pd.DataFrame({
            "order_id": o[oid[tot_at[0]]].astype(str).str.strip(), "date": o[o_date].dt.normalize(), "cust": cust,
            "name": shown.map(lambda v: " ".join(str(v).split()) if pd.notna(v) else None),
            "rank_key": o[cust_col].map(lambda v: str(v).strip() if pd.notna(v) else ""),
            "total": pd.to_numeric(o[tot_at[1]], errors="coerce"),
            "debt": meth.str.contains(r"\b(?:debt|ghi no)\b") & ~cancelled,
            "counted": ~cancelled & ~cod,
        })
        required = orders['counted'] | orders['debt']
        if (o.loc[required, oid[tot_at[0]]].isna().any()
                or orders.loc[required, 'order_id'].eq('').any()
                or orders.loc[required, ['date', 'total']].isna().any().any()
                or orders.loc[required, 'total'].isin([float('inf'), float('-inf')]).any()):
            return None
        orders = orders.dropna(subset=["order_id", "date", "total"])
        # A header may repeat across detail rows. Never select the first row
        # when those rows disagree on the fields that determine the debt.
        identity = ["date", "cust", "total", "debt", "counted"]
        if orders.groupby("order_id")[identity].nunique(dropna=False).gt(1).any().any():
            return None
        orders = orders.drop_duplicates("order_id")
        pay_ids = p[oid[pay_at[0]]]
        pay_amount = pd.to_numeric(p[pay_at[1]], errors="coerce")
        if (pay_ids.isna().any() or pay_ids.astype(str).str.strip().eq('').any()
                or p[p_dates[0]].isna().any() or pay_amount.isna().any()
                or pay_amount.isin([float('inf'), float('-inf')]).any()):
            return None
        pays = pd.DataFrame({"order_id": pay_ids.astype(str).str.strip(), "date": p[p_dates[0]].dt.normalize(),
                             "amount": pay_amount})
        if not set(pays['order_id']).issubset(set(orders['order_id'])):
            return None
        if orders.empty or pays.empty or not orders["debt"].any():
            return None
        return cls(orders, pays)

    # --- truy vấn ---------------------------------------------------------------
    def date_range(self) -> tuple[dt.date, dt.date]:
        return self.orders["date"].min().date(), self.orders["date"].max().date()

    def covers(self, period) -> bool:
        lo, hi = self.date_range()
        return not (period[1] < lo or period[0] > hi)

    def _owed_at(self, day: dt.date) -> pd.DataFrame:
        """Nợ từng đơn lúc hết ngày `day`: tổng đơn − các khoản trả đến hết ngày đó; chỉ đơn còn nợ > 0."""
        d = pd.Timestamp(day)
        o = self.orders[self.orders["counted"] & (self.orders["date"] <= d)]
        paid = self.pays[self.pays["date"] <= d].groupby("order_id")["amount"].sum()
        owed = o["total"] - o["order_id"].map(paid).fillna(0)
        return o.assign(owed=owed)[owed > 0]

    def owed_by_age(self, day: dt.date, lo: int, hi: int | None) -> int:
        """Tổng còn nợ lúc hết ngày `day` của các đơn có tuổi nợ (ngày − ngày đặt) trong [lo, hi] (hi None = không chặn)."""
        o = self._owed_at(day)
        age = (pd.Timestamp(day) - o["date"]).dt.days
        keep = (age >= lo) & ((age <= hi) if hi is not None else True)
        return int(round(o.loc[keep, "owed"].sum()))

    def total_owed(self, day: dt.date) -> int:
        return int(round(self._owed_at(day)["owed"].sum()))

    def debtor_count(self, day: dt.date) -> int:
        """Số khách còn nợ (khách lẻ không phải khách: không đếm)."""
        return int(self._owed_at(day)["cust"].dropna().nunique())

    def added(self, period, as_of) -> int:
        """Nợ thêm: phần tổng đơn chưa trả ngay trong ngày mua, của đơn ghi nợ đặt trong kỳ."""
        lo, hi = pd.Timestamp(period[0]), pd.Timestamp(min(period[1], as_of))
        o = self.orders[self.orders["debt"] & (self.orders["date"] >= lo) & (self.orders["date"] <= hi)]
        same_day = self.pays.merge(o[["order_id", "date"]], on=["order_id", "date"]).groupby("order_id")["amount"].sum()
        rest = (o["total"] - o["order_id"].map(same_day).fillna(0)).clip(lower=0)
        return int(round(rest.sum()))

    def repaid(self, period, as_of, cust: str | None = None) -> int:
        """Trả nợ / thu nợ: khoản trả sau ngày mua, có ngày trả trong kỳ, của mọi đơn ghi nợ (của một khách nếu có)."""
        lo, hi = pd.Timestamp(period[0]), pd.Timestamp(min(period[1], as_of))
        p = self.pays[(self.pays["date"] >= lo) & (self.pays["date"] <= hi)]
        o = self.orders[self.orders["debt"] & ((self.orders["cust"] == cust) if cust else True)]
        p = p.merge(o[["order_id", "date"]], on="order_id", suffixes=("", "_buy"))
        return int(round(p.loc[p["date"] > p["date_buy"], "amount"].sum()))

    def added_by(self, period, as_of, cust: str) -> int:
        """Nợ thêm trong kỳ của một khách (khoá `cust`)."""
        lo, hi = pd.Timestamp(period[0]), pd.Timestamp(min(period[1], as_of))
        o = self.orders[self.orders["debt"] & (self.orders["cust"] == cust) & (self.orders["date"] >= lo)
                        & (self.orders["date"] <= hi)]
        same_day = self.pays.merge(o[["order_id", "date"]], on=["order_id", "date"]).groupby("order_id")["amount"].sum()
        return int(round((o["total"] - o["order_id"].map(same_day).fillna(0)).clip(lower=0).sum()))

    def top_debtors(self, day: dt.date, k: int) -> list[tuple[str, int]] | None:
        """k khách nợ nhiều nhất lúc hết ngày `day` (bằng nhau: mã / tên tăng dần); None nếu thiếu tên / không đủ k."""
        o = self._owed_at(day).dropna(subset=["cust"])
        if o["name"].isna().any():
            return None
        g = o.groupby("cust").agg(owed=("owed", "sum"), name=("name", "first"), key=("rank_key", "first"))
        g["owed"] = g["owed"].round().astype(int)
        g = g.sort_values(["owed", "key"], ascending=[False, True], kind="stable")
        if len(g) < k:
            return None
        return list(zip(g["name"].head(k), g["owed"].head(k).map(int)))

    def customer_keys(self) -> dict[str, str]:
        """Tên khách đã chuẩn hoá (không dấu, thường) → khoá `cust` (chỉ tên xác định duy nhất một khách)."""
        o = self.orders.dropna(subset=["cust", "name"])
        by = o.assign(n=o["name"].map(lambda v: " ".join(fold(v).split()))).groupby("n")["cust"].unique()
        return {n: cs[0] for n, cs in by.items() if len(cs) == 1}
