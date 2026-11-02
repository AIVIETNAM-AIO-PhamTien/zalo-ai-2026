"""Tồn theo lô và hạn dùng (rulebook, nhà thuốc): mỗi dòng nhập là một lô; một dòng bán có thể lấy từ nhiều lô; hàng
khách trả về lại lô đã bán. Tồn của lô hết ngày D = nhập (đơn vị cơ bản, phiếu không huỷ, ngày nhập ≤ D) − bán từ lô
(đơn không huỷ, ngày đặt ≤ D) + khách trả lại lô đó.

Cột nhận theo tên (số lô, hạn dùng, số lượng đơn vị cơ bản, mã hàng, mã đơn); ngày / trạng thái đơn lấy từ OrderView.
Thiếu gì hoặc không chắc → None (để agent làm).
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

from datamate.imports import _CODE, _DATE, _NAME, _RECEIPT, _STATUS, _col, _key, _names_by_code
from datamate.loader import Shop
from datamate.semantic import classify_status, item_key

_LOT = re.compile(r"^(lo|malo|solo|sololo)$")
_EXPIRY = re.compile(r"^(hsd|hansudung|handung|handungden)$")
_LOT_IN = re.compile(r"^(slnhapquydoi|slnhapdvtcoban|soluongdvtcoban|soluongnhapdvtcoban)$")
_LOT_OUT = re.compile(r"^(slquydoi|slquyrale|soluongdvtcoban|slbandvtcoban)$")
_LOT_RET = re.compile(r"^(sltraquydoi|sltradvtcoban|soluongtradvtcoban)$")
_ORDER = re.compile(r"^(mahoadon|mahd|madon|madonhang|sohoadon|orderid)$")


def _one(df: pd.DataFrame, pat: re.Pattern) -> str | None:
    hits = [c for c in df.columns if pat.match(_key(c))]
    return hits[0] if len(hits) == 1 else None


@dataclass
class LotView:
    lots: pd.DataFrame  # lot, item (khoá tên), name, expiry, qty_in, date (ngày nhập), cancelled
    moves: pd.DataFrame  # lot, order_id, out, ret
    orders: pd.DataFrame | None = None  # order_id, date, cls (tự đọc khi shop không dựng được OrderView)

    @classmethod
    def from_shop(cls, shop: Shop) -> "LotView | None":
        lot_t = [t for t, df in shop.tables.items() if _one(df, _LOT) and _one(df, _EXPIRY) and _one(df, _LOT_IN)
                 and _one(df, _ORDER) is None]
        move_t = [t for t, df in shop.tables.items() if _one(df, _LOT) and _one(df, _ORDER) and _one(df, _LOT_OUT)]
        if len(lot_t) != 1 or len(move_t) != 1:
            return None
        L = shop.tables[lot_t[0]]
        lot, exp, qin, code = _one(L, _LOT), _one(L, _EXPIRY), _one(L, _LOT_IN), _col(L, _CODE)
        if code is None or not pd.api.types.is_datetime64_any_dtype(L[exp]) or L[lot].duplicated().any():
            return None
        names = _lot_names(shop, L, lot, code, lot_t[0])
        if names is None:
            return None
        lots = pd.DataFrame({"lot": L[lot].astype(str), "name": names,
                             "expiry": L[exp].dt.normalize(), "qty_in": pd.to_numeric(L[qin], errors="coerce")})
        if lots[["name", "qty_in"]].isna().any().any():
            return None
        lots["item"] = lots["name"].map(item_key)
        # ngày nhập / trạng thái phiếu: bảng khác có cùng số lô + cột ngày, hoặc nối theo mã phiếu nhập
        lots["date"], lots["cancelled"] = pd.NaT, False
        rc = _col(L, _RECEIPT)
        for t, df in shop.tables.items():
            if t == lot_t[0]:
                continue
            d, s = _col(df, _DATE), _col(df, _STATUS)
            if d is None or not pd.api.types.is_datetime64_any_dtype(df[d]):
                continue
            for k_lots, k_df in ((lot, _one(df, _LOT)), (rc, rc if rc in df.columns else None)):
                if k_lots is None or k_df is None or lots["date"].notna().all():
                    continue
                m = df.dropna(subset=[k_df]).drop_duplicates(k_df)
                idx = m[k_df].astype(str)
                keys = L[k_lots].astype(str)
                lots["date"] = lots["date"].fillna(keys.map(pd.Series(m[d].dt.normalize().values, index=idx)))
                if s is not None:
                    st = keys.map(pd.Series(m[s].values, index=idx)).map(classify_status)
                    lots["cancelled"] = lots["cancelled"] | (st == "cancelled")
        if lots["date"].isna().any():
            return None
        M = shop.tables[move_t[0]]
        ret = _one(M, _LOT_RET)
        moves = pd.DataFrame({"lot": M[_one(M, _LOT)].astype(str), "order_id": M[_one(M, _ORDER)].astype(str),
                              "out": pd.to_numeric(M[_one(M, _LOT_OUT)], errors="coerce").fillna(0),
                              "ret": pd.to_numeric(M[ret], errors="coerce").fillna(0) if ret else 0.0})
        if not moves["lot"].isin(lots["lot"]).mean() > 0.99:
            return None
        return cls(lots, moves, _orders(shop, moves["order_id"], move_t[0]))

    def stock(self, day: dt.date, orders: pd.DataFrame | None = None) -> pd.DataFrame:
        """Các lô kèm `left` = tồn hết ngày `day` (orders: order_id, date, cls của OrderView)."""
        end = pd.Timestamp(day)
        orders = self.orders if orders is None else orders
        if orders is None:
            raise ValueError("không có ngày / trạng thái đơn")
        o = orders[(orders["date"] <= end) & (orders["cls"] != "cancelled")]["order_id"]
        mv = self.moves[self.moves["order_id"].isin(set(o))]
        net = (mv["out"] - mv["ret"]).groupby(mv["lot"]).sum()
        lots = self.lots[(self.lots["date"] <= end) & ~self.lots["cancelled"]].copy()
        lots["left"] = lots["qty_in"] - lots["lot"].map(net).fillna(0)
        return lots

    def expiring(self, day: dt.date, until: dt.date, orders: pd.DataFrame | None = None) -> pd.DataFrame:
        """Lô còn tồn (hết ngày `day`) có hạn dùng từ `day` đến hết `until`."""
        s = self.stock(day, orders)
        return s[(s["left"] > 0) & (s["expiry"] >= pd.Timestamp(day)) & (s["expiry"] <= pd.Timestamp(until))]

    def nearest_expiry(self, item: str, day: dt.date, orders: pd.DataFrame | None = None) -> dt.date | None:
        """Lô gần hết hạn nhất của một mặt hàng: lô còn tồn, chưa quá hạn tại hôm nay, hạn sớm nhất."""
        s = self.stock(day, orders)
        s = s[(s["item"] == item) & (s["left"] > 0) & (s["expiry"] >= pd.Timestamp(day))]
        return None if s.empty else s["expiry"].min().date()


_ORDER_DATE = re.compile(r"^(ngaylap|ngaydat|ngayban|ngaytao|ngay|thoigianlap|thoigian|ngaygioban|thoigianban)$")


def _lot_names(shop: Shop, L: pd.DataFrame, lot: str, code: str, lot_table: str) -> pd.Series | None:
    """Tên hàng của từng lô: tra danh mục theo mã hàng, không được thì bảng nhập có cùng số lô + cột tên hàng."""
    lut = _names_by_code(shop, code, lot_table)
    if lut is not None:
        names = L[code].astype(str).map(lut)
        if names.notna().all():
            return names
    for t, df in shop.tables.items():
        name = _col(df, _NAME)
        if t == lot_table or lot not in df.columns or name is None:
            continue
        m = df.dropna(subset=[lot]).drop_duplicates(lot)
        names = L[lot].astype(str).map(pd.Series(m[name].values, index=m[lot].astype(str)))
        if names.notna().all():
            return names
    return None


def _orders(shop: Shop, ids: pd.Series, move_table: str) -> pd.DataFrame | None:
    """Bảng đơn (cột mã đơn chứa gần hết mã đơn của bảng bán theo lô) → order_id, date, cls."""
    want = set(ids)
    for t, df in shop.tables.items():
        ocol = _one(df, _ORDER)
        if t == move_table or ocol is None or df[ocol].duplicated().any():
            continue
        if pd.Series(list(want)).isin(set(df[ocol].astype(str))).mean() < 0.95:
            continue
        dates = [c for c in df.columns if _ORDER_DATE.match(_key(c)) and pd.api.types.is_datetime64_any_dtype(df[c])]
        st = _col(df, _STATUS)
        if not dates or st is None:
            continue
        o = pd.DataFrame({"order_id": df[ocol].astype(str), "date": df[dates[0]].dt.normalize(),
                          "cls": df[st].map(classify_status)})
        return o if o["cls"].notna().mean() > 0.99 else None
    return None
