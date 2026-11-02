"""Nhập hàng tất định: giá nhập bình quân một mặt hàng trong tháng (rulebook: bình quân theo số lượng nhập, theo
đơn vị nhập của dòng, các phiếu có ngày phiếu trong kỳ, bỏ phiếu huỷ).

Cột giá nhập lấy từ README: nghĩa cột 'giá nhập một đơn vị …' hoặc quy ước '**Nhập hàng**: X là giá nhập một đơn
vị'. Các cột còn lại (mã phiếu, số lượng, đơn vị, tên / mã hàng, ngày, trạng thái) nhận theo tên cột, ngày và trạng
thái có thể nằm ở bảng phiếu (nối theo mã phiếu). Thiếu gì hoặc không chắc → None (để agent làm).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold
from datamate.semantic import _tables_for, classify_status, item_key, shop_columns

_PRICE_NOTE = re.compile(r"\*\*Nhập hàng\*\*: [*`]{1,2}([^*`]+)[*`]{1,2} là giá nhập một đơn vị")
M_IMPORT_PRICE = "giá nhập một đơn vị"


def _key(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", fold(str(col)))


_RECEIPT = re.compile(r"^(ma|so)(phieu|phieunhap|nhaphang|pn)$")
_QTY = re.compile(r"^(soluong|sl|slnhap|soluongnhap)$")
_UNIT = re.compile(r"^(dvt|dvtnhap|donvitinh|donvi|donvinhap)$")
_NAME = re.compile(r"^(tenhang|tenmathang|tenhanghoa|tensanpham|tensp|product|tenthuoc|tengoi)$")
_CODE = re.compile(r"^(mahang|mahanghoa|masp|masanpham|mathuoc|mamon)$")
_DATE = re.compile(r"^(ngaynhap|ngaynhapkho|thoigian|ngay|ngayphieu|ngaylap)$")
_STATUS = re.compile(r"^(trangthai|tinhtrang)$")


def _col(df: pd.DataFrame, pat: re.Pattern) -> str | None:
    hits = [c for c in df.columns if pat.match(_key(c))]
    return hits[0] if len(hits) == 1 else None


def _price_refs(shop: Shop) -> list[tuple[str, str]]:
    """(bảng, cột giá nhập) theo README."""
    out = []
    for key, cols in shop_columns(shop).items():
        for t in _tables_for(shop, key):
            for c, m in cols.items():
                if fold(m).startswith(fold(M_IMPORT_PRICE)) and c in shop.tables[t].columns:
                    out.append((t, c))
    for ref in _PRICE_NOTE.findall(shop.readme or ""):
        table, _, col = ref.rpartition(".")
        out += [(t, col) for t, df in shop.tables.items() if (not table or t == table) and col in df.columns]
    return list(dict.fromkeys(out))


def _names_by_code(shop: Shop, code_col: str, line_table: str) -> pd.Series | None:
    """Mã hàng → tên hàng từ bảng danh mục có cùng tên cột mã và một cột tên hàng."""
    for t, df in shop.tables.items():
        if t == line_table or code_col not in df.columns:
            continue
        name = _col(df, _NAME)
        if name is None:
            continue
        lut = df.dropna(subset=[code_col, name]).drop_duplicates(code_col).set_index(code_col)[name]
        lut.index = lut.index.astype(str)
        return lut
    return None


@dataclass
class ImportView:
    lines: pd.DataFrame  # receipt, date, item (khoá tên), name, unit (khoá), qty, price, cancelled

    @classmethod
    def from_shop(cls, shop: Shop) -> "ImportView | None":
        refs = [(t, c) for t, c in _price_refs(shop) if _col(shop.tables[t], _RECEIPT)]
        if len(refs) != 1:
            return None
        t, price = refs[0]
        df = shop.tables[t]
        rc, qty, unit = _col(df, _RECEIPT), _col(df, _QTY), _col(df, _UNIT)
        if qty is None or unit is None:
            return None
        name, code = _col(df, _NAME), _col(df, _CODE)
        if name is not None:
            names = df[name]
        elif code is not None:
            lut = _names_by_code(shop, code, t)
            if lut is None:
                return None
            names = df[code].astype(str).map(lut)
            if names.isna().mean() > 0.01:
                return None
        else:
            return None
        lines = pd.DataFrame({"receipt": df[rc].astype(str), "name": names,
                              "unit": df[unit].map(lambda u: item_key(u) if pd.notna(u) else None),
                              "qty": pd.to_numeric(df[qty], errors="coerce"),
                              "price": pd.to_numeric(df[price], errors="coerce")})
        date, status = _col(df, _DATE), _col(df, _STATUS)
        if date is not None and not pd.api.types.is_datetime64_any_dtype(df[date]):
            date = None
        heads = [h for n, h in shop.tables.items() if n != t and rc in h.columns]
        if date is None or status is None:
            for h in heads:
                hd, hs = _col(h, _DATE), _col(h, _STATUS)
                m = h.dropna(subset=[rc]).drop_duplicates(rc)
                if date is None and hd is not None and pd.api.types.is_datetime64_any_dtype(h[hd]):
                    lines["date"] = lines["receipt"].map(pd.Series(m[hd].values, index=m[rc].astype(str)))
                    date = "@head"
                if status is None and hs is not None:
                    lines["status"] = lines["receipt"].map(pd.Series(m[hs].values, index=m[rc].astype(str)))
                    status = "@head"
        if date is None:
            return None
        if date != "@head":
            lines["date"] = df[date]
        if status is not None and status != "@head":
            lines["status"] = df[status]
        if lines["date"].isna().mean() > 0.01:
            return None
        lines["date"] = pd.to_datetime(lines["date"]).dt.normalize()
        lines["cancelled"] = (lines["status"].map(classify_status) == "cancelled") if "status" in lines else False
        lines["item"] = lines["name"].map(lambda v: item_key(v) if pd.notna(v) else None)
        return cls(lines.dropna(subset=["item"]))

    def avg_price(self, item: str, unit: str, period) -> int | None:
        """Bình quân gia quyền theo số lượng nhập, chỉ các dòng nhập theo `unit`; dòng của mặt hàng trong kỳ ghi đơn
        vị khác hoặc thiếu số → None (không chắc 'theo đơn vị nhập' nghĩa là gì ở câu đó)."""
        d = self.lines
        w = d[(d["item"] == item) & ~d["cancelled"] & (d["date"] >= pd.Timestamp(period[0]))
              & (d["date"] <= pd.Timestamp(period[1]))]
        if w.empty or set(w["unit"]) != {unit} or w[["qty", "price"]].isna().any().any() or w["qty"].sum() <= 0:
            return None
        return int(round((w["qty"] * w["price"]).sum() / w["qty"].sum()))
