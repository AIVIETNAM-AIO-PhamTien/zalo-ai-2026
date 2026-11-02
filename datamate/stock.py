"""Tồn kho cuối ngày hôm nay từ cột tồn mà README ghi là "tồn cuối ngày cuối có dữ liệu, theo đơn vị cơ bản"
(rulebook §5: tồn theo đơn vị lớn = [số đơn vị lớn nguyên, số đơn vị cơ bản lẻ]).

Chỉ dùng khi hôm nay đúng là ngày cuối có dữ liệu (cột tồn là ảnh chụp lúc đó); ngày khác → để agent.
Log lượt GPU 2: 60 câu tồn kho train chưa phủ, agent hay bỏ trống câu 'mấy hộp chẵn, mấy viên lẻ'.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold
from datamate.semantic import M_ITEM_NAME, _conversions, _find, item_key, shop_columns, catalog_base_units

_NOTE = re.compile(r"\*\*Tồn kho\*\*: [*`]{1,2}([^*`]+)[*`]{1,2} là tồn cuối ngày cuối có dữ liệu, theo đơn vị cơ bản")
_NAME_COLS = ("tensanpham", "tenhang", "tenhanghoa", "tengoi", "product", "ten")


def _key(c) -> str:
    return re.sub(r"[\s_]", "", fold(str(c)))


@dataclass
class StockView:
    stock: dict[str, tuple[str, int | float]]  # khoá tên hàng → (tên gốc, tồn đơn vị cơ bản)
    conv: dict[str, tuple[str, dict[str, float]]]  # khoá tên hàng → (đơn vị cơ bản, {đơn vị lớn: hệ số})
    last_day: dt.date

    @classmethod
    def from_shop(cls, shop: Shop, last_day: dt.date | None,
                  base_units: dict[str, str] | None = None) -> "StockView | None":
        m = _NOTE.search(shop.readme)
        if not m or last_day is None:
            return None
        ref = m.group(1).strip()
        table, _, col = ref.rpartition(".")
        hits = [t for t, df in shop.tables.items() if (not table or t == table) and col in df.columns]
        if len(hits) != 1:
            return None
        df = shop.tables[hits[0]]
        cols = shop_columns(shop)
        present = {}
        for cm in cols.values():
            present.update({c: v for c, v in cm.items() if c in df.columns})
        name = _find(present, M_ITEM_NAME) or next((c for c in df.columns if _key(c) in _NAME_COLS), None)
        names = df[name] if name else None
        if names is None:  # bảng tồn chỉ có mã: tra tên ở danh mục có cả mã và tên
            code = next((c for c in df.columns if _key(c) in ("mahang", "masanpham", "masku", "mahanghoa", "masp")), None)
            for t, cat in shop.tables.items():
                nm = next((c for c in cat.columns if _key(c) in _NAME_COLS), None)
                if t != hits[0] and code and code in cat.columns and nm and cat[code].is_unique:
                    names = df[code].map(cat.set_index(code)[nm])
                    break
        if names is None:
            return None
        qty = pd.to_numeric(df[col], errors="coerce")
        stock: dict[str, tuple[str, int | float]] = {}
        for n, q in zip(names, qty):
            if pd.isna(n) or pd.isna(q):
                continue
            k = item_key(n)
            if k in stock:  # một mặt hàng nhiều dòng (nhiều kho / chi nhánh): không chắc cộng hay lấy dòng nào
                stock[k] = (stock[k][0], float("nan"))
            else:
                # Keep an integer snapshot exact; converting it to float loses
                # units for values above 2**53 before any threshold is checked.
                stock[k] = (str(n), q)
        conv = _conversions(shop, cols)
        for k, u in catalog_base_units(shop, cols).items():
            conv.setdefault(k, (u, {}))
        if base_units:  # mặt hàng không có dòng quy đổi: đơn vị cơ bản theo bảng bán (OrderView.items)
            for k, u in base_units.items():
                conv.setdefault(k, (u, {}))
        return cls(stock, conv, last_day) if stock else None


def base_units(view) -> dict[str, str]:
    """Khoá mặt hàng → đơn vị cơ bản duy nhất theo các dòng bán (OrderView.items)."""
    it = getattr(view, "items", None)
    if it is None:
        return {}
    g = it.dropna(subset=["bunit"]).groupby("item")["bunit"].unique()
    return {k: u[0] for k, u in g.items() if len(u) == 1 and u[0]}
