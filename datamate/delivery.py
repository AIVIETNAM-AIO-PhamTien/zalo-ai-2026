"""Sổ giao hàng nhiều đợt (rulebook: mỗi đợt của mỗi đơn tính một đợt, theo ngày giao): đếm số đợt giao trong kỳ.

Log lượt GPU 2: agent đếm số dòng của sổ giao (một dòng mỗi mặt hàng mỗi đợt) → gấp ~2.2 lần đáp án, 7/7 câu sai.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import pandas as pd

from datamate.loader import Shop
from datamate.parsing import fold
from datamate.semantic import order_id_columns

_BATCH = re.compile(r"^(lan|luot|dot) ?(giao|cho|giao hang)?$|^(so )?dot$")


def _key(c) -> str:
    return " ".join(fold(str(c)).replace("_", " ").split())


@dataclass
class DeliveryView:
    """trips: order_id, batch, date (một dòng mỗi đợt của mỗi đơn)."""
    trips: pd.DataFrame

    @classmethod
    def from_shop(cls, shop: Shop) -> "DeliveryView | None":
        oid = order_id_columns(shop)
        found = []
        for t, df in shop.tables.items():
            batch = [c for c in df.columns if _BATCH.match(_key(c))]
            dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if len(batch) == 1 and len(dates) == 1 and t in oid:
                found.append((t, batch[0], dates[0]))
        if len(found) != 1:
            return None
        t, b, d = found[0]
        df = shop.tables[t]
        trips = pd.DataFrame({"order_id": df[oid[t]].astype(str), "batch": df[b].astype(str),
                              "date": df[d].dt.normalize()}).dropna().drop_duplicates(["order_id", "batch"])
        # một đợt phải có một ngày giao: lệch thì không chắc cách ghi sổ
        if trips.empty or df.dropna(subset=[d]).groupby([df[oid[t]].astype(str), df[b].astype(str)])[d].nunique().max() > 1:
            return None
        return cls(trips)

    def covers(self, period) -> bool:
        lo, hi = self.trips["date"].min().date(), self.trips["date"].max().date()
        return not (period[1] < lo or period[0] > hi)

    def count(self, period, as_of: dt.date) -> int:
        lo, hi = pd.Timestamp(period[0]), pd.Timestamp(min(period[1], as_of))
        return int(((self.trips["date"] >= lo) & (self.trips["date"] <= hi)).sum())
