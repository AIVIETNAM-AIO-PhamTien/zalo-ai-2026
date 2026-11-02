"""Normalized import receipts, distinguishing receipt totals from line amounts."""
import re

import numpy as np
import pandas as pd

from datamate.helpers import round_money
from datamate.parsing import fold
from datamate.semantic import classify_status, shop_columns, _tables_for


def key(s):
    return re.sub(r'[^a-z0-9]', '', fold(str(s)))


def one(frame, keys):
    hits = [c for c in frame if key(c) in keys]
    return hits[0] if len(hits) == 1 else None


class Receipts:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_shop(cls, shop):
        roles = {}
        for name, cols in shop_columns(shop).items():
            for table in _tables_for(shop, name):
                roles.setdefault(table, {}).update(cols)
        candidates = []
        for table, frame in shop.tables.items():
            receipt = one(frame, {'maphieunhap', 'manhaphang', 'sophieu', 'maphieu'})
            date = one(frame, {'ngaynhap', 'ngaynhapkho', 'thoigian', 'ngayphieu', 'ngay'})
            if not receipt or not date or not pd.api.types.is_datetime64_any_dtype(frame[date]):
                continue
            amounts = []
            for col in frame:
                meaning = fold(roles.get(table, {}).get(col, ''))
                note = fold(shop.readme)
                if 'tong tien phieu nhap' in meaning or (key(col) in {'cantrancc', 'tonggiatrinhap'} and fold(col) in note and 'tong tien cua ca phieu nhap' in note):
                    amounts.append((col, True))
                elif 'tien nhap cua ca dong' in meaning:
                    amounts.append((col, False))
            if len(amounts) != 1:
                continue
            amount, header = amounts[0]
            status = one(frame, {'trangthai', 'tinhtrang'})
            supplier = one(frame, {'mancc', 'manhacungcap', 'nhacungcap'})
            columns = [receipt, date, amount] + ([status] if status else []) + ([supplier] if supplier else [])
            if frame[[receipt, date, amount]].isna().any().any():
                continue
            header_cols = [date] + ([supplier] if supplier else []) + ([status] if status else [])
            if frame.groupby(receipt)[header_cols].nunique(dropna=False).gt(1).any().any():
                continue
            if header and frame.groupby(receipt)[columns[1:]].nunique(dropna=False).gt(1).any().any():
                continue
            rows = frame[columns].drop_duplicates(receipt) if header else frame[columns]
            rows = rows.copy()
            money = pd.to_numeric(rows[amount], errors='coerce')
            if money.isna().any() or not np.isfinite(money.to_numpy(dtype=float)).all():
                continue
            normalized = pd.DataFrame({'receipt': rows[receipt].astype(str), 'date': rows[date].dt.normalize(), 'amount': money,
                                       'supplier': rows[supplier].astype('string') if supplier else None,
                                       'cancelled': rows[status].map(classify_status).eq('cancelled') if status else False})
            candidates.append(cls(normalized))
        return candidates[0] if len(candidates) == 1 else None

    def value(self, period, as_of, best=False, most_receipts=False):
        rows = self.rows
        dates = rows['date']
        start, end = pd.Timestamp(period[0]), pd.Timestamp(min(period[1], as_of))
        if start > end or end < dates.min() or start > dates.max():
            return None
        rows = rows[(dates >= start) & (dates <= end) & ~rows['cancelled']]
        if best:
            if rows.empty or rows['supplier'].isna().any():
                return None
            amounts = rows.groupby('supplier')['receipt'].nunique() if most_receipts else rows.groupby('supplier')['amount'].sum()
            return str(min(amounts[amounts == amounts.max()].index, key=str))
        return round_money(rows['amount'].sum())
