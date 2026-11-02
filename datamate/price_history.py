"""Effective-dated basic-unit prices, used to locate a documented increase event."""
import re

import pandas as pd

from datamate.parsing import fold
from datamate.semantic import item_key, shop_columns, _tables_for, _find, M_ITEM_CODE, M_ITEM_NAME


def key(s):
    return re.sub('[^a-z0-9]', '', fold(str(s)))


def one(frame, names):
    cols = [c for c in frame if key(c) in names]
    return cols[0] if len(cols) == 1 else None


class PriceHistory:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_shop(cls, shop):
        note = fold(shop.readme)
        if not ('gia ban' in note and 'ap dung tu' in note and 'don vi co ban' in note):
            return None
        catalogs = []
        roles = shop_columns(shop)
        for source, cols in roles.items():
            for table in _tables_for(shop, source):
                frame = shop.tables[table]
                code = _find(cols, M_ITEM_CODE)
                name = _find(cols, M_ITEM_NAME) or next((c for c, v in cols.items() if fold(v).strip() == 'ten'), None)
                if code in frame and name in frame:
                    catalogs.append((frame[code].astype(str), frame[name].map(item_key)))
        candidates = []
        for table, frame in shop.tables.items():
            date = one(frame, {'ngayapdung', 'hieuluctu'})
            price = one(frame, {'giamoi', 'giabanle', 'giaban'})
            name = one(frame, {'tenhang', 'tensanpham', 'tengoi', 'mathang'})
            code = one(frame, {'mahang', 'mahanghoa', 'masp', 'masanpham'})
            if not date or not price or not pd.api.types.is_datetime64_any_dtype(frame[date]):
                continue
            if name:
                names = frame[name].map(item_key)
            elif code:
                wanted = frame[code].astype(str)
                lookups = []
                for codes, items in catalogs:
                    pairs = pd.DataFrame({'code': codes, 'item': items}).dropna()
                    if pairs.groupby('code').item.nunique().gt(1).any():
                        continue
                    lut = pairs.drop_duplicates('code').set_index('code').item
                    if set(wanted) <= set(lut.index):
                        lookups.append(wanted.map(lut))
                if not lookups or any(not n.equals(lookups[0]) for n in lookups[1:]):
                    continue
                names = lookups[0]
            else:
                continue
            rows = pd.DataFrame({'item': names, 'date': frame[date].dt.normalize(), 'price': pd.to_numeric(frame[price], errors='coerce')})
            if rows.isna().any().any() or (rows.price <= 0).any() or rows.groupby(['item', 'date']).price.nunique().gt(1).any():
                continue
            rows = rows.drop_duplicates(['item', 'date']).sort_values(['item', 'date'])
            rows['previous'] = rows.groupby('item').price.shift()
            candidates.append(cls(rows))
        return candidates[0] if len(candidates) == 1 else None

    def increase(self, item, price, day=None):
        rows = self.rows
        rows = rows[(rows.item == item) & (rows.price > rows.previous)]
        if price is not None:
            rows = rows[(rows.price - price).abs() < 1e-6]
        if day is not None:
            rows = rows[rows.date == pd.Timestamp(day)]
        return rows.date.min().date() if len(rows) else None


def effect(question, item, rest, as_of, view, history, guard):
    from datamate.baseline import _clean
    from datamate.helpers import pct_change
    import datetime as dt
    if not history or view.items is None or 'qty_sold' in guard.plan_blocked or 'Hiệu ứng tăng giá' in guard.changed:
        return None
    if not re.search(r'tang gia|doi gia|len gia', rest) or not re.search('%|phan tram', rest) or re.search(r'DOANHTHU|\btien\b', rest):
        return None
    windows = re.findall(r'\b(\d{1,3}) ngay\b', rest)
    if not windows or len(set(windows)) != 1 or int(windows[0]) < 1:
        return None
    n = int(windows[0])
    amount = re.search(r'\b(?:gia len|len gia|tang gia|gia) (\d{1,3}(?: \d{3})+|\d+)(?: ?dong|d)\b', rest)
    if not amount:
        return None
    price = int(amount[1].replace(' ', ''))
    remaining = rest.replace(amount[0], ' ')
    dates = set(re.findall(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', remaining))
    if len(dates) > 1:
        return None
    day = None
    if dates:
        d, m, y = map(int, next(iter(dates)))
        try:
            day = dt.date(y, m, d)
        except ValueError:
            return None
        remaining = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', ' ', remaining)
    remaining = re.sub(r'\b\d+ ngay\b', ' ngay ', remaining)
    units = set(view.items.loc[view.items.item == item, 'bunit'].dropna())
    if len(units) != 1:
        return None
    unit = next(iter(units))
    remaining = re.sub(r'\b' + re.escape(unit) + r'\b', ' ', remaining).replace('/', ' ')
    words = 'so ban ra ngay ke tu khi hoi tang gia len doi voi TRUOC do hay giam BAONHIEU % phan tram thay doi hon it nhieu lien dau thi tinh theo sau truong trung binh moi truoc'
    if not _clean(remaining, words):
        return None
    event = history.increase(item, price, day)
    if event is None:
        return None
    after, before = (event, event + dt.timedelta(days=n-1)), (event-dt.timedelta(days=n), event-dt.timedelta(days=1))
    if after[1] > as_of or not view.covers(after) or not view.covers(before):
        return None
    value = pct_change(view.qty_sold(after, as_of, unit, item=item), view.qty_sold(before, as_of, unit, item=item))
    return None if value is None else {'type': 'percent', 'answer': value}
