"""Additive sales on explicitly declared line ledgers WITHOUT order identities.

Every original row is an entry. No fabricated order count, customer, deduplication,
or average per order is supported. Definitions come from README and rulebook.
"""
import datetime as dt
import re

import pandas as pd

from datamate.helpers import round_money, pct_change
from datamate.parsing import fold
from datamate.semantic import (parse_readme_columns, _tables_for, _find, M_DATE,
    M_LINE_AMOUNT, M_ITEM_NAME, M_STATUS, _items, _conversions)


class LineLedger:
    def __init__(self, rows, rulebook=''):
        self.rows, self.rulebook = rows, rulebook

    @classmethod
    def from_shop(cls, shop, rulebook=''):
        text = fold(shop.readme)
        if not ('khong ghi ma don' in text and 'moi dong mot mat hang' in text):
            return None
        candidates = []
        roles = parse_readme_columns(shop.readme)
        for source, meanings in roles.items():
            for table in _tables_for(shop, source):
                frame = shop.tables[table]
                cols = {c: m for c, m in meanings.items() if c in frame}
                date, amount = _find(cols, M_DATE), _find(cols, M_LINE_AMOUNT)
                name, status = _find(cols, M_ITEM_NAME), _find(cols, M_STATUS)
                if not all([date, amount, name, status]) or not pd.api.types.is_datetime64_any_dtype(frame[date]):
                    continue
                from datamate.semantic import classify_status
                rows = pd.DataFrame({'date': frame[date].dt.normalize(), 'name': frame[name],
                    'amount': pd.to_numeric(frame[amount], errors='coerce'), 'cls': frame[status].map(classify_status)})
                if rows[['date', 'amount', 'name', 'cls']].isna().any().any():
                    continue
                # A temporary row locator aligns quantity conversion; it is never an order ID.
                temp = frame.copy()
                temp['__entry_locator'] = range(len(temp))
                items = _items(temp, '__entry_locator', cols, _conversions(shop, roles))
                if items is not None:
                    rows = rows.join(items.drop(columns=['order_id', 'name']), how='left')
                candidates.append(cls(rows, rulebook))
        return candidates[0] if len(candidates) == 1 else None

    def covers(self, period):
        dates = self.rows.date
        return not (pd.Timestamp(period[1]) < dates.min() or pd.Timestamp(period[0]) > dates.max())

    def revenue(self, period, as_of):
        r = self.rows
        r = r[r.date.between(pd.Timestamp(period[0]), pd.Timestamp(min(as_of, period[1]))) &
              r.cls.isin(['completed', 'delivering'])]
        return round_money(r.amount.sum())

    def answer(self, question, as_of, P):
        from datamate.business_query import canonical, periods, match_name, BASE, DIM
        from datamate.baseline import _clean
        from datamate.rule_guard import Guard
        from datamate.baseline_series import answer as series_answer
        guard = Guard(self.rulebook)
        if guard.core:
            return None
        s = canonical(question)
        # Monthly composition needs only additive revenue, never order identities.
        from datamate.business_query import readable
        result = series_answer(readable(s), as_of, self, P)
        if result is not None:
            return result
        key, left = match_name(s, {str(n): str(n) for n in self.rows.name.unique()})
        ps, rest = periods(left, P)
        labels = dict(ps)
        if 'this_month' in labels and 'today' in labels:
            ps = [p for p in ps if p[0] != 'today']
            rest = re.sub(r'\btu dau\b|\bden het\b', ' ', rest)
        if not ps or any(p[0] > as_of or label == 'BAD' or not self.covers(p) for label, p in ps):
            return None
        needs_orders = re.search(r'\b(?:so|BAONHIEU) don\b|\bdon\b.*\b(?:trung binh|binh quan)\b|\b(?:trung binh|binh quan)\b.*\bdon\b', rest)
        if needs_orders and _clean(rest, BASE + ' trung binh binh quan gia tri moi mot dem huy bi ti ty le % phan tram'):
            return {'type': 'unanswerable', 'reason': 'Sổ ghi từng dòng hàng, không có mã đơn để xác định số đơn hoặc tiền trung bình mỗi đơn.'}
        if re.search(r'\b(khong|ko|tru|ngoai|chi|khach|don|bill|nhom|ca)\b', rest):
            return None
        period = ps[0][1]
        r = self.rows
        selected = r[r.date.between(pd.Timestamp(period[0]), pd.Timestamp(min(as_of, period[1])))]
        if key is not None:
            selected = selected[selected.name == key]
        valid = selected[selected.cls.isin(['completed', 'delivering'])]
        words = BASE + ' ' + DIM
        def done(kind, value):
            return None if value is None else {'type': kind, 'answer': value}
        comparing = re.search(r'\b(so voi|so vs|tang|giam|chenh)\b', rest)
        if key is None and comparing:
            if not _clean(rest, BASE + ' % phan tram so voi vs tang hay giam chenh lech thang lien TRUOC cung ky nam thay doi'):
                return None
            if len(ps) == 2 and 'this_month' in labels and 'prev_month_same_days' in labels:
                period, old = labels['this_month'], labels['prev_month_same_days']
            elif len(ps) == 2:
                old = ps[1][1]
            elif re.search('cung ky nam TRUOC', rest) and ps[0][0] == 'month':
                old = P.month(period[0].month, period[0].year - 1)
            elif re.search(r'thang (lien )?TRUOC', rest):
                old = P.prev_month_same_days() if ps[0][0] == 'this_month' else P.prev_month(period[0].month, period[0].year)
            else:
                old = None
            if old is None or not self.covers(old):
                return None
            a, b = self.revenue(period, as_of), self.revenue(old, as_of)
            return done('percent', pct_change(a, b)) if re.search('%|phan tram', rest) else done('number', a-b)
        if len(ps) != 1:
            return None
        if 'DOANHTHU' in rest or 'tien' in rest.split():
            if re.search(r'ngay nao', rest) and re.search(r'(cao|BAONHIEU|DOANHTHU) nhat', rest):
                if key is not None or not _clean(rest, words + ' ngay nao cao nhat BAONHIEU xet theo') or valid.empty:
                    return None
                totals = valid.groupby('date').amount.sum()
                return done('date', min(totals[totals == totals.max()].index).strftime('%Y-%m-%d'))
            if _clean(rest, words):
                return done('number', round_money(valid.amount.sum()))
        if re.search(r'ban chay|hot nhat', rest) and key is None:
            if 'bqty' not in valid or valid.bqty.isna().any() or 'top_items' in guard.plan_blocked:
                return None
            m = re.search(r'\b(?:top )?(\d+)\b', rest)
            k = int(m[1]) if m else 1
            remaining = rest.replace(m[0], ' ') if m else rest
            if not 1 <= k <= 99 or not _clean(remaining, DIM + ' ban chay hot nhat nao top nhung gi liet ke'):
                return None
            totals = valid.dropna(subset=['item']).groupby('item').bqty.sum()
            names = valid.dropna(subset=['item']).groupby('item').name.first()
            keys = sorted(totals.index, key=lambda x: (-totals[x], str(names[x])))[:k]
            values = [str(names[x]) for x in keys]
            return done('list', values) if m else done('string', values[0] if values else None)
        if 'bunit' in selected and 'qty_sold' not in guard.plan_blocked:
            units = set(selected.bunit.dropna())
            hits = [u for u in units if re.search(r'\b' + re.escape(u) + r'\b', rest)]
            if len(hits) == 1:
                unit = hits[0]
                remaining = re.sub(r'\b' + re.escape(unit) + r'\b', ' ', rest)
                if key is not None and re.search(r'\bgia\b', remaining) and ps[0][0] in {'today', 'yesterday', 'day'}:
                    prices = selected.base_price if 'base_price' in selected else None
                    if prices is not None and not prices.empty and not prices.isna().any() and prices.max()-prices.min() < 1e-6 and _clean(remaining, DIM + ' gia ban don mot vao'):
                        return done('number', round_money(prices.iloc[0]))
                if re.search(r'\bban\b', remaining) and _clean(remaining, DIM + ' ban ra so'):
                    relevant = valid[valid.bunit == unit]
                    if (key is not None and selected.bunit.isna().any()) or relevant.bqty.isna().any():
                        return None
                    return done('number', float(relevant.bqty.sum()))
        return None
