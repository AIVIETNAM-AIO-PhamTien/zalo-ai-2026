"""Service visits are rows, including warranty visits; no sales order inference.

Enabled only by an explicit README visit-ledger declaration and named column roles.
"""
import re

import pandas as pd

from datamate.helpers import round_money, share, pct_change
from datamate.parsing import fold
from datamate.semantic import parse_readme_columns, _tables_for, _find


def staff_key(value):
    text = fold(str(value)).strip()
    return re.sub(r'^(?:a|anh|c|chi|chu|bac|co|ong|ba)\s+', '', text)


class ServiceLedger:
    def __init__(self, rows, staff, rulebook):
        self.rows, self.staff, self.rulebook = rows, staff, rulebook

    @classmethod
    def from_shop(cls, shop, rulebook=''):
        text = fold(shop.readme)
        if not ('moi dong mot luot xe' in text and 'bao hanh' in text):
            return None
        sources, registry = [], []
        for source, declared in parse_readme_columns(shop.readme).items():
            for table in _tables_for(shop, source):
                frame = shop.tables[table]
                cols = {c: m for c, m in declared.items() if c in frame}
                date, amount = _find(cols, 'ngày sửa'), _find(cols, 'tổng tiền cả phiếu')
                mechanic, note = _find(cols, 'thợ làm phiếu'), _find(cols, 'ghi chú')
                vehicle, jobs = _find(cols, 'dòng xe và biển số'), _find(cols, 'các việc đã làm')
                if all([date, amount, mechanic, note, vehicle, jobs]) and pd.api.types.is_datetime64_any_dtype(frame[date]):
                    rows = pd.DataFrame({'date': frame[date].dt.normalize(), 'amount': pd.to_numeric(frame[amount], errors='coerce'),
                        'staff': frame[mechanic].map(lambda x: staff_key(x) if pd.notna(x) else None),
                        'vehicle': frame[vehicle], 'jobs': frame[jobs].map(lambda x: fold(str(x)) if pd.notna(x) else ''),
                        'warranty': frame[note].fillna('').map(lambda x: bool(re.search(r'\bbao hanh\b', fold(str(x)))))})
                    if rows[['date', 'amount', 'staff']].isna().any().any() or (rows.amount < 0).any():
                        continue
                    # The documented zero-charge warranty distinction must hold in the source.
                    if (rows.loc[rows.warranty, 'amount'] != 0).any() or (rows.loc[~rows.warranty, 'amount'] <= 0).any():
                        continue
                    sources.append(rows)
                full, commission = _find(cols, 'họ tên thợ'), _find(cols, 'phần trăm hoa hồng')
                if full and commission:
                    registry.append((frame[full], pd.to_numeric(frame[commission], errors='coerce')))
        if len(sources) != 1:
            return None
        rows, names = sources[0], {}
        if len(registry) == 1:
            fulls, rates = registry[0]
            for nickname in rows.staff.unique():
                hits = [(str(n), float(rate)) for n, rate in zip(fulls, rates) if pd.notna(n) and pd.notna(rate)
                        and (fold(str(n)).strip() == nickname or fold(str(n)).strip().endswith(' ' + nickname))]
                if len(hits) == 1 and 0 <= hits[0][1] <= 100:
                    names[nickname] = hits[0]
        return cls(rows, names, rulebook)

    def answer(self, question, as_of, P):
        from datamate.business_query import canonical, periods, match_name, BASE
        from datamate.baseline import _clean
        from datamate.rule_guard import Guard
        guard = Guard(self.rulebook)
        if guard.core or set(guard.changed) & {'Lượt xe, phiếu', 'Bảo hành', 'Thợ làm nhiều nhất', 'Khách quen, xe tới nhiều lần'}:
            return None
        s = canonical(question)
        staff, left = None, s
        if re.search(r'\btho\b', s):
            staff, left = match_name(s, {k: v[0] for k, v in self.staff.items()})
            if staff is None:
                staff, left = match_name(s, {k: k for k in self.rows.staff.unique()})
        ps, rest = periods(left, P)
        if ps and all(label != 'BAD' for label, _ in ps) and 'DOANHTHU' in rest and re.search(r'\b(so voi|tang|giam|chenh)\b', rest):
            words = BASE + ' tho sua xe phieu % phan tram so voi hon kem cao thap hay tang giam thay doi chenh lech thang lien TRUOC do'
            if re.search(r'\b(khong|ko|chi|ngoai)\b', rest) or not _clean(rest, words):
                return None
            current = ps[0][1]
            if len(ps) == 2:
                labels = dict(ps)
                if 'this_month' in labels and 'prev_month_same_days' in labels:
                    current, previous = labels['this_month'], labels['prev_month_same_days']
                else:
                    previous = ps[1][1]
            elif len(ps) == 1 and re.search(r'thang (lien )?TRUOC', rest):
                previous = P.prev_month_same_days() if ps[0][0] == 'this_month' else P.prev_month(current[0].month, current[0].year) if ps[0][0] == 'month' else None
            else:
                previous = None
            if previous is None:
                return None
            def revenue(period):
                start, end = period
                if start > as_of or pd.Timestamp(end) < self.rows.date.min() or pd.Timestamp(start) > self.rows.date.max():
                    return None
                rows = self.rows[self.rows.date.between(pd.Timestamp(start), pd.Timestamp(min(end, as_of)))]
                if staff is not None:
                    rows = rows[rows.staff == staff]
                return round_money(rows.amount.sum())
            a, b = revenue(current), revenue(previous)
            if a is None or b is None:
                return None
            percent = bool(re.search(r'%|phan tram', rest))
            value = pct_change(a, b) if percent else a - b
            return None if value is None else {'type': 'percent' if percent else 'number', 'answer': value}
        if len(ps) != 1 or ps[0][0] == 'BAD':
            return None
        period = ps[0][1]
        if period[0] > as_of or pd.Timestamp(period[1]) < self.rows.date.min() or pd.Timestamp(period[0]) > self.rows.date.max():
            return None
        rows = self.rows[self.rows.date.between(pd.Timestamp(period[0]), pd.Timestamp(min(as_of, period[1])))]
        if staff is not None:
            rows = rows[rows.staff == staff]
        def done(kind, value):
            return None if value is None else {'type': kind, 'answer': value}
        if re.search(r'\b(khong|tru|chi|ngoai)\b', rest):
            return None
        words = BASE + ' tho sua xe luot phieu lam duoc tinh ke bao hanh'
        if re.search(r'hoa hong|hue hong', rest):
            if not _clean(rest, words + ' hoa hue hong lanh bang tung moi xep giam dan nhieu tien cao nhat ai nao'):
                return None
            totals = rows.groupby('staff').amount.sum()
            if set(totals.index) - set(self.staff):
                return None
            amounts = {k: round_money(v * self.staff[k][1] / 100) for k, v in totals.items()}
            if staff is not None:
                return done('number', amounts.get(staff, 0))
            keys = sorted(amounts, key=lambda k: (-amounts[k], self.staff[k][0]))
            if 'nhat' in rest:
                return done('string', self.staff[keys[0]][0] if keys else None)
            if re.search(r'\bbang\b|\btung\b|\bmoi\b', rest):
                return done('table', {'columns': ['tho', 'hoa_hong'], 'rows': [[self.staff[k][0], amounts[k]] for k in keys]})
            return None
        if re.search(r'\bbao hanh\b', rest) and re.search(r'%|phan tram|ti le|ty le', rest):
            if staff is not None or not _clean(rest, words + ' ti ty le % phan tram so voi co thu tien'):
                return None
            return done('percent', share(int(rows.warranty.sum()), int((~rows.warranty).sum())))
        if re.search(r'\bbao hanh\b', rest) and not re.search(r'\b(ke ca|tinh ca|gom ca)\b', rest):
            rows = rows[rows.warranty]
        if re.search(r'\btrung binh\b|\bbinh quan\b', rest):
            if not _clean(rest, words + ' trung binh binh quan mot co thu tien het'):
                return None
            paid = rows[~rows.warranty]
            return done('number', round_money(paid.amount.sum()/len(paid)) if len(paid) else None)
        if 'tho' in rest and staff is None and 'nhat' in rest and _clean(rest, words + ' nhat BAONHIEU ai nao'):
            counts = rows.groupby('staff').size()
            if counts.empty:
                return None
            k = min(counts[counts == counts.max()].index, key=lambda x: self.staff.get(x, (x, 0))[0])
            return done('string', self.staff.get(k, (k, 0))[0])
        if re.search(r'\bngay nao\b', rest) and 'DOANHTHU' in rest and 'nhat' in rest and _clean(rest, words + ' ngay nao cao BAONHIEU nhat'):
            totals = rows.groupby('date').amount.sum()
            return done('date', min(totals[totals == totals.max()].index).strftime('%Y-%m-%d') if len(totals) else None)
        if ('DOANHTHU' in rest or 'tien' in rest.split()) and _clean(rest, words):
            return done('number', round_money(rows.amount.sum()))
        if re.search(r'\b(luot|phieu)\b', rest) and _clean(rest, words):
            return done('number', len(rows))
        return None
