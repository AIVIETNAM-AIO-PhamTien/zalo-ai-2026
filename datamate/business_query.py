"""Compose a metric, a dimension and an operation, with complete residual checks.

No question IDs or reference answers are used here. Unknown conditions fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from datamate.baseline import norm, extract_periods, ATTR_MENTIONS, _clean
from datamate.helpers import pct_change, share


@lru_cache(maxsize=8192)
def canonical(question):
    s = norm(question.replace('≥', ' it nhat '))
    for pat, replacement in [
        (r'\bt\s*(\d{1,2})(?=/)', r'thang \1'),
        (r'\bq\s*([1-4])(?=/)', r'quy \1'),
        (r'\bth (nay|ni)\b|\bthang ni\b', 'thang nay'),
        (r'\bhom ni\b', 'HOMNAY'),
        (r'\b(dh|don hang)\b', 'don'),
        (r'\b(ncc)\b', 'nha cung cap'),
        (r'\b(nv)\b', 'nhan vien'),
        (r'\b(mik|mk)\b', 'minh'),
        (r'\b(wan|quan)\b', 'quan'),
        (r'\btrog\b', 'trong'),
        (r'\bnhut\b', 'nhat'),
        (r'\b(gium|jum)\b', 'giup'),
        (r'\b(tien tuoi)\b', 'tien mat'),
        (r'\b(cao hay thap hon|len hay xuong|len xuong)\b', 'tang hay giam'),
        (r'\b(cung thang nam roi)\b', 'cung ky nam TRUOC'),
        (r'\b([1-9]\d*) (?:trieu|tr)(?=\b)', r'\1 000 000 dong'),
        (r'\b([1-9]\d*)k\b', r'\1 000 dong'),
        (r'\b([1-9]\d*)tr\b', r'\1 000 000 dong'),
        (r'\b1 (don|phieu)\b', r'mot \1'),
        (r'\bbq\b', 'binh quan'),
        (r'\btu dau nam (?:nay )?(?:den|toi)(?: het)? HOMNAY\b', 'tu dau nam'),
        (r'\btu 1/1(?: nam nay)? (?:den|toi)(?: het)? HOMNAY\b', 'tu dau nam'),
        (r'\b(?:rua|he|hen|thanks)\b$', ' '),
        (r"\b(revenue|doanhthu)\b", "DOANHTHU"),
        (r"\b(nhiu|nhieu)\b", "BAONHIEU"),
        (r"\b(total)\b", "tong"),
        (r"\b(dc|dk)\b", "duoc"),
        (r"\b(j)\b", "gi"),
        (r"\b(order)\b", "don"),
        (r"^(?:e|ee)\b", " "),
        (r"\b(ta|ak)\b$", " "),
        (r"\bth TRUOC\b", "thang TRUOC"),
        (r"\btang/giam\b", "tang hay giam"),
        (r"\bcung (?:ky|thang) nam (?:ngoai|TRUOC)\b", "cung ky nam TRUOC"),
        (r"\bcung ky thang roi\b", "cung ky thang TRUOC"),
        (r"\bquy ([1-4]) nam (\d{4})\b", r"quy \1/\2"),
        (r"\b(ban|thu) (duoc )?BAONHIEU tien\b", "DOANHTHU"),
        (r"\bthu ve\b", "DOANHTHU"),
        (r"\bban dat nhat\b", "DOANHTHU cao nhat"),
        (r"\bban (duoc )?BAONHIEU nhat\b", "ban chay nhat"),
        (r"\bloi gop\b", "lai gop"),
    ]:
        s = re.sub(pat, replacement, s)
    return " ".join(s.split())


def readable(s):
    for a, b in [("DOANHTHU", "doanh thu"), ("BAONHIEU", "bao nhieu"),
                 ("HOMNAY", "hom nay"), ("HOMQUA", "hom qua"), ("TRUOC", "truoc")]:
        s = s.replace(a, b)
    return s


@dataclass(frozen=True)
class Metric:
    method: str
    args: tuple = ()

    def value(self, view, period, as_of):
        return getattr(view, self.method)(period, as_of, *self.args)


def match_name(s, names):
    """names: {khoá: tên} hoặc [(khoá, tên)] (một khoá nhiều cách viết)."""
    pairs = names.items() if isinstance(names, dict) else names
    hits = [(n, key) for key, name in pairs if (n := canonical(str(name)))
            and f" {n} " in f" {s} "]
    if not hits:
        return None, s
    longest = max(len(n) for n, _ in hits)
    hits = [(n, k) for n, k in hits if len(n) == longest]
    if len({k for _, k in hits}) != 1:
        return None, s
    n, key = hits[0]
    return key, " ".join(f" {s} ".replace(f" {n} ", " ", 1).split())


def periods(s, P):
    """Keep textual order, including explicit ranges; collapse a repeated denominator period later."""
    range_pat = r"\btu (?:ngay )?(\d{1,2}/\d{1,2}/\d{4}) den (?:het )?(?:ngay )?(\d{1,2}/\d{1,2}/\d{4})\b"
    m = re.search(range_pat, s)
    if m:
        import datetime as dt
        def date(t):
            d, mo, y = map(int, t.split('/'))
            return dt.date(y, mo, d)
        try:
            a, b = date(m[1]), date(m[2])
        except ValueError:
            return [], s
        if a > b:
            return [], s
        tail, rest = extract_periods(s.replace(m[0], ' '), P)
        return ([("range", (a, b))], rest) if not tail else ([], s)
    # Holiday windows come from the supplied calendar, never a hard-coded date.
    holiday = re.search(r'\b(?:dip |mua )?tet(?: am lich| nguyen dan)?(?: nam)? (\d{4})\b', s)
    if holiday:
        year = int(holiday[1])
        try:
            current = P.tet(year)
            left = s.replace(holiday[0], ' ', 1)
            before = re.search(r'\b(?:(\d+) ngay )?TRUOC tet\b', left)
            if before:
                n = int(before[1]) if before[1] else None
                prior = P.before_tet(year, n)
                left = left.replace(before[0], ' ')
                other, _ = extract_periods(left, P)
                return ([], s) if other else ([('tet', current), ('before_tet', prior)], left)
            other, _ = extract_periods(left, P)
            return ([], s) if other else ([('tet', current)], left)
        except (ValueError, KeyError):
            return [], s
    return extract_periods(s, P)


BASE = "DOANHTHU tien ban thu ve hang don hoa bill mua so"
DIM = "mat hang san pham sp mon thuoc nhom hoat chat cac co cua nha"
PERCENT = "% phan tram ti ty trong le chiem dong gop phan bang tren DOANHTHU tong ca"


def answer(question, as_of, view, P):
    from datamate.semantic import OrderView
    if not isinstance(view, OrderView):
        return None
    s = canonical(question)
    rules = getattr(view, '_sem_rules', None)
    from datamate.rule_guard import Guard
    guard = Guard(rules.rulebook if rules else '')
    if guard.core:
        return None
    # Resolve named entities BEFORE parsing periods (names may contain dates, %, or numbers).
    metric, rest = None, s
    if not hasattr(view, '_business_names'):
        view._business_names = [
            (r'\bhoat chat\b', view.ingredient_names(), 'ingredient_revenue'),
            (r'\bnhom\b', view.group_names(), 'group_revenue'),
            (r'.', view.item_name_pairs(), 'item_revenue'),
        ]
    for marker, names, method in view._business_names:
        if re.search(marker, rest) and not (method == 'group_revenue' and re.search(r'\bkhach\b', rest)):
            key, left = match_name(rest, names)
            if key is not None:
                metric, rest = Metric(method, (key,)), left
                break
    if metric is None:
        attrs = [(pat, lab) for pat, lab in ATTR_MENTIONS if re.search(pat, rest)]
        if len(attrs) == 1:
            pat, lab = attrs[0]
            metric = Metric('attr_revenue', lab)
            rest = re.sub(pat, ' ', rest)
    # A dimension stored on a header table uses the guarded SEM implementation.
    sem_dimension = None
    if metric is None and rules:
        for marker, binding, fn in [('khach', rules.groups, 'customer_group_revenue'),
                                    ('nhan vien', rules.staff, 'staff_revenue')]:
            if binding and marker in rest:
                table, oid, col = binding
                names = {str(v): str(v) for v in rules.shop.tables[table][col].dropna().unique()}
                key, left = match_name(rest, names)
                if key is not None:
                    sem_dimension = (fn, (*binding, key))
                    rest = left
                    break
    if metric and metric.method == 'item_revenue' and rules:
        from datamate.price_history import effect
        result = effect(question, metric.args[0], rest, as_of, view, getattr(rules, 'price_history', None), guard)
        if result is not None:
            return result
    ps, rest = periods(rest, P)
    if not ps or any(label == 'BAD' or p[0] > as_of for label, p in ps):
        return None
    if 'this_month' in [x[0] for x in ps] and 'today' in [x[0] for x in ps]:
        ps = [x for x in ps if x[0] != 'today']
        rest = re.sub(r'\btu dau\b|\bden het\b', ' ', rest)
    period = ps[0][1]

    def done(kind, value):
        return None if value is None else {'type': kind, 'answer': value}

    if metric is None and sem_dimension is None and len(ps) == 1 and view.covers(period):
        weekday = re.search(r'\b(?:thu (hai|ba|tu|nam|sau|bay|[2-7])|chu nhat)\b', rest)
        if weekday and re.search(r'trung binh|binh quan', rest) and 'weekday_avg' not in guard.plan_blocked:
            left = rest.replace(weekday[0], ' ')
            if _clean(left, BASE + ' trung binh binh quan moi ngay'):
                day = {'hai': 0, 'ba': 1, 'tu': 2, 'nam': 3, 'sau': 4, 'bay': 5}.get(weekday[1])
                day = (int(weekday[1])-2 if weekday[1] and weekday[1].isdigit() else 6) if day is None else day
                measure = 'revenue' if re.search(r'DOANHTHU|\btien\b', left) else 'orders' if 'don' in left.split() else None
                if measure:
                    return done('number', view.weekday_avg(period, as_of, day, measure))
        if re.search(r'ban chay|hot nhat', rest) and 'top_items' not in guard.plan_blocked:
            from datamate.baseline import TOP_WORDS
            m = re.search(r'\b(?:top )?(\d{1,2})\b', rest)
            k = int(m[1]) if m else 1
            left = rest.replace(m[0], ' ') if m else rest
            if 1 <= k <= 99 and _clean(left, TOP_WORDS):
                values = view.top_items(period, as_of, k)
                if values:
                    return done('list' if m else 'string', values if m else values[0])
        if re.search(r'\bhuy\b', rest) and re.search(r'%|phan tram|ti le|ty le|chiem', rest):
            if _clean(rest, 'don bi huy chiem ti ty le % phan tram'):
                return done('percent', view.cancel_rate(period, as_of))

    if metric is None and sem_dimension is None and len(ps) == 1 and view.covers(period):
        # Numeric thresholds are consumed as complete phrases before residual validation.
        m = re.search(r'\b(?:tu|dat|it nhat) (\d{1,3}(?: \d{3})+|\d+) dong(?: tro len)?\b', rest)
        if m and re.search(r'tro len|it nhat', m[0] + rest) and not re.search(r'\b(khong|ko|ngoai|tru|chi)\b', rest):
            threshold = int(m[1].replace(' ', ''))
            left = rest.replace(m[0], ' ')
            if _clean(left, BASE + ' ngay dem dat ma'):
                if 'ngay' in left and 'DOANHTHU' in left:
                    return done('number', view.days_revenue_min(period, as_of, threshold))
                if re.search(r'\b(don|bill)\b', left):
                    return done('number', view.order_count_min(period, as_of, threshold))
        m = re.search(r'\b(?:tu|it nhat|o) (\d+) don(?: hang)?(?: khac nhau)?(?: tro len)?\b', rest)
        if m and re.search(r'tro len|it nhat', m[0] + rest) and not re.search(r'\b(khong|ko|ngoai|tru|chi)\b', rest):
            left = rest.replace(m[0], ' ')
            if re.search(r'\b(mat hang|san pham)\b', left) and _clean(left, 'so mat hang san pham xuat hien o co mat duoc mua dem'):
                if view.items is None or 'item_order_count' in guard.plan_blocked:
                    return None
                ids = set(view._valid(view._orders_in(period, as_of)).order_id)
                rows = view.items[view.items.order_id.isin(ids)]
                return done('number', int((rows.groupby('item').order_id.nunique() >= int(m[1])).sum()))
        if re.search(r'\be\b|khong (?:ai mua|ban)|ko (?:ai mua|ban)', rest) and re.search(r'mat hang', rest):
            if _clean(rest, 'giai doan mat hang bi e han khong ko ai mua ban ra duoc cai mon nao van dang ma'):
                if view.item_lines is None or 'unsold_items' in guard.plan_blocked:
                    return None
                import datetime as dt
                lo, _ = view.date_range()
                before = (lo, period[0] - dt.timedelta(days=1))
                ids_before = set(view._valid(view._orders_in(before, as_of)).order_id)
                ids_now = set(view._valid(view._orders_in(period, as_of)).order_id)
                lines = view.item_lines
                old = set(lines.loc[lines.order_id.isin(ids_before), 'group'].dropna()) - {'phi phuc vu'}
                now = set(lines.loc[lines.order_id.isin(ids_now), 'group'].dropna())
                return done('number', len(old - now))
    # Conditions and negation must never be consumed as filler.
    if re.search(r'\b(khong|ko|ngoai|tru|it|thap|duoi|tren|rieng)\b|\bchi\b(?! tieu\b)', rest):
        return None

    receipts = getattr(rules, 'receipts', None)
    if receipts and len(ps) == 1 and re.search(r'\bnhap\b|nha cung cap chinh', rest) and 'import_total' not in guard.plan_blocked:
        words = 'nhap hang phieu gia tri tien tong bo ra vao het'
        principal = bool(re.search(r'nha cung cap chinh', rest))
        best = principal or bool(re.search(r'nha cung cap nao', rest) and re.search(r'\bnhat\b', rest))
        if _clean(rest, words + (' nha cung cap BAONHIEU cao lon nhat nao chinh ai' if best else '')):
            if principal and 'main_supplier' in guard.plan_blocked:
                return None
            return done('string' if best else 'number', receipts.value(period, as_of, best=best, most_receipts=principal))

    if len(ps) == 1 and metric and metric.method == 'group_revenue':
        # “Số đơn có mua hàng nhóm X” is an order-grain count: one order is
        # counted once even if it contains several X lines.  Do this before
        # the line-grain group item reducers.
        if (re.search(r'\b(?:so|BAONHIEU|may) don\b', rest)
                and re.search(r'\b(?:co mua|mua hang|co mat hang)\b', rest)
                and _clean(rest, DIM + ' don co mua hang co mat hang BAONHIEU so dem')):
            return done('number', view.group_order_count(period, as_of, metric.args[0]))
        from datamate.relational_sem import grouped_item_query
        grouped = grouped_item_query(view, metric.args[0], rest, period, as_of, guard)
        if grouped is not None:
            return grouped
    if len(ps) == 1 and sem_dimension and sem_dimension[0] == 'customer_group_revenue':
        if re.search(r'trung binh|binh quan', rest) and 'customer_avg_order_value' not in guard.plan_blocked and _clean(rest, BASE + ' ' + DIM + ' khach trung binh binh quan moi mot gia tri'):
            from datamate.relational_sem import customer_group_average
            return done('number', customer_group_average(rules, period, as_of, sem_dimension[1][-1]))

    if metric and metric.method == 'item_revenue' and len(ps) == 1:
        key = metric.args[0]
        if view.items is not None:
            it = view.items[view.items['item'] == key]
            units = set(it['bunit'].dropna())
            mentioned = [u for u in units if re.search(r'\b' + re.escape(u) + r'\b', rest)]
            if len(mentioned) == 1:
                unit = mentioned[0]
                left = re.sub(r'\b' + re.escape(unit) + r'\b', ' ', rest)
                if re.search(r'\bgia\b', left) and ps[0][0] in {'day', 'today', 'yesterday'}:
                    if 'base_price' not in it or not _clean(left, DIM + ' gia ban don mot vao') or 'Giá bán X ngày D' in guard.changed:
                        return None
                    ids = set(view._orders_in(period, as_of)['order_id'])
                    prices = it.loc[it['order_id'].isin(ids), 'base_price']
                    if prices.empty or prices.isna().any() or prices.max() - prices.min() > 1e-6:
                        return None
                    from datamate.helpers import round_money
                    return done('number', round_money(prices.iloc[0]))
                if re.search(r'\bban\b', left) and not re.search(r'DOANHTHU|\btien\b|%', left):
                    if _clean(left, DIM + ' so ban ra') and not guard.plan_blocked.intersection({'qty_sold'}):
                        return done('number', view.qty_sold(period, as_of, unit, item=key)) if view.covers(period) else None

    if metric is None and sem_dimension is None and len(ps) == 1:
        ranked = (re.search(r'\bkhach\b|\bai mua\b', rest) and re.search(r'\bnhat\b', rest)
                  and re.search(r'\btien\b|chi tieu', rest) and not re.search(r'\bdon\b', rest))
        if ranked:
            k_match = re.search(r'\b(?:top )?(\d{1,2}) khach\b', rest)
            k = int(k_match[1]) if k_match else 1
            left = rest.replace(k_match[0], ' khach ') if k_match else rest
            if 1 <= k <= 99 and _clean(left, BASE + ' khach chi tieu BAONHIEU nhat cao top liet ke ai nao nhung gi'):
                if not guard.plan_blocked.intersection({'top_spender', 'customer_revenue'}) and view.covers(period):
                    value = view.top_customers(period, as_of, k)
                    return done('list' if k_match else 'string', value if k_match else value[0] if value else None)
        # Highest-revenue day, including the colloquial "bán đắt nhất".
        if re.search(r'\bngay nao\b', rest) and 'DOANHTHU' in rest and re.search(r'\b(cao|BAONHIEU|DOANHTHU) nhat\b', rest):
            if _clean(rest, BASE + ' ngay nao cao BAONHIEU nhat xet theo') and view.covers(period):
                return done('date', view.best_revenue_day(period, as_of))

    has_money = bool(re.search(r'DOANHTHU|\btien\b', rest))
    if has_money and re.search(r'\b(?:va|kem) (?:so |BAONHIEU )?don\b|\bso don\b.*\bva\b.*DOANHTHU', rest):
        return None  # Two measures require a list/table contract, never one scalar.
    if metric is not None or sem_dimension:
        if not has_money and not re.search(r'%|phan tram|chiem', rest):
            return None
        # Named item quantity must not accidentally become revenue.
        words = BASE + ' ' + DIM + ' ' + PERCENT + ' khach nhan vien ma tra bang qua dong gop phan'
    else:
        if has_money:
            metric = Metric('revenue')
        elif re.search(r'\b(don|bill|hoa don)\b', rest):
            metric = Metric('order_count')
        else:
            return None
        words = BASE

    def value(p):
        if not view.covers(p):
            return None
        if sem_dimension:
            from datamate.sem_api import SemAPI
            sem = SemAPI(rules.shop, as_of, rules.rulebook)
            sem._view, sem._built = view, True
            fn, args = sem_dimension
            return getattr(sem, fn)(p, *args)
        aliases = {'attr_revenue': 'payment_revenue' if metric.args[0] == 'pay' else 'channel_revenue'} if metric.method == 'attr_revenue' else {}
        if aliases.get(metric.method, metric.method) in guard.plan_blocked:
            return None
        return metric.value(view, p, as_of)

    is_pct = bool(re.search(r'%|phan tram|\b(ti|ty) (le|trong)\b', rest))
    comparing = bool(re.search(r'\b(so voi|so vs|hon|kem|thay doi|tang|giam|chenh)\b', rest))
    if comparing:
        if not _clean(rest, words + ' % phan tram so voi vs hon kem hay tang giam thay doi chenh lech thang lien TRUOC cung ky nam truong dau'):
            return None
        if len(ps) == 1:
            if re.search(r'cung ky nam TRUOC', rest) and ps[0][0] == 'month':
                old = P.month(period[0].month, period[0].year - 1)
            elif re.search(r'thang (lien )?TRUOC', rest):
                old = P.prev_month_same_days() if ps[0][0] == 'this_month' else P.prev_month(period[0].month, period[0].year)
            else:
                return None
        elif len(ps) == 2:
            labels = dict(ps)
            if 'this_month' in labels and 'prev_month_same_days' in labels:
                period, old = labels['this_month'], labels['prev_month_same_days']
            else:
                old = ps[1][1]
        else:
            return None
        new_v, old_v = value(period), value(old)
        if new_v is None or old_v is None:
            return None
        return done('percent' if is_pct else 'number', pct_change(new_v, old_v) if is_pct else new_v - old_v)
    if is_pct:
        # Repeated period in "X tháng T / tổng doanh thu tháng T" is one scope.
        if any(p != period for _, p in ps) or not (sem_dimension or metric.method != 'revenue'):
            return None
        if _clean(rest, words + ' ' + PERCENT):
            return done('percent', share(value(period), view.revenue(period, as_of)) if view.covers(period) else None)
        return None
    if len(ps) != 1 or not _clean(rest, words):
        return None
    return done('number', value(period))
