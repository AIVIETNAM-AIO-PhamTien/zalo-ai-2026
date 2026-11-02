"""Reusable scopes on already validated sales rows; no guessed joins."""
import re

from datamate.helpers import round_money


def item_groups(view):
    items, groups = view.items, view.groups
    if items is None or groups is None or not items.index.is_unique or not groups.index.is_unique:
        return None
    linked = groups.reindex(items.index)
    if linked['group'].isna().any() or not items.order_id.eq(linked.order_id).all():
        return None
    data = items[['item']].assign(group=linked['group'])
    if data.groupby('item')['group'].nunique().gt(1).any():
        return None
    return data.drop_duplicates('item').set_index('item')['group']


def grouped_item_query(view, group, rest, period, as_of, guard):
    from datamate.baseline import _clean
    from datamate.business_query import DIM, BASE
    mapping = item_groups(view)
    if mapping is None or not view.covers(period):
        return None
    ids = set(view._valid(view._orders_in(period, as_of)).order_id)
    keys = set(mapping[mapping == group].index)
    items = view.items[view.items.item.isin(keys)]
    active = items[items.order_id.isin(ids)]
    money = bool(re.search(r'DOANHTHU|\btien\b', rest))
    words = DIM + ' ' + BASE + ' nhat cao BAONHIEU nao gi ra ban chay hot'
    if re.search(r'\bnhat\b', rest):
        unit_mentions = [u for u in set(items.bunit.dropna()) if re.search(r'\b' + re.escape(u) + r'\b', rest)]
        if len(unit_mentions) > 1:
            return None
        if unit_mentions:
            active = active[active.bunit == unit_mentions[0]]
            words += ' ' + unit_mentions[0]
        if not _clean(rest, words) or active.empty:
            return None
        if money:
            frame = view.item_lines
            if frame is None:
                return None
            frame = frame[frame.order_id.isin(ids) & frame['group'].isin(keys)]
            totals = frame.groupby('group').amount.sum()
        else:
            if 'top_items' in guard.plan_blocked or active.bqty.isna().any():
                return None
            totals = active.groupby('item').bqty.sum()
        if totals.empty:
            return None
        ranks = items.groupby('item').rank_key.first() if 'rank_key' in items and items.rank_key.notna().all() else items.groupby('item').name.first()
        best = min(totals[totals == totals.max()].index, key=lambda x: str(ranks[x]))
        names = items.groupby('item').name.first()
        return {'type': 'string', 'answer': str(names[best])}
    if money or 'qty_sold' in guard.plan_blocked:
        return None
    units = [u for u in set(items.bunit.dropna()) if re.search(r'\b' + re.escape(u) + r'\b', rest)]
    if len(units) != 1:
        return None
    unit = units[0]
    left = re.sub(r'\b' + re.escape(unit) + r'\b', ' ', rest)
    if not re.search(r'\bban\b', left) or not _clean(left, DIM + ' ban ra so') or items.bunit.isna().any():
        return None
    rows = active[active.bunit == unit]
    if rows.bqty.isna().any():
        return None
    return {'type': 'number', 'answer': float(rows.bqty.sum())}


def customer_group_average(rules, period, as_of, label):
    from datamate.sem_api import SemAPI
    view = rules.view
    table, oid, col = rules.groups
    frame = rules.shop.tables[table][[oid, col]].copy()
    frame[oid] = frame[oid].astype(str)
    if frame.groupby(oid)[col].nunique(dropna=False).gt(1).any():
        return None
    ids = set(view._valid(view._orders_in(period, as_of)).order_id)
    ids &= set(view.customers.order_id) if view.customers is not None else set()
    selected = set(frame.loc[frame[col] == label, oid]) & ids
    sem = SemAPI(rules.shop, as_of, rules.rulebook)
    sem._view, sem._built = view, True
    total = sem.customer_group_revenue(period, *rules.groups, label)
    return round_money(total/len(selected)) if selected and total is not None else None
