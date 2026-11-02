"""Train §5: giá vốn dòng bán, lãi gộp/ròng. Không suy giá vốn từ phiếu nhập."""
import numpy as np
import pandas as pd

from datamate.helpers import round_money
from datamate.semantic import clean_order_rows, item_key


def sale_costs(shop, view, qty_col, cost_col, item_col=None):
    """Chi phí khớp chính xác với nguồn dòng hàng đã kiểm của OrderView."""
    source = view.source or {}
    frame = shop.tables.get(source.get("table"))
    oid, amount = source.get("order_id"), source.get("amount")
    cols = [oid, amount, qty_col, cost_col] + ([item_col] if item_col else [])
    if frame is None or any(c not in frame for c in cols):
        return None
    frame = clean_order_rows(shop, source.get("table"))
    out = pd.DataFrame({"order_id": frame[oid].astype(str),
                        "amount": pd.to_numeric(frame[amount], errors="coerce"),
                        "qty": pd.to_numeric(frame[qty_col], errors="coerce"),
                        "cost": pd.to_numeric(frame[cost_col], errors="coerce")})
    if item_col:
        out["item"] = frame[item_col].map(lambda x: item_key(x) if pd.notna(x) else None)
    # Không chỉ khớp tổng shop: đối chiếu từng đơn để chặn nối sai/mất dòng.
    got = out.groupby("order_id")["amount"].sum(min_count=1)
    expected = view.lines.groupby("order_id")["amount"].sum(min_count=1)
    if set(got.index) != set(expected.index) or not np.isclose(got.reindex(expected.index).to_numpy(dtype=float, na_value=float("nan")), expected.to_numpy(dtype=float, na_value=float("nan")),
                                                                              atol=0.001, rtol=0).all():
        return None
    return out


def profit(shop, view, period, as_of, qty_col, cost_col, item_col=None, item=None):
    rows = sale_costs(shop, view, qty_col, cost_col, item_col)
    if rows is None:
        return None
    ids = set(view._valid(view._orders_in(period, as_of))["order_id"])
    if item is not None:
        if not item_col or item_key(item) not in set(rows["item"].dropna()):
            return None
        rows = rows[rows["item"] == item_key(item)]
    rows = rows[rows["order_id"].isin(ids)]
    values = rows[["amount", "qty", "cost"]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float, na_value=float("nan"))).all() or (rows[["qty", "cost"]] < 0).any().any():
        return None
    return round_money((rows["amount"] - rows["qty"] * rows["cost"]).sum())


def expenses(shop, period, as_of, table, date_col, amount_col):
    frame = shop.tables.get(table)
    if frame is None or date_col not in frame or amount_col not in frame or date_col == amount_col:
        return None
    dates = frame[date_col]
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any():
        return None
    start, end = pd.Timestamp(period[0]), pd.Timestamp(min(period[1], as_of))
    # Sổ không theo đơn: không xoá dòng giống nhau, sửa chữa/thiết bị vẫn tính.
    rows = frame.loc[(dates.dt.normalize() >= start) & (dates.dt.normalize() <= end), amount_col]
    values = pd.to_numeric(rows, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float, na_value=float("nan"))).all():
        return None
    return round_money(values.sum())


def order_dimension(shop, view, period, as_of, table, id_col, group_col, label, best=False):
    frame = shop.tables.get(table)
    if frame is None or id_col not in frame or group_col not in frame:
        return None
    data = frame[[id_col, group_col]].copy()
    data[id_col] = data[id_col].astype(str)
    if data.groupby(id_col)[group_col].nunique(dropna=False).gt(1).any():
        return None
    data = data.drop_duplicates(id_col).set_index(id_col)[group_col]
    orders = view._valid(view._orders_in(period, as_of))
    if set(orders.order_id) - set(data.index):
        return None
    amounts = view.order_money().reindex(orders.order_id, fill_value=0)  # NaN: tiền đơn chưa biết
    if amounts.isna().any():
        return None
    chosen = data.reindex(orders.order_id)
    if best and chosen.isna().any():
        return None
    totals = amounts.groupby(chosen).sum()
    if best:
        return None if totals.empty else min(map(str, totals[totals == totals.max()].index))
    return round_money(totals.get(label, 0)) if label in set(data.dropna()) else None


def best_profit_item(shop, view, period, as_of, qty, cost, name, code):
    rows = sale_costs(shop, view, qty, cost, name)
    if rows is None:
        return None
    frame = shop.tables[(view.source or {})["table"]]
    if code not in frame or name not in frame:
        return None
    cat = pd.DataFrame({"item": frame[name].map(item_key), "code": frame[code]})
    if cat["code"].isna().any() or cat.groupby("item").code.nunique().gt(1).any():
        return None
    ids = set(view._valid(view._orders_in(period, as_of)).order_id)
    rows = rows[rows.order_id.isin(ids)]
    rows = rows[rows.item != "phi phuc vu"]
    if rows.empty or rows[["amount", "qty", "cost"]].isna().any().any():
        return None
    if not np.isfinite(rows[["amount", "qty", "cost"]].to_numpy(dtype=float, na_value=float("nan"))).all():
        return None
    if (rows[["qty", "cost"]] < 0).any().any():
        return None
    values = (rows.amount - rows.qty * rows.cost).groupby(rows.item).sum()
    ranks = cat.drop_duplicates("item").set_index("item").code.astype(str)
    best = min(values[values == values.max()].index, key=lambda k: ranks[k])
    return str(cat.loc[cat.item == best, "code"].iloc[0])
