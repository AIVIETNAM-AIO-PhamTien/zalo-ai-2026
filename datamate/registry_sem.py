"""Phép tính trên bảng danh mục/phiếu với tên cột được chỉ định rõ.

Không đoán cột hoặc ngành từ tên file; dữ liệu phải đã qua loader.
"""
import pandas as pd

from datamate.semantic import classify_status


def dated_rows(shop, table, columns, date_column, period, as_of):
    frame = shop.tables.get(table)
    if frame is None or len(set(columns)) != len(columns) or any(c not in frame for c in columns):
        return None
    dates = frame[date_column]
    if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any() or frame.empty:
        return None
    start, end = map(pd.Timestamp, period)
    end = min(end.normalize(), pd.Timestamp(as_of))
    if start > end or end < dates.min().normalize() or start > dates.max().normalize():
        return None
    return frame.loc[(dates.dt.normalize() >= start.normalize()) & (dates.dt.normalize() <= end)].copy()


def new_members(shop, period, as_of, table, customer, registered):
    frame = shop.tables.get(table)
    if frame is None or customer not in frame or registered not in frame:
        return None
    # Một khách chỉ có một ngày đăng ký; không đếm một mã hai lần.
    if frame[customer].isna().any() or frame[customer].astype(str).str.strip().eq("").any():
        return None
    if frame.groupby(customer)[registered].nunique(dropna=False).gt(1).any():
        return None
    rows = dated_rows(shop, table, [customer, registered], registered, period, as_of)
    return None if rows is None else int(rows[customer].nunique())


def main_supplier(shop, period, as_of, table, receipt, supplier, date, status):
    frame = shop.tables.get(table)
    cols = [receipt, supplier, date, status]
    if frame is None or any(c not in frame for c in cols) or frame[cols].isna().any().any():
        return None
    if any(frame[c].astype(str).str.strip().eq("").any() for c in [receipt, supplier, status]):
        return None
    # Bảng chi tiết được phép lặp mã phiếu, nhưng thông tin đầu phiếu phải nhất quán.
    if frame.groupby(receipt)[[supplier, date, status]].nunique().gt(1).any().any():
        return None
    rows = dated_rows(shop, table, cols, date, period, as_of)
    if rows is None:
        return None
    rows = rows[rows[status].map(classify_status) != "cancelled"].drop_duplicates(receipt)
    counts = rows.groupby(supplier)[receipt].nunique()
    if counts.empty:
        return None
    # supplier phải là mã NCC (hoặc tên nếu shop không có mã): đúng thứ tự hoà §9.
    return min(map(str, counts[counts == counts.max()].index))


def import_total(shop, period, as_of, table, receipt, date, amount, status):
    """Bảng đầu phiếu nhập: tiền cả phiếu lặp ở dòng chi tiết chỉ tính một lần."""
    from datamate.helpers import round_money
    import numpy as np

    frame = shop.tables.get(table)
    cols = [receipt, date, amount, status]
    if frame is None or any(c not in frame for c in cols) or frame[cols].isna().any().any():
        return None
    if frame.groupby(receipt)[[date, amount, status]].nunique().gt(1).any().any():
        return None
    rows = dated_rows(shop, table, cols, date, period, as_of)
    if rows is None:
        return None
    rows = rows[rows[status].map(classify_status) != "cancelled"].drop_duplicates(receipt)
    money = pd.to_numeric(rows[amount], errors="coerce")
    if money.isna().any() or not np.isfinite(money).all():
        return None
    return round_money(money.sum())
