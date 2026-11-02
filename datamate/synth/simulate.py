"""Mô phỏng sổ bán hàng của một cửa hàng ở dạng chuẩn (canonical): đây là "sự thật" để tính đáp án. Cách xuất ra file
POS (tên cột, định dạng, README) nằm ở render.py, nên mỗi đáp án không phụ thuộc cách xuất."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from datamate.synth.worlds import Archetype

SURNAMES = "Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Đinh Mai Trịnh Lâm".split()
MIDDLES = "Văn Thị Minh Ngọc Thanh Hoàng Đức Thu Quốc Gia Bảo Kim Hữu".split()
GIVEN = ("An Bình Chi Dũng Giang Hà Hải Hạnh Hiếu Hoa Huy Hương Khoa Lan Linh Long Mai Nam Nga Nhung Phúc Phương "
         "Quân Sơn Tâm Thảo Thắng Trang Trung Tuấn Vy Yến").split()


@dataclass
class World:
    biz_id: str
    arch: Archetype
    start: dt.date
    end: dt.date  # ngày cuối có dữ liệu
    as_of: dt.date
    items: pd.DataFrame  # item_id, name, group, bunit, conv (dict), cost
    prices: pd.DataFrame  # item_id, date, price (một đơn vị cơ bản, áp dụng từ date)
    customers: pd.DataFrame  # cust_id, name
    orders: pd.DataFrame  # order_id, ts, date, cust_id, channel, payment, status, order_discount, ship_fee
    lines: pd.DataFrame  # order_id, line_no, item_id, unit, factor, qty, unit_price, line_discount, amount, bqty, ret_qty
    payments: pd.DataFrame  # order_id, date, amount (đơn ghi nợ)
    receipts: pd.DataFrame  # receipt_id, date, status, supplier, opening
    receipt_lines: pd.DataFrame  # receipt_id, item_id, unit, qty, unit_cost


def _names(rng, n: int) -> list[str]:
    out = [f"{rng.choice(SURNAMES)} {rng.choice(MIDDLES)} {rng.choice(GIVEN)}" for _ in range(n)]
    for i in rng.choice(n, size=max(1, n // 20), replace=False)[1:]:  # vài khách trùng tên khác mã (rulebook §3)
        out[i] = out[0]
    return out


def simulate(arch: Archetype, biz_id: str, seed: int) -> World:
    rng = np.random.default_rng(seed)
    end = dt.date(2026, 5, 1) + dt.timedelta(days=int(rng.integers(0, 150)))
    start = dt.date(end.year - 1, int(rng.integers(1, 7)), 1)
    as_of = end - dt.timedelta(days=int(rng.choice([0, 0, 0, 3, 9])))
    days = pd.date_range(start, end).date

    items = pd.DataFrame([{"item_id": f"SP{i + 1:04d}", "name": n, "group": g, "bunit": u, "price": p, "conv": c,
                           "pop": w, "cost": int(round(p * rng.uniform(0.55, 0.8), -2))}
                          for i, (n, g, u, p, c, w) in enumerate(arch.items)])
    # đổi giá: ~40% mặt hàng tăng giá 1 lần; sức mua sau tăng giá đổi theo hệ số ngẫu nhiên
    price_rows, demand_shift = [], {}
    for r in items.itertuples():
        price_rows.append((r.item_id, start, r.price))
        if rng.random() < 0.4:
            d = days[int(rng.integers(40, len(days) - 40))]
            price_rows.append((r.item_id, d, int(round(r.price * rng.uniform(1.04, 1.15), -2))))
            demand_shift[r.item_id] = (d, rng.uniform(0.55, 1.25))
    prices = pd.DataFrame(price_rows, columns=["item_id", "date", "price"])

    n_cust = int(max(20, arch.orders_per_day * 8))
    customers = pd.DataFrame({"cust_id": [f"KH{i + 1:04d}" for i in range(n_cust)], "name": _names(rng, n_cust)})
    cust_w = 1 / np.arange(1, n_cust + 1) ** 0.8
    cust_w /= cust_w.sum()

    online = [c for c in arch.channels[1:]]
    orders, lines, payments = [], [], []
    oid = 0
    for d in days:
        season = 1 + 0.25 * np.sin(2 * np.pi * (d.timetuple().tm_yday / 365)) + (0.2 if d.weekday() >= 5 else 0)
        for _ in range(rng.poisson(arch.orders_per_day * season)):
            oid += 1
            order_id = f"HD{oid:06d}"
            hour = int(rng.integers(arch.open_hours[0], arch.open_hours[1]))
            ts = dt.datetime.combine(d, dt.time(hour, int(rng.integers(0, 60)), int(rng.integers(0, 60))))
            channel = arch.channels[0] if not online or rng.random() < 0.65 else str(rng.choice(online))
            cust = str(rng.choice(customers["cust_id"], p=cust_w)) if rng.random() < arch.known_customer_rate else None
            pays = [p for p in arch.payments if p != "COD" or channel != arch.channels[0]]
            payment = str(rng.choice(pays))
            if arch.debt and cust and channel == arch.channels[0] and rng.random() < 0.18:
                payment = "Ghi nợ"
            status = "completed"
            u = rng.random()
            if u < 0.04:
                status = "cancelled"
            elif u < 0.07:
                status = "returned"
            elif channel != arch.channels[0] and (end - d).days <= 3:
                status = "delivering"
            w = items["pop"].to_numpy(float).copy()
            for i, iid in enumerate(items["item_id"]):
                if iid in demand_shift and d >= demand_shift[iid][0]:
                    w[i] *= demand_shift[iid][1]
            picks = rng.choice(len(items), size=min(len(items), 1 + rng.poisson(0.8)), replace=False, p=w / w.sum())
            total = 0
            for ln, i in enumerate(picks):
                it = items.iloc[i]
                price = int(prices[(prices["item_id"] == it.item_id) & (prices["date"] <= d)]["price"].iloc[-1])
                unit, factor = it.bunit, 1
                if it.conv and rng.random() < 0.15:
                    unit, factor = next(iter(it.conv.items()))
                qty = int(1 + rng.poisson(0.6 if factor > 1 else 1.2))
                unit_price = price * factor
                disc = int(round(unit_price * qty * 0.05, -3)) if rng.random() < 0.08 else 0
                amount = unit_price * qty - disc
                total += amount
                lines.append((order_id, ln + 1, it.item_id, unit, factor, qty, unit_price, disc, amount, qty * factor, 0))
            if status == "returned":  # trả một phần dòng đầu (rulebook §1: vẫn bỏ cả đơn)
                last = len(lines) - len(picks)
                row = list(lines[last])
                row[10] = int(rng.integers(1, row[5] + 1))
                lines[last] = tuple(row)
            odisc = int(round(total * 0.1, -3)) if total > 200_000 and rng.random() < 0.1 else 0
            ship = 20_000 if channel != arch.channels[0] and rng.random() < 0.5 else 0
            orders.append((order_id, ts, d, cust, channel, payment, status, odisc, ship))
            if payment == "Ghi nợ" and status != "cancelled":
                due = total - odisc + ship
                paid_now = int(round(due * float(rng.choice([0, 0, 0.3, 0.5])), -3))
                if paid_now:
                    payments.append((order_id, d, paid_now))
                left, pd_ = due - paid_now, d
                while left > 0 and rng.random() < 0.75:
                    pd_ = pd_ + dt.timedelta(days=int(rng.integers(3, 40)))
                    if pd_ > end:
                        break
                    amt = left if rng.random() < 0.6 else int(round(left * 0.5, -3)) or left
                    payments.append((order_id, pd_, amt))
                    left -= amt

    receipts, rlines = [], []
    if arch.imports:
        rid = 0
        for k, d in enumerate(days[::7]):
            rid += 1
            st = "cancelled" if k and rng.random() < 0.05 else "done"
            receipts.append((f"PN{rid:05d}", d, st, str(rng.choice(["NCC Minh Phát", "NCC Hoàng Gia", "NCC Thành Công"])),
                             k == 0))
            for it in items.sample(n=max(2, len(items) // 2), random_state=int(rng.integers(1 << 30))).itertuples():
                big = list(it.conv.items())
                unit, factor = big[0] if big and rng.random() < 0.8 else (it.bunit, 1)
                cost = int(round(it.cost * factor * rng.uniform(0.95, 1.06), -2))
                rlines.append((f"PN{rid:05d}", it.item_id, unit, int(rng.integers(1, 12)) * (1 if factor > 1 else 5), cost))

    return World(
        biz_id, arch, start, end, as_of, items.drop(columns=["price", "pop"]), prices, customers,
        pd.DataFrame(orders, columns=["order_id", "ts", "date", "cust_id", "channel", "payment", "status",
                                      "order_discount", "ship_fee"]),
        pd.DataFrame(lines, columns=["order_id", "line_no", "item_id", "unit", "factor", "qty", "unit_price",
                                     "line_discount", "amount", "bqty", "ret_qty"]),
        pd.DataFrame(payments, columns=["order_id", "date", "amount"]),
        pd.DataFrame(receipts, columns=["receipt_id", "date", "status", "supplier", "opening"]),
        pd.DataFrame(rlines, columns=["receipt_id", "item_id", "unit", "qty", "unit_cost"]),
    )
