"""Nhận câu hỏi đầy đủ → SEM, không gọi LLM, không dùng id shop/câu hỏi.

Chỉ nhận kỳ đơn và ý định đơn; cột phải duy nhất theo README/schema và tên
có nghĩa rõ. Thất bại/thiếu dữ liệu trả None để giữ đường fallback hiện có.
"""
from __future__ import annotations

import re

import pandas as pd

from datamate.baseline import _clean, extract_periods, norm
from datamate.helpers import Periods
from datamate.parsing import fold
from datamate.sem_api import SemAPI
from datamate.loader import Shop
from datamate.semantic import M_ITEM_NAME, M_ORDER_ID, _find, _qty_col, _tables_for, shop_columns
from datamate.semantic import order_id_columns, _name_key
from datamate.semantic import parse_readme_columns


def _key(text):
    return re.sub(r"[^a-z0-9]", "", fold(text))


def _one(columns, predicate):
    hits = [c for c in columns if predicate(c)]
    return hits[0] if len(hits) == 1 else None


class SemRules:
    def __init__(self, shop, view, rulebook="", calendar=None, public_rules=True):
        self.shop, self.view, self.rulebook, self.calendar = shop, view, rulebook, calendar
        self.public_rules = public_rules
        self.roles = {}
        for key, cols in shop_columns(shop).items():
            for table in _tables_for(shop, key):
                self.roles.setdefault(table, {}).update({c: m for c, m in cols.items() if c in shop.tables[table]})
        # A registry customer key is not an order key, even if an auxiliary card says so.
        for key, cols in parse_readme_columns(shop.readme).items():
            for table in _tables_for(shop, key):
                self.roles.setdefault(table, {}).update({c: m for c, m in cols.items() if c in shop.tables[table]})
        # Additional lexical roles are independent of shop IDs and of column values.
        declared = {table: set(cols) for key, cols in parse_readme_columns(shop.readme).items()
                    for table in _tables_for(shop, key)}
        aliases = {'makhachhang': 'mã khách', 'makhach': 'mã khách', 'makh': 'mã khách',
                   'nhomkhachhang': 'nhóm khách', 'loaikhachhang': 'nhóm khách',
                   'manhanvien': 'mã nhân viên bán', 'manv': 'mã nhân viên bán'}
        for table, frame in shop.tables.items():
            for col in frame:
                if col not in declared.get(table, set()) and _key(col) in aliases:
                    self.roles.setdefault(table, {})[col] = aliases[_key(col)]
        self.cost = self._cost_columns()
        from datamate.receipt_sem import Receipts
        self.receipts = Receipts.from_shop(shop)
        from datamate.price_history import PriceHistory
        self.price_history = PriceHistory.from_shop(shop)
        self.expenses = self._expense_columns()
        self.staff = self._dimension("mã nhân viên bán")
        self.groups = self._dimension("nhóm khách")
        if self.groups is None and view.customers is not None:
            candidates = []
            for table, cols in self.roles.items():
                code, group = _find(cols, 'mã khách'), _find(cols, 'nhóm khách')
                if not code or not group:
                    continue
                frame = shop.tables[table][[code, group]].dropna(subset=[code]).copy()
                frame[code] = frame[code].astype(str).str.strip()
                if frame.groupby(code)[group].nunique(dropna=False).gt(1).any():
                    continue
                lookup = frame.drop_duplicates(code).set_index(code)[group]
                if set(view.customers.customer) - set(lookup.index):
                    continue
                bound = view.orders[['order_id']].merge(view.customers[['order_id', 'customer']],
                                                       on='order_id', how='left', validate='one_to_one')
                bound['group'] = bound.customer.map(lookup)
                candidates.append(bound[['order_id', 'group']])
            if len(candidates) == 1:
                table = '__sem_customer_groups'
                # Keep the generated binding local, without changing the caller's Shop.
                shop = Shop(shop.biz_id, shop.readme, {**shop.tables, table: candidates[0]},
                            {**shop.meta, table: {}})
                self.groups = (table, 'order_id', 'group')
        self.members = self._member_columns() if public_rules else None
        self.adjustments = self._adjustment_columns() if public_rules else {}
        source = self.view.source or {}
        frame = self.shop.tables.get(source.get("table"))
        self.item = None
        if frame is not None:
            name = _find(self.roles.get(source.get("table"), {}), M_ITEM_NAME)
            code = _one(frame, lambda c: _key(c) in {"mahang", "mahanghoa", "masp", "masanpham"})
            if name and code:
                self.item = (name, code)
        # Giữ đúng các bảng cần cho luật mới, không giữ toàn bộ dữ liệu thô mỗi shop.
        needed = {binding[0] for binding in (self.expenses, self.staff, self.groups, self.members) if binding}
        needed.update(binding[0] for binding in self.adjustments.values())
        if self.cost:
            needed.add(source.get("table"))
        self.shop = Shop(shop.biz_id, shop.readme,
                         {t: shop.tables[t] for t in needed if t in shop.tables},
                         {t: shop.meta[t] for t in needed if t in shop.meta})

    def _cost_columns(self):
        source = self.view.source or {}
        table = source.get("table")
        frame = self.shop.tables.get(table)
        if frame is None:
            return None
        qty = _qty_col(self.roles.get(table, {}))
        cost = _one(frame, lambda c: _key(c) in {"giavon", "giavonlucban", "giavonmotdonviban"})
        # Đừng lấy giá vốn hiện tại trong danh mục thay cho giá vốn ghi trên dòng bán.
        declared = re.search(r"gia von.{0,100}mot don vi ban", fold(self.shop.readme))
        return (qty, cost) if qty and cost and declared else None

    def _expense_columns(self):
        hits = []
        if "so chi phi" not in fold(self.shop.readme):
            return None
        for table, frame in self.shop.tables.items():
            if "chiphi" not in _key(table):
                continue
            date = _one(frame, lambda c: pd.api.types.is_datetime64_any_dtype(frame[c]))
            money = _one(frame, lambda c: _key(c) in {"sotien", "tienchi", "chiphi"})
            if date and money:
                hits.append((table, date, money))
        return hits[0] if len(hits) == 1 else None

    def _dimension(self, meaning):
        hits = []
        for table, cols in self.roles.items():
            oid, role = _find(cols, M_ORDER_ID), _find(cols, meaning)
            if oid and role and set(self.view.orders.order_id) <= set(self.shop.tables[table][oid].astype(str)):
                hits.append((table, oid, role))
        return hits[0] if len(hits) == 1 else None

    def _member_columns(self):
        hits = []
        for table, frame in self.shop.tables.items():
            date = _one(frame, lambda c: _key(c) in {"ngaydangky", "ngaythamgiathanhvien"})
            if not date or not pd.api.types.is_datetime64_any_dtype(frame[date]):
                continue
            code = _one(frame, lambda c: _key(c) in {"makhach", "makhachhang", "makh", "customerid"})
            if code is None:
                name = _one(frame, lambda c: _key(c) in {"tenkh", "tenkhach", "tenkhachhang"})
                if name and frame[name].notna().all() and frame[name].map(_name_key).is_unique:
                    code = name
            if code:
                hits.append((table, code, date))
        return hits[0] if len(hits) == 1 else None

    def _adjustment_columns(self):
        keys = {"shipping": {"phiship", "phigiaohang", "phivanchuyen"},
                "voucher": {"tiengiamvoucher", "giamvoucher", "giatrivoucher"},
                "points": {"tiengiamdiem", "tiengiamnhodungdiem", "tienduocgiamtudiem"},
                "trade_in": {"thumaycu", "tienthumaycu"}}
        hits = {name: [] for name in keys}
        source = self.view.source or {}
        ids = order_id_columns(self.shop)
        if source.get("table") and source.get("order_id"):
            ids.setdefault(source["table"], source["order_id"])
        for table, oid in ids.items():
            for name, names in keys.items():
                col = _one(self.shop.tables[table], lambda c: _key(c) in names)
                if col:
                    hits[name].append((table, oid, col))
        return {name: values[0] for name, values in hits.items() if len(values) == 1}

    def answer(self, question, as_of):
        s = norm(question)
        # norm dùng 'may' cho 'mấy'; trong cụm nghiệp vụ này nó là danh từ 'máy'.
        s = re.sub(r"\bthu BAONHIEU cu\b", "thu may cu", s)
        periods, rest = extract_periods(s, Periods(as_of, self.calendar))
        if len(periods) != 1 or periods[0][0] == "BAD":
            return None
        period = periods[0][1]
        if period[0] > as_of or not self.view.covers(period):
            return None
        # Cách diễn đạt đầy đủ phép trừ của lãi ròng, không phải lọc bỏ một khoản.
        if "lai rong" in rest or "loi nhuan rong" in rest:
            rest = re.sub(r"\btru het chi phi roi thi\b", " ", rest)
        # _clean cho phép từ đệm; các điều kiện này không phải từ đệm ở tầng mới.
        if re.search(r"\b(khong|tru|ngoai|rieng|it|thap|so voi|vs)\b|%|phan tram|ti le|ty le|\bchi\b(?! (phi|tieu)\b)", rest):
            return None
        sem = SemAPI(self.shop, as_of, self.rulebook)
        sem._view, sem._built = self.view, True
        fn, args, kind = None, (), "number"
        ranked = None
        for name, pattern, words, result_kind in [
            ("loyal_customer", r"khach.*trung thanh.*nhat", "khach hang trung thanh nhat ai", "string"),
            ("big_spender", r"khach.*sop.*nhat", "khach hang sop nhat ai", "string"),
            ("fickle_customer", r"khach.*doi y.*nhat", "khach hang hay doi y nhat ai", "string"),
            ("popular_item", r"(mat hang|san pham|mon).*ua chuong.*nhat", "mat hang san pham mon ua chuong nhat ai gi", "string"),
            ("busiest_day", r"ngay.*dat khach.*nhat", "ngay dat khach nhat nao", "date"),
            ("peak_hour", r"gio cao diem", "gio cao diem khung bat dau nao", "number"),
            ("customer_most_orders", r"khach.*don.*nhat", "khach hang mua co don hoa bill BAONHIEU nhat ai", "string"),
            ("top_spender", r"khach.*(BAONHIEU tien|chi tieu).*nhat", "khach hang mua chi tieu BAONHIEU tien nhat ai", "string"),
        ]:
            if not self.public_rules:
                break
            if re.search(pattern, rest) and _clean(rest, words):
                ranked = (name, result_kind)
        adjustment = None
        for name, pattern, words in [
            ("shipping", r"phi ship|phi giao hang|phi van chuyen", "phi ship giao hang van chuyen"),
            ("voucher", r"voucher", "giam voucher"),
            ("points", r"tien giam.*diem|giam nho dung diem", "giam nho dung diem"),
            ("trade_in", r"thu may cu", "thu may cu"),
        ]:
            if name == "trade_in" and "tien" not in rest:
                continue
            if name in self.adjustments and re.search(pattern, rest) and _clean(rest, "tong tien " + words):
                adjustment = name
        if ranked:
            name, kind = ranked
            fn = getattr(sem, name)
        elif adjustment:
            fn, args = sem.order_adjustment_total, self.adjustments[adjustment]
        elif self.members and re.search(r"thanh vien moi|khach (hang )?(dang ky|moi dang ky)", rest):
            if _clean(rest, "khach hang thanh vien moi dang ky them"):
                fn, args = sem.new_members, self.members
        elif self.cost and self.item and re.search(r"\bloi nhat\b|lai gop.*cao nhat", rest):
            if _clean(rest, "mat hang san pham nao loi nhat tong lai gop cao nhat ca thang"):
                fn, args, kind = sem.best_profit_item, (*self.cost, *self.item), "string"
        elif re.search(r"\b(lai gop|loi nhuan gop)\b", rest) and self.cost:
            if _clean(rest, "lai gop loi nhuan"):
                fn, args = sem.gross_profit, self.cost
        elif re.search(r"\b(lai rong|loi nhuan rong)\b", rest) and self.cost and self.expenses:
            if _clean(rest, "lai rong loi nhuan con"):
                fn, args = sem.net_profit, (*self.cost, *self.expenses)
        elif re.search(r"\bchi phi\b", rest) and self.expenses:
            # 'chi' là một phần của ý định chi phí, xử lý riêng ở đầu hàm.
            if _clean(rest, "chi phi"):
                fn, args = sem.expense_total, self.expenses
        elif self.staff and re.search(r"\bnhan vien\b", rest):
            if re.search(r"gioi nhat|DOANHTHU cao nhat", rest) and _clean(rest, "nhan vien ban gioi nhat DOANHTHU cao ma ten ai"):
                fn, args, kind = sem.staff_best, self.staff, "string"
            else:
                table, oid, col = self.staff
                labels = self.shop.tables[table][col].dropna().astype(str).unique()
                matches = [label for label in labels if f" {norm(label)} " in f" {rest} "]
                if len(matches) == 1:
                    label = matches[0]
                    remaining = f" {rest} ".replace(f" {norm(label)} ", " ", 1)
                    if re.search(r"DOANHTHU|tien|tien hang", remaining) and _clean(remaining, "nhan vien ma DOANHTHU ban tien hang don hoa cong lai"):
                        fn, args = sem.staff_revenue, (*self.staff, label)
        elif re.search(r"\btra hang\b", rest) and re.search(r"BAONHIEU|\bso don\b", rest) and _clean(rest, "don hoa bill bi tra hang"):
            fn = sem.returned_count
        elif re.search(r"BAONHIEU khach|\bso khach\b", rest) and _clean(rest, "khach hang da mua"):
            fn = sem.customer_count
        elif re.search(r"\btrung binh\b|\bbinh quan\b", rest):
            if re.search(r"\bngay\b", rest) and _clean(rest, "DOANHTHU trung binh binh quan moi ngay"):
                fn = sem.avg_daily_revenue if "DOANHTHU" in rest else None
            elif re.search(r"\bngay\b", rest) and _clean(rest, "don hoa bill trung binh binh quan moi ngay"):
                fn = sem.avg_daily_orders
        elif self.groups and "DOANHTHU" in rest:
            table, oid, col = self.groups
            labels = self.shop.tables[table][col].dropna().astype(str).unique()
            matches = [label for label in labels if f" {norm(label)} " in f" {rest} "]
            if len(matches) == 1:
                label = matches[0]
                remaining = f" {rest} ".replace(f" {norm(label)} ", " ", 1)
                if _clean(remaining, "DOANHTHU nhom khach hang"):
                    fn, args = sem.customer_group_revenue, (*self.groups, label)
        if fn is None:
            return None
        if not sem._allowed(fn.__name__):
            return None
        value = fn(period, *args)
        from datamate.answer_type import expected_type
        expected = expected_type(question)
        # Hai ý định đã nhận đủ có bẫy bỏ dấu: 'mới tháng' ≠ 'mỗi tháng',
        # 'thu máy' ≠ 'thứ mấy'. Kiểu từ hợp đồng SEM rõ hơn heuristic chung.
        explicit_public_metric = getattr(fn, "__name__", "") in {"new_members", "order_adjustment_total"}
        if expected and expected != kind and not explicit_public_metric:
            return None
        return None if value is None else {"type": kind, "answer": value}
