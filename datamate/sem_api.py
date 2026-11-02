"""`SEM` trong môi trường chạy code của agent: các phép tính chuẩn theo rulebook đã kiểm chứng trên train
(semantic layer, sai 0), để agent gọi thẳng hoặc đối chiếu kết quả pandas tự viết (TroVE: thư viện hàm dùng lại).

Dựng lười (chỉ khi agent dùng tới). Mọi hàm trả None khi shop không dựng được bảng chuẩn hoặc không chắc —
khi đó agent tự tính bằng pandas.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

HELP = """SEM — phép tính chuẩn theo rulebook (None = không dựng được / không chắc → tự tính bằng pandas):
  SEM.answer() gọi lại bộ giải cho toàn bộ câu gốc; SEM.explain() xem trạng thái hỗ trợ (None không phải unanswerable)
  SEM.revenue(ky)            doanh thu (đơn hợp lệ §1)          SEM.order_count(ky)      số đơn hợp lệ
  SEM.cancelled_count(ky)    số đơn huỷ                         SEM.cancel_rate(ky)      tỉ lệ huỷ (%)
  SEM.avg_order_value(ky)    giá trị trung bình một đơn         SEM.best_revenue_day(ky) ngày doanh thu cao nhất
  SEM.customer_most_orders(ky) khách nhiều đơn nhất   SEM.loyal_customer(ky) khách trung thành (nhiều ngày mua nhất)
  SEM.big_spender(ky)        khách sộp (TB một đơn cao nhất, ≥3 đơn)   SEM.fickle_customer(ky) khách nhiều đơn huỷ nhất
  SEM.popular_item(ky)       mặt hàng có trong nhiều đơn nhất   SEM.busiest_day(ky)  ngày nhiều đơn nhất
  SEM.peak_hour(ky)          giờ cao điểm (giờ bắt đầu khung 1 giờ có nhiều đơn nhất)
  SEM.weekday_avg(ky, thu, chi_so)  TB mỗi ngày thứ X (thu 2..7, CN = 8; chi_so 'doanh_thu' / 'so_don')
  SEM.qty_sold(ky, don_vi, mat_hang=None, size=None)  số lượng bán quy về đơn vị cơ bản ('ly', 'kg'…; size 'm'/'l')
  SEM.top_items(ky, k)       k mặt hàng bán chạy nhất (theo số lượng đơn vị cơ bản)
  SEM.top_spender(ky)  khách mua nhiều tiền nhất (tổng tiền các đơn)
  SEM.item_revenue(ky, mat_hang)  doanh thu một mặt hàng   SEM.best_item(ky)  mặt hàng bán chạy nhất
  SEM.group_order_count(ky, nhom)  số đơn có mặt hàng thuộc nhóm   SEM.orders_min(ky, nguong)  số đơn có doanh thu ≥ ngưỡng
  SEM.days_revenue_min(ky, nguong)  số ngày doanh thu ≥ ngưỡng
  SEM.customers_min_orders(ky, n, ti_le=False)  số khách (hoặc % khách đã mua) có từ n đơn trở lên
  SEM.group_revenue(ky, nhom) doanh thu một nhóm hàng (đúng tên nhóm; None nếu không có nhóm đó)
  SEM.ingredient_revenue(ky, hoat_chat)  doanh thu thuốc có đúng hoạt chất đó
  SEM.rx_revenue(ky, ke_don=True)  doanh thu thuốc kê đơn (False = hàng không kê đơn)
  SEM.payment_revenue(ky, 'cod'/'cash'/'transfer'/'ewallet'/'debt')   SEM.channel_revenue(ky, 'store'/'phone'/'zalo'/'market')
  SEM.debt_total(ngay)       tổng khách còn nợ lúc hết ngày     SEM.debtor_count(ngay)   số khách còn nợ
  SEM.debt_added(ky)         nợ thêm trong kỳ                   SEM.debt_repaid(ky)      thu nợ trong kỳ
  SEM.debt_top(ngay, k)      [(tên, còn nợ)] k khách nợ nhiều nhất   SEM.debt_customer(ky, ten) [nợ thêm, trả nợ] của một khách
  SEM.debt_by_age(ngay, tu_ngay, den_ngay=None)  tổng còn nợ của các đơn có tuổi nợ trong khoảng (ngày)
  SEM.import_avg_price(ky, mat_hang, don_vi)  giá nhập bình quân một đơn vị nhập ('hop', 'thung'…)
  SEM.lot_nearest_expiry(mat_hang)  hạn dùng lô gần hết hạn nhất còn tồn   SEM.lots_expiring(den_ngay) lô còn tồn sắp hết hạn
  SEM.price_effect(mat_hang, ngay, n_ngay)  % đổi số lượng bán N ngày kể từ ngày đổi giá so với N ngày trước
  SEM.delivery_count(ky)     số đợt giao trong kỳ (mỗi đợt của mỗi đơn)
  SEM.shift_revenue(ky, ca)  doanh thu một ca ('sang'/'chieu'/'toi')   SEM.shift_order_count(ky, ca)  số đơn một ca
  SEM.sales_lines()          DataFrame dòng hàng chuẩn: order_id, date, cls (completed/cancelled/returned/delivering),
                             item, name, bqty (số lượng đơn vị cơ bản), bunit
  SEM.customer_count(ky)     số khách đã mua (không gồm khách lẻ, không phải thành viên đăng ký mới)
  SEM.customer_revenue(ky, khach) tổng doanh thu của khách (mã hoặc tên duy nhất)
  SEM.customer_order_count(ky, khach) số đơn của khách   SEM.customer_avg_order_value(ky, khach) TB tiền/đơn khách
  SEM.item_order_count(ky, mat_hang) số đơn chứa mặt hàng, mỗi đơn chỉ đếm một lần
  SEM.returned_count(ky)     số đơn trả hàng theo ngày đặt (không phải lượt trả trong kỳ)
  SEM.active_days(ky)        số ngày có đơn hợp lệ (không phải số ngày mở cửa)
  SEM.avg_daily_revenue(ky)  TB doanh thu/ngày lịch, kể cả ngày không bán
  SEM.avg_daily_orders(ky)   TB đơn/ngày lịch, kể cả ngày không bán
  SEM.payment_order_count(ky, hinh_thuc) số đơn theo hình thức thanh toán (nhãn như payment_revenue)
  SEM.channel_order_count(ky, kenh) số đơn theo kênh (nhãn như channel_revenue)
  SEM.new_members(ky, bang, cot_khach, cot_dang_ky) số khách đăng ký mới, kể cả chưa mua; bảng danh mục đầy đủ
  SEM.gross_profit(ky, cot_sl, cot_gia_von, cot_hang=None, mat_hang=None) lãi gộp từ giá vốn/đơn vị BÁN ghi trên dòng
  SEM.expense_total(ky, bang, cot_ngay, cot_tien) mọi chi phí, kể cả thiết bị; không xoá dòng trùng sổ chi phí
  SEM.net_profit(ky, cot_sl, cot_gia_von, bang_chi_phi, cot_ngay_chi, cot_tien_chi) lãi gộp trừ mọi chi phí kỳ
  SEM.staff_best(ky, bang, cot_don, cot_nhan_vien) mã nhân viên doanh thu cao nhất (không phải số đơn)
  SEM.staff_revenue(ky, bang, cot_don, cot_nhan_vien, nhan_vien) doanh thu của đúng mã/tên nhân viên ghi ở cột
  SEM.best_profit_item(ky, cot_sl, cot_gia_von, cot_ten_hang, cot_ma_hang) mã hàng có tổng lãi gộp cao nhất
  SEM.customer_group_revenue(ky, bang, cot_don, cot_nhom, nhom) doanh thu nhóm khách; bang phải chứa mỗi đơn
  SEM.main_supplier(ky, bang, cot_phieu, cot_ncc, cot_ngay, cot_trang_thai) mã NCC nhiều phiếu nhập nhất;
      cot_ncc là mã NCC (không có mã mới dùng tên), không phải nhà cung cấp nhiều tiền nhất
  SEM.import_total(ky, bang, cot_phieu, cot_ngay, cot_tong_phieu, cot_trang_thai) tiền nhập theo đầu phiếu, bỏ huỷ
  SEM.order_adjustment_total(ky, bang, cot_don, cot_tien) tổng số dương tiền voucher/điểm/ship/thu máy cũ ghi ở cấp đơn
  SEM.unsold_items(ky, cot_ten_hang, cot_ma_hang) mã hàng từng bán trước kỳ nhưng không bán trong kỳ, mã tăng dần
  SEM.unsold_count(ky, cot_ten_hang, cot_ma_hang) số mặt hàng như trên
ky = P.month(5, 2025), P.this_month()…; ngay = TODAY hoặc pd.Timestamp('2026-05-31')."""


def _d(x) -> dt.date:
    return pd.Timestamp(x).date()


def _period(ky) -> tuple[dt.date, dt.date]:
    return _d(ky[0]), _d(ky[1])


class SemAPI:
    def __init__(self, shop, as_of: dt.date, rulebook: str = "", *, question=None, on_answer=None):
        self._shop, self._as_of, self._rulebook = shop, as_of, rulebook
        self._view = None
        self._built = False
        self._question, self._on_answer = question, on_answer
        self._calls = []

    def answer(self, question=None):
        """Recall the full deterministic solver; None is not evidence of missing data.

        Only an exact original-question call can certify the result. A rephrasing
        can omit conditions, so its result remains an ordinary agent calculation.
        """
        from datamate.rule_recall import RuleRecall
        question = self._question if question is None else question
        if not isinstance(question, str) or not question.strip():
            return None
        recall = self._lazy('_recall', lambda: RuleRecall(self))
        result = recall.answer(question)
        self._calls.append({'method': 'answer', 'status': recall.diagnostics.get('last_status'),
                            'original_question': question == self._question})
        if result is not None and self._on_answer is not None and question == self._question:
            self._on_answer(result)
        return result

    def explain(self):
        """Capabilities and the last recall status, without inventing an answer."""
        from datamate.rule_recall import RuleRecall
        recall = self._lazy('_recall', lambda: RuleRecall(self))
        recall._prepare()
        return dict(recall.diagnostics)

    @property
    def view(self):
        if not self._built:
            self._built = True
            try:
                from datamate.semantic import OrderView
                self._view = OrderView.from_shop(self._shop)
                if self._view is not None and self._rulebook:
                    from datamate.semantic import parse_shifts
                    self._view.shifts = parse_shifts(self._rulebook)
            except Exception:
                self._view = None
        return self._view

    def help(self) -> None:
        print(HELP)

    def __repr__(self) -> str:
        return HELP

    def _item(self, mat_hang) -> str:
        """Khoá mặt hàng: mọi cách viết tên của cùng mã hàng về một khoá (OrderView.item_of)."""
        from datamate.semantic import item_key
        return self.view.item_of(mat_hang) if self.view is not None else item_key(mat_hang)

    def _call(self, name, ky, *args, **kw):
        permission = {'customers_with_orders': 'customers_min_orders', 'order_count_min': 'orders_min'}.get(name, name)
        if name == 'attr_revenue' and args:
            permission = 'payment_revenue' if args[0] == 'pay' else 'channel_revenue'
        if not self._allowed(permission):
            self._calls.append({'method': name, 'status': 'blocked_by_changed_rules'})
            return None
        v = self.view
        if v is None:
            self._calls.append({'method': name, 'status': 'missing_order_view'})
            return None
        p = _period(ky)
        if p[0] > p[1] or p[0] > self._as_of or not v.covers(p):
            self._calls.append({'method': name, 'status': 'invalid_or_uncovered_period'})
            return None
        value = getattr(v, name)(p, self._as_of, *args, **kw)
        self._calls.append({'method': name, 'status': 'ok' if value is not None else 'missing_binding_or_unsupported',
                            'period': [d.isoformat() for d in p]})
        return value

    def revenue(self, ky):
        return self._call("revenue", ky)

    def _common(self, public_name, method, ky, *args):
        """Các SEM bổ sung tự kiểm guard cả khi agent gọi trực tiếp."""
        from datamate.rule_guard import Guard
        guard = self._lazy("_common_guard", lambda: Guard(self._rulebook))
        if guard is None or guard.core or public_name in guard.plan_blocked:
            return None
        p = _period(ky)
        if p[0] > p[1] or p[0] > self._as_of:
            return None
        return self._call(method, ky, *args)

    def returned_count(self, ky):
        return self._common("returned_count", "returned_count", ky)

    def item_order_count(self, ky, mat_hang: str):
        from datamate.semantic import item_key
        return self._common("item_order_count", "item_order_count", ky, self._item(mat_hang))

    def customer_count(self, ky):
        result = self._common("customer_count", "customers_with_orders", ky, 1)
        return None if result is None else result[1]

    def customer_revenue(self, ky, khach: str):
        return self._common("customer_revenue", "customer_metric", ky, khach, "revenue")

    def customer_order_count(self, ky, khach: str):
        return self._common("customer_order_count", "customer_metric", ky, khach, "orders")

    def customer_avg_order_value(self, ky, khach: str):
        return self._common("customer_avg_order_value", "customer_metric", ky, khach, "average")

    def active_days(self, ky):
        return self._common("active_days", "active_days", ky)

    def avg_daily_revenue(self, ky):
        return self._common("avg_daily_revenue", "daily_average", ky, "revenue")

    def avg_daily_orders(self, ky):
        return self._common("avg_daily_orders", "daily_average", ky, "orders")

    def payment_order_count(self, ky, hinh_thuc: str):
        if hinh_thuc not in {"cod", "cash", "transfer", "ewallet", "debt"}:
            return None
        return self._common("payment_order_count", "attr_order_count", ky, "pay", hinh_thuc)

    def channel_order_count(self, ky, kenh: str):
        if kenh not in {"store", "phone", "zalo", "market"}:
            return None
        return self._common("channel_order_count", "attr_order_count", ky, "channel", kenh)

    def _registry(self, name, ky, *columns):
        from datamate import registry_sem
        from datamate.rule_guard import Guard
        guard = self._lazy("_common_guard", lambda: Guard(self._rulebook))
        if guard is None or guard.core or name in guard.plan_blocked:
            return None
        try:
            return getattr(registry_sem, name)(self._shop, _period(ky), self._as_of, *columns)
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    def new_members(self, ky, bang, cot_khach, cot_dang_ky):
        return self._registry("new_members", ky, bang, cot_khach, cot_dang_ky)

    def main_supplier(self, ky, bang, cot_phieu, cot_ncc, cot_ngay, cot_trang_thai):
        return self._registry("main_supplier", ky, bang, cot_phieu, cot_ncc, cot_ngay, cot_trang_thai)

    def import_total(self, ky, bang, cot_phieu, cot_ngay, cot_tong_phieu, cot_trang_thai):
        return self._registry("import_total", ky, bang, cot_phieu, cot_ngay, cot_tong_phieu, cot_trang_thai)

    def _receipts(self):
        from datamate.receipt_sem import Receipts
        return self._lazy("_receipt_rows", lambda: Receipts.from_shop(self._shop))

    def import_spend(self, ky):
        """Tiền nhập hàng trong kỳ theo sổ nhập shop tự nhận diện (bỏ phiếu huỷ); không cần chỉ cột."""
        r = self._receipts()
        if r is None or not self._allowed("import_total"):
            return None
        return r.value(_period(ky), self._as_of)

    def top_supplier(self, ky, theo_so_phieu=False):
        """Nhà cung cấp nhập nhiều tiền nhất (theo_so_phieu: nhiều phiếu nhất) trong kỳ; bằng nhau lấy tên nhỏ nhất."""
        r = self._receipts()
        if r is None or not self._allowed("main_supplier"):
            return None
        return r.value(_period(ky), self._as_of, best=True, most_receipts=bool(theo_so_phieu))

    def order_adjustment_total(self, ky, bang, cot_don, cot_tien):
        from datamate.helpers import round_money
        import math
        v, p = self.view, _period(ky)
        if not self._allowed("order_adjustment_total") or v is None or not v.covers(p) or p[0] > min(p[1], self._as_of):
            return None
        frame = self._shop.tables.get(bang)
        if frame is None or cot_don not in frame or cot_tien not in frame or cot_don == cot_tien:
            return None
        frame = frame.copy()
        meta = self._shop.meta.get(bang, {})
        if meta.get("continuation_rows"):
            cc = [c for c in meta.get("continuation_cols", []) if c in frame]
            frame[cc] = frame[cc].ffill()
        data = pd.DataFrame({"id": frame[cot_don].astype(str), "amount": pd.to_numeric(frame[cot_tien], errors="coerce")})
        ids = set(v._valid(v._orders_in(p, self._as_of)).order_id)
        data = data[data.id.isin(ids)]
        if ids - set(data.id) or data.amount.isna().any() or not data.amount.map(math.isfinite).all():
            return None
        if data.groupby("id").amount.nunique(dropna=False).gt(1).any():
            return None
        return round_money(data.drop_duplicates("id").amount.abs().sum())

    def unsold_items(self, ky, cot_ten_hang, cot_ma_hang):
        from datamate.semantic import item_key
        v, p = self.view, _period(ky)
        if not self._allowed("unsold_items") or v is None or v.items is None or not v.covers(p) or p[0] > min(p[1], self._as_of):
            return None
        frame = self._shop.tables.get((v.source or {}).get("table"))
        if frame is None or cot_ten_hang not in frame or cot_ma_hang not in frame:
            return None
        cat = pd.DataFrame({"item": frame[cot_ten_hang].map(item_key), "code": frame[cot_ma_hang]})
        if cat.code.isna().any() or cat.groupby("item").code.nunique().gt(1).any():
            return None
        before = set(v._valid(v.orders[v.orders.date < pd.Timestamp(p[0])]).order_id)
        during = set(v._valid(v._orders_in(p, self._as_of)).order_id)
        keys = set(v.items.loc[v.items.order_id.isin(before), "item"]) - set(v.items.loc[v.items.order_id.isin(during), "item"])
        keys -= {"phi phuc vu"}
        mapping = cat.drop_duplicates("item").set_index("item").code
        if keys - set(mapping.index):
            return None
        return sorted(str(mapping[k]) for k in keys)

    def unsold_count(self, ky, cot_ten_hang, cot_ma_hang):
        """Số mặt hàng ế trong kỳ (unsold_items)."""
        codes = self.unsold_items(ky, cot_ten_hang, cot_ma_hang)
        return None if codes is None else len(codes)

    def _allowed(self, name):
        from datamate.rule_guard import Guard
        guard = self._lazy("_common_guard", lambda: Guard(self._rulebook))
        return guard is not None and not guard.core and name not in guard.plan_blocked

    def gross_profit(self, ky, cot_sl, cot_gia_von, cot_hang=None, mat_hang=None):
        from datamate.finance_sem import profit
        v, p = self.view, _period(ky)
        if not self._allowed("gross_profit") or v is None or not v.covers(p) or p[0] > min(p[1], self._as_of):
            return None
        return profit(self._shop, v, p, self._as_of, cot_sl, cot_gia_von, cot_hang, mat_hang)

    def expense_total(self, ky, bang, cot_ngay, cot_tien):
        from datamate.finance_sem import expenses
        p = _period(ky)
        if not self._allowed("expense_total") or p[0] > min(p[1], self._as_of):
            return None
        v = self.view
        if v is not None and not v.covers(p):
            return None
        if v is None:
            frame = self._shop.tables.get(bang)
            if frame is None or cot_ngay not in frame or frame.empty:
                return None
            dates = frame[cot_ngay]
            if not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any():
                return None
            if p[1] < dates.min().date() or p[0] > dates.max().date():
                return None
        return expenses(self._shop, p, self._as_of, bang, cot_ngay, cot_tien)

    def net_profit(self, ky, cot_sl, cot_gia_von, bang_chi_phi, cot_ngay_chi, cot_tien_chi):
        if not self._allowed("net_profit"):
            return None
        gross = self.gross_profit(ky, cot_sl, cot_gia_von)
        cost = self.expense_total(ky, bang_chi_phi, cot_ngay_chi, cot_tien_chi)
        return None if gross is None or cost is None else gross - cost

    def _dimension(self, name, ky, bang, cot_don, cot_nhom, label=None, best=False):
        from datamate.finance_sem import order_dimension
        v, p = self.view, _period(ky)
        if not self._allowed(name) or v is None or not v.covers(p) or p[0] > min(p[1], self._as_of):
            return None
        return order_dimension(self._shop, v, p, self._as_of, bang, cot_don, cot_nhom, label, best)

    def staff_best(self, ky, bang, cot_don, cot_nhan_vien):
        """Cột nhân viên là mã; không có mã mới dùng tên. Hoà lấy mã tăng dần."""
        return self._dimension("staff_best", ky, bang, cot_don, cot_nhan_vien, best=True)

    def staff_revenue(self, ky, bang, cot_don, cot_nhan_vien, nhan_vien):
        return self._dimension("staff_revenue", ky, bang, cot_don, cot_nhan_vien, label=nhan_vien)

    def best_profit_item(self, ky, cot_sl, cot_gia_von, cot_ten_hang, cot_ma_hang):
        from datamate.finance_sem import best_profit_item
        v, p = self.view, _period(ky)
        if not self._allowed("best_profit_item") or v is None or not v.covers(p) or p[0] > min(p[1], self._as_of):
            return None
        return best_profit_item(self._shop, v, p, self._as_of, cot_sl, cot_gia_von, cot_ten_hang, cot_ma_hang)

    def customer_group_revenue(self, ky, bang, cot_don, cot_nhom, nhom):
        """Khách lẻ loại khỏi nhóm, nhưng vẫn có trong SEM.revenue để tính tỉ trọng."""
        v = self.view
        if v is None or v.customers is None:
            return None
        # Bảng nhóm khách phải chỉ tới đúng các đơn và có thể ghi nhóm 'lẻ'.
        # Chỉ giữ nhóm cho đơn có khoá khách; khách vãng lai luôn bị loại.
        from datamate.finance_sem import order_dimension
        from datamate.loader import Shop
        frame = self._shop.tables.get(bang)
        if frame is None or cot_don not in frame or cot_nhom not in frame or not self._allowed("customer_group_revenue"):
            return None
        copy = frame.copy()
        copy[cot_nhom] = copy[cot_nhom].where(copy[cot_don].astype(str).isin(v.customers.order_id))
        p = _period(ky)
        if not v.covers(p) or p[0] > min(p[1], self._as_of):
            return None
        return order_dimension(Shop("bound", "", {bang: copy}), v, p, self._as_of,
                               bang, cot_don, cot_nhom, nhom)

    def order_count(self, ky):
        return self._call("order_count", ky)

    def cancelled_count(self, ky):
        return self._call("cancelled_count", ky)

    def cancel_rate(self, ky):
        return self._call("cancel_rate", ky)

    def avg_order_value(self, ky):
        return self._call("avg_order_value", ky)

    def best_revenue_day(self, ky):
        return self._call("best_revenue_day", ky)

    def customer_most_orders(self, ky):
        return self._call("customer_most_orders", ky)

    def loyal_customer(self, ky):
        return self._call("loyal_customer", ky)

    def top_spender(self, ky):
        return self._call("top_spender", ky)

    def top_customers(self, ky, k: int = 1):
        """Top-k khách theo doanh thu, hoà theo mã; kết quả luôn là list."""
        return self._call("top_customers", ky, k)

    def big_spender(self, ky):
        return self._call("big_spender", ky)

    def fickle_customer(self, ky):
        return self._call("fickle_customer", ky)

    def popular_item(self, ky):
        return self._call("popular_item", ky)

    def busiest_day(self, ky):
        return self._call("busiest_day", ky)

    def peak_hour(self, ky):
        return self._call("peak_hour", ky)

    def weekday_avg(self, ky, thu: int, chi_so: str = "doanh_thu"):
        """thu: 2..7 = thứ Hai..thứ Bảy, 8 = Chủ nhật; chi_so 'doanh_thu' / 'so_don'."""
        if thu not in range(2, 9) or chi_so not in ("doanh_thu", "so_don"):
            return None
        return self._call("weekday_avg", ky, thu - 2, "revenue" if chi_so == "doanh_thu" else "orders")

    def qty_sold(self, ky, don_vi: str, mat_hang: str | None = None, size: str | None = None):
        from datamate.semantic import item_key
        return self._call("qty_sold", ky, item_key(don_vi), item=self._item(mat_hang) if mat_hang else None,
                          size=size.lower() if size else None)

    def top_items(self, ky, k: int = 1):
        return self._call("top_items", ky, k)

    def item_revenue(self, ky, mat_hang: str):
        from datamate.semantic import item_key
        return self._call("item_revenue", ky, self._item(mat_hang))

    def unit_price(self, ngay, mat_hang: str):
        """Giá bán một đơn vị cơ bản của mặt hàng ngày D (rulebook 'Giá bán X ngày D'): đơn giá trên dòng bán của các
        đơn đặt ngày D, trước chiết khấu dòng, tính cả đơn huỷ / trả; trong ngày có nhiều giá → None."""
        from datamate.helpers import round_money
        from datamate.rule_guard import Guard
        from datamate.semantic import item_key
        v = self.view
        guard = self._lazy("_common_guard", lambda: Guard(self._rulebook))
        if v is None or v.items is None or "base_price" not in v.items or guard is None or guard.core \
                or "Giá bán X ngày D" in guard.changed:
            return None
        day = ngay if isinstance(ngay, dt.date) else dt.date.fromisoformat(str(ngay))
        if day > self._as_of or not v.covers((day, day)):
            return None
        it = v.items[v.items["item"] == self._item(mat_hang)]
        ids = set(v._orders_in((day, day), self._as_of)["order_id"])
        prices = it.loc[it["order_id"].isin(ids), "base_price"]
        if prices.empty or prices.isna().any() or prices.max() - prices.min() > 1e-6:
            return None
        return round_money(prices.iloc[0])

    def group_order_count(self, ky, nhom: str):
        from datamate.semantic import item_key
        return self._call("group_order_count", ky, item_key(nhom))

    def days_revenue_min(self, ky, nguong: float):
        return self._call("days_revenue_min", ky, float(nguong))

    def orders_min(self, ky, nguong: float):
        return self._call("order_count_min", ky, float(nguong))

    def customers_min_orders(self, ky, n: int, ti_le: bool = False):
        """Số khách có từ n đơn hợp lệ trở lên trong kỳ; ti_le=True → % trên số khách đã mua trong kỳ."""
        r = self._call("customers_with_orders", ky, int(n))
        if r is None:
            return None
        k, total = r
        from datamate.helpers import share
        return share(k, total) if ti_le else k

    def best_item(self, ky):
        r = self.top_items(ky, 1)
        return r[0] if r else None

    def group_revenue(self, ky, nhom: str):
        from datamate.semantic import item_key
        v = self.view
        if v is None or v.groups is None or item_key(nhom) not in v.group_names():
            return None
        return self._call("group_revenue", ky, item_key(nhom))

    def _debt(self):
        from datamate.rule_guard import Guard
        guard = self._lazy('_common_guard', lambda: Guard(self._rulebook))
        if guard is None or guard.core or guard.debt:
            return None
        v = self.view
        d = getattr(v, "debt", None) if v is not None else None
        if d is None and not hasattr(self, "_debt_alone"):  # bảng công nợ dựng được cả khi bảng đơn chuẩn không
            try:
                from datamate.debt import DebtView
                self._debt_alone = DebtView.from_shop(self._shop)
            except Exception:
                self._debt_alone = None
        return d if d is not None else getattr(self, "_debt_alone", None)

    def debt_total(self, ngay):
        d = self._debt()
        return None if d is None else d.total_owed(min(_d(ngay), self._as_of))

    def debtor_count(self, ngay):
        if not self._allowed('customer_count'):
            return None
        d = self._debt()
        return None if d is None else d.debtor_count(min(_d(ngay), self._as_of))

    def debt_added(self, ky):
        d = self._debt()
        return None if d is None else d.added(_period(ky), self._as_of)

    def debt_repaid(self, ky):
        d = self._debt()
        return None if d is None else d.repaid(_period(ky), self._as_of)

    def sales_lines(self) -> pd.DataFrame | None:
        v = self.view
        if v is None or v.items is None:
            return None
        return v.items.merge(v.orders[["order_id", "date", "cls"]], on="order_id", how="left")

    def payment_revenue(self, ky, hinh_thuc: str):
        """hinh_thuc: 'cod' / 'cash' / 'transfer' / 'ewallet' / 'debt' (theo hình thức ghi trên đơn)."""
        return self._attr(ky, "pay", hinh_thuc)

    def channel_revenue(self, ky, kenh: str):
        """kenh: 'store' (tại quầy) / 'phone' / 'zalo' / 'market' (sàn TMĐT)."""
        return self._attr(ky, "channel", kenh)

    def _attr(self, ky, kind, label):
        v = self.view
        if v is None or label not in v.attr_labels().get(kind, set()):
            return None
        return self._call("attr_revenue", ky, kind, label)

    def ingredient_revenue(self, ky, hoat_chat: str):
        """Doanh thu các thuốc có hoạt chất ghi đúng như vậy trong danh mục (phối hợp là loại riêng)."""
        from datamate.semantic import item_key
        v = self.view
        if v is None or v.ingredients is None or item_key(hoat_chat) not in v.ingredient_names():
            return None
        return self._call("ingredient_revenue", ky, item_key(hoat_chat))

    # --- các bộ giải tất định khác, gọi theo tên agent đã tìm bằng find_values --------------------------------
    def _lazy(self, attr, build):
        if not hasattr(self, attr):
            try:
                setattr(self, attr, build())
            except Exception:
                setattr(self, attr, None)
        return getattr(self, attr)

    def import_avg_price(self, ky, mat_hang: str, don_vi: str):
        """Giá nhập bình quân một `don_vi` của mặt hàng trong kỳ (gia quyền theo SL nhập, bỏ phiếu huỷ)."""
        from datamate.imports import ImportView
        from datamate.parsing import fold
        from datamate.semantic import item_key
        if not self._allowed('import_avg_price'):
            return None
        p = _period(ky)
        p = (p[0], min(p[1], self._as_of))
        if p[0] > p[1]:
            return None
        v = self.view
        iv = getattr(v, "imports", None) if v is not None else None
        iv = iv or self._lazy("_imports", lambda: ImportView.from_shop(self._shop))
        if iv is None:
            return None
        return iv.avg_price(item_key(mat_hang), " ".join(fold(don_vi).split()), p)

    def _lots(self):
        from datamate.rule_guard import Guard
        guard = self._lazy('_common_guard', lambda: Guard(self._rulebook))
        if guard is None or guard.core or guard.lot:
            return None
        from datamate.lots import LotView
        return self._lazy("_lotview", lambda: LotView.from_shop(self._shop))

    def lot_nearest_expiry(self, mat_hang: str):
        """Hạn dùng của lô gần hết hạn nhất còn tồn (hôm nay) của mặt hàng; None nếu không có bảng lô."""
        from datamate.baseline import norm
        lv = self._lots()
        if lv is None:
            return None
        keys = {norm(str(n)): k for k, n in lv.lots.drop_duplicates("item")[["item", "name"]].itertuples(index=False)}
        k = keys.get(norm(mat_hang))
        d = None if k is None else lv.nearest_expiry(k, self._as_of)
        return None if d is None else d.isoformat()

    def lots_expiring(self, den_ngay):
        """DataFrame các lô còn tồn (hôm nay) có hạn dùng từ hôm nay đến hết `den_ngay` (cột item, name, expiry, left)."""
        lv = self._lots()
        return None if lv is None else lv.expiring(self._as_of, _d(den_ngay))

    def debt_top(self, ngay, k: int):
        """[(tên khách, còn nợ)] k khách nợ nhiều nhất lúc hết ngày (bằng nhau: mã tăng dần)."""
        d = self._debt() if self._allowed('top_spender') else None
        return None if d is None else d.top_debtors(min(_d(ngay), self._as_of), k)

    def debt_by_age(self, ngay, tu_ngay: int, den_ngay: int | None = None):
        """Tổng còn nợ lúc hết ngày của các đơn có tuổi nợ từ `tu_ngay` đến `den_ngay` ngày (None = không chặn)."""
        d = self._debt()
        return None if d is None else d.owed_by_age(min(_d(ngay), self._as_of), tu_ngay, den_ngay)

    def debt_customer(self, ky, ten: str):
        """[nợ thêm, trả nợ] trong kỳ của một khách (tên như trong dữ liệu); None nếu tên không xác định một khách."""
        from datamate.parsing import fold
        d = self._debt() if self._allowed('customer_revenue') else None
        if d is None:
            return None
        cust = d.customer_keys().get(" ".join(fold(ten).split()))
        if cust is None:
            return None
        p = _period(ky)
        return [d.added_by(p, self._as_of, cust), d.repaid(p, self._as_of, cust)]

    def price_effect(self, mat_hang: str, ngay, n_ngay: int):
        """% thay đổi số lượng bán (đơn vị cơ bản, bỏ đơn huỷ / trả) N ngày kể từ `ngay` (tính cả ngày đó) so với
        N ngày ngay trước — câu 'từ ngày D tăng / đổi giá, so N ngày kể từ D với N ngày trước đó'."""
        from datamate.helpers import pct_change
        from datamate.semantic import item_key
        if not self._allowed('price_effect') or not self._allowed('qty_sold'):
            return None
        v = self.view
        if v is None or v.items is None:
            return None
        item = self._item(mat_hang)
        units = set(v.items.loc[v.items["item"] == item, "bunit"].dropna())
        if len(units) != 1:
            return None
        day, n = _d(ngay), int(n_ngay)
        after = (day, day + dt.timedelta(days=n - 1))
        before = (day - dt.timedelta(days=n), day - dt.timedelta(days=1))
        if after[1] > self._as_of or not (v.covers(after) and v.covers(before)):
            return None
        unit = units.pop()
        return pct_change(v.qty_sold(after, self._as_of, unit, item=item), v.qty_sold(before, self._as_of, unit, item=item))

    def delivery_count(self, ky):
        """Số đợt giao trong kỳ (mỗi đợt của mỗi đơn một đợt, theo ngày giao); None nếu không có sổ giao nhiều đợt."""
        if not self._allowed('delivery_count'):
            return None
        from datamate.delivery import DeliveryView
        dl = self._lazy("_deliveries", lambda: DeliveryView.from_shop(self._shop))
        return None if dl is None or not dl.covers(_period(ky)) else dl.count(_period(ky), self._as_of)

    def shift_revenue(self, ky, ca: str):
        """Doanh thu một ca theo giờ tạo đơn, khung giờ ca theo rulebook ('sang' / 'chieu' / 'toi')."""
        from datamate.parsing import fold
        v = self.view
        if v is None or fold(ca) not in (getattr(v, "shifts", None) or {}):
            return None
        return self._call("shift_revenue", ky, fold(ca))

    def shift_order_count(self, ky, ca: str):
        from datamate.parsing import fold
        v = self.view
        if v is None or fold(ca) not in (getattr(v, "shifts", None) or {}):
            return None
        return self._call("shift_order_count", ky, fold(ca))

    def rx_revenue(self, ky, ke_don: bool = True):
        """Doanh thu thuốc kê đơn (ke_don=True) hoặc mọi hàng còn lại (False), theo cờ kê đơn trong danh mục thuốc."""
        v = self.view
        if v is None or getattr(v, "rx", None) is None:
            return None
        return self._call("rx_revenue", ky, bool(ke_don))
