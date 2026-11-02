import datetime as dt

import pandas as pd

from datamate.baseline import debt_any
from datamate.debt import DebtView
from datamate.helpers import Periods

D = dt.date
T = pd.Timestamp


def view():
    orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "date": [T("2026-01-02"), T("2026-01-10"), T("2026-01-20"), T("2026-01-25")],
        "cust": ["kh02", "kh01", "kh01", None],
        "name": ["Lê Văn B", "Trần Thị A", "Trần Thị A", None],
        "rank_key": ["KH02", "KH01", "KH01", ""],
        "total": [300_000, 200_000, 100_000, 50_000],
        "debt": [True, True, True, True], "counted": [True, True, True, True]})
    # O1 trả 100k cùng ngày mua (không phải trả nợ), O2 trả 0, O3 trả sau 50k
    pays = pd.DataFrame({"order_id": ["O1", "O3"], "date": [T("2026-01-02"), T("2026-01-28")],
                         "amount": [100_000, 50_000]})
    return DebtView(orders, pays)


def test_top_debtors_ties_by_code():
    dv = view()
    # lúc hết 31/01: KH02 còn 200k, KH01 còn 200k + 50k = 250k; khách lẻ không xếp hạng
    assert dv.top_debtors(D(2026, 1, 31), 2) == [("Trần Thị A", 250_000), ("Lê Văn B", 200_000)]
    # 15/01: hai khách cùng nợ 200k → mã tăng dần (KH01 trước KH02)
    assert dv.top_debtors(D(2026, 1, 15), 2) == [("Trần Thị A", 200_000), ("Lê Văn B", 200_000)]


def test_conflicting_order_headers_do_not_produce_a_guessed_debt():
    from datamate.loader import Shop
    orders = pd.DataFrame({'Mã đơn': ['a', 'a', 'b'], 'Ngày': pd.to_datetime(['2026-01-01'] * 3),
        'Mã khách': ['K1', 'K1', 'K2'], 'Tổng': [100, 100, 50],
        'Trạng thái': ['Hoàn thành', 'Hoàn thành', 'Huỷ'], 'Thanh toán': ['Ghi nợ', 'Ghi nợ', 'COD']})
    payments = pd.DataFrame({'Mã đơn': ['  a  '], 'Ngày thu': pd.to_datetime(['2026-01-02']), 'Thu': [40]})
    readme = '**Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **Thu** là số tiền trả lần đó; **Tổng** là số khách phải trả cho cả đơn.'
    shop = Shop('synthetic', readme, {'orders': orders, 'payments': payments}, {})
    ledger = DebtView.from_shop(shop)
    assert ledger is not None and ledger.total_owed(D(2026, 1, 2)) == 60
    orders.loc[1, 'Tổng'] = 120
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Tổng'] = None
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Tổng'] = 100
    payments.loc[0, 'Mã đơn'] = 'missing'
    assert DebtView.from_shop(shop) is None
    payments.loc[0, 'Mã đơn'] = 'a'
    payments.loc[0, 'Thu'] = None
    assert DebtView.from_shop(shop) is None
    payments.loc[0, 'Thu'] = float('inf')
    assert DebtView.from_shop(shop) is None
    payments.loc[0, 'Thu'] = 40
    orders.loc[1, 'Tổng'] = float('inf')
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Tổng'] = 100
    orders.loc[1, 'Mã đơn'] = '   '
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Mã đơn'] = 'a'
    orders.loc[1, 'Trạng thái'] = None
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Trạng thái'] = 'không rõ'
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Trạng thái'] = 'Hoàn thành'
    orders.loc[1, 'Thanh toán'] = None
    assert DebtView.from_shop(shop) is None
    orders.loc[1, 'Thanh toán'] = 'Ghi nợ'
    orders.loc[:1, 'Trạng thái'] = 'Đã lấy'
    orders.loc[2, 'Trạng thái'] = 'Không giao được'
    assert DebtView.from_shop(shop) is not None
    orders.loc[2, 'Trạng thái'] = 'Đổi trả'
    assert DebtView.from_shop(shop) is not None


def test_readme_names_rebind_to_renamed_columns():
    """README giữ tên cũ ('Giá trị', 'Tổng sau giảm giá', 'Mã KH') còn cột đã đổi cách viết / dịch."""
    from datamate.loader import Shop
    orders = pd.DataFrame({'id_order': ['a', 'b'], 'time': pd.to_datetime(['2026-01-01'] * 2),
        'id_kh': ['K1', 'K2'], 'total_sau_giam_price': [100, 50],
        'status': ['Hoàn thành', 'Huỷ'], 'phuong_thuc_payment': ['Ghi nợ', 'COD']})
    payments = pd.DataFrame({'id_order': ['a'], 'time': pd.to_datetime(['2026-01-02']), 'price_tri': [40]})
    readme = ('**Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **Giá trị** là số tiền trả lần đó; '
              '**Tổng sau giảm giá** là số khách phải trả cho cả đơn.')
    ledger = DebtView.from_shop(Shop('synthetic', readme, {'orders': orders, 'payments': payments}, {}))
    assert ledger is not None and ledger.total_owed(D(2026, 1, 2)) == 60
    # hai cột cùng tương đương → không đoán
    payments['PriceTri'] = payments['price_tri']
    assert DebtView.from_shop(Shop('synthetic', readme, {'orders': orders, 'payments': payments}, {})) is None


def test_explicit_note_status_allows_blank_completed_orders():
    from datamate.loader import Shop
    orders = pd.DataFrame({'Mã đơn': ['a', 'b'], 'Ngày': pd.to_datetime(['2026-01-01'] * 2),
        'Mã khách': ['K1', 'K2'], 'Tổng': [100, 50],
        'DienGiai': [None, 'Huỷ'], 'Thanh toán': ['Ghi nợ', 'COD']})
    payments = pd.DataFrame({'Mã đơn': ['a'], 'Ngày thu': pd.to_datetime(['2026-01-02']), 'Thu': [40]})
    readme = '**Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **Thu** là số tiền trả lần đó; **Tổng** là số khách phải trả cho cả đơn.'
    ledger = DebtView.from_shop(Shop('synthetic', readme, {'orders': orders, 'payments': payments}, {}))
    assert ledger is not None and ledger.total_owed(D(2026, 1, 2)) == 60


def test_status_like_values_in_an_unrelated_column_are_not_evidence():
    from datamate.loader import Shop
    orders = pd.DataFrame({'Mã đơn': ['a', 'b'], 'Ngày': pd.to_datetime(['2026-01-01'] * 2),
        'Mã khách': ['K1', 'K2'], 'Tổng': [100, 50],
        'Loại hàng': ['Hoàn thành', 'Huỷ'], 'Thanh toán': ['Ghi nợ', 'COD']})
    payments = pd.DataFrame({'Mã đơn': ['a'], 'Ngày thu': pd.to_datetime(['2026-01-02']), 'Thu': [40]})
    readme = '**Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **Thu** là số tiền trả lần đó; **Tổng** là số khách phải trả cho cả đơn.'
    assert DebtView.from_shop(Shop('synthetic', readme, {'orders': orders, 'payments': payments}, {})) is None
    documented = readme + '\n## Cột trong `orders.csv`\n- **Loại hàng**: trạng thái đơn\n'
    meta = {'orders': {'sources': ['orders.csv']}, 'payments': {'sources': ['payments.csv']}}
    ledger = DebtView.from_shop(Shop('synthetic', documented, {'orders': orders, 'payments': payments}, meta))
    assert ledger is not None and ledger.total_owed(D(2026, 1, 2)) == 60


def test_relative_snapshot_and_outstanding_debt_wording():
    dv = view()
    day = D(2026, 1, 25)
    p = Periods(day)
    assert debt_any('Hôm nay có bao nhiêu khách chưa trả hết nợ?', day, dv, p) == {'type': 'number', 'answer': 2}
    assert debt_any('Tính đến hết hôm nay khách còn thiếu tiền tổng cộng bao nhiêu?', day, dv, p) == {'type': 'number', 'answer': 550000}
    assert debt_any('Hôm nay khách Lan còn thiếu tiền bao nhiêu?', day, dv, p) is None
    assert debt_any('Hôm nay có bao nhiêu khách đã trả hết nợ?', day, dv, p) is None


def test_age_buckets_and_customer_flow():
    dv = view()
    day = D(2026, 1, 31)  # tuổi: O1 29, O2 21, O3 11, O4 6
    assert dv.owed_by_age(day, 0, 7) == 50_000
    assert dv.owed_by_age(day, 8, 15) == 50_000
    assert dv.owed_by_age(day, 16, None) == 400_000
    jan = (D(2026, 1, 1), D(2026, 1, 31))
    assert dv.added_by(jan, day, "kh02") == 200_000 and dv.repaid(jan, day, "kh01") == 50_000


def test_debt_questions():
    dv = view()
    as_of = D(2026, 2, 5)
    P = Periods(as_of)
    q = "Tính đến hết ngày 31/01/2026, 2 khách nào đang nợ nhiều nhất, mỗi người còn nợ bao nhiêu?"
    assert debt_any(q, as_of, dv, P) == {"type": "table", "answer": {
        "columns": ["khach_hang", "so_tien_no"], "rows": [["Trần Thị A", 250_000], ["Lê Văn B", 200_000]]}}
    q = "Tháng 1/2026 khách Lê Văn B ghi nợ thêm bao nhiêu và trả nợ được bao nhiêu? Ghi hai số theo thứ tự."
    assert debt_any(q, as_of, dv, P) == {"type": "list", "answer": [200_000, 0]}
    q = "Tính đến hết ngày 31/01/2026, tổng số tiền khách còn nợ ở các đơn đã đặt quá 15 ngày là bao nhiêu?"
    assert debt_any(q, as_of, dv, P) == {"type": "number", "answer": 400_000}
    q = "Trong tháng 1/2026, tiền khách trả bớt nợ các đơn mua chịu được bao nhiêu?"
    assert debt_any(q, as_of, dv, P) == {"type": "number", "answer": 50_000}


def test_delivery_batches_counted_once_per_order_batch():
    from datamate.baseline import delivery_answer
    from datamate.delivery import DeliveryView
    from datamate.loader import Shop
    giao = pd.DataFrame({"Mã đơn hàng": ["D1", "D1", "D1", "D2"], "Lần giao": [1, 1, 2, 1],
                         "Ngày giao hàng": pd.to_datetime(["2026-03-02", "2026-03-02", "2026-04-01", "2026-03-05"]),
                         "Tên hàng hóa": ["Cát", "Gạch", "Cát", "Xi măng"]})
    dl = DeliveryView.from_shop(Shop("b", "", {"giao": giao}, {}))
    as_of = D(2026, 4, 10)
    assert delivery_answer("Tháng 3/2026 xe đã giao tổng cộng mấy đợt hàng?", as_of, dl, Periods(as_of)) == \
        {"type": "number", "answer": 2}
