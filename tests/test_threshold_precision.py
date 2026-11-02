import datetime as dt
import pandas as pd
from datamate.semantic import OrderView


def test_thresholds_compare_exact_sums_without_rounding_to_integer():
    orders = pd.DataFrame({'order_id': ['a', 'b', 'c'],
        'date': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03']),
        'cls': ['completed', 'completed', 'cancelled']})
    lines = pd.DataFrame({'order_id': ['a', 'b', 'b', 'c'], 'amount': [9.6, 0.1, 9.9, 100]})
    view = OrderView(lines, orders)
    period = (dt.date(2026, 1, 1), dt.date(2026, 1, 3))
    assert view.order_count_min(period, period[1], 10) == 1
    assert view.days_revenue_min(period, period[1], 10) == 1


def test_missing_or_nonfinite_amounts_do_not_become_zero_or_a_count():
    orders = pd.DataFrame({'order_id': ['a'], 'date': pd.to_datetime(['2026-01-01']), 'cls': ['completed']})
    view = OrderView(pd.DataFrame({'order_id': ['a'], 'amount': [float('nan')]}), orders)
    period = (dt.date(2026, 1, 1), dt.date(2026, 1, 1))
    assert view.order_count_min(period, period[1], 10) is None
    assert view.days_revenue_min(period, period[1], 10) is None
    for missing in (None, pd.NA, float('inf'), 'invalid'):
        view.lines['amount'] = pd.Series([missing], dtype='object')
        assert view.order_count_min(period, period[1], 10) is None
        assert view.days_revenue_min(period, period[1], 10) is None


def test_stock_snapshot_keeps_integer_precision_after_loading():
    from datamate.loader import Shop
    from datamate.stock import StockView
    from datamate.baseline import stock_answer
    day = dt.date(2026, 1, 1)
    frame = pd.DataFrame({'Tên hàng': ['Hạt đậu'], 'Tồn': pd.Series([9007199254740993], dtype='Int64')})
    shop = Shop('synthetic', '**Tồn kho**: **Tồn** là tồn cuối ngày cuối có dữ liệu, theo đơn vị cơ bản',
                {'stock': frame}, {'stock': {'sources': ['stock.csv']}})
    stock = StockView.from_shop(shop, day, {'hat dau': 'hat'})
    assert stock is not None and stock.stock['hat dau'][1] == 9007199254740993
    assert stock_answer('Hôm nay Hạt đậu còn bao nhiêu hạt trong kho?', day, stock) == {
        'type': 'number', 'answer': 9007199254740993}


def test_integer_quantity_sum_is_not_converted_through_float():
    day = dt.date(2026, 1, 1)
    orders = pd.DataFrame({'order_id': ['a'], 'date': pd.to_datetime(['2026-01-01']), 'cls': ['completed']})
    items = pd.DataFrame({'order_id': ['a'], 'item': ['x'], 'name': ['X'],
                          'bunit': ['cai'], 'bqty': pd.Series([9007199254740993], dtype='Int64')})
    view = OrderView(pd.DataFrame({'order_id': ['a'], 'amount': [1]}), orders, items=items)
    assert view.qty_sold((day, day), day, 'cai', item='x') == 9007199254740993
    view.items = pd.DataFrame({'order_id': ['a', 'a'], 'item': ['x', 'x'], 'name': ['X', 'X'],
                               'bunit': ['cai', 'cai'], 'bqty': pd.Series([2**62, 2**62], dtype='Int64')})
    assert view.qty_sold((day, day), day, 'cai', item='x') == 2**63


def test_money_totals_and_average_do_not_overflow_or_ignore_missing():
    day = dt.date(2026, 1, 1)
    orders = pd.DataFrame({'order_id': ['a', 'b'], 'date': pd.to_datetime(['2026-01-01'] * 2),
                           'cls': ['completed', 'completed']})
    view = OrderView(pd.DataFrame({'order_id': ['a', 'b'], 'amount': pd.Series([2**62, 2**62], dtype='Int64')}), orders)
    assert view.revenue((day, day), day) == 2**63
    assert view.avg_order_value((day, day), day) == 2**62
    view.lines['amount'] = pd.Series([1, None], dtype='Int64')
    assert view.revenue((day, day), day) is None
    assert view.avg_order_value((day, day), day) is None
    view.lines['amount'] = pd.Series([0.1, 0.4])
    assert view.revenue((day, day), day) == 1
