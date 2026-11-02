import datetime as dt

import pandas as pd
import pytest

from datamate.baseline import answer, prepare_sem
from datamate.business_query import canonical, readable
from datamate.helpers import Periods
from datamate.loader import Shop
from datamate.price_history import PriceHistory
from datamate.semantic import OrderView, _items


@pytest.mark.parametrize('source,target', [('T9/26', 'thang 9/26'), ('Q2/2026', 'quy 2/2026'),
    ('th ni', 'thang nay'), ('hôm ni', 'hom nay'), ('180k', '180 000 dong'), ('15 triệu', '15 000 000 dong')])
def test_time_and_number_notation_is_lexical_not_question_specific(source, target):
    assert readable(canonical(source)) == target


def test_relative_week_and_month_words_are_not_removed_as_courtesy():
    assert readable(canonical('tuần rồi')) == 'tuan roi'
    assert readable(canonical('cùng kỳ tháng rồi')) == 'cung ky thang truoc'


def test_catalog_unit_can_supply_missing_sales_unit_but_not_quantity():
    f = pd.DataFrame({'id': ['a', 'b'], 'N': ['Thuốc A', 'Thuốc A'], 'Q': [2, 1], 'BQ': [20, 10], 'P': [100, 100]})
    roles = {'N': 'tên hàng', 'Q': 'số lượng', 'BQ': 'số lượng đã đổi ra đơn vị cơ bản', 'P': 'đơn giá'}
    assert _items(f, 'id', roles, {}) is None
    items = _items(f, 'id', roles, {}, {'thuoc a': 'vien'})
    assert items.bunit.tolist() == ['vien', 'vien']
    assert items.base_price.tolist() == [10, 10]


@pytest.fixture
def history_view():
    days = pd.to_datetime(['2026-04-01', '2026-04-02', '2026-04-03', '2026-04-04', '2026-04-04'])
    o = pd.DataFrame({'order_id': list('abcde'), 'date': days, 'cls': ['completed'] * 4 + ['cancelled']})
    l = pd.DataFrame({'order_id': list('abcde'), 'amount': [8, 8, 20, 20, 999]})
    items = pd.DataFrame({'order_id': list('abcde'), 'name': ['Nước A']*5, 'item': ['nuoc a']*5,
        'bunit': ['chai']*5, 'bqty': [1, 1, 2, 2, 100]})
    v = OrderView(l, o, items=items, item_lines=l.assign(group='nuoc a', gname='Nước A'))
    frame = pd.DataFrame({'Tên hàng': ['Nước A']*5,
        'Ngày áp dụng': pd.to_datetime(['2026-01-01', '2026-04-03', '2026-04-05', '2026-04-06', '2026-04-08']),
        'Giá mới': [8, 10, 10, 9, 10]})
    shop = Shop('synthetic', '**Giá bán**: Giá mới theo đơn vị cơ bản áp dụng từ Ngày áp dụng.',
                {'history': frame}, {'history': {'sources': ['history.csv']}})
    prepare_sem(v, shop)
    return v, shop


def test_price_increase_uses_first_actual_increase_and_equal_windows(history_view):
    v, _ = history_view
    day = dt.date(2026, 4, 9)
    q = 'Nước A tăng giá lên 10đ: so 2 ngày kể từ ngày tăng với 2 ngày ngay trước thì số chai bán ra tăng hay giảm bao nhiêu %?'
    assert answer(q, day, v, Periods(day)) == {'type': 'percent', 'answer': 100.0}
    assert answer(q.replace('2 ngày ngay trước', '3 ngày ngay trước'), day, v, Periods(day)) is None
    assert answer(q.replace('10đ', '11đ'), day, v, Periods(day)) is None


def test_price_history_conflicting_same_day_is_rejected(history_view):
    _, shop = history_view
    f = shop.tables['history']
    f.loc[5] = ['Nước A', pd.Timestamp('2026-04-03'), 12]
    assert PriceHistory.from_shop(shop) is None


def test_group_quantity_and_best_item_use_different_measures():
    day = dt.date(2026, 5, 31)
    orders = pd.DataFrame({'order_id': ['a','b','c'], 'date': pd.to_datetime(['2026-05-01']*3), 'cls': ['completed','completed','cancelled']})
    lines = pd.DataFrame({'order_id': ['a','b','c'], 'amount': [100,60,999]})
    items = pd.DataFrame({'order_id': ['a','b','c'], 'item': ['a','b','b'], 'name': ['Món A','Món B','Món B'],
        'bunit': ['ly']*3, 'bqty': [1,3,100], 'rank_key': ['P2','P1','P1']})
    groups = lines.assign(group='tra', gname='Trà')
    v = OrderView(lines, orders, items=items, groups=groups, item_lines=lines.assign(group=items.item, gname=items.name))
    assert answer('Tháng 5/2026 nhóm Trà bán bao nhiêu ly?', day, v, Periods(day)) == {'type': 'number', 'answer': 4}
    assert answer('Tháng 5/2026 nhóm Trà mặt hàng nào bán nhiều ly nhất?', day, v, Periods(day)) == {'type': 'string', 'answer': 'Món B'}
    assert answer('Tháng 5/2026 nhóm Trà mặt hàng nào bán chạy nhất?', day, v, Periods(day)) == {'type': 'string', 'answer': 'Món B'}
    assert answer('Tháng 5/2026 nhóm Trà mặt hàng nào bán được nhiều nhất?', day, v, Periods(day)) == {'type': 'string', 'answer': 'Món B'}
    assert answer('Tháng 5/2026 nhóm Trà mặt hàng nào thu về nhiều tiền nhất?', day, v, Periods(day)) == {'type': 'string', 'answer': 'Món A'}


def test_weekday_average_includes_calendar_days_without_sales():
    day = dt.date(2026, 6, 30)
    o = pd.DataFrame({'order_id': ['a', 'b'], 'date': pd.to_datetime(['2026-06-01', '2026-06-30']), 'cls': ['completed']*2})
    v = OrderView(pd.DataFrame({'order_id': ['a','b'], 'amount': [100,50]}), o)
    assert answer('Tháng 6/2026 trung bình mỗi thứ hai doanh thu bao nhiêu?', day, v, Periods(day)) == {'type': 'number', 'answer': 20}
