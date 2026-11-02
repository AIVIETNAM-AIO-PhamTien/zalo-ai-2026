import datetime as dt

import pandas as pd
import pytest

from datamate.baseline import answer, prepare_sem
from datamate.helpers import Periods
from datamate.loader import Shop
from datamate.semantic import OrderView, _items


DAY = dt.date(2026, 3, 15)


@pytest.fixture
def view():
    orders = pd.DataFrame({'order_id': list('abcdef'),
        'date': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03', '2026-03-16', '2026-02-01', '2026-03-04']),
        'cls': ['completed', 'completed', 'cancelled', 'completed', 'completed', 'returned']})
    lines = pd.DataFrame({'order_id': list('abcdef'), 'amount': [40, 40, 900, 500, 100, 1000]})
    customers = pd.DataFrame({'order_id': list('abcdef'), 'customer': ['C2', 'C1', 'C3', 'C4', 'C1', 'C3'],
                              'display': ['Bình', 'An', 'Chi', 'Dũng', 'An', 'Chi']})
    item_lines = lines.assign(group='ca phe', gname='Cà phê')
    return OrderView(lines, orders, customers=customers, item_lines=item_lines)


def solve(view, q):
    return answer(q, DAY, view, Periods(DAY))


def test_aspect_word_in_order_counts_and_average_money_paraphrases(view):
    assert solve(view, 'Tính từ đầu năm đến hết hôm nay đã có bao nhiêu đơn?') == {'type': 'number', 'answer': 3}
    assert solve(view, 'Tháng 3/2026 bình quân mỗi đơn đem về bao nhiêu doanh thu?') == {'type': 'number', 'answer': 40}
    assert solve(view, 'Tháng 3/2026 bq mỗi đơn mang lại bao nhiêu doanh thu?') == {'type': 'number', 'answer': 40}
    assert solve(view, 'Tính từ đầu năm đến hết hôm nay đã có bao nhiêu đơn của khách Lan?') is None
    assert solve(view, 'Tháng 3/2026 bình quân mỗi đơn đem về bao nhiêu doanh thu và số lượng?') is None


def test_top_k_uses_customer_code_for_ties_and_excludes_invalid_or_future(view):
    assert solve(view, 'Top 2 khách chi tiêu nhiều nhất tháng 3/2026') == {'type': 'list', 'answer': ['An', 'Bình']}
    assert solve(view, 'Khách nào mua nhiều đơn nhất tháng 3/2026') is None
    from datamate.sem_api import SemAPI
    sem = SemAPI(Shop('synthetic', '', {}), DAY)
    sem._view, sem._built = view, True
    assert sem.top_customers(Periods(DAY).this_month(), 2) == ['An', 'Bình']


def test_named_item_revenue_and_repeated_period_share(view):
    assert solve(view, 'Tháng 3/2026 mặt hàng Cà phê thu về bao nhiêu tiền') == {'type': 'number', 'answer': 80}
    assert solve(view, 'Doanh thu Cà phê tháng 3/2026 bằng bao nhiêu % tổng doanh thu tháng 3/2026') == {'type': 'percent', 'answer': 100.0}
    assert solve(view, 'Doanh thu Cà phê tháng 3/2026 bằng bao nhiêu % tổng doanh thu tháng 2/2026') is None


def test_group_order_count_uses_distinct_order_grain(view):
    # A group can contribute multiple line rows to one order; the requested
    # measure is the number of orders, never the number of those rows.
    view.groups = pd.concat([view.item_lines, view.item_lines.iloc[:1]], ignore_index=True)
    assert solve(view, 'Tháng 3/2026 số đơn có mua hàng nhóm Cà phê là bao nhiêu?') == {
        'type': 'number', 'answer': 2}


@pytest.mark.parametrize('condition', ['của khách Lan', 'không tính tiền mặt', 'chỉ các đơn COD', 'giá vốn', 'dưới 10 triệu', 'và số đơn'])
def test_unknown_conditions_are_not_dropped(view, condition):
    assert solve(view, 'Doanh thu Cà phê tháng 3/2026 ' + condition) is None


def test_comparison_uses_current_month_even_when_reference_is_mentioned_first(view):
    assert solve(view, 'So với cùng kỳ th trước, doanh thu từ đầu tháng này đến hết hôm nay tăng hay giảm bao nhiêu %') == {'type': 'percent', 'answer': -20.0}
    assert solve(view, 'Doanh thu Cà phê tháng 3/2026 chênh tháng 2/2026 bao nhiêu tiền') == {'type': 'number', 'answer': -20}


def test_year_over_year_and_slash_increase_decrease(view):
    view.orders.loc[view.orders.order_id == 'e', 'date'] = pd.Timestamp('2025-03-01')
    assert solve(view, 'So cùng kỳ năm trước, doanh thu tháng 3/2026 tăng/giảm mấy %') == {'type': 'percent', 'answer': -20.0}


def test_zero_denominator_abstains(view):
    view.lines.loc[view.lines.order_id == 'e', 'amount'] = 0
    assert solve(view, 'Doanh thu tháng 3/2026 tăng bao nhiêu % so với tháng 2/2026') is None


def test_price_is_before_discount_in_base_units_including_cancelled(view):
    raw = pd.DataFrame({'id': ['a', 'c'], 'name': ['Cà phê', 'Cà phê'],
                        'qty': [2, 1], 'base': [12, 6], 'unit': ['hop', 'hop'], 'price': [600, 600]})
    # Explicit basic unit row establishes the base; the cancelled row remains price evidence.
    raw.loc[2] = ['b', 'Cà phê', 1, 1, 'ly', 100]
    view.items = _items(raw, 'id', {'name': 'tên hàng', 'qty': 'số lượng',
        'base': 'số lượng đã đổi ra đơn vị cơ bản', 'unit': 'đơn vị tính', 'price': 'đơn giá'}, {})
    assert solve(view, 'Ngày 03/03/2026 bán Cà phê giá bao nhiêu một ly') == {'type': 'number', 'answer': 100}


def test_receipt_totals_not_multiplied_and_supplier_ranked_by_money(view):
    frame = pd.DataFrame({'Mã nhập hàng': ['R1', 'R1', 'R2', 'R3'],
        'Thời gian': pd.to_datetime(['2026-03-01'] * 4), 'Mã NCC': ['S1', 'S1', 'S2', 'S3'],
        'Cần trả NCC': [500, 500, 100, 9999], 'Trạng thái': ['Hoàn thành'] * 3 + ['Huỷ']})
    readme = '## Cột trong `nhap.csv`\n- **Cần trả NCC**: tổng tiền phiếu nhập\n'
    shop = Shop('synthetic', readme, {'nhap': frame}, {'nhap': {'sources': ['nhap.csv']}})
    prepare_sem(view, shop)
    assert solve(view, 'Tháng 3/2026 tổng giá trị các phiếu nhập hàng là bao nhiêu') == {'type': 'number', 'answer': 600}
    assert solve(view, 'Tháng 3/2026 nhập nhiều tiền hàng nhất từ nhà cung cấp nào') == {'type': 'string', 'answer': 'S1'}
    assert solve(view, 'Tháng 3/2026 tiền nhập hàng từ nhà cung cấp nào cao nhất') == {'type': 'string', 'answer': 'S1'}
    assert solve(view, 'Tháng 3/2026 nhập bao nhiêu tiền từ nhà cung cấp nào') is None


def test_customer_group_join_is_many_to_one_and_keeps_walkin_in_denominator(view):
    frame = pd.DataFrame({'Mã khách': ['C1', 'C2', 'C3', 'C4'], 'Nhóm': ['Sỉ', 'Lẻ', 'Sỉ', 'Sỉ']})
    readme = '## Cột trong `khach.csv`\n- **Mã khách**: mã khách\n- **Nhóm**: nhóm khách\n'
    shop = Shop('synthetic', readme, {'khach': frame}, {'khach': {'sources': ['khach.csv']}})
    shop.card = {'khach': {'Mã khách': 'mã đơn'}}
    prepare_sem(view, shop)
    assert solve(view, 'Doanh thu khách nhóm sỉ tháng 3/2026 chiếm bao nhiêu % tổng doanh thu') == {'type': 'percent', 'answer': 50.0}
    view.customers = view.customers[view.customers.order_id != 'a']
    prepare_sem(view, shop)
    assert solve(view, 'Doanh thu khách nhóm sỉ tháng 3/2026 chiếm bao nhiêu % tổng doanh thu') == {'type': 'percent', 'answer': 50.0}
    frame.loc[4] = ['C1', 'Lẻ']
    prepare_sem(view, shop)
    assert solve(view, 'Doanh thu khách nhóm sỉ tháng 3/2026 chiếm bao nhiêu % tổng doanh thu') is None


def test_thresholds_use_distinct_orders_and_valid_daily_revenue(view):
    assert solve(view, 'Tháng 3/2026 số đơn có doanh thu ≥ 40 đồng là bao nhiêu') == {'type': 'number', 'answer': 2}
    assert solve(view, 'Tháng 3/2026 có mấy ngày doanh thu đạt 40 đồng trở lên') == {'type': 'number', 'answer': 2}
    assert solve(view, 'Tháng 3/2026 số đơn doanh thu trên 40 đồng là bao nhiêu') is None
    assert solve(view, 'Tháng 3/2026 số đơn có doanh thu ≥ 40 đồng của khách An') is None
    assert solve(view, 'Tháng 3/2026 số đơn có doanh thu không đạt 40 đồng trở lên') is None


def test_unsold_means_previously_sold_not_every_catalog_item(view):
    view.item_lines.loc[view.item_lines.order_id == 'e', ['group', 'gname']] = ['tra', 'Trà']
    assert solve(view, '7 ngày gần nhất có bao nhiêu mặt hàng bị ế') == {'type': 'number', 'answer': 2}
    assert solve(view, 'Từ ngày 01/03/2026 đến hết ngày 15/03/2026 có bao nhiêu mặt hàng bị ế') == {'type': 'number', 'answer': 1}


def test_changed_rule_blocks_composed_customer_ranking(view):
    shop = Shop('synthetic', '', {}, {})
    prepare_sem(view, shop, '| Xác định khách | quy tắc mới |')
    assert solve(view, 'Top 2 khách chi tiêu nhiều nhất tháng 3/2026') is None


def test_receipt_conflicting_header_fails_closed(view):
    from datamate.receipt_sem import Receipts
    frame = pd.DataFrame({'Mã nhập hàng': ['R1', 'R1'],
        'Thời gian': pd.to_datetime(['2026-03-01'] * 2), 'Cần trả NCC': [100, 200]})
    readme = '## Cột trong `nhap.csv`\n- **Cần trả NCC**: tổng tiền phiếu nhập\n'
    shop = Shop('synthetic', readme, {'nhap': frame}, {'nhap': {'sources': ['nhap.csv']}})
    assert Receipts.from_shop(shop) is None
