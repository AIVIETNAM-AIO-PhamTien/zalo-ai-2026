import datetime as dt
import pandas as pd

from datamate.loader import Shop
from datamate.semantic import OrderView, _items
from datamate.semantic import order_id_columns


def test_detail_source_wins_over_header_total_regardless_of_table_order():
    headers = pd.DataFrame({'ID': ['a', 'b'], 'Date': pd.to_datetime(['2026-01-01', '2026-01-02']),
                            'Status': ['Hoàn thành', 'Huỷ'], 'Goods': [50, 90]})
    details = pd.DataFrame({'ID': ['a', 'a', 'b'], 'Name': ['X', 'Y', 'X'],
                            'Amount': [20, 30, 90], 'Qty': [2, 3, 9], 'Base': [2, 3, 9], 'Unit': ['cái'] * 3})
    readme = '''## Cột trong `header.csv`
- **ID**: mã đơn
- **Date**: ngày bán
- **Status**: trạng thái đơn
- **Goods**: tiền hàng của đơn
## Cột trong `detail.csv`
- **ID**: mã đơn
- **Name**: tên hàng
- **Amount**: thành tiền sau giảm
- **Qty**: số lượng
- **Base**: số lượng đã đổi ra đơn vị cơ bản
- **Unit**: đơn vị tính
'''
    for tables in ({'header': headers, 'detail': details}, {'detail': details, 'header': headers}):
        shop = Shop('synthetic', readme, tables, {k: {'sources': [k + '.csv']} for k in tables})
        view = OrderView.from_shop(shop)
        assert view is not None
        assert view.source['table'] == 'detail'
        day = dt.date(2026, 1, 2)
        assert view.revenue((dt.date(2026, 1, 1), day), day) == 50
        assert view.qty_sold((dt.date(2026, 1, 1), day), day, 'cai') == 5


def test_unknown_base_unit_does_not_erase_known_items():
    raw = pd.DataFrame({'ID': ['a', 'b'], 'Name': ['Known', 'Unknown'], 'Base': [4, 8]})
    items = _items(raw, 'ID', {'Name': 'tên hàng', 'Base': 'số lượng đã đổi ra đơn vị cơ bản'}, {}, {'known': 'cai'})
    assert items is not None
    assert items.loc[items.item == 'known', 'bunit'].iloc[0] == 'cai'
    assert items.loc[items.item == 'unknown', 'bunit'].isna().all()


def test_card_cannot_promote_customer_total_above_documented_goods_total():
    frame = pd.DataFrame({'ID': ['a'], 'Date': pd.to_datetime(['2026-01-01']),
                          'Status': ['Hoàn thành'], 'Goods': [100], 'Due': [120]})
    readme = '''## Cột trong `header.csv`
- **ID**: mã đơn
- **Date**: ngày bán
- **Status**: trạng thái đơn
- **Goods**: tiền hàng của đơn
- **Due**: số khách phải trả cho cả đơn, gồm phí giao
'''
    shop = Shop('synthetic', readme, {'header': frame}, {'header': {'sources': ['header.csv']}})
    shop.card = {'header': {'Due': 'thành tiền sau giảm'}}
    view = OrderView.from_shop(shop)
    assert view is not None and view.source['amount'] == 'Goods'


def test_document_number_alias_does_not_match_customer_or_item_ids():
    shop = Shop('synthetic', '', {
        'orders': pd.DataFrame(columns=['Đơn số', 'Mã khách', 'Mã hàng']),
        'payments': pd.DataFrame(columns=['Ma don', 'Mã phiếu thu']),
        'catalogue': pd.DataFrame(columns=['Mã khách', 'Mã hàng'])}, {})
    assert order_id_columns(shop) == {'orders': 'Đơn số', 'payments': 'Ma don'}
