import pandas as pd

from datamate.schema_roles import business_roles
from datamate.schema_roles import align_declared_columns


def test_roles_are_invariant_to_case_spacing_and_camelcase():
    frame = pd.DataFrame(columns=['MaHangHoa', 'TenGoi', 'DonViCoSo', 'ThanhPhanChinh', 'BanTheoDon'])
    assert business_roles(frame) == {'MaHangHoa': 'mã hàng', 'DonViCoSo': 'đơn vị cơ bản',
                                    'ThanhPhanChinh': 'hoạt chất', 'BanTheoDon': 'thuốc kê đơn', 'TenGoi': 'tên hàng'}


def test_ambiguous_roles_and_generic_labels_are_not_guessed():
    assert business_roles(pd.DataFrame(columns=['Item Code', 'Product ID', 'Name', 'Nhóm'])) == {}
    assert business_roles(pd.DataFrame(columns=['Name', 'Nhóm'])) == {}
    roles = business_roles(pd.DataFrame(columns=['Customer ID', 'Name', 'Nhóm']))
    assert roles == {'Customer ID': 'mã khách', 'Name': 'tên khách'}


def test_customer_name_cannot_become_product_name_in_mixed_header():
    roles = business_roles(pd.DataFrame(columns=['Mã hàng', 'Mã khách', 'Tên gọi']))
    assert 'Tên gọi' not in roles
    assert business_roles(pd.DataFrame(columns=['SKU', 'Category', 'Base Unit'])) == {
        'SKU': 'mã hàng', 'Category': 'nhóm hàng', 'Base Unit': 'đơn vị cơ bản'}


def test_product_category_aliases_require_a_product_entity_context():
    assert business_roles(pd.DataFrame(columns=['Mã hàng', 'Loại sản phẩm'])) == {
        'Mã hàng': 'mã hàng', 'Loại sản phẩm': 'nhóm hàng'}
    # A customer table's classification is not silently promoted to a product group.
    assert business_roles(pd.DataFrame(columns=['Mã khách', 'Loại sản phẩm'])) == {'Mã khách': 'mã khách'}


def test_schema_card_cannot_promote_prices_or_names_to_product_groups():
    from datamate.loader import Shop
    from datamate.semantic import shop_columns
    frame = pd.DataFrame({'Giá 1 đv': [12_000, 15_000], 'Họ tên': ['An', 'Bình'],
                          'Nhóm hàng': ['Nước', 'Bánh']})
    shop = Shop('synthetic', '', {'products': frame}, {})
    shop.card = {'products': {'Giá 1 đv': 'nhóm hàng', 'Họ tên': 'nhóm hàng', 'Nhóm hàng': 'nhóm hàng'}}
    roles = shop_columns(shop)
    bound = {column: meaning for columns in roles.values() for column, meaning in columns.items()}
    assert bound.get('Nhóm hàng') == 'nhóm hàng'
    assert 'Giá 1 đv' not in bound and 'Họ tên' not in bound


def test_readme_rebinding_is_unique_ordered_and_not_positional():
    frame = pd.DataFrame(columns=['amount', 'name_customer', 'id_order_item', 'price_san_pham'])
    declared = {'Mã đơn hàng': 'mã đơn', 'Tên khách hàng': 'tên khách',
                'Thành tiền': 'thành tiền sau giảm', 'Giá sản phẩm': 'đơn giá', 'Mã hàng': 'mã hàng'}
    assert align_declared_columns(frame, declared) == {
        'id_order_item': 'mã đơn', 'name_customer': 'tên khách', 'amount': 'thành tiền sau giảm', 'price_san_pham': 'đơn giá'}
    ambiguous = pd.DataFrame(columns=['Amount', 'amount'])
    assert align_declared_columns(ambiguous, {'Thành tiền': 'thành tiền sau giảm'}) == {}


def test_datetime_and_date_aliases_must_agree():
    from datamate.semantic import infer_meanings
    frame = pd.DataFrame({'ThoiGianLap': pd.to_datetime(['2026-01-10 15:30']),
                          'NgayLap': pd.to_datetime(['2026-01-10'])})
    assert infer_meanings(frame)['ThoiGianLap'] == 'ngày bán'
    frame.loc[0, 'NgayLap'] = pd.Timestamp('2026-01-11')
    assert 'ngày bán' not in infer_meanings(frame).values()


def test_word_by_word_translated_headers_keep_their_role():
    # strange2: 'Tên hàng' → 'name_item', 'SL quy đổi' → 'qty_quy_doi', 'Mã KH' → 'id_kh'
    frame = pd.DataFrame(columns=['id_order', 'id_item', 'name_item', 'unit_tinh', 'qty', 'qty_quy_doi'])
    assert business_roles(frame) == {'id_item': 'mã hàng', 'name_item': 'tên hàng', 'unit_tinh': 'đơn vị tính',
                                     'qty': 'số lượng', 'qty_quy_doi': 'số lượng đã đổi ra đơn vị cơ bản'}
    assert business_roles(pd.DataFrame(columns=['id_order', 'id_kh'])) == {'id_kh': 'mã khách'}
    # hai cột cùng tương đương: không đoán
    assert 'tên hàng' not in business_roles(pd.DataFrame(columns=['name_item', 'NameItem'])).values()


def test_order_customers_accept_translated_customer_columns():
    from datamate.semantic import _order_customers
    df = pd.DataFrame({'id_order': ['A', 'B', 'C'], 'id_kh': ['K1', 'K2', None], 'customer': ['An', 'Bình', 'Khách lẻ']})
    out = _order_customers([(df, 'id_order')])
    assert out.set_index('order_id')['customer'].to_dict() == {'A': 'K1', 'B': 'K2'}
    assert out.set_index('order_id')['display'].to_dict() == {'A': 'An', 'B': 'Bình'}
