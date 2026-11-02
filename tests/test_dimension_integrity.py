import pandas as pd
from datamate.loader import Shop
from datamate.semantic import _line_groups


def test_conflicting_catalogue_values_do_not_silently_select_first():
    sales = pd.DataFrame({'sku': ['a', 'b']})
    catalogue = pd.DataFrame({'sku': ['a', 'a', 'b'], 'group': ['A', 'B', 'B']})
    shop = Shop('synthetic', '', {'catalog': catalogue}, {'catalog': {'sources': ['catalog.csv']}})
    roles = {'catalog.csv': {'sku': 'mã hàng', 'group': 'nhóm hàng'}}
    assert _line_groups(shop, roles, sales, {'sku': 'mã hàng'}) is None
    catalogue.loc[1, 'group'] = 'A'
    assert _line_groups(shop, roles, sales, {'sku': 'mã hàng'}).tolist() == ['A', 'B']
