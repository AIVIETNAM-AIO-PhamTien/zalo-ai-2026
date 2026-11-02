

def test_row_grain_and_join_keys():
    import pandas as pd
    from datamate.brief import join_keys, row_grain
    from datamate.loader import Shop
    lines = pd.DataFrame({"MaHD": ["A", "A", "B", "C", "D"], "MaSP": ["x", "y", "x", "w", "z"]})
    orders = pd.DataFrame({"MaHD": ["A", "B", "C", "D"]})
    products = pd.DataFrame({"MaSP": ["x", "y", "z", "w"]})
    assert "nunique" in row_grain(lines, "MaHD")
    assert "một dòng = một đơn" in row_grain(orders, "MaHD")
    keys = join_keys(Shop("b", "", {"ct": lines, "dh": orders, "sp": products}, {}))
    assert any("'ct']['MaHD']" in k and "'dh']['MaHD']" in k for k in keys)
    assert not any("MaHD" in k and "MaSP" in k for k in keys)
