import datetime as dt
from types import SimpleNamespace

from datamate.baseline import stock_answer


def test_product_name_is_removed_before_metric_guards():
    day = dt.date(2026, 6, 20)
    stock = SimpleNamespace(last_day=day, stock={'x': ('Lọ tinh dầu Giá Tốt', 17)}, conv={'x': ('lo', {})})
    assert stock_answer('Cuối ngày hôm nay Lọ tinh dầu Giá Tốt còn bao nhiêu lọ trong kho?', day, stock) == {'type': 'number', 'answer': 17}
    assert stock_answer('Giá trị tồn kho Lọ tinh dầu Giá Tốt hôm nay bao nhiêu tiền?', day, stock) is None
    assert stock_answer('Lọ tinh dầu Giá Tốt hôm nay còn bao nhiêu lọ hết hạn?', day, stock) is None
    assert stock_answer('Lọ tinh dầu Giá Tốt hôm nay còn bao nhiêu lọ của khách Lan?', day, stock) is None
    assert stock_answer('Lọ tinh dầu Giá Tốt hôm nay còn bao nhiêu lọ?', day - dt.timedelta(days=1), stock) is None
