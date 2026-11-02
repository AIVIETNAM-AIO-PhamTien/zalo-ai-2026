import datetime as dt

import pandas as pd
import pytest

from datamate.loader import Shop
from datamate.semantic import OrderView, _catalog_item_names


@pytest.fixture
def shop():
    lines = pd.DataFrame({"oid": ["A", "A", "B"], "sku": ["S1", "S2", "S1"],
                          "qty": [2, 3, 4], "base": [2, 3, 4], "unit": ["ly", "cai", "ly"],
                          "amount": [20, 30, 40]})
    heads = pd.DataFrame({"oid": ["A", "B"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "status": ["Xong", "Xong"]})
    catalogue = pd.DataFrame({"code": ["S1", "S2"], "label": ["Trà", "Bánh"]})
    readme = """## Cột trong `lines.csv`
- **oid**: mã đơn
- **sku**: mã hàng
- **qty**: số lượng
- **base**: số lượng đã đổi ra đơn vị cơ bản
- **unit**: đơn vị tính
- **amount**: thành tiền sau giảm
## Cột trong `heads.csv`
- **oid**: mã đơn
- **date**: ngày bán
- **status**: trạng thái đơn
## Cột trong `catalogue.csv`
- **code**: mã hàng
- **label**: tên hàng
"""
    tables = {"lines": lines, "heads": heads, "catalogue": catalogue}
    return Shop("synthetic", readme, tables, {t: {"sources": [f"{t}.csv"]} for t in tables})


def test_catalogue_enables_item_metrics_without_changing_revenue_or_rows(shop):
    v = OrderView.from_shop(shop)
    assert v.items is not None and len(v.items) == 3
    assert v.source["item_name_catalog"] == "catalogue"
    assert not any(c.startswith("__catalog") for c in shop.tables["lines"])
    p, today = (dt.date(2026, 1, 1), dt.date(2026, 1, 31)), dt.date(2026, 1, 31)
    assert v.revenue(p, today) == 90
    assert v.item_revenue(p, today, "tra") == 60
    assert v.qty_sold(p, today, "ly") == 6
    assert v.items.loc[v.items.item == "tra", "rank_key"].unique().tolist() == ["S1"]


@pytest.mark.parametrize("case", ["missing", "duplicate_conflict", "blank", "conflicting_tables"])
def test_uncertain_catalogue_never_selects_a_convenient_first_row(shop, case):
    if case == "missing":
        shop.tables["catalogue"] = shop.tables["catalogue"].iloc[:1]
    elif case == "duplicate_conflict":
        shop.tables["catalogue"] = pd.concat([shop.tables["catalogue"], pd.DataFrame({"code": ["S1"], "label": ["Cà phê"]})])
    elif case == "blank":
        shop.tables["catalogue"].loc[0, "label"] = " "
    else:
        shop.tables["other"] = pd.DataFrame({"code": ["S1", "S2"], "label": ["Cà phê", "Bánh"]})
        shop.meta["other"] = {"sources": ["other.csv"]}
        shop.readme += "\n## Cột trong `other.csv`\n- **code**: mã hàng\n- **label**: tên hàng\n"
    v = OrderView.from_shop(shop)
    assert v is not None and v.items is None and v.item_lines is None
    assert v.revenue((dt.date(2026, 1, 1), dt.date(2026, 1, 31)), dt.date(2026, 1, 31)) == 90


def test_duplicate_identical_catalogue_rows_do_not_multiply_sales(shop):
    shop.tables["catalogue"] = pd.concat([shop.tables["catalogue"], shop.tables["catalogue"]])
    assert len(OrderView.from_shop(shop).items) == 3


def test_generic_catalogue_name_is_safe_only_with_a_declared_product_code(shop):
    shop.readme = shop.readme.replace("**label**: tên hàng", "**label**: tên")
    assert OrderView.from_shop(shop).items is not None


def test_card_can_fill_a_described_item_name_without_changing_core_columns(shop):
    shop.tables["lines"]["Hàng"] = ["Trà", "Bánh", "Trà"]
    shop.readme = shop.readme.replace("## Cột trong `heads.csv`", "- **Hàng**: hàng bán ghi tắt\n## Cột trong `heads.csv`")
    shop.card = {"lines": {"Hàng": "tên hàng"}}
    v = OrderView.from_shop(shop)
    assert v.items is not None
    assert "item_name_catalog" not in v.source
    assert ("lines", "Hàng") in shop.card_added


def test_card_cannot_replace_brand_or_customer_with_product_name(shop):
    shop.tables["lines"]["Hãng"] = ["Brand"] * 3
    shop.readme = shop.readme.replace("## Cột trong `heads.csv`", "- **Hãng**: hãng sản xuất\n## Cột trong `heads.csv`")
    shop.card = {"lines": {"Hãng": "tên hàng"}}
    assert OrderView.from_shop(shop).source["item_name_catalog"] == "catalogue"


def test_old_card_cannot_replace_declared_quantities_with_unit_price(shop):
    shop.tables["lines"]["price"] = [10, 10, 10]
    shop.readme = shop.readme.replace("## Cột trong `heads.csv`", "- **price**: đơn giá\n## Cột trong `heads.csv`")
    shop.card = {"lines": {"price": "số lượng đã đổi ra đơn vị cơ bản"}}
    v = OrderView.from_shop(shop)
    assert v.items.bqty.tolist() == [2, 3, 4]
    assert ("lines", "price") not in shop.card_added


def test_card_quantity_cannot_point_to_a_monetary_column_without_readme():
    from datamate.semantic import _card_ok
    frame = pd.DataFrame({"Đơn giá": [100, 200], "SL quy đổi": [1, 2]})
    assert _card_ok(frame, {"Đơn giá": "số lượng đã đổi ra đơn vị cơ bản", "SL quy đổi": "số lượng đã đổi ra đơn vị cơ bản"}) == {
        "SL quy đổi": "số lượng đã đổi ra đơn vị cơ bản"}
