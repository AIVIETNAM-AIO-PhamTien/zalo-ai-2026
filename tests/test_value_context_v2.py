import pandas as pd

from datamate.value_context import ValueIndex


def test_value_hits_keep_source_rows_and_name_collisions():
    products = pd.DataFrame({"name": ["Coca Cola", "Trà sữa", "Coca Cola"]}, dtype=object)
    products.attrs["physical_rows"] = [2, 3, 4]
    customers = pd.DataFrame({"name": ["Coca Cola"]}, dtype=object)
    customers.attrs["physical_rows"] = [9]
    index = ValueIndex({"products": products, "customers": customers})
    result = index.search("Doanh thu Coca Cola là bao nhiêu?")
    assert {(m["table"], m["column"]) for m in result["matches"]} == {
        ("products", "name"), ("customers", "name")}
    product_hit = next(m for m in result["matches"] if m["table"] == "products")
    assert product_hit["count"] == 2
    assert product_hit["physical_rows"] == [2, 4]
    assert all(m["status"] == "candidate" for m in result["matches"])
    assert all(m["match"] == "exact_value" for m in result["matches"])
    assert len(index.search("Trà sữa", tables=["products"])["matches"]) == 1
    assert not index.search("Tra sua")["matches"]  # no accent stripping


def test_value_index_bounds_and_substring_guard():
    frame = pd.DataFrame({"name": ["An", "Samsung", "Sales", "Sony"]}, dtype=object)
    index = ValueIndex({"items": frame}, max_cells=1)
    assert index.truncated
    assert not index.search("Sung")["matches"]
    assert len(index.search("Samsung")["matches"]) == 1
    assert index.search("Samsung", limit=0)["truncated"]


def test_question_phrase_can_retrieve_longer_cell_without_asserting_identity():
    frame = pd.DataFrame({"product": ["Coca Cola 330ml", "Pepsi 330ml"]}, dtype=object)
    result = ValueIndex({"items": frame}).search("Doanh thu Coca Cola bao nhiêu?")
    assert [m["value"] for m in result["matches"]] == ["Coca Cola 330ml"]
    assert result["matches"][0]["match"] == "question_phrase_in_value"
    assert result["matches"][0]["status"] == "candidate"
    assert not ValueIndex({"items": pd.DataFrame({"unit": ["bao"]})}).search("bao nhiêu?")["matches"]


def test_numeric_columns_do_not_exhaust_text_retrieval_budget():
    frame = pd.DataFrame({"amount": [1, 2, 3, 4], "name": ["Coca Cola"] * 4}, dtype=object)
    index = ValueIndex({"items": frame}, max_cells=4)
    assert not index.truncated
    assert index.search("Coca Cola")["matches"][0]["count"] == 4


def test_low_cardinality_names_are_indexed_before_unique_ids():
    frame = pd.DataFrame({"id": ["id-001", "id-002", "id-003"],
                          "name": ["Coca Cola"] * 3}, dtype=object)
    profiles = {"items": {"columns": {"id": {"distinct": 3}, "name": {"distinct": 1}}}}
    index = ValueIndex({"items": frame}, profiles=profiles, max_values=1)
    assert index.truncated
    assert index.search("Coca Cola")["matches"][0]["count"] == 3
