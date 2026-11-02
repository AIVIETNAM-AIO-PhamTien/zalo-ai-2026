import json

import pandas as pd

from datamate.llm import ScriptedLLM
from datamate.loader import Shop
from datamate.schema_card import make_card, parse_card


def test_prompt_keeps_late_tables_and_all_column_names():
    from datamate.schema_card import build_prompt
    tables = {f"table_{i}": pd.DataFrame({f"column_{i}": ["x" * 500]}) for i in range(30)}
    prompt = build_prompt(Shop("s", "", tables), "prefix" * 10000, brief_chars=1000)
    assert "T['table_29']" in prompt and "'column_29'" in prompt
    assert "T['table_0']" in prompt


def test_single_unit_quantities_are_not_mistaken_for_row_numbers():
    df = pd.DataFrame({"Mã đơn": ["A", "B", "C"], "SL": [1, 1, 1], "Tiền": [10, 20, 30]})
    card = parse_card('{"t": {"Mã đơn": "order_id", "SL": "qty", "Tiền": "line_amount"}}', Shop("s", "", {"t": df}))
    assert card["t"]["SL"] == "số lượng"


def test_empty_numeric_column_cannot_be_a_valid_role():
    assert parse_card('{"t": {"empty": "line_amount"}}', Shop("s", "", {"t": pd.DataFrame({"empty": [None, None]})})) == {}


def _shop():
    df = pd.DataFrame({"Đơn số": ["A1", "A1", "A2"], "Ngày": pd.to_datetime(["2026-01-01"] * 3),
                       "Thành tiền": [10, 20, 30]})
    return Shop("b", "", {"hd": df})


def test_parse_card_ignores_braces_in_think():
    text = '<think>xem {cột} nào</think>{"hd": {"Đơn số": "order_id", "Thành tiền": "line_amount", "X": "qty"}}'
    assert parse_card(text, _shop()) == {"hd": {"Đơn số": "mã đơn", "Thành tiền": "thành tiền sau giảm"}}


def test_make_card_refines_once_then_reads_cache(tmp_path):
    llm = ScriptedLLM(['{"hd": {"Ngày": "order_date"}}'])
    card = make_card(_shop(), "brief", llm, tmp_path, "b")
    assert card == {"hd": {"Ngày": "ngày bán"}}
    assert json.loads((tmp_path / "b.card.json").read_text(encoding="utf-8")) == card
    assert len(llm.calls) == 2  # bảng đơn chưa dựng được → hỏi lại một lượt kèm lý do
    assert "chưa dựng được" in llm.calls[1][-1]["content"]
    make_card(_shop(), "brief", llm, tmp_path, "b")
    assert len(llm.calls) == 2


def test_make_card_keeps_refined_card_only_if_view_builds(tmp_path):
    df = pd.DataFrame({"Đơn số": ["A1", "A1", "A2"], "Ngày": pd.to_datetime(["2026-01-01"] * 3),
                       "Thành tiền": [10, 20, 30], "Ghi chú": [None, None, "khách trả lại"]})
    shop = Shop("b", "", {"hd": df}, {"hd": {}})
    good = '{"hd": {"Đơn số": "order_id", "Ngày": "order_date", "Thành tiền": "line_amount"}}'
    card = make_card(shop, "brief", ScriptedLLM(['{"hd": {"Ngày": "order_date"}}', good]), tmp_path, "b")
    assert card["hd"]["Đơn số"] == "mã đơn"


def test_make_card_bad_json_gives_empty(tmp_path):
    assert make_card(_shop(), "brief", ScriptedLLM(["không biết"]), tmp_path, "b") == {}
    assert not (tmp_path / "b.card.json").exists()


def test_card_table_key_wrapped_and_price_vs_line_amount():
    df = pd.DataFrame({"Mã đơn": ["A", "A", "B"], "SL": [2, 1, 3], "Giá": [10, 50, 7], "Tổng tiền": [20, 50, 21]})
    shop = Shop("b", "", {"hd": df})
    text = '{"T[\'hd\']": {"Mã đơn": "order_id", "SL": "qty", "Giá": "line_amount", "Tổng tiền": "order_goods"}}'
    card = parse_card(text, shop)
    # 'Giá' × SL = 'Tổng tiền' → Giá là đơn giá, tiền dòng là 'Tổng tiền'; tiền cả đơn khác nhau trong đơn A → bỏ
    assert card == {"hd": {"Mã đơn": "mã đơn", "Tổng tiền": "thành tiền sau giảm", "SL": "số lượng"}}


def test_cache_changes_when_data_changes_even_if_prompt_statistics_match(tmp_path):
    shop = _shop()
    llm = ScriptedLLM(['{"hd": {"Ngày": "order_date"}}'])
    make_card(shop, "brief", llm, tmp_path, "b")
    assert len(llm.calls) == 2
    shop.tables["hd"].loc[1, "Thành tiền"] = 21  # cùng min/max/số dòng
    make_card(shop, "brief", llm, tmp_path, "b")
    assert len(llm.calls) == 4
    shop.readme = "x" * 7000 + "before"
    make_card(shop, "brief", llm, tmp_path, "b")
    assert len(llm.calls) == 6
    shop.readme = "x" * 7000 + "after"  # ngoài prefix 6.000 ký tự của prompt
    make_card(shop, "brief", llm, tmp_path, "b")
    assert len(llm.calls) == 8
    make_card(shop, "brief", llm, tmp_path, "b")
    assert len(llm.calls) == 8


def test_corrupted_card_is_not_accepted_from_a_matching_cache(tmp_path):
    llm = ScriptedLLM(['{"hd": {"Ngày": "order_date"}}'])
    shop = _shop()
    make_card(shop, "brief", llm, tmp_path, "b")
    (tmp_path / "b.card.json").write_text("invalid", encoding="utf-8")
    assert make_card(shop, "brief", llm, tmp_path, "b") == {"hd": {"Ngày": "ngày bán"}}
    assert len(llm.calls) == 4


def test_server_failure_does_not_poison_future_cache(tmp_path):
    class Broken:
        def chat(self, *args, **kwargs):
            raise RuntimeError("server offline")
    assert make_card(_shop(), "brief", Broken(), tmp_path, "b") == {}
    assert not (tmp_path / "b.card.json").exists()
    llm = ScriptedLLM(['{"hd": {"Ngày": "order_date"}}'])
    assert make_card(_shop(), "brief", llm, tmp_path, "b")
