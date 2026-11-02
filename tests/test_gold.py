import pandas as pd

from datamate.gold import _shift, gold_report


def test_shift_follows_rulebook_hours():
    assert _shift(pd.Timestamp("2025-01-01 06:00")) == "sang"
    assert _shift(pd.Timestamp("2025-01-01 13:59")) == "sang"
    assert _shift(pd.Timestamp("2025-01-01 14:00")) == "chieu"
    assert _shift(pd.Timestamp("2025-01-01 22:30")) is None


def test_gold_report_counts_rule_violations():
    o = pd.DataFrame({"order_id": ["A", "A", "B"], "cls": ["completed", "returned", "completed"],
                      "date": [pd.Timestamp("2025-01-01")] * 2 + [None],
                      "counts_revenue": [True, True, True], "revenue": [10, 10, None]})
    ln = pd.DataFrame({"order_id": ["A", "C"], "item_key": ["x", None], "name": ["X", None], "bqty": [1, None],
                       "bunit": ["ly", None], "base_price": [1, None]})
    r = gold_report({"fact_order": o, "fact_line": ln})
    assert r["orders_multi_status"] == 1 and r["orders_no_date"] == 1 and r["revenue_orders_no_amount"] == 1
    assert r["lines_orphan"] == 1 and r["lines_no_base_qty"] == 1 and r["violations"] == 6


def test_stt_is_order_key_only_when_it_repeats():
    from datamate.semantic import infer_meanings
    base = [1, 1, 2, 3, 3, 3, 4, 5, 5, 6]
    orders = pd.DataFrame({"STT": base + [x + 6 for x in base] + [x + 12 for x in base], "Hàng": ["a"] * 30})
    assert infer_meanings(orders).get("STT") == "mã đơn"
    rows = pd.DataFrame({"STT": range(1, 31), "Hàng": ["a"] * 30})
    assert infer_meanings(rows).get("STT") != "mã đơn"


def test_abbreviated_order_and_amount_headers():
    from datamate.semantic import infer_meanings
    df = pd.DataFrame({"HĐ": ["A", "A", "B"] * 10, "SL": [1, 2, 1] * 10, "ĐG": [10, 5, 7] * 10, "TT": [10, 10, 7] * 10})
    m = infer_meanings(df)
    assert m["HĐ"] == "mã đơn" and m["TT"] == "thành tiền sau giảm"


def test_text_unit_conversion_keeps_equals_sign():
    from types import SimpleNamespace

    from datamate.semantic import _conversions, item_key
    df = pd.DataFrame({"Tên hàng": ["Tã Bobby L"], "ĐVT": ["bịch"], "Quy đổi": ["1 thùng = 4 bịch"]})
    shop = SimpleNamespace(tables={"DM": df}, meta={"DM": {"sources": ["DM.xlsx"]}})
    cols = {"DM.xlsx": {"Tên hàng": "tên hàng", "ĐVT": "đơn vị cơ bản", "Quy đổi": "quy đổi đơn vị lớn ra ĐVT"}}
    assert _conversions(shop, cols)[item_key("Tã Bobby L")] == ("bich", {"thung": 4.0})


def test_missing_conversion_is_a_data_gap_not_a_violation():
    o = pd.DataFrame({"order_id": ["A"], "cls": ["completed"], "date": [pd.Timestamp("2025-01-01")],
                      "counts_revenue": [True], "revenue": [10]})
    ln = pd.DataFrame({"order_id": ["A", "A"], "item_key": ["ong pvc", "keo"], "name": ["Ống PVC", "Keo"],
                       "bqty": [None, None], "bunit": [None, None], "base_price": [None, None]})
    r = gold_report({"fact_order": o, "fact_line": ln, "convertible": {"keo"}})
    assert r["data_gaps"] == 1 and r["violations"] == 1  # 'keo' có quy đổi mà vẫn thiếu: lỗi; 'ong pvc': thiếu thông tin


def test_top_k_breaks_ties_by_code():
    from datamate.helpers import top_k
    df = pd.DataFrame({"ma": ["B", "A", "C"], "sl": [5, 5, 7]})
    assert top_k(df, "sl", "ma")["ma"].tolist() == ["C", "A", "B"]
    assert top_k(df, "sl", "ma", 2, ascending=True)["ma"].tolist() == ["A", "B"]


def test_lint_flags_bare_idxmax_and_nan_sum():
    from datamate.lint import LintHints, lint
    msgs = lint("best = df.groupby('ma')['sl'].sum().idxmax()\nx = G['don']['revenue'].sum()", LintHints())
    assert any("top_k" in m for m in msgs) and any("NaN" in m for m in msgs)
    assert lint("best = top_k(t, 'sl', 'ma', 1)", LintHints()) == []


def test_agent_tables_money_frames_and_report():
    from datamate.gold import agent_tables, gold_dictionary
    from datamate.loader import load_shop
    G = agent_tables(load_shop("data/train/data/biz_101"), "2026-06-30")
    assert {"don", "dong", "tien_mat_hang"} <= set(G) and "tien_mat_hang" in G["_bao_cao"]["tables"]
    assert {"key", "name", "amount", "counts_revenue"} <= set(G["tien_mat_hang"].columns)
    assert (G["don"]["date"] <= pd.Timestamp("2026-06-30")).all()
    assert "{" not in gold_dictionary() and "counts_revenue" in gold_dictionary()


def test_agent_tables_ledgers_reported():
    from datamate.gold import agent_tables
    from datamate.loader import load_shop
    G = agent_tables(load_shop("data/train/data/biz_101"), "2026-06-30")
    rep = G["_bao_cao"]
    for t in ("cong_no_don", "nhap_hang", "lo_hang"):
        assert (t in G) != (t in rep["missing"])  # có bảng, hoặc ghi lý do không có
    if "cong_no_tra" in G:
        assert (G["cong_no_tra"]["date"] <= pd.Timestamp("2026-06-30")).all()


def test_agent_tables_store_reused_and_isolated(tmp_path):
    """Phần nặng của G lưu ra đĩa: tiến trình sau đọc lại giống hệt; code agent sửa bảng không lan sang lần sau."""
    import pickle
    import datamate.gold as gold
    from datamate.loader import load_shop
    shop = load_shop("data/train/data/biz_101")
    store = tmp_path / "g.pkl"
    a = gold.agent_tables(shop, "2026-06-30", None, store)
    assert store.is_file()
    gold._BUILT.clear()
    b = gold.agent_tables(pickle.loads(pickle.dumps(shop)), "2026-06-30", None, store)
    assert a.keys() == b.keys() and all(a[k].equals(b[k]) for k in a if k != "_bao_cao")
    a["don"].loc[:, "revenue"] = 0
    c = gold.agent_tables(shop, "2026-06-30", None, store)
    assert not (c["don"]["revenue"] == 0).all()
    assert gold.gold_store_path("x.pkl", "r1") != gold.gold_store_path("x.pkl", "r2")
    assert gold.gold_store_path("x.pkl", "r", {"a": 1}) != gold.gold_store_path("x.pkl", "r")


def test_gold_brief_and_lint_nudge():
    from datamate.gold import agent_tables, gold_brief
    from datamate.loader import load_shop
    from datamate.lint import LintHints, lint
    G = agent_tables(load_shop("data/train/data/biz_101"), "2026-06-30")
    brief = gold_brief(G)
    assert "G['don']" in brief and "counts_revenue" in brief and "{" not in brief
    h = LintHints(gold_tables=set(G["_bao_cao"]["tables"]))
    raw = "df = T['orders']\ndf[df['st'] == 'xong']['tong'].sum()"
    assert any("G['don']" in n for n in lint(raw, h))
    assert not any("G['don']" in n for n in lint("G['don']['revenue'].sum()", h))
    assert not any("G['don']" in n for n in lint(raw, LintHints()))  # shop không có G: không nhắc
