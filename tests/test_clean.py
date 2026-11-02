import pandas as pd

from datamate.clean import entity_groups, infer_contracts


def test_entity_groups_merges_spelling_variants_only():
    vals = ["Trà Sữa"] * 5 + ["tra sua", "TRÀ SỮA ", "Trà Sũa"] + ["Size 1", "Size 2"] * 3
    m = entity_groups(vals)
    assert m["tra sua"] == "Trà Sữa" and m["TRÀ SỮA "] == "Trà Sữa"
    assert m["Trà Sũa"] == "Trà Sữa"  # chỉ khác dấu
    from datamate.clean import near_duplicates
    m2 = entity_groups(["Trà Sữa", "Trà Sữaa"])
    assert m2 == {}  # khác chữ: rulebook coi là hai thứ, không gộp
    assert near_duplicates(["Trà Sữa ô long", "Trà Sữa ô lông", "Trà Sữa ô longg"]) == [("tra sua o long", "tra sua o longg")]
    assert "Size 1" not in m and "Size 2" not in m


def test_infer_contracts_flags_rare_violations():
    df = pd.DataFrame({"qty": [1] * 99 + [-3], "id": [f"D{i}" for i in range(99)] + ["D5"]})
    c = infer_contracts(df, {"qty": "int", "id": "text"})
    assert c["qty"]["non_negative"] == [99]
    assert c["id"]["explained"]["unique_key"]["by"] == "one_to_many"  # hai dòng khác nhau: một-nhiều


def test_unseen_messy_shop_loads_clean(tmp_path):
    """Dữ liệu kiểu private chưa từng thấy: header tiếng Anh, ký tự vô hình, NBSP, biến thể tên, dòng trùng, số âm lạ."""
    import random

    from datamate.loader import load_shop
    from tests.test_loader import _xlsx
    rnd = random.Random(0)
    names = ["Bánh Mì Pate", "banh mi pate", "BÁNH MÌ PATE ", "Ca\u0300 phe\u0302 sư\u0303a", "Cà phê sữa"]
    rows = [("SALES LOG",), (), ("Invoice No", "Date", "Item\u200b", "Qty", "Amount")]
    for i in range(200):
        rows.append((f"INV{i // 2}", f"{1 + i % 28:02d}/03/2026", "\u00a0" + rnd.choice(names) + "\t",
                     -5 if i == 7 else rnd.randint(1, 4), f"{rnd.randint(10, 90)}.000\u00a0đ"))
    rows.append(rows[-1])
    biz = tmp_path / "biz_unseen"
    biz.mkdir()
    _xlsx(biz / "sales.xlsx", {"Sheet1": rows})
    shop = load_shop(biz)
    (t, df), = shop.tables.items()
    meta = shop.meta[t]
    assert "error" not in meta["clean"]
    text = [v for c in df.columns if df[c].dtype == object for v in df[c].dropna() if isinstance(v, str)]
    assert all(v == v.strip() and "\u00a0" not in v and "\u200b" not in v for v in text)
    item = next(c for c in df.columns if c.startswith("Item"))
    ent = meta["clean"]["entities"][item]
    canon = {ent.get(v, v) for v in df[item].dropna()}
    assert len(canon) == 2
    assert meta["quality"]["exact_duplicate_rows"] >= 1


def test_violation_explained_by_category_is_not_an_error():
    df = pd.DataFrame({"qty": [2] * 197 + [-1] * 3, "st": ["ok"] * 197 + ["return"] * 3})
    c = infer_contracts(df, {"qty": "int", "st": "text"})
    assert "non_negative" not in c["qty"] and c["qty"]["explained"]["non_negative"]["value"] == "returned"


def test_return_rows_with_varied_status_spelling_are_explained():
    st = ["Hoàn thành"] * 197 + ["Trả hàng", "trả lại", "Đổi trả"]
    df = pd.DataFrame({"qty": [2] * 197 + [-1] * 3, "st": st})
    c = infer_contracts(df, {"qty": "int", "st": "text"})
    assert c["qty"]["explained"]["non_negative"]["value"] == "returned"


def test_one_to_many_key_with_category_does_not_crash():
    from datamate.clean import clean_profile
    ids = [f"D{i}" for i in range(199)] + ["D3"]
    df = pd.DataFrame({"id": ids, "pay": list(range(200)), "kind": ["cash"] * 200})
    prof = clean_profile(df, {"id": "text", "pay": "int", "kind": "text"})
    assert "error" not in prof and prof["contracts"]["id"]["explained"]["unique_key"]["by"] == "one_to_many"


def test_cleaner_follows_rulebook_dedup_and_name_policy():
    import pandas as pd
    from datamate.loader import Shop
    from datamate.rule_policy import infer
    from datamate.semantic import _order_customers, clean_order_rows
    rb = open("data/train/rulebook.md", encoding="utf-8").read()
    pol, _ = infer(rb)
    assert pol.dedup_same_order and pol.merge_name_variants  # train: luật mặc định
    changed = rb.replace("lỗi xuất file, tính một lần", "mỗi dòng là một lần bán, tính cả hai").replace(
        "hai tên **chỉ khác cách viết** là cùng một khách", "mỗi cách viết là một khách riêng")
    pol2, resolved = infer(changed)
    assert not pol2.dedup_same_order and not pol2.merge_name_variants and len(resolved) == 2
    df = pd.DataFrame({"Mã đơn": ["A", "A"], "Hàng": ["x", "x"]})
    shop = Shop("s", "", {"t": df}, {})
    assert len(clean_order_rows(shop, "t")) == 1
    shop.rule_policy = pol2
    assert len(clean_order_rows(shop, "t")) == 2
    names = pd.DataFrame({"Mã đơn": ["1", "2"], "Tên khách": ["Chị Lan", "chị lan"]})
    assert _order_customers([(names, "Mã đơn")])["customer"].nunique() == 1
    assert _order_customers([(names, "Mã đơn")], merge_variants=False)["customer"].nunique() == 2
