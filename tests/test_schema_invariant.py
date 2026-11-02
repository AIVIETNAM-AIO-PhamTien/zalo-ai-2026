"""Bước 5: tên cột lạ → vai trò cột tiền chấm bằng bất biến số (tiền = SL × giá − giảm)."""
import numpy as np
import pandas as pd

from datamate.semantic import infer_meanings


def _lines(n=200, seed=0):
    rng = np.random.default_rng(seed)
    qty = rng.integers(1, 4, n)
    qty[: int(n * 0.9)] = 1  # phần lớn SL = 1: đổi vai giá ↔ tiền cũng gần khớp
    price = rng.choice([7000, 42000, 45000, 55000], n)
    disc = np.where(rng.random(n) < 0.2, 5000, 0)
    disc[:3] = [1000, 2000, 3000]
    return pd.DataFrame({"id_order": [f"HD{i // 2}" for i in range(n)], "stt": [i % 2 + 1 for i in range(n)],
                         "qty": qty, "qty_quy_doi": qty, "order_price": price, "giam_price": disc,
                         "amount": qty * price - disc, "price_von": price // 3})


def test_money_roles_from_invariant_with_odd_names():
    m = infer_meanings(_lines())
    assert m["id_order"] == "mã đơn"
    assert m["amount"] == "thành tiền sau giảm"
    assert m["order_price"] == "đơn giá"
    assert m["qty"] == "số lượng"


def test_no_money_role_when_invariant_fails():
    df = _lines()
    df["amount"] = df["amount"] * 1.3 + 17
    assert "thành tiền sau giảm" not in infer_meanings(df).values()
