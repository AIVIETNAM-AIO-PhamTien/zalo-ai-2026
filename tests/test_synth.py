import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from datamate.loader import load_shop
from datamate.synth.questions import Oracle, generate
from datamate.synth.render import Style, write_shop
from datamate.synth.simulate import simulate
from datamate.synth.worlds import ARCHETYPES

D = dt.date


@pytest.fixture(scope="module")
def world():
    arch = dataclasses.replace(next(a for a in ARCHETYPES if a.key == "thu_cung"), orders_per_day=3)
    return simulate(arch, "syn_t", seed=7)


def test_world_follows_rulebook_shapes(world):
    w = world
    assert w.start <= w.as_of <= w.end
    assert set(w.orders["status"]) <= {"completed", "cancelled", "returned", "delivering"}
    ln = w.lines
    assert ((ln["unit_price"] * ln["qty"] - ln["line_discount"]) == ln["amount"]).all()
    assert (ln["bqty"] == ln["qty"] * ln["factor"]).all()
    # đơn trả hàng có ít nhất một dòng trả; đơn khác không có
    ret = set(ln.loc[ln["ret_qty"] > 0, "order_id"])
    assert ret == set(w.orders.loc[w.orders["status"] == "returned", "order_id"])
    # ghi nợ chỉ có ở khách có mã
    assert w.orders.loc[w.orders["payment"] == "Ghi nợ", "cust_id"].notna().all()


def test_oracle_consistency(world):
    st = Style.sample(np.random.default_rng(0), world)
    O = Oracle(world, st)
    p = (world.start, world.end)
    total = O.revenue(p, world.as_of)
    by_group = sum(O.revenue(p, world.as_of, group=g) for g in world.items["group"].unique())
    assert total == by_group > 0
    assert O.revenue(p, world.as_of) == O.day_revenue(p, world.as_of).sum()
    owed, n = O.debt_owed(world.as_of)
    assert owed >= 0 and n >= 0


def test_generate_and_write_roundtrip(world, tmp_path):
    rng = np.random.default_rng(3)
    st = Style.sample(rng, world)
    st.fmt = "csv"
    write_shop(world, st, tmp_path / "syn_t")
    shop = load_shop(tmp_path / "syn_t")
    assert len(shop.tables) >= 3 and "Dữ liệu có đến hết" in shop.readme
    qs = generate(world, st, rng, 40)
    assert len(qs) == 40 and len({q for q, _, _ in qs}) == 40
    for q, rec, intent in qs:
        assert rec["type"] in ("number", "percent", "date", "list", "string", "unanswerable", "boolean")
