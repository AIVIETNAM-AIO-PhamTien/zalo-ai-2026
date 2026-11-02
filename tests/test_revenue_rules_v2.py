import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from datamate.engine import execute
from datamate.ir import PlanError
from datamate.revenue_rules import SUPPORTED, revenue_policy, revenue_relation
from datamate.temporal import resolve_window
from datamate.verifier import ExactEvaluator
from test_engine_v2 import plan


def documents():
    return {"$rulebook": "\n".join(f"| {k} | {v} |" for k, v in SUPPORTED.items())}


def source():
    frame = pd.DataFrame({"ref": ["a", "a", "b", "b", "c", "d"],
        "net": [Decimal(v) for v in (10, 20, 100, 5, 40, 80)],
        "placed": [dt.datetime(2026, 2, 1)] * 5 + [dt.datetime(2026, 3, 1)],
        "state": ["completed", "completed", "completed", "returned", "shipping", "completed"],
        "vat": [999] * 6, "order_discount": [999] * 6}, dtype=object)
    return frame


def relation(frame):
    return revenue_relation(revenue_policy(documents()), "lines", frame,
        {"order": "ref", "amount": "net", "placed": "placed", "status": "state"},
        {"completed", "shipping"}, {"returned", "cancelled"}, resolve_window("tháng 2/2026"))


def test_partial_return_excludes_whole_order_and_shipping_is_included():
    frame = source()
    rel = relation(frame)
    schemas = {"lines": {"ref": "text", "net": "number", "placed": "date", "state": "text", "vat": "number", "order_discount": "number"}}
    assert execute(plan(rel), {"lines": frame}, schemas).rows == [(Decimal(70),)]
    assert ExactEvaluator({"lines": frame}).evaluate(rel)[1] == [(Decimal(70),)]


@pytest.mark.parametrize("mutation", ["unknown_status", "missing_amount", "conflicting_dates", "missing_id"])
def test_ambiguous_source_preconditions_fail(mutation):
    frame = source()
    if mutation == "unknown_status":
        frame.loc[0, "state"] = "maybe done"
    elif mutation == "missing_amount":
        frame.loc[0, "net"] = None
    elif mutation == "conflicting_dates":
        frame.loc[1, "placed"] = dt.datetime(2026, 3, 1)
    else:
        frame.loc[0, "ref"] = None
    with pytest.raises(PlanError):
        relation(frame)


def test_changed_rules_and_partial_rulebooks_are_not_accepted():
    docs = documents()
    docs["$rulebook"] = docs["$rulebook"].replace("không cộng thuế VAT", "cộng thuế VAT")
    with pytest.raises(PlanError):
        revenue_policy(docs)
    with pytest.raises(PlanError):
        revenue_policy({"$rulebook": ""})


def test_no_eligible_orders_is_valid_zero_not_missing_data():
    frame = source()
    frame["state"] = "cancelled"
    rel = relation(frame)
    schema = {"lines": {"ref": "text", "net": "number", "placed": "date", "state": "text", "vat": "number", "order_discount": "number"}}
    assert execute(plan(rel), {"lines": frame}, schema).rows == [(0,)]
    assert ExactEvaluator({"lines": frame}).evaluate(rel)[1] == [(0,)]


def test_train_rulebook_policy_is_recognized_without_reading_answers():
    from pathlib import Path
    path = Path("data/train/rulebook.md")
    if not path.exists():
        pytest.skip("training rulebook unavailable")
    docs = {"$rulebook": path.read_text(encoding="utf-8")}
    policy = revenue_policy(docs)
    assert len(policy.evidence) == 3
    assert all(e.valid(docs) for e in policy.evidence)
