import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from datamate.engine import execute
from datamate.ir import PlanError
from datamate.temporal import PeriodPolicy, reference_date, resolve_window
from datamate.verifier import ExactEvaluator
from test_engine_v2 import aggregate, col, plan


POLICY = PeriodPolicy(0, True, True, "Synthetic rulebook: Monday weeks; periods to date; compare elapsed month days.")


@pytest.mark.parametrize("phrase,reference,start,end", [
    ("hôm qua", "2026-01-01", "2025-12-31", "2026-01-01"),
    ("tháng này", "2024-02-29", "2024-02-01", "2024-03-01"),
    ("cùng kỳ tháng trước", "2024-03-31", "2024-02-01", "2024-03-01"),
    ("tháng trước", "2026-01-02", "2025-12-01", "2026-01-01"),
    ("tuần này", "2026-10-29", "2026-10-26", "2026-10-30"),
    ("tuần trước", "2026-10-29", "2026-10-19", "2026-10-26"),
    ("7 ngày gần nhất", "2026-01-03", "2025-12-28", "2026-01-04"),
    ("từ đầu năm", "2026-10-29", "2026-01-01", "2026-10-30"),
])
def test_relative_windows(phrase, reference, start, end):
    window = resolve_window(phrase, reference_date(question=reference), POLICY)
    assert (window.start.isoformat(), window.end_exclusive.isoformat()) == (start, end)
    assert window.reference.source == "question.as_of_date"


def test_reference_priority_and_invalid_high_priority_never_falls_back():
    ref = reference_date(question="2026-01-01", rulebook="2025-01-01", calendar_reference="2024-01-01")
    assert ref.date == dt.date(2026, 1, 1)
    assert reference_date(rulebook="2025-01-01").source == "rulebook"
    with pytest.raises(PlanError):
        reference_date(question="invalid", rulebook="2025-01-01")
    with pytest.raises(PlanError):
        reference_date()


@pytest.mark.parametrize("phrase", ["2026-02-30", "tháng 13/2026", "từ 2026-02-02 đến 2026-01-01",
                                        "0 ngày gần nhất", "tháng này hoặc tháng trước"])
def test_invalid_or_partial_temporal_phrases_fail(phrase):
    with pytest.raises(PlanError):
        resolve_window(phrase, reference_date(question="2026-10-29"), POLICY)


def test_relative_period_requires_policy_and_future_data_is_not_a_clock():
    with pytest.raises(PlanError, match="policy"):
        resolve_window("tháng này", reference_date(question="2026-10-29"))
    with pytest.raises(PlanError, match="reference"):
        resolve_window("hôm nay")


def test_half_open_filter_is_executed_at_service_timestamp_boundaries():
    frames = {"repairs": pd.DataFrame({
        "when": [dt.datetime(2026, 1, 31, 23, 59, 59), dt.datetime(2026, 2, 1),
                 dt.datetime(2026, 2, 28, 23, 59, 59), dt.datetime(2026, 3, 1)],
        "minutes": [Decimal(1000), Decimal(10), Decimal(20), Decimal(2000)]}, dtype=object)}
    schemas = {"repairs": {"when": "date", "minutes": "number"}}
    window = resolve_window("tháng 2/2026")
    rel = aggregate(window.apply({"kind": "scan", "table": "repairs"}, "when"),
                    {"kind": "aggregate", "op": "sum", "value": col("minutes")})
    assert execute(plan(rel), frames, schemas).rows == [(Decimal(30),)]
    assert ExactEvaluator(frames).evaluate(rel)[1] == [(Decimal(30),)]


def test_explicit_range_includes_end_date_and_policy_is_business_specific():
    window = resolve_window("từ 2026-01-01 đến 2026-01-02")
    assert window.end_exclusive == dt.date(2026, 1, 3)
    policy = PeriodPolicy(6, False, False, "Synthetic: Sunday week, whole periods")
    window = resolve_window("tuần này", reference_date(question="2026-10-29"), policy)
    assert (window.start, window.end_exclusive) == (dt.date(2026, 10, 25), dt.date(2026, 11, 1))
    with pytest.raises(PlanError):
        resolve_window("cùng kỳ tháng trước", reference_date(question="2026-10-29"), policy)
