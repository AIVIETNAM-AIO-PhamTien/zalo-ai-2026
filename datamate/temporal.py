"""Evidence-bearing temporal windows and executable half-open predicates.

This module resolves a temporal phrase, not the metric or the whole question.
Period policies are supplied by a tested rulebook adapter, never inferred from data.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
import unicodedata
from dataclasses import dataclass

from datamate.ir import PlanError


def iso_date(value):
    if type(value) is dt.date:
        return value
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise PlanError("temporal_reference", "reference must be a complete ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise PlanError("temporal_reference", "invalid calendar date") from exc


@dataclass(frozen=True)
class ReferenceDate:
    date: dt.date
    source: str


def reference_date(question=None, rulebook=None, calendar_reference=None, explicit=None):
    for source, value in (("question.as_of_date", question), ("rulebook", rulebook),
                          ("calendar", calendar_reference), ("question.evidence", explicit)):
        if value is not None:
            return ReferenceDate(iso_date(value), source)
    raise PlanError("temporal_reference", "relative period has no evidenced reference date")


@dataclass(frozen=True)
class PeriodPolicy:
    week_start: int
    current_to_date: bool
    compare_month_elapsed_days: bool
    evidence: str

    def __post_init__(self):
        if (type(self.week_start) is not int or not 0 <= self.week_start <= 6
                or type(self.current_to_date) is not bool or type(self.compare_month_elapsed_days) is not bool
                or not self.evidence):
            raise ValueError("explicit valid period policy and evidence required")


@dataclass(frozen=True)
class TimeWindow:
    start: dt.date
    end_exclusive: dt.date
    phrase: str
    reference: ReferenceDate | None = None
    policy_evidence: str | None = None

    def __post_init__(self):
        if self.start >= self.end_exclusive:
            raise PlanError("temporal_range", "empty or reversed time interval")

    def predicate(self, column):
        def comparison(op, value):
            return {"kind": "binary", "op": op, "left": {"kind": "column", "name": column},
                    "right": {"kind": "literal", "type": "date", "value": value.isoformat()}}
        return {"kind": "binary", "op": "and", "left": comparison("ge", self.start),
                "right": comparison("lt", self.end_exclusive)}

    def apply(self, relation, column):
        return {"kind": "filter", "input": relation, "predicate": self.predicate(column)}


def month_start(year, month):
    try:
        return dt.date(year, month, 1)
    except ValueError as exc:
        raise PlanError("temporal_range", "invalid month/year") from exc


def next_month(value):
    return month_start(value.year + (value.month == 12), value.month % 12 + 1)


def resolve_window(phrase: str, reference: ReferenceDate | None = None,
                   policy: PeriodPolicy | None = None) -> TimeWindow:
    # Date punctuation is syntax; fuzzy-name normalization would destroy it.
    text = unicodedata.normalize("NFD", phrase.lower())
    text = " ".join("".join(c for c in text if unicodedata.category(c) != "Mn").replace("đ", "d").split())
    one = dt.timedelta(days=1)
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            start = iso_date(text)
            return TimeWindow(start, start + one, phrase)
        match = re.fullmatch(r"tu (\d{4}-\d{2}-\d{2}) den (\d{4}-\d{2}-\d{2})", text)
        if match:
            return TimeWindow(iso_date(match[1]), iso_date(match[2]) + one, phrase)
        match = re.fullmatch(r"thang (\d{1,2})(?:/| nam )(\d{4})", text)
        if match:
            start = month_start(int(match[2]), int(match[1]))
            return TimeWindow(start, next_month(start), phrase)
        if reference is None:
            raise PlanError("temporal_reference", "relative or unresolved period requires reference evidence")
        today = reference.date
        if text in {"hom nay", "hom qua"}:
            start = today - (one if text == "hom qua" else dt.timedelta())
            return TimeWindow(start, start + one, phrase, reference)
        match = re.fullmatch(r"(\d+) ngay gan nhat", text)
        if match and 1 <= int(match[1]) <= 36600:
            return TimeWindow(today - dt.timedelta(days=int(match[1]) - 1), today + one, phrase, reference)
        if policy is None:
            raise PlanError("temporal_policy", "period policy must be grounded in the applicable rulebook")
        end = today + one
        if text == "thang nay":
            start = today.replace(day=1)
            if not policy.current_to_date:
                end = next_month(start)
        elif text in {"tuan nay", "tuan truoc"}:
            start = today - dt.timedelta(days=(today.weekday() - policy.week_start) % 7)
            if text == "tuan truoc":
                start, end = start - dt.timedelta(days=7), start
            elif not policy.current_to_date:
                end = start + dt.timedelta(days=7)
        elif text in {"thang truoc", "cung ky thang truoc"}:
            last = today.replace(day=1) - one
            start, end = last.replace(day=1), today.replace(day=1)
            if text == "cung ky thang truoc":
                if not policy.compare_month_elapsed_days:
                    raise PlanError("temporal_policy", "elapsed-day month comparison is not authorized")
                end = last.replace(day=min(today.day, calendar.monthrange(last.year, last.month)[1])) + one
        elif text == "tu dau nam":
            start = today.replace(month=1, day=1)
        else:
            raise PlanError("temporal_unsupported", "temporal phrase is not fully supported: " + phrase)
        return TimeWindow(start, end, phrase, reference, policy.evidence)
    except OverflowError as exc:
        raise PlanError("temporal_range", "date outside supported calendar range") from exc
