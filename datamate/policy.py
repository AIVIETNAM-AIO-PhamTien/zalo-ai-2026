"""Evidence-based selection without consensus-as-truth or invented probabilities."""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, localcontext

from datamate.engine import ExecutionResult
from datamate.verifier import Verification


@dataclass
class Candidate:
    plan: dict
    result: ExecutionResult
    verification: Verification
    strategy: str
    tokens: int = 0
    llm_calls: int = 0
    runtime_seconds: float = 0


def cell(value, percent=False):
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("nonfinite decimal")
        with localcontext() as ctx:
            ctx.prec = 80
            value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if percent else value
            if value == value.to_integral_value():
                return int(value)
            converted = float(value)
            if not math.isfinite(converted) or Decimal(str(converted)) != value:
                raise ValueError("cannot represent exact decimal in official JSON numeric value")
            return converted
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, (str, bool, int)):
        return value
    raise ValueError("unsupported or unverified answer cell")


def format_answer(candidate: Candidate) -> dict:
    if not candidate.verification.accepted:
        raise ValueError("candidate is not verified")
    result = candidate.result
    rows = candidate.verification.exact_result if candidate.verification.exact_result is not None else result.rows
    typ = candidate.plan["answer_type"]
    if typ == "table":
        return {"type": typ, "answer": {"columns": result.columns, "rows": [[cell(v) for v in r] for r in rows]}}
    if typ == "list":
        if len(result.columns) != 1:
            raise ValueError("list requires one column")
        return {"type": typ, "answer": [cell(r[0]) for r in rows]}
    if len(rows) != 1 or len(result.columns) != 1:
        raise ValueError("scalar answer shape mismatch")
    value = cell(rows[0][0], percent=typ == "percent")
    if typ in {"number", "percent"} and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError("numeric output type mismatch")
    if typ == "boolean" and not isinstance(value, bool):
        raise ValueError("boolean output type mismatch")
    if typ in {"string", "date"} and not isinstance(value, str):
        raise ValueError("text output type mismatch")
    if typ == "date":
        dt.date.fromisoformat(value)
    return {"type": typ, "answer": value}


@dataclass
class Selection:
    answer: dict
    reason: str
    selected: int | None = None


def select(candidates: list[Candidate]) -> Selection:
    accepted = []
    for i, candidate in enumerate(candidates):
        try:
            answer = format_answer(candidate)
        except (ValueError, ArithmeticError):
            continue
        accepted.append((i, candidate, answer))
    if not accepted:
        return Selection({"type": "abstain"}, "no fully grounded candidate")
    # Same answer can hide different sources/filters/grain; compare plans as well.
    signatures = {json.dumps({"relation": c.plan["relation"], "contracts": c.plan["contract_ids"], "answer": a},
                             sort_keys=True, ensure_ascii=False) for _, c, a in accepted}
    if len(signatures) != 1:
        return Selection({"type": "abstain"}, "conflicting grounded interpretations require more evidence")
    i, _, answer = min(accepted, key=lambda item: (item[1].llm_calls, item[1].tokens))
    return Selection(answer, "structural, semantic and execution evidence passed", i)


@dataclass
class Budget:
    max_candidates: int = 3
    max_repairs: int = 2
    max_tokens: int = 8192
    max_seconds: float = 30

    def next_action(self, candidates: list[Candidate], repairs: int, elapsed: float) -> str:
        if candidates and select(candidates).selected is not None:
            return "answer"
        if (elapsed >= self.max_seconds or sum(c.tokens for c in candidates) >= self.max_tokens
                or len(candidates) >= self.max_candidates):
            return "abstain"
        if candidates and repairs < self.max_repairs:
            return "repair"
        return "explore" if not candidates else "abstain"
