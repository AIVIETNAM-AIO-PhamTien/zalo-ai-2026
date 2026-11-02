"""Offline evaluation only: grouped retrospective splits and full PA accounting."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

from datamate.scoring import score_item


def business_split(questions: list[dict], salt: str = "datamate2", validation_fraction: float = 0.25):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction outside (0,1)")
    assignment = {}
    for question in questions:
        business = question["business_id"]
        value = int(hashlib.sha256((salt + "\0" + business).encode()).hexdigest()[:16], 16) / 2**64
        assignment[business] = "validation" if value < validation_fraction else "development"
    return {split: [q["id"] for q in questions if assignment[q["business_id"]] == split]
            for split in ("development", "validation")}


def metrics(scores):
    n = len(scores)
    correct, wrong, abstain = scores.count(1), scores.count(-1), scores.count(0)
    answered = correct + wrong
    return {"total": n, "correct": correct, "wrong": wrong, "abstain": abstain,
            "pa": (correct - wrong) / n if n else 0,
            "precision": correct / answered if answered else None,
            "coverage": answered / n if n else 0}


def evaluate(predictions: list[dict], references: list[dict], questions: list[dict],
             previous: list[dict] | None = None, traces: list[dict] | None = None):
    def unique(rows):
        out = {}
        for row in rows:
            if row["id"] in out:
                raise ValueError("duplicate question id")
            out[row["id"]] = row
        return out
    preds, refs, qs = unique(predictions), unique(references), unique(questions)
    if set(refs) != set(qs) or set(preds) - set(qs):
        raise ValueError("evaluation ID scope mismatch")
    old = unique(previous) if previous is not None else None
    scores, by_business, by_type, errors = [], defaultdict(list), defaultdict(list), defaultdict(int)
    regressions, recoveries = 0, 0
    for qid, ref in refs.items():
        pred = preds.get(qid)
        score = score_item(pred, ref)
        scores.append(score)
        by_business[qs[qid]["business_id"]].append(score)
        by_type[ref["type"]].append(score)
        if score == -1:
            errors["answer_type" if pred.get("type") != ref["type"] else "value_or_semantics"] += 1
        elif score == 0:
            errors["missing_prediction" if pred is None else "abstain"] += 1
        if old is not None:
            before = score_item(old.get(qid), ref)
            regressions += before == 1 and score == -1
            recoveries += before == -1 and score == 1
    costs = None
    if traces is not None:
        costs = {key: sum(t.get(key, 0) for t in traces) for key in ("tokens", "llm_calls", "runtime_seconds")}
    return {"metrics": metrics(scores), "by_business": {b: metrics(s) for b, s in by_business.items()},
            "by_answer_type": {t: metrics(s) for t, s in by_type.items()}, "errors": dict(errors),
            "correct_to_wrong": regressions if old is not None else None,
            "wrong_to_correct": recoveries if old is not None else None, "costs": costs,
            "calibration": "not fitted; evidence-only policy",
            "split_status": "retrospective training validation, not truly unseen"}


def reliability(probabilities: list[float], correct: list[bool], bins: int = 10):
    if len(probabilities) != len(correct) or bins < 1:
        raise ValueError("invalid reliability data")
    grouped = defaultdict(list)
    for p, y in zip(probabilities, correct):
        if not math.isfinite(p) or not 0 <= p <= 1 or not isinstance(y, bool):
            raise ValueError("probability or label invalid")
        grouped[min(int(p * bins), bins - 1)].append((p, int(y)))
    result = []
    for index, rows in sorted(grouped.items()):
        result.append({"bin": index, "n": len(rows), "mean_probability": sum(p for p, _ in rows) / len(rows),
                       "observed_accuracy": sum(y for _, y in rows) / len(rows)})
    return {"bins": result,
            "brier": sum((p - y) ** 2 for p, y in zip(probabilities, correct)) / len(correct) if correct else None,
            "ece": sum(r["n"] * abs(r["mean_probability"] - r["observed_accuracy"]) for r in result) / len(correct) if correct else None}
