"""Bộ chấm Penalized Accuracy cục bộ, theo luật chấm của đề và rulebook §9.

Mỗi câu: +1 đúng, 0 abstain (hoặc thiếu / dòng không đọc được), -1 sai.
"""
from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

VALUE_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table"}
FLOAT_TOL = 1e-6


def normalize_string(s: str) -> str:
    """NFC → lowercase → ký tự không phải chữ/số/khoảng trắng thành khoảng trắng → gộp khoảng trắng.
    Giữ dấu tiếng Việt."""
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(s.split())


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _num_eq(p, r) -> bool:
    # chỉ bỏ qua sai số dấu phẩy động: tiền đúng từng đồng, đếm đúng từng đơn vị
    return _is_number(p) and _is_number(r) and abs(p - r) <= max(FLOAT_TOL, 1e-12 * abs(r))


def _pct_eq(p, r) -> bool:
    return _is_number(p) and _is_number(r) and abs(round(p, 2) - round(r, 2)) <= FLOAT_TOL


def _as_alias_list(alias) -> list:
    if alias is None:
        return []
    return alias if isinstance(alias, list) else [alias]


def _str_eq(p, r, alias=None) -> bool:
    if not isinstance(p, str) or not isinstance(r, str):
        return False
    np_ = normalize_string(p)
    accepted = [r] + [a for a in _as_alias_list(alias) if isinstance(a, str)]
    return any(np_ == normalize_string(a) for a in accepted)


def _cell_eq(p, r, alias=None) -> bool:
    if isinstance(r, str):
        return _str_eq(p, r, alias)
    if _is_number(r):
        return _num_eq(p, r)
    return p == r


def _list_eq(p, r, alias) -> bool:
    if not isinstance(p, list) or len(p) != len(r):
        return False
    alias = alias if isinstance(alias, list) else []
    for i, (pi, ri) in enumerate(zip(p, r)):
        if not _cell_eq(pi, ri, alias[i] if i < len(alias) else None):
            return False
    return True


def _table_eq(p, r, alias) -> bool:
    if not isinstance(p, dict) or not isinstance(p.get("columns"), list) or not isinstance(p.get("rows"), list):
        return False
    rcols, pcols = r["columns"], p["columns"]
    if len(rcols) != len(pcols):
        return False
    rnorm = [normalize_string(str(c)) for c in rcols]
    pnorm = [normalize_string(str(c)) for c in pcols]
    # cột khớp theo tên đã chuẩn hoá; tên lệch thì theo vị trí
    order = [pnorm.index(c) for c in rnorm] if sorted(rnorm) == sorted(pnorm) and len(set(pnorm)) == len(pnorm) \
        else list(range(len(rcols)))
    prows = []
    for row in p["rows"]:
        if not isinstance(row, list) or len(row) != len(rcols):
            return False
        prows.append([row[j] for j in order])
    rrows = r["rows"]
    if len(prows) != len(rrows):
        return False
    alias = alias if isinstance(alias, list) else []
    used = [False] * len(prows)
    # hàng khớp như tập: ghép từng hàng reference với một hàng dự đoán chưa dùng
    for i, rrow in enumerate(rrows):
        row_alias = alias[i] if i < len(alias) else None
        for j, prow in enumerate(prows):
            if used[j]:
                continue
            if all(_cell_eq(pc, rc, row_alias if k == 0 else None) for k, (pc, rc) in enumerate(zip(prow, rrow))):
                used[j] = True
                break
        else:
            return False
    return True


def _is_blank(pred) -> bool:
    if pred is None or not isinstance(pred, dict):
        return True
    if "reason" in pred and not isinstance(pred["reason"], str):
        return True  # rulebook §7: reason không phải chuỗi → dòng không đọc được
    return pred.get("type") == "abstain" or pred.get("type") not in VALUE_TYPES | {"unanswerable"}


def score_item(pred: dict | None, ref: dict) -> int:
    if pred is None or _is_blank(pred):
        return 0
    ptype, rtype = pred["type"], ref["type"]
    if ptype != rtype:
        return -1
    if rtype == "unanswerable":
        return 1
    if "answer" not in pred:
        return -1
    p, r, alias = pred["answer"], ref["answer"], ref.get("alias")
    ok = {
        "number": lambda: _num_eq(p, r),
        "percent": lambda: _pct_eq(p, r),
        "string": lambda: _str_eq(p, r, alias),
        "date": lambda: isinstance(p, str) and p == r,
        "boolean": lambda: isinstance(p, bool) and p == r,
        "list": lambda: _list_eq(p, r, alias),
        "table": lambda: _table_eq(p, r, alias),
    }[rtype]()
    return 1 if ok else -1


def _read_jsonl(path) -> list:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append(None)
    return out


def score_file(submission_path, answers_path) -> dict:
    refs = {r["id"]: r for r in _read_jsonl(answers_path) if r}
    preds = {}
    for p in _read_jsonl(submission_path):
        if isinstance(p, dict) and "id" in p:
            preds[p["id"]] = p
    per_item = {qid: score_item(preds.get(qid), r) for qid, r in refs.items()}
    c = Counter(per_item.values())
    n = len(refs)
    return {
        "n": n, "correct": c[1], "wrong": c[-1], "abstain": c[0],
        "pa": (c[1] - c[-1]) / n if n else 0.0,
        "per_item": per_item,
    }
