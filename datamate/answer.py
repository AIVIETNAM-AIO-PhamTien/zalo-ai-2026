"""Chuẩn hoá đáp án agent về đúng format nộp (rulebook §9) và tạo khoá so sánh để vote."""
from __future__ import annotations

import json
import datetime as dt
import math
import re

from datamate.helpers import round2, round_money
from datamate.scoring import normalize_string

_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, str):
        s = x.strip()
        if not re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
            return None
        x = float(s)
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    if float(x).is_integer():
        return int(x)
    return round2(x)


def _cell(x):
    if isinstance(x, str):
        return x.strip()
    n = _num(x)
    return n if n is not None else None


def _money(x):
    """Rulebook: tiền (kể cả trung bình của tiền) làm tròn về đồng, nửa đồng lên. Số lẻ từ 1000 trở lên coi là tiền;
    log agent Colab B9 v4: 72270.25 / 445333.33 / 69726.28 lệch đáp án hoặc làm vỡ đồng thuận giữa các mẫu."""
    if isinstance(x, float) and abs(x) >= 1000 and x != int(x):
        return round_money(x)
    return x


def normalize_answer(a) -> dict | None:
    """Trả về dict hợp lệ để ghi submission, hoặc None nếu sai format."""
    if not isinstance(a, dict):
        return None
    t = a.get("type")
    if t == "unanswerable":
        r = a.get("reason", "")
        return {"type": "unanswerable", "reason": r if isinstance(r, str) else str(r)}
    if t == "abstain":
        return {"type": "abstain"}
    v = a.get("answer")
    if t == "number":
        n = _num(v)
        return None if n is None else {"type": t, "answer": _money(n)}
    if t == "percent":
        n = _num(v)
        return None if n is None else {"type": t, "answer": round2(n)}
    if t == "string":
        return {"type": t, "answer": v.strip()} if isinstance(v, str) and v.strip() else None
    if t == "date":
        if not isinstance(v, str) or not _DATE.fullmatch(v.strip()):
            return None
        try:
            dt.date.fromisoformat(v.strip() + "-01" if len(v.strip()) == 7 else v.strip())
        except ValueError:
            return None
        return {"type": t, "answer": v.strip()}
    if t == "boolean":
        if isinstance(v, bool):
            return {"type": t, "answer": v}
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return {"type": t, "answer": v.strip().lower() == "true"}
        return None
    if t == "list":
        if not isinstance(v, list) or not v:
            return None
        cells = [_cell(x) for x in v]
        return None if any(c is None for c in cells) else {"type": t, "answer": [_money(c) for c in cells]}
    if t == "table":
        if not isinstance(v, dict) or not isinstance(v.get("columns"), list) or not isinstance(v.get("rows"), list):
            return None
        cols = [str(c) for c in v["columns"]]
        rows = []
        for r in v["rows"]:
            if not isinstance(r, list) or len(r) != len(cols):
                return None
            cells = [_cell(x) for x in r]
            if any(c is None for c in cells):
                return None
            rows.append(cells)
        return {"type": t, "answer": {"columns": cols, "rows": rows}}
    return None


def _key_cell(x):
    return normalize_string(x) if isinstance(x, str) else x


def answer_key(a: dict | None) -> str | None:
    """Hai đáp án cùng khoá thì được chấm như nhau (chuỗi chuẩn hoá, hàng bảng không xét thứ tự)."""
    if a is None:
        return None
    t = a["type"]
    if t in ("unanswerable", "abstain"):
        return t
    v = a["answer"]
    if t == "string":
        v = normalize_string(v)
    elif t == "list":
        v = [_key_cell(x) for x in v]
    elif t == "table":
        columns = [normalize_string(c) for c in v['columns']]
        # Equal cell values under different measure columns are not agreement.
        order = sorted(range(len(columns)), key=lambda i: columns[i]) if len(set(columns)) == len(columns) else list(range(len(columns)))
        v = [[columns[i] for i in order], sorted(json.dumps([_key_cell(r[i]) for i in order], ensure_ascii=False) for r in v['rows'])]
    return json.dumps([t, v], ensure_ascii=False)
