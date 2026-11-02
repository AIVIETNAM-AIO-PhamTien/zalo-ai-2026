"""Bộ chọn đã hiệu chỉnh (Bước 3, docs/KE_HOACH_PA_V1.md): mỗi ứng viên (tầng tất định, DSL, agent) → đặc trưng →
logistic P(đúng); chọn ứng viên P cao nhất, trả lời khi P > ngưỡng. Hệ số và ngưỡng học bằng
scripts/calibrate_select.py --save; run.py --selector dùng file đó."""
from __future__ import annotations

import json
import math
from pathlib import Path

from datamate.agent import calib_features, is_zero_answer
from datamate.answer import answer_key

SRC = ("det", "dsl", "agent")
TYPES = ("number", "percent", "string", "list", "table", "date", "boolean")


def ok(a) -> bool:
    return bool(a) and a.get("type") not in (None, "abstain", "unanswerable")


def key(a):
    try:
        return answer_key(a)
    except Exception:
        return json.dumps(a, ensure_ascii=False, sort_keys=True)


def _plans(info) -> list:
    return [p for p in (info or {}).get("plans") or [] if ok(p.get("answer"))]


def candidates(r):
    """→ [(nguồn, đáp án, đặc trưng riêng của nguồn)]."""
    out = []
    if r["det"]:
        out.append(("det", r["det"], []))
    plans = r["dsl"]
    if plans:
        k0 = key(plans[0]["answer"])
        raw = plans[0].get("raw") or ""
        out.append(("dsl", plans[0]["answer"],
                    [sum(key(p["answer"]) == k0 for p in plans) / len(plans), float('"bo_sot": []' not in raw),
                     raw.count('"fn"') / 3]))
    d = r["agent"]
    if d:
        cand, x = calib_features(d.get("answers") or [], d.get("traces") or [], bool(d.get("reconciled")))
        if cand is not None:
            out.append(("agent", cand, x[1:10]))
    return out


WIDTH = {"det": 0, "dsl": 3, "agent": 9}
CTX = 5  # đặc trưng chất lượng dữ liệu / đường đọc / luật (Bước 9 A1, A8, B6, H4, H5)


def shop_context(shop, rulebook: str | None = None) -> dict:
    """Đặc trưng theo shop: không dựng được gold (A8), tỉ lệ đơn thiếu tiền trong gold (A1), rulebook có dòng luật
    đổi mà chưa diễn đạt được (H5). Lỗi → coi như thiếu gold."""
    from datamate.gold import agent_tables
    changed = 0.0
    if rulebook:
        from datamate.rule_guard import Guard
        changed = float(bool(Guard(rulebook, policy_aware=True).changed))
    try:
        g = agent_tables(shop)
    except Exception:
        return {"gold_missing": 1.0, "gold_nan": 0.0, "rule_changed": changed}
    o = g.get("don")
    if o is None or o.empty:
        return {"gold_missing": 1.0, "gold_nan": 0.0, "rule_changed": changed}
    return {"gold_missing": 0.0, "gold_nan": float(o["revenue"].isna().mean()), "rule_changed": changed}


def raw_reads(info) -> float:
    """B6: tỉ lệ lượt code của agent đọc bảng thô T (cổng thoát) — 0 khi chỉ dùng SEM / G."""
    turns = [t.get("reply") or "" for tr in (info or {}).get("traces") or [] for t in tr if "```" in (t.get("reply") or "")]
    return sum("T[" in t for t in turns) / len(turns) if turns else 0.0


def guard_violations(question: str, info) -> float:
    """H4: tỉ lệ mẫu agent có code vi phạm guard rulebook (arbiter.violations: thiếu valid_order, thiếu lọc tồn…).
    Đặc trưng, không phủ quyết (Bước 0: dùng làm cổng chặn nhầm nhiều câu đúng)."""
    from datamate.agent import _last_code
    from datamate.arbiter import violations
    pairs = [(a, t) for a, t in zip((info or {}).get("answers") or [], (info or {}).get("traces") or []) if a]
    if not question or not pairs:
        return 0.0
    return sum(bool(violations(question, a, _last_code(t)[0])) for a, t in pairs) / len(pairs)


def ctx_vec(src, r) -> list[float]:
    c = r.get("ctx") or {}
    agent = src == "agent"
    return [float(c.get("gold_missing", 0.0)), float(c.get("gold_nan", 0.0)),
            raw_reads(r.get("agent")) if agent else 0.0,
            guard_violations(r.get("question", ""), r.get("agent")) if agent else 0.0,
            float(c.get("rule_changed", 0.0))]


def vec(src, ans, own, others, ctx=None):
    k = key(ans)
    agree = sum(key(o) == k for _, o, _ in others)
    x = [1.0, *[float(src == s) for s in SRC], agree / 2, (len(others) - agree) / 2,
         float(any(s == "det" and key(o) == k for s, o, _ in others)),
         float(is_zero_answer(ans)), *[float(ans.get("type") == t) for t in TYPES]]
    for s in SRC:  # khối đặc trưng riêng, 0 khi không phải nguồn này
        x += own if s == src else [0.0] * WIDTH[s]
    if ctx is not None:  # cấu hình cũ (không có khối này) vẫn chạy: choose zip theo độ dài hệ số
        x += ctx
    return x


def choose(r: dict, cfg: dict) -> tuple[str | None, dict | None, float]:
    """r = {"det", "dsl", "agent"} như candidates; cfg = {"w", "thr"} → (nguồn, đáp án, P); P ≤ ngưỡng → (None, None, P)."""
    cs = candidates(r)
    best = (None, None, 0.0)
    for s, ans, own in cs:
        x = vec(s, ans, own, [c for c in cs if c[0] != s], ctx_vec(s, r) if cfg.get("ctx") else None)
        p = 1 / (1 + math.exp(-sum(wi * xi for wi, xi in zip(cfg["w"], x))))
        if p > best[2]:
            best = (s, ans, p)
    return best if best[2] > cfg["thr"] else (None, None, best[2])


def load_cfg(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
