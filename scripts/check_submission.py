"""Kiểm tra submission.jsonl trước khi nộp: đủ mọi id, không trùng, mỗi dòng đúng format.

    python scripts/check_submission.py <submission.jsonl> <questions.jsonl>
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.answer import normalize_answer  # noqa: E402


def check(sub_path, q_path) -> int:
    qids = [json.loads(l)["id"] for l in Path(q_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, bad = [], 0
    for i, l in enumerate(Path(sub_path).read_text(encoding="utf-8").splitlines(), 1):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            print(f"dòng {i}: không phải JSON")
            bad += 1
            continue
        rows.append(r)
        body = {k: v for k, v in r.items() if k != "id"}
        if "id" not in r or normalize_answer(body) is None:
            print(f"dòng {i}: sai format: {l[:200]}")
            bad += 1
    ids = Counter(r.get("id") for r in rows)
    dup = [k for k, c in ids.items() if c > 1]
    missing = set(qids) - set(ids)
    extra = set(ids) - set(qids)
    types = Counter(r.get("type") for r in rows)
    print(f"{len(rows)} dòng / {len(qids)} câu; thiếu {len(missing)}, thừa {len(extra)}, trùng {len(dup)}, sai format {bad}")
    print("type:", dict(types.most_common()))
    return 0 if not (missing or extra or dup or bad) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(check(sys.argv[1], sys.argv[2]))
