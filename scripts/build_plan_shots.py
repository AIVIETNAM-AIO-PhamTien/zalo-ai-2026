"""Kho few-shot cho tầng kế hoạch (--plan) từ kho chương trình đúng (scripts/program_search.py).

    python scripts/build_plan_shots.py outputs/programs/train_v6.jsonl --out datamate/plan_shots.jsonl

Mỗi dòng {id, biz, type, question, as_of, plan}: chương trình có căn cứ đầu tiên của câu, bỏ tham số bảng / cột
(lúc chạy plan.fill_columns gắn cột của shop đang hỏi). `type` là ngành suy từ dữ liệu (datamate/shop_type.py) để
lúc chạy không lấy ví dụ cùng ngành với shop đang hỏi. Chỉ dựng từ split có đáp án.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.loader import CACHE_VERSION  # noqa: E402
from datamate.plan import COLUMN_ARGS  # noqa: E402
from datamate.shop_type import shop_type  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("programs")
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--out", default="datamate/plan_shots.jsonl")
    a = ap.parse_args()
    if "public" in a.programs:
        sys.exit("chỉ dựng từ split có đáp án")
    rows = [json.loads(l) for l in open(a.programs, encoding="utf-8") if l.strip()]
    types = {}
    out = []
    for r in rows:
        prog = next((p for p in r["programs"] if p.get("grounded")), None)
        if prog is None:
            continue
        b = r["business_id"]
        if b not in types:
            types[b] = shop_type(pickle.loads((Path(a.cache) / f"{b}.v{CACHE_VERSION}.pkl").read_bytes()))
        calls = [{"fn": c["fn"], "args": {k: v for k, v in c["args"].items() if k not in COLUMN_ARGS}}
                 for c in prog["calls"]]
        out.append({"id": r["id"], "biz": b, "type": types[b], "question": r["question"], "as_of": r["as_of_date"],
                    "plan": {"op": prog["op"], "calls": calls}})
    Path(a.out).write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in out), encoding="utf-8")
    by_type = {}
    for s in out:
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1
    print(f"{len(out)} ví dụ → {a.out}; theo ngành {by_type}")


if __name__ == "__main__":
    main()
