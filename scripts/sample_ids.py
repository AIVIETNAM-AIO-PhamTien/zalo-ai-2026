"""In N id câu train lấy mẫu cân theo business (giữ phân bố type tự nhiên), cố định seed.

    python scripts/sample_ids.py data/train 200 > ids.txt
    python scripts/sample_ids.py data/train 400 1 > ids.txt   # seed khác → bộ câu khác
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def sample(split_dir: Path, n: int, seed: int = 0) -> list[str]:
    qs = {json.loads(l)["id"]: json.loads(l) for l in (split_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    strata = defaultdict(list)
    for qid, q in sorted(qs.items()):
        strata[q["business_id"]].append(qid)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    # vòng tròn qua các shop để shop nào cũng có câu
    out, keys = [], sorted(strata)
    while len(out) < min(n, len(qs)):
        for k in keys:
            if strata[k] and len(out) < n:
                out.append(strata[k].pop())
    return out


if __name__ == "__main__":
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print("\n".join(sample(Path(sys.argv[1]), int(sys.argv[2]), seed)))
