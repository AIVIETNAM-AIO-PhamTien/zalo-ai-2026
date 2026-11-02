"""Chấm submission.jsonl với answers.jsonl, in PA tổng và theo type đáp án.

    python scripts/score.py <submission.jsonl> <answers.jsonl> [--ids ids.txt]
"""
import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.scoring import score_file  # noqa: E402


def main(sub, ans, ids=None):
    lines = [l for l in Path(ans).read_text(encoding="utf-8").splitlines() if l.strip()]
    if ids:
        keep = set(Path(ids).read_text(encoding="utf-8").split())
        lines = [l for l in lines if json.loads(l)["id"] in keep]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        ans = fh.name
    res = score_file(sub, ans)
    print(f"PA={res['pa']:.4f}  n={res['n']}  correct={res['correct']}  wrong={res['wrong']}  abstain={res['abstain']}")
    types = {json.loads(l)["id"]: json.loads(l)["type"] for l in lines}
    by_type = defaultdict(list)
    for qid, s in res["per_item"].items():
        by_type[types[qid]].append(s)
    for t, ss in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:13s} n={len(ss):5d}  PA={sum(ss) / len(ss):+.4f}  "
              f"correct={ss.count(1)}  wrong={ss.count(-1)}  abstain={ss.count(0)}")
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("answers")
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    main(a.submission, a.answers, a.ids)
