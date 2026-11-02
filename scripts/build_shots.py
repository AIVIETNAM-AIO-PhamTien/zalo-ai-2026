"""Dựng kho ví dụ cho few-shot động từ log các lượt chạy: câu agent trả lời đúng, lấy các cell code chạy
không lỗi của một mẫu ra đúng đáp án đó, chọn lời giải ngắn nhất cho mỗi câu.

    python scripts/build_shots.py data/train/answers.jsonl datamate/shots.jsonl outputs/vast/train_fix1_n400 [...]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.answer import answer_key  # noqa: E402
from datamate.scoring import score_item  # noqa: E402

MAX_CODE = 2500  # ký tự; lời giải dài quá làm phình prompt
# lời giải ra đúng nhờ may: tự liệt kê trạng thái / so ngày bằng == → không đưa làm ví dụ
_BAD = re.compile(r"isin\(\s*\[\s*['\"]|(in|==)\s*\[\s*['\"]|status(es)?\s*=\s*\[")


def _code(trace) -> str:
    cells = [s["reply"] for s in trace if s.get("reply") and s.get("err") is None and "out" in s]
    blocks = []
    for reply in cells:
        start = reply.find("```")
        body = reply[start:].split("\n", 1)[1] if start >= 0 else ""
        blocks.append(body.rsplit("```", 1)[0].strip())
    return "\n\n".join(b for b in blocks if b)


def main(answers, out, run_dirs):
    refs = {}
    for line in Path(answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            refs[r["id"]] = r
    best = {}
    for run in map(Path, run_dirs):
        for line in (run / "submission.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pred = json.loads(line)
            qid = pred["id"]
            log_p = run / "logs" / f"{qid}.json"
            if qid not in refs or not log_p.exists() or score_item(pred, refs[qid]) != 1:
                continue
            log = json.loads(log_p.read_text(encoding="utf-8"))
            if log.get("source") != "agent" or pred["type"] == "unanswerable":
                continue
            final = {k: v for k, v in pred.items() if k != "id"}
            for ans, trace in zip(log["answers"], log["traces"]):
                if ans is None or answer_key(ans) != answer_key(final):
                    continue
                code = _code(trace)
                if _BAD.search(code) and "valid_order" not in code:
                    continue
                if code and len(code) <= MAX_CODE and (qid not in best or len(code) < len(best[qid]["code"])):
                    best[qid] = {"id": qid, "biz": log["business_id"], "question": log["question"],
                                 "code": code, "answer": final}
    Path(out).write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in best.values()), encoding="utf-8")
    print(f"{len(best)} ví dụ → {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    main(a.answers, a.out, a.runs)
