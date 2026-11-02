"""Few-shot động: lấy các câu train tương tự đã giải đúng (kèm code) đưa vào prompt.

Kho ví dụ (jsonl, dựng bằng scripts/build_shots.py): mỗi dòng {id, biz, question, code, answer}.
Tìm theo mẫu câu hỏi (bỏ dấu, số thay bằng 0); bỏ câu cùng shop để giống điều kiện shop mới khi chấm.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

from datamate.parsing import fold


def _pattern(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", fold(question))).strip()


class ShotBank:
    def __init__(self, path: str | Path):
        self.shots = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self._keys = [_pattern(s["question"]) for s in self.shots]

    def find(self, question: str, biz_id: str, k: int = 3, exclude_id: str | None = None,
             min_score: float = 65, max_gap: float = 12, exclude_type: str | None = None) -> list[dict]:
        """Tối đa k ví dụ khác shop; bỏ ví dụ kém câu giống nhất quá max_gap điểm (thường là mẫu câu khác).
        exclude_type: bỏ cả ví dụ cùng ngành (shop private có thể thuộc ngành chưa gặp; đo bỏ-một-ngành)."""
        hits = process.extract(_pattern(question), self._keys, scorer=fuzz.token_sort_ratio,
                               limit=k * 10 + 10, score_cutoff=min_score)
        out, seen, top = [], set(), None
        for _, score, i in hits:
            s = self.shots[i]
            if (s["biz"] == biz_id or s["id"] == exclude_id or self._keys[i] in seen
                    or (exclude_type and exclude_type != "khac" and s.get("type") == exclude_type)):
                continue
            top = score if top is None else top
            if score < top - max_gap:
                break
            seen.add(self._keys[i])
            out.append(s)
            if len(out) == k:
                break
        return out


def format_shots(shots: list[dict]) -> str:
    if not shots:
        return ""
    parts = ["# Ví dụ câu tương tự đã giải đúng ở cửa hàng khác",
             "Chỉ tham khảo cách làm: tên bảng, tên cột, cách ghi dữ liệu của cửa hàng này có thể khác, "
             "luôn kiểm tra lại trên dữ liệu thật."]
    for n, s in enumerate(shots, 1):
        parts.append(f"## Ví dụ {n}\nCâu hỏi: {s['question']}\n```python\n{s['code'].strip()}\n```\n"
                     f"<answer>{json.dumps(s['answer'], ensure_ascii=False)}</answer>")
    return "\n\n".join(parts)
