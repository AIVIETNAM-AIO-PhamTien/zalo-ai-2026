"""Dựng datamate/vi_vocab.json cho bộ sửa lỗi gõ Telex (parsing.telex_fix), chỉ từ dữ liệu TRAIN:
- "vi": âm tiết có dấu xuất hiện trong câu hỏi, rulebook, README và giá trị chữ của bảng train;
- "ascii": từ không dấu trong giá trị chữ của bảng train (tên hàng, nhãn hiệu: 'max', 'gas', 'tiger'…), để không
  đọc nhầm 'max' thành 'mã'.

    python scripts/build_vi_vocab.py data/train
"""
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.loader import load_shop  # noqa: E402


def has_mark(t: str) -> bool:
    return "đ" in t or any(unicodedata.category(ch) == "Mn" for ch in unicodedata.normalize("NFD", t))


def main():
    split = Path(sys.argv[1] if len(sys.argv) > 1 else "data/train")
    texts = [json.loads(l)["question"] for l in open(split / "questions.jsonl", encoding="utf-8")]
    texts.append((split / "rulebook.md").read_text(encoding="utf-8"))
    values = []
    for biz in sorted((split / "data").iterdir()):
        texts.append((biz / "README.md").read_text(encoding="utf-8") if (biz / "README.md").exists() else "")
        shop = load_shop(biz)
        for df in shop.tables.values():
            for c in df.columns:
                s = df[c].dropna()
                if s.dtype == object or str(s.dtype).startswith("str"):
                    values.extend(str(v) for v in s.unique()[:5000])
    tok = lambda s: re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", s.lower()))  # noqa: E731
    vi = {t for s in texts + values for t in tok(s) if has_mark(t)}
    asc = collections.Counter(t for s in values for t in tok(s) if not has_mark(t) and t.isascii() and len(t) > 1)
    out = {"vi": sorted(vi), "ascii": sorted(t for t, n in asc.items() if n >= 2)}
    p = Path(__file__).resolve().parents[1] / "datamate" / "vi_vocab.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out['vi'])} âm tiết có dấu, {len(out['ascii'])} từ không dấu → {p}")


if __name__ == "__main__":
    main()
