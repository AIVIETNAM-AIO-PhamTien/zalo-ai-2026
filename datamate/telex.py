"""Sửa lỗi gõ Telex chưa bật bộ gõ trong câu hỏi: 'soos đơn hoom nay' → 'số đơn hôm nay', 'cuwar hàng' → 'cửa hàng',
'dduwowcj' → 'được', 'thangs' → 'tháng' (train + public: hàng trăm câu kiểu này; regex tầng tất định và agent hiểu sai).

Chỉ sửa khi chắc: từ thuần ASCII; có dấu hiệu Telex (aa ee oo aw ow uw dd) và ra âm tiết hợp lệ, hoặc chỉ có phím
dấu cuối (s f r x j) và kết quả là âm tiết có thật trong từ điển (dựng từ train, scripts/build_vi_vocab.py). Từ không
dấu có trong tên hàng train ('max', 'gas', 'cool') không bao giờ sửa.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_TONES = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}
_MARKS = [("uwow", "ươ"), ("uow", "ươ"), ("uw", "ư"), ("ow", "ơ"), ("aw", "ă"), ("aa", "â"), ("ee", "ê"), ("oo", "ô")]
_V = "aăâeêioôơuưy"
_SYL = re.compile(rf"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnpqrstvx])?[{_V}]{{1,3}}(ng|nh|ch|[cmnpt])?$")
_WORD = re.compile(r"(?<![\w-])[A-Za-z]+(?![\w-])")  # từ dính gạch nối / số là nhãn hiệu, mã hàng: 'D-nee'


@lru_cache(maxsize=1)
def _vocab():
    p = Path(__file__).with_name("vi_vocab.json")
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"vi": [], "ascii": []}
    index = {}
    for w in d["vi"]:
        nfd = unicodedata.normalize("NFD", w)
        tone = next((k for k, m in _TONES.items() if m in nfd), None)
        bare = unicodedata.normalize("NFC", nfd.replace(_TONES[tone], "")) if tone else w
        index.setdefault((bare, tone), w)
    return index, set(d["ascii"])


def _place(bare: str, tone: str | None) -> str:
    """Đặt dấu thanh vào âm tiết chưa có trong từ điển (quy tắc kiểu mới, đủ cho gợi ý)."""
    if not tone:
        return bare
    m = re.search(rf"[{_V}]+", bare[2:] if bare.startswith(("gi", "qu")) and len(bare) > 2 else bare)
    off = 2 if bare.startswith(("gi", "qu")) and len(bare) > 2 else 0
    s, e = m.start() + off, m.end() + off
    vow = bare[s:e]
    special = [i for i, ch in enumerate(vow) if ch in "ăâêôơư"]
    if special:
        i = special[-1]
    elif len(vow) >= 2 and (e < len(bare) or vow in ("oa", "oe", "uy")):
        i = 1
    else:
        i = len(vow) // 2 if len(vow) == 3 else 0
    pos = s + i
    return unicodedata.normalize("NFC", bare[:pos + 1] + _TONES[tone] + bare[pos + 1:])


def decode(token: str) -> str | None:
    """Một từ ASCII gõ Telex → âm tiết có dấu, hoặc None nếu không chắc."""
    t = token.lower()
    index, ascii_words = _vocab()
    if not t.isascii() or t in ascii_words or len(t) < 2:
        return None
    tone = t[-1] if t[-1] in _TONES and len(t) > 2 else None
    base = t[:-1] if tone else t
    marked = base[2:] if base.startswith("dd") else base
    marked = ("đ" if base.startswith("dd") else "") + marked
    for a, b in _MARKS:
        marked = marked.replace(a, b)
    strong = marked != base
    if not strong and not tone:
        return None
    if (marked, tone) in index:
        return index[(marked, tone)]
    if strong and _SYL.match(marked):
        return _place(marked, tone)
    return None


def telex_fix(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Sửa các từ gõ Telex trong câu; trả (câu đã sửa, [(từ gốc, từ sửa)])."""
    fixed = []

    def sub(m):
        d = decode(m.group(0))
        if d is None:
            return m.group(0)
        fixed.append((m.group(0), d))
        return d

    return _WORD.sub(sub, text), fixed
