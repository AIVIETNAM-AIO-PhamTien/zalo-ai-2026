"""Gợi ý theo từng câu hỏi, đặt ngay dưới câu hỏi:
- luật rulebook liên quan (NVIDIA KGMON: trích đúng mục luật thay vì để model tự tìm trong cả rulebook dài;
  rulebook đọc lúc chạy từ split nên không cứng hoá luật nào);
- cột ứng viên (schema linking kiểu RAT-SQL / ReFoRCE column exploration): cột có tên hoặc nghĩa README trùng từ
  trong câu hỏi.
Chỉ dựa trên chữ, không dùng LLM; gợi ý, không ép.
"""
from __future__ import annotations

import re

from datamate.loader import Shop
from datamate.parsing import fold

STOP = set("""
cua hang quan tiem shop la bao nhieu nhiu bnhiu bn may the nao vay z v nhi nhe ha do a oi cho toi minh em anh chi
hoi xem giup gium coi check biet duoc dc tong cong ca ban tat het trong ngay rieng thi luc tinh den tu va voi co
khong roi nao nhieu it tien so mot nhu sao ne nha xiu chut dang hien tai thu ve thang nam tuan hom nay qua truoc
nay lan luot theo thu tu ghi cac nhung moi ky gi ai o vao ra di len xuong sau tren duoi khi neu
""".split())
_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$")

# Nguồn: data/public_test/rulebook.md, §3–6. Chỉ ánh xạ cách hỏi → tên
# luật; nội dung luôn lấy từ rulebook của split đang chạy (không chép công thức).
_RULE_BUNDLES = [
    (r"tra gop|tien gop|ky gop|qua han", ["Trả góp", "Đơn được tính vào nợ", "Nợ một đơn tại ngày X", "Tuổi nợ"]),
    (r"thu may|doi may|may cu|khach phai tra", ["Thu máy cũ", "Đơn tính vào doanh thu", "Doanh thu một đơn"]),
    (r"dong may|\bhang\b.*(iphone|apple)|dung luong", ["Dòng máy, hãng"]),
    (r"voucher|phi ship|tich diem|dung diem|giam.*diem", ["Tiền giảm voucher, tiền giảm nhờ dùng điểm, phí ship"]),
    (r"thanh vien moi|dang ky|khach moi", ["Khách thành viên mới trong tháng"]),
    (r"bim|\bta\b|size|so sua", ["Số sữa, size bỉm"]),
    (r"luot xe|so phieu|mot phieu|moi phieu|bao hanh", ["Lượt xe, phiếu", "Bảo hành"]),
    (r"khach quen|xe nao|bien so|chiec xe|dong xe", ["Khách quen, xe tới nhiều lần", "Dòng xe, biển số", "Lượt xe, phiếu"]),
    (r"nhot|ruot|va xe|sua dien|ac quy|bong den|lam gi", ["Việc ghi trong ô Làm gì", "Lượt xe, phiếu"]),
    (r"\btho\b|nhan vien.*cham chi", ["Thợ làm nhiều nhất", "Nhân viên, thợ chăm chỉ nhất", "Lượt xe, phiếu"]),
    (r"nha cung cap|\bncc\b", ["Nhà cung cấp chính", "Tiền nhập hàng trong kỳ"]),
    (r"topping", ["Topping ghi chung ô với món", "Bán chạy"]),
    (r"tien mat|bao cao ca", ["Báo cáo tiền mặt theo ca", "Ca sáng, ca chiều"]),
    (r"trung thanh|\bsop\b|doi y|khach.*nhat", ["Xác định khách"]),
]


def _words(s: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z0-9]", " ", fold(s)).split() if len(w) > 1 and w not in STOP and not w.isdigit()}


def _bigrams(s: str) -> set[str]:
    toks = [w for w in re.sub(r"[^a-z0-9]", " ", fold(s)).split() if not w.isdigit()]
    return {f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOP or b not in STOP}


def rule_rows(rulebook: str) -> list[tuple[set[str], set[str], str]]:
    """Các dòng bảng luật '| Tên luật | Mặc định |' → (từ khoá tên luật, cụm hai từ, dòng gốc)."""
    out = []
    section = ""
    entries = []
    for line in rulebook.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        m = _ROW.match(line.strip())
        if not m or set(m.group(1)) <= set("-: ") or fold(m.group(1)) in ("luat", "muc", "khai niem"):
            continue
        entries.append((m.group(1), line.strip(), section))
    # Public có hai luật cùng tên “Bảo hành” nhưng khác mẫu số theo ngành.
    # Giữ cả hai và ghi nguồn mục để agent không trộn công thức.
    from collections import Counter
    counts = Counter(name for name, _, _ in entries)
    for name, text, section in entries:
        if counts[name] > 1 and section:
            text += f"\n(Mục rulebook: {section})"
        out.append((_words(name), _bigrams(name), text))
    return out


def relevant_rules(question: str, rows, k: int = 3) -> list[str]:
    if k <= 0:
        return []
    rows = list(rows)
    qw, qb = _words(question), _bigrams(question)
    scored = []
    for words, bigrams, text in rows:
        if not words:
            continue
        hit = len(words & qw) / len(words) + 0.5 * len(bigrams & qb)
        if hit >= 0.6:
            scored.append((hit, text))
    scored.sort(key=lambda x: -x[0])
    selected = [t for _, t in scored[:k]]
    # k giới hạn kết quả từ khoá; các luật phụ thuộc phải đi cùng nhau.
    names = {fold(name) for pattern, bundle in _RULE_BUNDLES
             if re.search(pattern, fold(question)) for name in bundle}
    for _, _, text in rows:
        m = _ROW.match(text.splitlines()[0])
        if m and fold(m.group(1).strip()) in names and text not in selected:
            selected.append(text)
    return selected


def column_index(shop: Shop) -> list[tuple[str, str, set[str], str]]:
    """(bảng, cột, từ khoá tên + nghĩa README, nghĩa)."""
    from datamate.semantic import _tables_for, parse_readme_columns  # tránh import vòng
    meaning: dict[tuple[str, str], str] = {}
    for key, cols in parse_readme_columns(shop.readme).items():
        for t in _tables_for(shop, key):
            for c, m in cols.items():
                meaning.setdefault((t, c), m)
    out = []
    for t, df in shop.tables.items():
        for c in df.columns:
            if c == "_sheet":
                continue
            m = meaning.get((t, c), "")
            name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(c))  # 'MaHangHoa' → 'Ma Hang Hoa'
            out.append((t, c, _words(name) | _words(m), m))
    return out


def relevant_columns(question: str, index, k: int = 6) -> list[str]:
    qw = _words(question)
    scored = []
    for t, c, words, m in index:
        hit = len(words & qw)
        if hit >= 2:  # một từ trùng ('vỉ' ~ 'vi_tri') quá nhiễu
            scored.append((hit, -len(words), t, c, m))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    out, seen = [], set()
    for _, _, t, c, m in scored:
        if (t, c) in seen:
            continue
        seen.add((t, c))
        out.append(f"T[{t!r}][{c!r}]" + (f" ({m[:60]})" if m else ""))
        if len(out) == k:
            break
    return out


def hints(question: str, rows, index) -> str:
    parts = []
    rules = relevant_rules(question, rows) if rows else []
    if rules:
        parts.append("Luật rulebook có thể liên quan:\n" + "\n".join(rules))
    cols = relevant_columns(question, index) if index else []
    if cols:
        parts.append("Cột có thể liên quan (gợi ý theo tên / nghĩa cột, tự kiểm tra lại): " + "; ".join(cols))
    return "\n\n".join(parts)
