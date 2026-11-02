"""Guard nghiệp vụ soi trên code cuối của các mẫu agent: lọc đơn hợp lệ, lô còn tồn, phá hoà khi xếp hạng, trả tên
khi hỏi "ai / … nào". Câu hỏi đòi guard nào thì đáp án mà MỌI mẫu ra nó đều vi phạm guard đó bị phủ quyết.

Chỉ guard trong VETO_GUARDS được phủ quyết thật. Replay log agent GPU cũ (scripts/replay_arbiter.py, 30/10/2026):
train 1,280 câu: valid_order chặn 28 sai / 111 đúng, tie_break 27 / 121, in_stock 6 / 1, name_not_code 0 / 2;
strange 200 câu: valid_order 4 / 22, tie_break 4 / 16, in_stock 0 / 0. Chọn đáp án theo guard khi các mẫu lệch nhau
(trọng tài) trên câu abstain cũ: cứu đúng 2, chọn sai 3 → không dùng. Các guard còn lại để làm đặc trưng cho bộ chọn.
"""
import re
import unicodedata

from datamate.answer import answer_key
from datamate.parsing import fold

VETO_GUARDS = frozenset({"in_stock"})

_SALES = re.compile(r"doanh thu|dthu|\bdt\b|\bban\b|\bdon\b|ban chay|so luong|\bsl\b")
# lô / hạn dùng: đọc trên chữ có dấu vì bỏ dấu thì 'lọ' = 'lô', 'hạn đúng' = 'hạn dùng', 'hết hàng' ⊃ 'het han';
# câu viết không dấu mới dùng bản gấp dấu (log public 30/10: 8/10 câu bị phủ nhầm vì các cặp này)
_LOTS_MARKED = re.compile(r"\blô\b|hạn (sử )?dùng|hết hạn\b|cận (date|hạn)|\bhsd\b|\bdate\b")
_LOTS_PLAIN = re.compile(r"\blo (hang|thuoc|nao|gan|sap)\b|\bcac lo\b|han (su )?dung|het han\b|can (date|han)\b|"
                         r"\bhsd\b")
_RANK = re.compile(r"nhat\b|\btop\b|\bxep hang\b|\bdan dau\b")
_NAME_Q = re.compile(r"\bai\b|(khach|mat hang|san pham|\bsp\b|mon|hang|tho|nhan vien|\bnv\b|nha cung cap|thuoc)"
                     r"( \w+)? nao\b")
_CODE_Q = re.compile(r"\bma\b")
_CODE_LIKE = re.compile(r"^[A-Za-z]{1,5}[-_]?\d{2,}$")

_VALID = re.compile(r"valid_order\(|order_status\(|SEM\.\w+\(")
_STOCK = re.compile(r"(ton|left|stock|con_lai|con lai|so_luong|qty|sl|remain)\w*['\"\]]*\s*\)?\s*>\s*0|SEM\.lots?_|"
                    r"lots_expiring|lot_nearest_expiry|remaining\s*=|con_lai\s*=|ton_con\w*\s*=", re.I)
_MULTI_SORT = re.compile(r"sort_values\(\s*(by\s*=\s*)?\[[^\]]+,[^\]]+\]|SEM\.\w+\(|sorted\([^)]*key\s*=\s*lambda")


def required(question: str) -> set[str]:
    """Các guard câu hỏi đòi (theo nghĩa câu, không theo shop)."""
    s = fold(question)
    need = set()
    if _SALES.search(s):
        need.add("valid_order")
    raw = unicodedata.normalize("NFC", question).lower()
    lots = _LOTS_PLAIN.search(s) if raw == s else _LOTS_MARKED.search(raw)
    if lots:
        need.add("in_stock")
    if _RANK.search(s):
        need.add("tie_break")
    if _NAME_Q.search(s) and not _CODE_Q.search(s):
        need.add("name_not_code")
    return need


def violations(question: str, answer: dict, code: str) -> set[str]:
    """Guard bị vi phạm bởi một mẫu (code cuối + đáp án)."""
    need, bad = required(question), set()
    if "valid_order" in need and not _VALID.search(code):
        bad.add("valid_order")
    if "in_stock" in need and not _STOCK.search(code):
        bad.add("in_stock")
    if "tie_break" in need and not _MULTI_SORT.search(code) and re.search(r"idxmax|idxmin|\.iloc\[0\]|nlargest|head\(",
                                                                           code):
        bad.add("tie_break")
    if "name_not_code" in need:
        vals = answer.get("answer")
        vals = vals if isinstance(vals, list) else [vals]
        if any(isinstance(v, str) and _CODE_LIKE.match(v.strip()) for v in vals):
            bad.add("name_not_code")
    return bad


FEATURE_GUARDS = ("valid_order", "tie_break", "name_not_code", "in_stock")


def guard_features(question: str, cand: dict, samples: list[tuple[dict | None, str]]) -> list[float]:
    """Đặc trưng cho bộ chọn: với mỗi guard, câu có đòi không và tỉ lệ mẫu ra `cand` vi phạm nó."""
    need = required(question)
    k = answer_key(cand)
    codes = [code or "" for ans, code in samples if ans and answer_key(ans) == k] or [""]
    bad = [violations(question, cand, c) for c in codes]
    return [x for g in FEATURE_GUARDS
            for x in (float(g in need), sum(g in b for b in bad) / len(bad))]


def veto(question: str, answer: dict, samples: list[tuple[dict | None, str]],
         guards=VETO_GUARDS) -> set[str]:
    """Guard (trong `guards`) mà MỌI mẫu ra đáp án `answer` đều vi phạm; rỗng = không phủ quyết."""
    k = answer_key(answer)
    codes = [code for ans, code in samples if ans and answer_key(ans) == k]
    if not codes:
        return set()
    common = set(guards)
    for code in codes:
        common &= violations(question, answer, code or "")
    return common
