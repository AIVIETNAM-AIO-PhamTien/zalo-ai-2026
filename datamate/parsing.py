"""Parser tất định cho giá trị bẩn trong dữ liệu tiểu thương: số tiền, số lượng, ngày, tên."""
from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata

_MULT = {"tỷ": 1e9, "tỉ": 1e9, "triệu": 1e6, "tr": 1e6, "nghìn": 1e3, "ngàn": 1e3, "k": 1e3}
_NUM_RE = re.compile(
    r"^(?P<num>[-+]?\d[\d.,\s]*)\s*(?P<mult>tỷ|tỉ|triệu|tr|nghìn|ngàn|k)?\s*(?:đồng|vnđ|vnd|đ|₫)?$"
)
_EXCEL_EPOCH = dt.datetime(1899, 12, 30)


def _to_num(x: float):
    if isinstance(x, int):
        return x
    return int(x) if float(x).is_integer() else x


def _digits_to_float(s: str, has_mult: bool) -> int | float | None:
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s).strip()
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        # một dấu tách và đúng 3 chữ số phía sau (không kèm đơn vị tr/k) → phân cách nghìn
        if len(parts) > 2 or (len(parts[1]) == 3 and not has_mult):
            s = s.replace(sep, "")
        else:
            s = s.replace(sep, ".")
    try:
        # Preserve integer literals before any float conversion: binary floats
        # cannot represent every integer above 2**53.
        return int(s) if re.fullmatch(r'[+-]?\d+', s) else float(s)
    except ValueError:
        return None


def parse_number(x) -> int | float | None:
    """'5,000' → 5000; '115,000.0000' → 115000; '270.000đ' → 270000; '250k' → 250000; '16,5tr' → 16500000."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if isinstance(x, float) and math.isnan(x) else _to_num(x)
    s = unicodedata.normalize("NFC", str(x)).strip().lower()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    # kiểu viết tắt: '409tr663' = 409 triệu 663 nghìn; '1tr2' = 1,2 triệu; '16k5' = 16,5 nghìn
    if s.startswith("-"):
        neg, s = not neg, s[1:].strip()
    m = re.match(r"^(\d+)\s*(tr|k)\s*(\d{1,3})\s*(?:đồng|vnđ|vnd|đ|₫)?$", s)
    if m:
        whole, unit, tail = int(m.group(1)), m.group(2), m.group(3)
        base = _MULT[unit]
        val = whole * base + (int(tail) * base / 1000 if len(tail) == 3 else float("0." + tail) * base)
        return _to_num(round(-val if neg else val, 6))
    m = _NUM_RE.match(s)
    if not m:
        return None
    mult = m.group("mult")
    val = _digits_to_float(m.group("num"), mult is not None)
    if val is None:
        return None
    if mult:
        val = round(val * _MULT[mult], 6)
    return _to_num(-val if neg else val)


def parse_quantity(x) -> tuple[int | float, str | None] | None:
    """'4 chai' → (4, 'chai'); '1' → (1, None)."""
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _to_num(x), None
    whole = parse_number(x)
    if whole is not None:
        return whole, None
    # đơn vị bắt đầu bằng chữ cái: 'chai', 'm3', 'xe 4m3', 'hộp 30' (không nhận ngày '1 thg 6 2025', '02/06')
    if parse_date(x) is not None:
        return None
    m = re.match(r"^\s*([-+]?\d[\d.,]*)\s*([^\W\d_][\w .]*?)?\s*$", str(x))
    if not m:
        return None
    n = parse_number(m.group(1))
    if n is None:
        return None
    return n, (m.group(2) or None)


def _mk(y, mo, d, h=0, mi=0, s=0):
    try:
        return dt.datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), int(s or 0))
    except ValueError:
        return None


_TIME = r"(?:[ T]+(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?)?"
_DATE_PATTERNS = [
    re.compile(r"^(?P<y>\d{4})[-/.](?P<mo>\d{1,2})[-/.](?P<d>\d{1,2})" + _TIME + r"$"),
    re.compile(r"^(?P<d>\d{1,2})[-/.](?P<mo>\d{1,2})[-/.](?P<y>\d{4})" + _TIME + r"$"),
    re.compile(r"^(?P<d>\d{1,2})\s*thg\s*(?P<mo>\d{1,2}),?\s*(?P<y>\d{4})" + _TIME + r"$"),
    re.compile(r"^ngày\s*(?P<d>\d{1,2})\s*tháng\s*(?P<mo>\d{1,2})\s*năm\s*(?P<y>\d{4})" + _TIME + r"$"),
]
_NO_YEAR = re.compile(r"^(?P<d>\d{1,2})[-/.](?P<mo>\d{1,2})$")


def parse_date(x, default_year: int | None = None) -> dt.datetime | None:
    """dd/mm/yyyy (ưu tiên kiểu Việt), ISO, '1 thg 7 2025', serial Excel; 'dd/mm' cần default_year
    (rulebook §8: ngày không ghi năm lấy năm của sheet)."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day)
    if isinstance(x, (int, float)):
        serial = float(x)
    else:
        s = unicodedata.normalize("NFC", str(x)).strip().lower()
        try:
            serial = float(s)
        except ValueError:
            serial = None
        if serial is None:
            for pat in _DATE_PATTERNS:
                m = pat.match(s)
                if m:
                    g = m.groupdict()
                    return _mk(g["y"], g["mo"], g["d"], g.get("h"), g.get("mi"), g.get("s"))
            m = _NO_YEAR.match(s)
            if m and default_year:
                return _mk(default_year, m.group("mo"), m.group("d"))
            return None
    # serial Excel: chỉ nhận khoảng ~1954–2119 để không nhầm với số thường
    if not 20000 <= serial <= 80000:
        return None
    return (_EXCEL_EPOCH + dt.timedelta(days=serial)).replace(microsecond=0)


def fold(s) -> str:
    """Bỏ dấu, đ→d, lowercase, bỏ ký tự đặc biệt, gộp khoảng trắng — dùng để so khớp mờ."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d")
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


_WALKIN = {"", "kl", "khach le", "khach vang lai", "vang lai", "le"}


def is_walkin(name) -> bool:
    """Khách lẻ / vãng lai: ô trống, 'kl', 'Khách lẻ'…"""
    return fold(name) in _WALKIN
