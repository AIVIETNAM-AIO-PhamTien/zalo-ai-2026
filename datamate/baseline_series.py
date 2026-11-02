"""Ghép phép tính doanh thu đã có qua nhiều tháng; không dùng LLM/id shop."""
import re

from datamate.baseline import _clean, extract_periods, norm


def _month_index(day):
    return day.year * 12 + day.month - 1


def _month(index, periods):
    year, month = divmod(index, 12)
    return periods.month(month + 1, year)


def answer(question, as_of, view, periods):
    text = norm(question)
    # The noun "tháng" between a count and an explicit month belongs to the
    # enumeration, and must not disappear together with the first month token.
    # A year before the next month ("8/2025 tháng 9/2025") is not
    # a declared count of 2025 months. Keep date components intact.
    enumeration = re.search(r'(?<![\d/])\b(\d{1,2}) thang (?=\d{1,2}/)', text)
    parse_text = text.replace(enumeration[0], ' ', 1) if enumeration else text
    found, rest = extract_periods(parse_text, periods)
    if not found or any(label != "month" for label, _ in found):
        return None
    months = [p for _, p in found]
    count = re.search(r"\b(\d+) thang (?:lien nhau|lien tiep) ket thuc (?:o |tai )?", text)
    if count:
        n = int(count[1])
        if not 1 <= n <= 24 or len(months) != 1:
            return None
        last = _month_index(months[0][0])
        months = [_month(i, periods) for i in range(last - n + 1, last + 1)]
        rest = re.sub(r"\b\d+ thang (lien nhau|lien tiep) ket thuc (o|tai)?\b", " ", rest)
    elif len(months) == 2 and re.search(r"\btu thang \d+(?:/\d+)? (den|toi) thang \d+", text):
        first, last = _month_index(months[0][0]), _month_index(months[1][0])
        if not 0 <= last - first < 24:
            return None
        months = [_month(i, periods) for i in range(first, last + 1)]
    if len(months) < 2 or len(months) != len(set(months)):
        return None
    if enumeration and int(enumeration[1]) != len(months):
        return None
    declared = re.search(r'\b(\d+) thang(?: lien tiep| lien nhau)?\b', rest)
    if declared:
        if int(declared[1]) != len(months):
            return None
        rest = rest.replace(declared[0], ' ')
    rest = re.sub(r'\bthang sau co luon cao hon thang TRUOC\b', ' tang lien tuc ', rest)
    if any(p[0] > as_of or not view.covers(p) for p in months):
        return None
    # Một ý định đơn, không nuốt điều kiện theo hàng/khách/nhân viên/kênh.
    if "DOANHTHU" not in rest and not re.search(r"ban.*tien", rest):
        return None
    base = "DOANHTHU ban tien thang cac xet tu den toi qua lan luot moi cu moi ghi theo thu tu do thoi gian liet ke tung"
    if "tang lien tuc" in rest:
        if not _clean(rest, base + " tang lien tuc"):
            return None
        values = [view.revenue(p, as_of) for p in months]
        return {"type": "boolean", "answer": all(b > a for a, b in zip(values, values[1:]))}
    if re.search(r"thang nao|vao thang nao", rest) and re.search(r"cao nhat|lon nhat|BAONHIEU(?: tien)? nhat", rest):
        if not _clean(rest, base + " nao vao cao lon nhat"):
            return None
        values = [view.revenue(p, as_of) for p in months]
        best = min(p[0] for p, v in zip(months, values) if v == max(values))
        return {"type": "date", "answer": best.strftime("%Y-%m")}
    if count or "lan luot" in rest or 'liet ke' in rest:
        if not _clean(rest, base + " tu den moi thang"):
            return None
        return {"type": "list", "answer": [view.revenue(p, as_of) for p in months]}
    return None
