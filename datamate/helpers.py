"""Hàm tiện ích tất định cho agent: kỳ thời gian theo rulebook §2, Tết theo calendar.csv,
làm tròn, phần trăm, khớp tên. Mọi kỳ trả về (ngày đầu, ngày cuối), tính cả hai đầu."""
from __future__ import annotations

import calendar as _cal
import datetime as dt
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from datamate.parsing import fold

Period = tuple[dt.date, dt.date]


def _month_end(y: int, m: int) -> dt.date:
    return dt.date(y, m, _cal.monthrange(y, m)[1])


class Periods:
    """Các cách nói thời gian, quy về khoảng ngày theo rulebook §2. `today` = as_of_date."""

    def __init__(self, as_of: dt.date, calendar_csv: str | Path | None = None):
        self.as_of = as_of if isinstance(as_of, dt.date) else dt.date.fromisoformat(str(as_of))
        self._lunar = None
        if calendar_csv and Path(calendar_csv).exists():
            self._lunar = pd.read_csv(calendar_csv, dtype=str)

    def today(self) -> Period:
        return self.as_of, self.as_of

    def day(self, d) -> Period:
        d = d if isinstance(d, dt.date) else dt.date.fromisoformat(str(d))
        return d, d

    def yesterday(self) -> Period:
        d = self.as_of - dt.timedelta(days=1)
        return d, d

    def this_week(self) -> Period:
        """Tuần này: thứ Hai của tuần chứa hôm nay → hết hôm nay."""
        return self.as_of - dt.timedelta(days=self.as_of.weekday()), self.as_of

    def last_week(self) -> Period:
        """Tuần trước / tuần rồi: cả tuần liền trước, thứ Hai → Chủ nhật."""
        start = self.as_of - dt.timedelta(days=self.as_of.weekday() + 7)
        return start, start + dt.timedelta(days=6)

    def this_month(self) -> Period:
        """Tháng này: ngày 1 → hết hôm nay."""
        return self.as_of.replace(day=1), self.as_of

    def prev_month_same_days(self) -> Period:
        """Tháng trước để so với tháng này: ngày 1 → ngày trùng số với hôm nay (hoặc hết tháng)."""
        y, m = (self.as_of.year, self.as_of.month - 1) if self.as_of.month > 1 else (self.as_of.year - 1, 12)
        end = _month_end(y, m)
        return dt.date(y, m, 1), min(end, dt.date(y, m, min(self.as_of.day, end.day)))

    def _resolve_year(self, start_month: int, year: int | None) -> int:
        if year is not None:
            return year
        y = self.as_of.year
        return y if dt.date(y, start_month, 1) <= self.as_of else y - 1

    def month(self, m: int, y: int | None = None) -> Period:
        """Tháng M/Y cả tháng. Không ghi năm: năm nay nếu không sau hôm nay, ngược lại năm trước."""
        y = self._resolve_year(m, y)
        return dt.date(y, m, 1), _month_end(y, m)

    def prev_month(self, m: int, y: int | None = None) -> Period:
        y = self._resolve_year(m, y)
        return self.month(12, y - 1) if m == 1 else self.month(m - 1, y)

    def quarter(self, q: int, y: int | None = None) -> Period:
        first = 3 * (q - 1) + 1
        y = self._resolve_year(first, y)
        return dt.date(y, first, 1), _month_end(y, first + 2)

    def year(self, y: int) -> Period:
        return dt.date(y, 1, 1), dt.date(y, 12, 31)

    def year_to_date(self) -> Period:
        return dt.date(self.as_of.year, 1, 1), self.as_of

    def last_n_days(self, n: int) -> Period:
        """N ngày gần nhất: kể cả hôm nay (luật train); `last_n_include_today = False` → N ngày, ngày cuối là hôm qua."""
        end = self.as_of if getattr(self, "last_n_include_today", True) else self.as_of - dt.timedelta(days=1)
        return end - dt.timedelta(days=n - 1), end

    def _solar(self, lunar_day: int, lunar_month: int, lunar_year: int) -> dt.date:
        if self._lunar is None:
            raise ValueError("chưa nạp calendar.csv")
        c = self._lunar
        row = c[(c.lunar_day == str(lunar_day)) & (c.lunar_month == str(lunar_month))
                & (c.lunar_year == str(lunar_year)) & (c.lunar_leap.str.lower() == "false")]
        if row.empty:
            raise ValueError(f"không có ngày âm {lunar_day}/{lunar_month}/{lunar_year} trong calendar")
        return dt.date.fromisoformat(row.iloc[0]["date"])

    def tet(self, y: int) -> Period:
        """Tết năm Y: 23 tháng Chạp (năm âm Y-1) → hết mùng 10 tháng Giêng (năm âm Y)."""
        return self._solar(23, 12, y - 1), self._solar(10, 1, y)

    def before_tet(self, y: int, n_days: int | None = None) -> Period:
        """N ngày ngay trước 23 tháng Chạp; mặc định N = số ngày của dịp Tết."""
        start, end = self.tet(y)
        n = n_days or (end - start).days + 1
        return start - dt.timedelta(days=n), start - dt.timedelta(days=1)

    @staticmethod
    def mask(series: pd.Series, period: Period) -> pd.Series:
        """Dòng có ngày (bỏ giờ) nằm trong kỳ."""
        d = pd.to_datetime(series, errors="coerce").dt.normalize()
        return d.between(pd.Timestamp(period[0]), pd.Timestamp(period[1])).fillna(False)

    @staticmethod
    def n_days(period: Period) -> int:
        return (period[1] - period[0]).days + 1


class TsPeriods:
    """Periods cho agent: kỳ trả (pd.Timestamp đầu ngày, pd.Timestamp cuối ngày) để so trực tiếp với cột
    datetime64 (so `date` với cột datetime64 báo TypeError và làm model mất lượt)."""

    def __init__(self, periods: Periods):
        self._p = periods

    @staticmethod
    def _ts(period):
        start, end = period
        return pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    def __getattr__(self, name):
        attr = getattr(self._p, name)
        if name in ("mask", "n_days") or not callable(attr):
            return attr
        return lambda *a, **k: self._ts(attr(*a, **k))

    @staticmethod
    def mask(series: pd.Series, period) -> pd.Series:
        d = pd.to_datetime(series, errors="coerce").dt.normalize()
        return d.between(pd.Timestamp(period[0]).normalize(), pd.Timestamp(period[1]).normalize()).fillna(False)

    @staticmethod
    def n_days(period) -> int:
        return (pd.Timestamp(period[1]).normalize() - pd.Timestamp(period[0]).normalize()).days + 1


def round_money(x) -> int:
    """Tiền làm tròn về đồng, nửa đồng làm tròn lên."""
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def round2(x) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def pct_change(new, old) -> float | None:
    """Phần trăm thay đổi so với kỳ trước, làm tròn 2 chữ số; giảm ra số âm."""
    if old in (0, None) or new is None:
        return None
    return round2((new - old) / old * 100)


def share(part, whole) -> float | None:
    if whole in (0, None) or part is None:
        return None
    return round2(part / whole * 100)


_PHONE = re.compile(r"[\s\-–(]*\(?\+?\d[\d .]{7,}\d\)?\s*$")


def strip_phone(name: str) -> str:
    """'Đàm Bảo Bảo (0797929008)', 'Đào Dương Giang - 0912 365 003' → tên không kèm SĐT."""
    return _PHONE.sub("", str(name)).strip(" -–()") if name is not None else name


def find_names(query: str, candidates, limit: int = 5) -> list[tuple[str, float]]:
    """Khớp mờ không dấu: trả [(tên ứng viên, điểm 0–100)] giảm dần."""
    cands = list(dict.fromkeys(c for c in candidates if c is not None))
    folded = [fold(c) for c in cands]
    res = process.extract(fold(query), folded, scorer=fuzz.WRatio, limit=limit)
    return [(cands[i], round(score, 1)) for _, score, i in res]


def find_values(tables: dict, query: str, limit: int = 8, min_score: float = 60) -> list[tuple]:
    """Tìm giá trị chữ gần với `query` trong MỌI cột chữ của mọi bảng (tên hàng, khách, nhân viên, nhóm…).
    → [(giá trị, bảng, cột, điểm)] giảm dần. Dùng trước khi kết luận "không có" hay trả 0."""
    q = fold(query)
    hits: dict[tuple, float] = {}
    for tname, df in tables.items():
        for col in df.columns:
            s = df[col]
            if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
                continue
            vals = [v for v in s.dropna().unique() if isinstance(v, str)]
            if not vals or len(vals) > 50000:
                continue
            folded = [fold(v) for v in vals]
            for _, score, i in process.extract(q, folded, scorer=fuzz.WRatio, limit=limit,
                                               score_cutoff=min_score):
                key = (vals[i], tname, col)
                hits[key] = max(hits.get(key, 0), round(score, 1))
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])[:limit]
    return [(v, t, c, sc) for (v, t, c), sc in ranked]


def _name_key(s) -> str:
    return re.sub(r"\s+", " ", fold(str(s))).strip()


def name_mask(series: pd.Series, name: str) -> pd.Series:
    """Mask các dòng ghi cùng tên với `name`, bỏ qua hoa/thường, dấu, khoảng trắng thừa
    ('Bộ van dây gas Namilux' = 'BỘ  VAN DÂY GAS NAMILUX')."""
    key = _name_key(name)
    hit = {v for v in series.dropna().unique() if _name_key(v) == key}
    return series.isin(hit)


def order_status(series: pd.Series) -> pd.Series:
    """Trạng thái đơn → completed / cancelled / returned / delivering (rulebook §8), không phân biệt cách ghi.

    Ô trống (sau khi dòng tiếp nối đã kế thừa dòng đầu đơn) là đơn không ghi huỷ / trả → completed (rulebook §1);
    sổ chỉ ghi chú khi có chuyện ('Huỷ', 'đang giao') để trống mọi đơn bình thường."""
    from datamate.semantic import classify_status  # tránh import vòng
    return series.map({v: classify_status(v) for v in series.dropna().unique()}).fillna("completed")


def fill_order_status(df: pd.DataFrame, order_col: str) -> pd.DataFrame:
    """Trạng thái là của cả đơn: dòng để trống (Voucher, phí ship…) lấy trạng thái ghi ở dòng khác của cùng đơn
    (rulebook: ô trống ở các dòng sau của nhóm giống dòng đầu, không có nghĩa là đơn hoàn thành).
    Cột trạng thái = cột chữ có giá trị huỷ / trả; ô vẫn trống sau bước này là đơn không ghi gì → completed."""
    from datamate.semantic import classify_status  # tránh import vòng
    out = df
    for c in df.columns:
        if c == order_col or not (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object):
            continue
        s = df[c]
        if not s.isna().any():
            continue
        vals = s.dropna().unique()
        if len(vals) > 50 or not any(classify_status(v) in ("cancelled", "returned") for v in vals):
            continue
        filled = s.groupby(df[order_col]).transform("first")
        if out is df:
            out = df.copy()
        out[c] = s.fillna(filled)
    return out


def valid_order(series: pd.Series) -> pd.Series:
    """Mask đơn hợp lệ để tính doanh thu / số đơn: không huỷ, không trả hàng (đang giao vẫn tính)."""
    return order_status(series).isin(["completed", "delivering"])


_ANCHOR = re.compile(r"(\d{1,3}) ngay (dau )?(ke tu|tu|bat dau tu|sau) ngay (\d{1,2})/(\d{1,2})/(20\d\d)")
_BEFORE = re.compile(r"\d{1,3} ngay (ngay |lien |ngay lien )?(truoc|trc|roi|lien truoc)")


def anchored_windows(s: str) -> tuple[Period, Period] | None:
    """'N ngày kể từ ngày D … với N ngày ngay trước đó' (s: câu đã norm + lower) → (N ngày từ D, N ngày liền trước D)."""
    m = _ANCHOR.search(s)
    if not m or not _BEFORE.search(s):
        return None
    n = int(m.group(1))
    try:
        d = dt.date(int(m.group(6)), int(m.group(5)), int(m.group(4)))
    except ValueError:
        return None
    if not 1 <= n <= 366:
        return None
    return (d, d + dt.timedelta(days=n - 1)), (d - dt.timedelta(days=n), d - dt.timedelta(days=1))


def top_k(df, by: str, key: str, k: int | None = None, ascending: bool = False):
    """Xếp hạng theo rulebook: số đo `by` giảm dần (ascending=True: tăng dần cho 'ít nhất'), bằng nhau thì `key` (mã)
    tăng dần. k=None → cả bảng đã xếp; k → k dòng đầu. Dòng thiếu số đo (NaN) đứng cuối."""
    out = df.sort_values([by, key], ascending=[ascending, True], na_position="last", kind="mergesort")
    return out if k is None else out.head(k)
