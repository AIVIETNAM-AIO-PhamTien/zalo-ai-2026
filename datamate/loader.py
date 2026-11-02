"""Nạp mọi file của một shop (csv / xlsx / sqlite) thành bảng pandas đã gán kiểu.

Không ánh xạ ngữ nghĩa ở đây: chỉ dò header, gộp sheet/file cùng cấu trúc (mỗi tháng một sheet,
file tách _1/_2/_2026…), bỏ dòng tổng phụ ("Tổng ngày …"), đổi cột số / ngày bẩn sang kiểu chuẩn.
"""
from __future__ import annotations

import csv
import hashlib
import io
import datetime as dt
import re
import sqlite3
import unicodedata

from datamate.clean import clean_profile
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import numpy as np
import pandas as pd

from datamate.parsing import fold, parse_date, parse_number, parse_quantity

CACHE_VERSION = 26  # CSV UTF-16 / cp1258, bỏ file trùng nguyên byte
# v14: # ô chữ chuẩn hoá NFC + bỏ ký tự vô hình / khoảng trắng thừa; meta["quality"]
NUM_RATE = 0.97
DATE_RATE = 0.9
# 'Tổng ngày 01/04', 'Cộng ngày 01/05/2025', 'Tổng tháng 3', 'Tổng cộng', 'Cộng' — nhưng KHÔNG phải tên
# người 'Tống Thắng Thu' (bỏ dấu cũng thành 'tong thang …'): sau nhãn phải là số hoặc hết chuỗi
_SUBTOTAL = re.compile(r"^(tong|cong)( (ngay|thang|tuan|ca|ky|nam))?( \d[\d ]*)?$|^tong cong( .*)?$")
_TEXT_NAME_HINT = re.compile(r"(^|\b)(sdt|so dien thoai|phone|dien thoai|imei|barcode|ma vach|ma |stk|cmnd|cccd)")
_DATE_NAME_HINT = re.compile(r"(^|[\s_])(ngay|date|thoi gian|thoi diem|han dung|han su dung|hsd)($|[\s_])")
_SERIAL_MIN, _SERIAL_MAX = 36526, 58440  # số serial Excel của 1/1/2000 và 31/12/2059
_STEM_SUFFIX = re.compile(r"(_(t|thang)?\d{1,2}_\d{4}|_\d{4}|_\d{1,2})$", re.I)
_INVISIBLE = re.compile("[​‌‍⁠﻿­]")


def clean_text(v):
    """Ô chữ đạt chuẩn khi nạp: Unicode NFC (dấu tiếng Việt dựng sẵn thay vì tổ hợp: 'Hoà' gõ hai kiểu là một
    giá trị), bỏ ký tự vô hình (zero-width, BOM, soft hyphen), NBSP / tab / xuống dòng thành dấu cách, gộp khoảng
    trắng, bỏ đầu cuối. Không đổi hoa / thường, không bỏ dấu: đáp án tên phải giữ đúng cách viết."""
    if not isinstance(v, str):
        return v
    out = " ".join(_INVISIBLE.sub("", unicodedata.normalize("NFC", v)).split())
    return out or None


@dataclass
class Shop:
    biz_id: str
    readme: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    meta: dict[str, dict] = field(default_factory=dict)
    card: dict[str, dict[str, str]] | None = None  # vai trò cột do LLM gán (datamate/schema_card.py), gắn lúc chạy


def _nonempty(v) -> bool:
    return v is not None and not (isinstance(v, str) and not v.strip())


def detect_header(rows: list) -> int:
    """Dòng header = dòng đầu tiên (trong 10 dòng) có số ô khác rỗng ≥ 80% dòng nhiều ô nhất,
    và mọi ô khác rỗng là chữ (bỏ qua dòng tiêu đề báo cáo, dòng nhóm cột phía trên)."""
    head = rows[:10]
    counts = [sum(_nonempty(v) for v in r) for r in head]
    if not counts or max(counts) == 0:
        return 0
    need = max(2, 0.8 * max(counts))
    for i, r in enumerate(head):
        cells = [v for v in r if _nonempty(v)]
        if len(cells) >= need and all(isinstance(v, str) for v in cells):
            return i
    return 0


def _header_names(raw: list) -> list[str]:
    names, seen = [], {}
    for j, v in enumerate(raw):
        n = clean_text(str(v)) if _nonempty(v) else f"col{j}"
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        names.append(n)
    return names


def _year_hint(*names: str) -> int | None:
    for n in names:
        m = re.search(r"(20\d\d)", n or "")
        if m:
            return int(m.group(1))
    return None


def _read_xlsx(path: Path):
    try:  # calamine (Rust) nhanh hơn openpyxl ~10 lần, cho cùng giá trị ô
        import python_calamine as pc
        wb = pc.CalamineWorkbook.from_path(str(path))
        for name in wb.sheet_names:
            rows = wb.get_sheet_by_name(name).to_python(skip_empty_area=False)
            yield name, [tuple(None if v == "" else v for v in r) for r in rows]
        return
    except ImportError:
        pass
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        yield ws.title, [tuple(r) for r in ws.iter_rows(values_only=True)]


def _csv_text(path: Path) -> str:
    """Nội dung CSV theo đúng bảng mã: BOM UTF-16/UTF-8 trước, rồi UTF-8, rồi cp1258 (Windows tiếng Việt, dấu tổ hợp →
    clean_text gộp NFC). Không đoán được thì latin-1 (không sập cả shop; ô chữ lạ hiện ra ở báo cáo chất lượng)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp1258"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _read_csv(path: Path):
    yield None, [tuple(r) for r in csv.reader(io.StringIO(_csv_text(path), newline=""))]


def _frame(rows: list, sheet: str | None):
    h = detect_header(rows)
    cols = _header_names(rows[h])
    body, notes, repeats = [], 0, 0
    src_rows, dropped = [], []  # số dòng gốc (1 = dòng đầu file / sheet) của mỗi dòng giữ; dòng bỏ kèm lý do
    head = [fold(str(v)).strip() if v is not None else "" for v in rows[h]]
    for i, r in enumerate(rows[h + 1:], start=h + 2):
        # dòng tiêu đề lặp lại giữa bảng (in / xuất theo trang): không phải dữ liệu
        if sum(1 for a, b in zip(head, r) if a and isinstance(b, str) and fold(b).strip() == a) >= max(2, len([a for a in head if a]) // 2):
            repeats += 1
            dropped.append((i, "tiêu đề lặp", r))
            continue
        r = list(r) + [None] * (len(cols) - len(r))
        r = [clean_text(v) for v in r[:len(cols)]]
        filled = [v for v in r if v is not None]
        if not filled:
            continue
        # dòng ghi chú / tiêu đề lẻ một ô chữ trong bảng nhiều cột ("Ghi chú: số liệu ghi tay")
        if len(cols) >= 4 and len(filled) == 1 and isinstance(filled[0], str):
            notes += 1
            dropped.append((i, "ghi chú", r))
            continue
        body.append(r)
        src_rows.append(i)
    df = pd.DataFrame(body, columns=cols, dtype=object)
    df = df[[c for c in cols if not (c.startswith("col") and df[c].isna().all())]]
    df.attrs["note_rows"] = notes
    df.attrs["header_repeats"] = repeats
    df.attrs["src_rows"] = src_rows
    df.attrs["dropped"] = dropped
    return df, h, sheet


def _subtotal_mask(df: pd.DataFrame) -> pd.Series:
    """Dòng tổng phụ: có ô chữ dạng 'Tổng ngày …', 'Cộng ngày …', 'Tổng cộng'. Xét trên giá trị duy nhất."""
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        uniq = {v for v in df[c].dropna().unique() if isinstance(v, str)}
        hits = [v for v in uniq if _SUBTOTAL.match(fold(v))]
        if hits:
            mask |= df[c].isin(hits)
    return mask


def _continuation(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Dòng nối tiếp của nhóm (rulebook §8): cột đầu trống VÀ các cột thông tin nhóm khác
    (luôn có giá trị ở dòng đầu nhóm) cũng trống. Cột đầu trống đơn thuần (vd mã khách của khách lẻ)
    thì không phải."""
    if df.empty:
        return 0, []
    first = df.columns[0]
    empty = df[first].isna()
    if not empty.any() or empty.all() or (~empty).sum() / len(df) < 0.3:
        return 0, []
    head, cont = df[~empty], df[empty]
    cols = [c for c in df.columns[1:] if head[c].notna().mean() >= 0.95 and cont[c].isna().mean() >= 0.9]
    return (int(empty.sum()), [first] + cols) if cols else (0, [])


def _map_unique(series: pd.Series, fn, *, preserve_object: bool = False) -> pd.Series:
    """Áp fn trên giá trị duy nhất rồi map lại — nhanh hơn nhiều với cột lặp giá trị."""
    cache = {}
    for v in series.dropna().unique():
        try:
            cache[v] = fn(v)
        except TypeError:  # giá trị không hash được
            pass
    def convert(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return cache[v]
        except (KeyError, TypeError):
            return fn(v)
    if preserve_object:
        # Series.map may coerce Python ints plus None to float64, losing bits
        # before the caller can convert to nullable Int64.
        return pd.Series([convert(v) for v in series], index=series.index, name=series.name, dtype=object)
    return series.map(convert)


def _id_like(vals: list) -> bool:
    """Chuỗi số có số 0 đầu hoặc rất dài (SĐT, IMEI, mã vạch) → giữ dạng chữ."""
    s = [str(v).strip() for v in vals if isinstance(v, (str, int)) and not isinstance(v, bool)]
    digits = [x for x in s if x.isdigit()]
    if not digits:
        return False
    lead0 = sum(1 for x in digits if len(x) > 1 and x.startswith("0"))
    long_ = [x for x in digits if len(x) >= 9]
    # mã dài (SĐT, IMEI) gần như không lặp; một số dài lặp ở mọi dòng (999999999 = 'không giới hạn') là số
    return lead0 / len(digits) > 0.05 or (len(long_) / len(digits) > 0.5 and len(set(long_)) >= 0.2 * len(long_))


def _type_column(series: pd.Series, years: pd.Series | None) -> tuple[pd.Series, str, int]:
    vals = series.dropna().tolist()
    if not vals:
        return series, "empty", 0
    if all(isinstance(v, (dt.datetime, dt.date)) for v in vals):
        return pd.to_datetime(series, errors="coerce"), "date", 0
    name_hint = _TEXT_NAME_HINT.search(fold(series.name))
    if name_hint or _id_like(vals):
        return series.map(lambda v: None if v is None else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))), "text", 0

    n = len(vals)
    nums = _map_unique(series, parse_number, preserve_object=True)
    ok = int(nums.notna().sum())
    if ok / n >= NUM_RATE:
        kind = "int" if all(float(x).is_integer() for x in nums.dropna().unique()) else "float"
        # cột ngày ghi bằng số serial Excel (45717 = 2025-03-01): tên cột là ngày và mọi giá trị trong 2000–2060
        if _DATE_NAME_HINT.search(fold(series.name)) and nums.dropna().between(_SERIAL_MIN, _SERIAL_MAX).all():
            days = pd.to_timedelta(nums.astype("Float64").astype(float), unit="D")
            return pd.Timestamp("1899-12-30") + days, "date", n - ok
        return nums.astype("Int64" if kind == "int" else "Float64"), kind, n - ok

    dates = _map_unique(series, parse_date).astype(object)
    if years is not None and dates.notna().sum() < n:
        # ngày không ghi năm ('05/03') lấy năm của sheet (rulebook §8)
        miss = dates.isna() & series.notna()
        dates[miss] = [parse_date(v, default_year=y) for v, y in zip(series[miss], years[miss])]
    okd = int(dates.notna().sum())
    if okd / n >= DATE_RATE:
        return pd.to_datetime(dates, errors="coerce"), "date", n - okd

    qty = _map_unique(series, parse_quantity)
    okq = int(qty.notna().sum())
    units = {q[1] for q in qty.dropna().unique() if q[1]}
    # cột số lượng thật có ít loại đơn vị ngắn, lặp lại ('2 ly', '5 kg'); '386 Lê Lợi, Q1' (địa chỉ) có gần như mỗi dòng
    # một 'đơn vị' khác nhau → không phải số lượng, giữ nguyên chữ (đọc nhầm làm mất phần chữ của ô)
    if okq / n >= NUM_RATE and units and len(units) <= 30 and max(len(str(u)) for u in units) <= 15:
        out = qty.map(lambda q: None if q is None or (isinstance(q, float) and pd.isna(q)) else q[0])
        kind = "int" if all(float(x).is_integer() for x in out.dropna().unique()) else "float"
        return out.astype("Int64" if kind == "int" else "Float64"), kind + "(qty)", n - okq
    return series.map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)), "text", 0


def _short(r) -> tuple:
    return tuple(v for v in r if v is not None and not (isinstance(v, float) and pd.isna(v)))[:8]


def _partition(parts: list, sources: list[str]) -> list[str]:
    """Sheet / file gộp vào cùng bảng mà trùng nguyên nội dung (bản sao → cộng hai lần). Chồng từng dòng không xét:
    dòng nối tiếp của nhóm (chỉ tên hàng, giá) giống nhau giữa các tháng là bình thường."""
    out, seen = [], {}
    for k, (df, _, _) in enumerate(parts):
        if len(parts) < 2 or not len(df):
            continue
        key = (len(df), int(pd.util.hash_pandas_object(df.astype(str), index=False).sum()))
        if key in seen:
            out.append(f"{sources[k]} trùng nội dung {sources[seen[key]]}")
        seen.setdefault(key, k)
    return out


def _finalize(name: str, parts: list, sources: list[str]) -> tuple[pd.DataFrame, dict]:
    frames, src_part, src_row, dropped = [], [], [], []
    for k, (df, _, _) in enumerate(parts):
        rows = df.attrs.get("src_rows") or list(range(2, len(df) + 2))  # sqlite: dòng 1 = bản ghi đầu
        src_part += [k] * len(df)
        src_row += rows
        dropped += [(sources[k], i, why, _short(r)) for i, why, r in df.attrs.get("dropped", [])]
    notes = sum(df.attrs.get("note_rows", 0) for df, _, _ in parts)
    repeats = sum(df.attrs.get("header_repeats", 0) for df, _, _ in parts)
    for df, sheet, year in parts:
        df = df.copy()
        if len(parts) > 1:
            df["_sheet"] = sheet
        df["_year"] = year
        frames.append(df)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    mask = _subtotal_mask(df.drop(columns=["_sheet", "_year"], errors="ignore"))
    for k, i, r in zip([src_part[x] for x in mask[mask].index], [src_row[x] for x in mask[mask].index],
                       df[mask].itertuples(index=False)):
        dropped.append((sources[k], i, "tổng phụ", _short(r)))
    keep = (~mask).to_numpy()
    src_part = np.asarray(src_part, dtype=np.int32)[keep]
    src_row = np.asarray(src_row, dtype=np.int32)[keep]
    n_sub = int(mask.sum())
    sub_raw = df[mask].drop(columns=["_sheet", "_year"], errors="ignore").reset_index(drop=True)
    df = df[~mask].reset_index(drop=True)
    years = df["_year"] if df["_year"].notna().any() else None
    types, unparsed, examples, cells = {}, {}, {}, {}
    for c in df.columns:
        if c in ("_sheet", "_year"):
            continue
        raw = df[c]
        df[c], kind, bad = _type_column(raw, years)
        types[c] = kind
        if bad:
            unparsed[c] = bad
            lost = raw[raw.notna() & df[c].isna()]
            examples[c] = list(dict.fromkeys(map(str, lost)))[:5]
            if kind.startswith(("int", "float", "date")):
                # chữ trong cột số / ngày ('tặng', 'hủy', 'miễn phí') là thông tin nghiệp vụ: giữ nguyên theo dòng
                cells[c] = {int(i): str(v) for i, v in lost.items()}
    df = df.drop(columns=["_year"])
    continuation, cont_cols = _continuation(df.drop(columns=["_sheet"], errors="ignore"))
    # chỉ ffill cột thông tin nhóm (chữ, ngày); cột tiền / số lượng trống ở dòng nối tiếp là thật
    cont_cols = [c for c in cont_cols if not types.get(c, "").startswith(("int", "float"))]
    meta = {
        "sources": sources, "rows": len(df), "types": types, "unparsed": unparsed, "unparsed_examples": examples,
        "subtotal_rows_dropped": n_sub, "note_rows_dropped": notes, "header_repeats_dropped": repeats,
        "continuation_rows": continuation, "continuation_cols": cont_cols,
        # provenance (Bước 2): dòng i của bảng ← sources[src_part[i]], dòng src_row[i] của file / sheet gốc
        "src_part": src_part, "src_row": src_row,
        "dropped_rows": dropped,  # (nguồn, dòng gốc, lý do, giá trị) — bỏ khỏi bảng nhưng không mất dấu
        "partition": _partition(parts, sources),
        "unparsed_cells": cells,  # cột → {dòng: chữ gốc} của ô không đọc được thành số / ngày
        "subtotal_rows": sub_raw,  # dòng tổng phụ nguyên vẹn (giá trị thô), dùng để kiểm chứng
        "subtotal_check": check_subtotals(df, sub_raw, types),
        "quality": quality_report(df, types, unparsed),
        "clean": clean_profile(df, types),  # thực thể gộp + ràng buộc tự suy (datamate/clean.py)
    }
    return df, meta


def _shape(v: str) -> str:
    """Hình dạng của một giá trị: chữ → A, số → 9, gộp lặp ('ORD000123' → 'A9', '0912 345 678' → '9 9 9')."""
    return re.sub(r"(.)+", r"", re.sub(r"\d", "9", re.sub(r"[^\W\d_]", "A", v)))


def quality_report(df: pd.DataFrame, types: dict, unparsed: dict) -> dict:
    """Hồ sơ chất lượng của bảng sau khi nạp, theo các chiều chuẩn (Deequ / ISO 8000): đầy đủ (tỉ lệ ô có giá trị),
    hợp lệ (tỉ lệ ô đọc được đúng kiểu cột), duy nhất (dòng trùng nguyên vẹn; độ duy nhất cột chữ), nhất quán (cột
    chữ có cùng giá trị khác hoa / thường hoặc dấu). Chỉ đo, không sửa: dòng trùng có thể là dòng hàng hợp lệ (sổ ghi
    không theo đơn), việc bỏ trùng theo rulebook làm ở tầng ngữ nghĩa (semantic.clean_order_rows)."""
    data = df.drop(columns=["_sheet"], errors="ignore")
    n = len(data)
    cols = {}
    for c in data.columns:
        s = data[c]
        filled = int(s.notna().sum())
        col = {"type": types.get(c, "?"), "completeness": round(filled / n, 4) if n else 1.0,
               "validity": round(1 - unparsed.get(c, 0) / max(filled + unparsed.get(c, 0), 1), 4)}
        if col["type"] == "text" and filled:
            vals = s.dropna().astype(str)
            col["distinct"] = int(vals.nunique())
            col["uniqueness"] = round(col["distinct"] / filled, 4)
            # cùng một giá trị viết khác hoa / thường / dấu ('Trà sữa' vs 'TRÀ SỮA', 'tra sua'): không gộp, chỉ báo
            col["variant_groups"] = int(col["distinct"] - vals.map(lambda v: " ".join(fold(v).split())).nunique())
            # độ khớp định dạng (pattern profiling): tỉ lệ giá trị cùng 'hình dạng' với dạng trội (ORD000123 → A9);
            # chỉ đo cột mà dạng trội chiếm ≥ 80% (cột mã, SĐT…), cột chữ tự do để None
            shape = vals.map(_shape).value_counts(normalize=True)
            col["conformity"] = round(float(shape.iloc[0]), 4) if shape.iloc[0] >= 0.8 else None
        cols[c] = col
    dup = int(data.astype(str).duplicated().sum()) if n else 0
    return {"rows": n, "exact_duplicate_rows": dup, "columns": cols,
            "score": round(float(np.mean([min(v["completeness"], 1) * v["validity"] for v in cols.values()]))
                           if cols else 1.0, 4)}


def check_subtotals(df: pd.DataFrame, sub: pd.DataFrame, types: dict) -> dict:
    """Bất biến số học từ dòng tổng phụ: tổng các dòng 'Tổng ngày/tháng …' (hoặc dòng 'Tổng cộng') của một cột số
    phải bằng tổng các dòng chi tiết. Mỗi cấp ('tong ngay', 'tong thang', 'tong cong') kiểm riêng, vì sổ có thể có
    cả tổng ngày lẫn tổng tháng. Trả {cấp: {cột: {"sub", "detail", "ok"}}}; chỉ là chẩn đoán, không sửa dữ liệu."""
    if sub is None or not len(sub):
        return {}
    label = pd.Series("", index=sub.index)
    for c in sub.columns:
        hit = sub[c].map(lambda v: isinstance(v, str) and bool(_SUBTOTAL.match(fold(v))))
        label[hit & (label == "")] = sub[c][hit].map(fold)
    def lv(s):
        w = s.split()
        if len(w) > 1 and w[1] in ("ngay", "thang", "tuan", "ca", "ky", "nam"):
            return "tong " + w[1]
        return "tong cong" if s in ("tong", "cong") or s.startswith("tong cong") else "tong"  # 'Tổng 01/04': không rõ cấp
    level = label.map(lv)
    out = {}
    for lv, rows in sub.groupby(level):
        for c, kind in types.items():
            if c not in rows.columns or not kind.startswith(("int", "float")):
                continue
            vals = [parse_number(v) for v in rows[c] if v is not None]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            s, d = float(sum(vals)), float(pd.to_numeric(df[c], errors="coerce").sum())
            out.setdefault(lv, {})[c] = {"sub": s, "detail": d, "ok": abs(s - d) <= max(1.0, 1e-6 * abs(d))}
    return out


def load_shop(biz_dir) -> Shop:
    biz_dir = Path(biz_dir)
    readme = (biz_dir / "README.md").read_text(encoding="utf-8") if (biz_dir / "README.md").exists() else ""
    shop = Shop(biz_dir.name, readme)
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []

    seen: dict[bytes, str] = {}  # băm nội dung → file đầu tiên: file xuất trùng nguyên byte ('x (1).csv') bỏ, ghi lại
    shop.duplicate_files = []
    # bản sao kiểu 'x (1).csv' / 'x - Copy.csv' xếp sau bản gốc để bản gốc được giữ khi trùng nguyên byte
    copy = re.compile(r"\s*(\(\d+\)|-\s*copy|copy)$", re.I)
    for f in sorted(biz_dir.iterdir(), key=lambda p: (copy.sub("", p.stem), bool(copy.search(p.stem)), p.name)):
        suf = f.suffix.lower()
        if f.is_file() and suf in (".csv", ".xlsx", ".xlsm", ".sqlite", ".db"):
            h = hashlib.sha1(f.read_bytes()).digest()
            if h in seen:
                shop.duplicate_files.append((f.name, seen[h]))
                continue
            seen[h] = f.name
        if suf == ".sqlite" or suf == ".db":
            con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
            for (t,) in con.execute("select name from sqlite_master where type='table' order by name"):
                df = pd.read_sql_query(f'select * from "{t}"', con).astype(object)
                df = df.map(clean_text, na_action="ignore")
                df = df.astype(object).where(df.notna(), None)  # ô trống giữ None (không thành NaN → chuỗi "nan")
                tdf, meta = _finalize(t, [(df, None, None)], [f"{f.name}:{t}"])
                shop.tables[t], shop.meta[t] = tdf, meta
            con.close()
            continue
        if suf not in (".csv", ".xlsx", ".xlsm"):
            continue
        reader = _read_csv if suf == ".csv" else _read_xlsx
        stem = _STEM_SUFFIX.sub("", f.stem)
        for sheet, rows in reader(f):
            if not rows or not any(any(_nonempty(v) for v in r) for r in rows):
                continue
            df, _, _ = _frame(rows, sheet)
            key = (stem, tuple(c for c in df.columns))
            if key not in groups:
                groups[key] = {"parts": [], "sources": [], "first_sheet": sheet}
                order.append(key)
            groups[key]["parts"].append((df, sheet, _year_hint(sheet or "", f.stem)))
            groups[key]["sources"].append(f.name + (f":{sheet}" if sheet else ""))

    per_stem: dict[str, int] = {}
    for stem, _ in order:
        per_stem[stem] = per_stem.get(stem, 0) + 1
    # stem chỉ có một nhóm → tên stem; nhiều nhóm → nhóm gộp nhiều sheet (sổ theo tháng) giữ tên stem,
    # nhóm một sheet thêm tên sheet: 'SoBanHang' + 'SoBanHang.Ghi tắt', 'SoNo.Sổ nợ', 'SoNo.Trả góp'
    for key in order:
        stem = key[0]
        g = groups[key]
        name = stem if per_stem[stem] == 1 or len(g["parts"]) > 1 else f"{stem}.{g['first_sheet']}"
        if name in shop.tables:
            name = f"{stem}.{g['first_sheet']}"
        df, meta = _finalize(name, g["parts"], g["sources"])
        shop.tables[name], shop.meta[name] = df, meta
    return shop


def source_of(shop: Shop, table: str, i: int) -> str:
    """Provenance: dòng i (chỉ số sau khi nạp) của bảng → 'file:sheet dòng N' trong dữ liệu gốc."""
    m = shop.meta[table]
    return f"{m['sources'][int(m['src_part'][i])]} dòng {int(m['src_row'][i])}"
