"""Lossless typed projection selection from native types and documented meanings.

Competing total conversions must agree on every value. No majority coercion,
unit stripping, missing-to-zero, or guessing locale from column names.
"""
import datetime as dt
from decimal import Decimal

from datamate.foundation import decimal_value, date_value
from datamate.parsing import fold


def documented_type(descriptions):
    meanings = {fold(s) for s in descriptions}
    if len(meanings) != 1:
        return None
    meaning = next(iter(meanings))
    if any(word in meaning for word in ("ma don", "ma khach", "identifier", "barcode", "so dien thoai", "phone")):
        return None
    numeric = any(word in meaning for word in ("thanh tien", "don gia", "so luong", "doanh thu", "duration", "minutes", "thoi luong", "so phut"))
    temporal = any(word in meaning for word in ("ngay ", "date", "timestamp"))
    if numeric == temporal:
        return None
    return "number" if numeric else "date"


def infer_spec(values, descriptions):
    present = [v for v in values if v is not None]
    if not present:
        return None
    intent = documented_type(descriptions)
    if intent != "date" and all(isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in present):
        return {"type": "number", "locale": "en", "basis": "native_numeric_values"}
    if intent is None:
        return None
    # Evaluate conversion functions once per distinct typed cell value. This is
    # computational caching only; source rows and duplicates are untouched.
    present = list({(type(v), repr(v)): v for v in present}.values())
    candidates = []
    if intent == "number":
        for locale in ("en", "vi"):
            conversions = [decimal_value(v, locale) for v in present]
            if all(not c.error for c in conversions):
                candidates.append(({"type": "number", "locale": locale}, [c.value for c in conversions]))
    else:
        for format in (None, "%d/%m/%Y", "%m/%d/%Y", "vi_named", "dayfirst_mixed", "monthfirst_mixed"):
            conversions = [date_value(v, format=format) for v in present]
            if all(not c.error for c in conversions):
                candidates.append(({"type": "date", "format": format}, [c.value for c in conversions]))
    if not candidates or any(result != candidates[0][1] for _, result in candidates[1:]):
        return None
    return {**candidates[0][0], "basis": "documented_type_and_unanimous_total_conversion"}
