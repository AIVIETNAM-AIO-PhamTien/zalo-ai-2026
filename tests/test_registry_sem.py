import datetime as dt

import pandas as pd

from datamate.loader import Shop
from datamate.sem_api import SemAPI

KY = (dt.date(2026, 1, 1), dt.date(2026, 1, 31))


def api(table):
    return SemAPI(Shop("synthetic", "", {"registry": table}), dt.date(2026, 1, 20))


def test_members_includes_nonbuyers_deduplicates_and_excludes_future():
    table = pd.DataFrame({"id": ["A", "A", "B", "C", "D"],
                          "joined": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-20",
                                                    "2026-01-21", "2025-12-31"])})
    sem = api(table)
    assert sem.new_members(KY, "registry", "id", "joined") == 2
    table.loc[1, "joined"] = pd.Timestamp("2026-01-02")
    assert sem.new_members(KY, "registry", "id", "joined") is None


def test_registry_missing_bad_dates_and_outside_period():
    table = pd.DataFrame({"id": ["A"], "joined": pd.to_datetime(["2026-01-01"])})
    sem = api(table)
    assert sem.new_members(KY, "missing", "id", "joined") is None
    assert sem.new_members(KY, "registry", "unknown", "joined") is None
    assert sem.new_members((dt.date(2025, 1, 1), dt.date(2025, 1, 31)), "registry", "id", "joined") is None
    table["joined"] = ["01/01/2026"]
    assert sem.new_members(KY, "registry", "id", "joined") is None


def test_supplier_counts_receipts_not_lines_or_amount_and_ties_by_code():
    table = pd.DataFrame({"receipt": ["1", "1", "2", "3", "4", "5"],
                          "supplier": ["B", "B", "A", "A", "B", "B"],
                          "day": pd.to_datetime(["2026-01-01"] * 5 + ["2026-01-21"]),
                          "status": ["Đã nhập"] * 4 + ["Đã huỷ", "Đã nhập"],
                          "amount": [999999, 999999, 1, 1, 999999, 999999]})
    sem = api(table)
    args = (KY, "registry", "receipt", "supplier", "day", "status")
    assert sem.main_supplier(*args) == "A"
    table.loc[3, "status"] = "Đã huỷ"
    assert sem.main_supplier(*args) == "A"  # hoà số phiếu, lấy mã tăng dần
    table.loc[1, "supplier"] = "C"
    assert sem.main_supplier(*args) is None


def test_registry_direct_calls_guard_changed_rules():
    table = pd.DataFrame({"id": ["A"], "joined": pd.to_datetime(["2026-01-01"])})
    sem = api(table)
    sem._rulebook = "| Khách thành viên mới trong tháng | định nghĩa khác |"
    assert sem.new_members(KY, "registry", "id", "joined") is None
    sem = api(table)
    sem._rulebook = "| Nhà cung cấp chính | định nghĩa khác |"
    assert sem.main_supplier(KY, "registry", "id", "id", "joined", "id") is None


def test_import_total_deduplicates_header_not_receipt_lines():
    frame = pd.DataFrame({"receipt": ["1", "1", "2", "3"],
                          "date": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-21"]),
                          "total": [100, 100, 999, 999],
                          "status": ["Nhập đầu kỳ", "Nhập đầu kỳ", "Huỷ", "Đã nhập"]})
    sem = api(frame)
    assert sem.import_total(KY, "registry", "receipt", "date", "total", "status") == 100
    frame.loc[1, "total"] = 50
    assert sem.import_total(KY, "registry", "receipt", "date", "total", "status") is None
