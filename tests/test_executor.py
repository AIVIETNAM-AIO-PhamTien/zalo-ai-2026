import datetime as dt
import pickle

import pandas as pd
import pytest

from datamate.executor import Executor
from datamate.loader import Shop


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    shop = Shop("biz_t", "readme", {"ban_hang": pd.DataFrame({"x": [1, 2, 3]})}, {})
    p = tmp_path_factory.mktemp("c") / "biz_t.pkl"
    p.write_bytes(pickle.dumps(shop))
    return p


def test_persistent_namespace_and_helpers(cache):
    with Executor(cache, dt.date(2026, 3, 4), None, timeout=60) as ex:
        out, err = ex.run("s = T['ban_hang']['x'].sum()\nprint(s)")
        assert err is None and out.strip() == "6"
        out, err = ex.run("print(s * 2, TODAY, P.last_week())")
        assert err is None
        assert out.strip().startswith("12 2026-03-04")
        assert "2026-02-23 00:00:00" in out and "2026-03-01 23:59:59" in out
        out, err = ex.run("print(callable(checked_merge))")
        assert err is None and out.strip() == "True"


def test_error_is_reported_and_state_kept(cache):
    with Executor(cache, dt.date(2026, 3, 4), None, timeout=60) as ex:
        ex.run("a = 5")
        out, err = ex.run("1/0")
        assert "ZeroDivisionError" in err
        out, err = ex.run("print(a)")
        assert out.strip() == "5"


def test_last_expression_is_displayed(cache):
    with Executor(cache, dt.date(2026, 3, 4), None, timeout=60) as ex:
        out, err = ex.run("x = 3\nx + 4")
        assert out.strip() == "7"


def test_missing_cache_fails_fast(tmp_path):
    import time
    t0 = time.time()
    with pytest.raises(RuntimeError):
        Executor(tmp_path / "nope.pkl", dt.date(2026, 3, 4), None, timeout=60)
    assert time.time() - t0 < 30


def test_timeout_restarts_and_replays(cache):
    with Executor(cache, dt.date(2026, 3, 4), None, timeout=8) as ex:
        ex.run("keep = 42")
        out, err = ex.run("while True: pass")
        assert "Timeout" in err
        out, err = ex.run("print(keep)")
        assert out.strip() == "42"


def test_agent_view_fills_continuation_rows_and_compares_dates(tmp_path):
    df = pd.DataFrame({"don": ["O1", None, "O2"],
                       "ngay": pd.to_datetime(["2026-02-28", None, "2026-03-02"]),
                       "tien": [10, 20, 30]})
    shop = Shop("biz_c", "readme", {"bh": df},
                {"bh": {"continuation_rows": 1, "continuation_cols": ["don", "ngay"]}})
    p = tmp_path / "biz_c.pkl"
    p.write_bytes(pickle.dumps(shop))
    with Executor(p, dt.date(2026, 3, 4), None, timeout=60) as ex:
        out, err = ex.run("k = P.month(2, 2026)\nb = T['bh']\n"
                          "int(b[(b['ngay'] >= k[0]) & (b['ngay'] <= k[1])]['tien'].sum())")
        assert err is None and out.strip() == "30"
        out, err = ex.run("int(T['bh'][P.mask(T['bh']['ngay'], P.this_month())]['tien'].sum()), P.n_days(P.this_month())")
        assert err is None and out.strip() == "(30, 4)"


def test_identical_lines_of_one_order_count_once(tmp_path):
    lines = pd.DataFrame({"MaHoaDon": ["O1", "O1", "O2"], "Hàng": ["A", "A", "A"], "tien": [10, 10, 10]})
    log = pd.DataFrame({"Hàng": ["A", "A"], "tien": [5, 5]})  # sổ không ghi theo đơn: giữ cả hai
    shop = Shop("biz_d", "readme", {"ct": lines, "so": log}, {"ct": {}, "so": {}})
    p = tmp_path / "biz_d.pkl"
    p.write_bytes(pickle.dumps(shop))
    with Executor(p, dt.date(2026, 3, 4), None, timeout=60) as ex:
        out, err = ex.run("int(T['ct']['tien'].sum()), int(T['so']['tien'].sum())")
        assert err is None and out.strip() == "(20, 10)"


def test_today_compares_with_date_columns(tmp_path):
    shop = Shop("biz_t2", "readme", {"bh": pd.DataFrame({"ngay": pd.to_datetime(["2026-03-03", "2026-03-04"]),
                                                          "tien": [1, 2]})}, {})
    p = tmp_path / "biz_t2.pkl"
    p.write_bytes(pickle.dumps(shop))
    with Executor(p, dt.date(2026, 3, 4), None, timeout=60) as ex:
        out, err = ex.run("b = T['bh']\nint(b[b['ngay'] == TODAY - dt.timedelta(days=1)]['tien'].sum())")
        assert err is None and out.strip() == "1"


def test_key_error_hints():
    import pandas as pd
    from datamate.executor import _exec_cell
    T = {"ban": pd.DataFrame({"Thành tiền": [1, 2]}), "hd": pd.DataFrame({"Mã": [1]})}
    assert "có ở T['ban']" in _exec_cell("T['hd']['Thành tiền']", {"T": T})[1]
    assert "gần giống" in _exec_cell("T['ban']['Thanh tien']", {"T": T})[1]
    assert ".iloc[-1]" in _exec_cell("T['ban']['Thành tiền'].sort_values(ascending=False)[-1]", {"T": T})[1]


def test_english_order_id_column_still_dedups(tmp_path):
    # cột mã đơn tên tiếng Anh, README không mô tả: vai trò suy từ dữ liệu (shop_columns) → vẫn bỏ dòng lặp §8
    lines = pd.DataFrame({"Invoice No": ["A1", "A1", "A2"], "Item": ["x", "x", "y"], "Qty": [1, 1, 2],
                          "UnitPrice": [10, 10, 5], "LineTotal": [10, 10, 10]})
    shop = Shop("biz_e", "", {"sales": lines}, {"sales": {}})
    p = tmp_path / "biz_e.pkl"
    p.write_bytes(pickle.dumps(shop))
    with Executor(p, dt.date(2026, 3, 4), None, timeout=60) as ex:
        out, err = ex.run("int(T['sales']['LineTotal'].sum())")
        assert err is None and out.strip() == "20"
