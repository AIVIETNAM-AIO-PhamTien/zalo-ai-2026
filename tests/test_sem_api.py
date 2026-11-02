import datetime as dt

import pandas as pd

from datamate.loader import Shop
from datamate.sem_api import SemAPI


def test_sem_returns_none_for_shop_without_order_table():
    sem = SemAPI(Shop("b", "", {"t": pd.DataFrame({"a": [1]})}, {"t": {"sources": ["t.csv"]}}), dt.date(2026, 1, 1))
    ky = (pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31 23:59:59"))
    assert sem.revenue(ky) is None and sem.debt_total(pd.Timestamp("2025-12-31")) is None
    assert sem.sales_lines() is None
    assert "SEM.revenue" in repr(sem)
