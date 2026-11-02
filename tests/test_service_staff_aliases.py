import pandas as pd
import datetime as dt
from datamate.helpers import Periods

from datamate.loader import Shop
from datamate.service_ledger import ServiceLedger


def test_documented_nicknames_with_titles_join_unique_staff_registry():
    rows = pd.DataFrame({'Date': pd.to_datetime(['2026-01-02'] * 3), 'Money': [100, 200, 300],
                         'Who': ['Đức', 'a Đức', 'A.Đức'], 'Note': [''] * 3,
                         'Vehicle': ['V'] * 3, 'Jobs': ['sửa'] * 3})
    registry = pd.DataFrame({'Name': ['Lương Chi Đức'], 'Rate': [25]})
    readme = '''Mỗi dòng một lượt xe, có phiếu bảo hành.
## Cột trong `visits.csv`
- **Date**: ngày sửa
- **Money**: tổng tiền cả phiếu
- **Who**: thợ làm phiếu
- **Note**: ghi chú
- **Vehicle**: dòng xe và biển số
- **Jobs**: các việc đã làm
## Cột trong `staff.csv`
- **Name**: họ tên thợ
- **Rate**: phần trăm hoa hồng
'''
    def build():
        return ServiceLedger.from_shop(Shop('synthetic', readme, {'visits': rows, 'staff': registry},
            {'visits': {'sources': ['visits.csv']}, 'staff': {'sources': ['staff.csv']}}))
    ledger = build()
    assert set(ledger.rows.staff) == {'duc'}
    assert ledger.staff == {'duc': ('Lương Chi Đức', 25.0)}
    registry.loc[1] = ['Trần Văn Đức', 30]
    assert build().staff == {}  # ambiguous nickname must not select the first person


def test_service_revenue_comparison_uses_same_days_and_preserves_conditions():
    rows = pd.DataFrame({'date': pd.to_datetime(['2026-02-01', '2026-02-20', '2026-03-01', '2026-03-20']),
                         'amount': [100, 900, 150, 9000], 'staff': ['duc'] * 4, 'warranty': [False] * 4})
    ledger = ServiceLedger(rows, {}, '')
    day = dt.date(2026, 3, 10)
    def ask(q):
        return ledger.answer(q, day, Periods(day))
    assert ask('Doanh thu tháng này so với cùng kỳ tháng trước tăng bao nhiêu %?') == {'type': 'percent', 'answer': 50.0}
    assert ask('Doanh thu tháng 3/2026 so với tháng trước chênh bao nhiêu tiền?') == {'type': 'number', 'answer': -850}
    assert ask('Doanh thu tháng này không tính bảo hành so với tháng trước tăng bao nhiêu %?') is None
    assert ask('Số lượt xe tháng này so với tháng trước tăng bao nhiêu %?') is None
