import datetime as dt

import pandas as pd
import pytest

from datamate.auxiliary import prepare, answer
from datamate.helpers import Periods
from datamate.loader import Shop
from datamate.line_ledger import LineLedger
from datamate.service_ledger import ServiceLedger


DAY = dt.date(2026, 4, 20)


@pytest.fixture
def sales():
    # Repeated identical entries have no order identifiers and must both count.
    frame = pd.DataFrame({'D': pd.to_datetime(['2026-04-01'] * 2 + ['2026-04-02', '2026-04-03', '2026-04-21', '2026-03-01']),
        'N': ['Bột A'] * 6, 'A': [60, 60, 500, 700, 900, 100], 'S': ['Hoàn thành'] * 2 + ['Huỷ', 'Trả hàng', 'Hoàn thành', 'Hoàn thành'],
        'Q': [1] * 6, 'BQ': [1] * 6, 'U': ['kg'] * 6, 'P': [60] * 6})
    readme = '''Sổ: mỗi dòng một mặt hàng khách mua, không ghi mã đơn.
## Cột trong `ledger.csv`
- **D**: ngày bán
- **N**: tên hàng
- **A**: thành tiền sau giảm
- **S**: trạng thái đơn
- **Q**: số lượng
- **BQ**: số lượng đã đổi ra đơn vị cơ bản
- **U**: đơn vị tính
- **P**: đơn giá
'''
    return Shop('synthetic', readme, {'ledger': frame}, {'ledger': {'sources': ['ledger.csv']}})


def solve(shop, question):
    return answer(question, DAY, prepare(shop), Periods(DAY))


def test_line_ledger_keeps_duplicate_entries_filters_status_and_future(sales):
    assert solve(sales, 'Doanh thu T4/2026 là bao nhiêu?') == {'type': 'number', 'answer': 120}
    assert solve(sales, 'Tháng 4/2026 Bột A bán được bao nhiêu kg?') == {'type': 'number', 'answer': 2}
    assert solve(sales, 'Ngày nào doanh thu cao nhất tháng 4/2026?') == {'type': 'date', 'answer': '2026-04-01'}


@pytest.mark.parametrize('question', [
    'Doanh thu tháng 4/2026 của khách Hoa?', 'Doanh thu tháng 4/2026 không tính Bột A?',
])
def test_line_ledger_never_infers_order_identities(sales, question):
    assert solve(sales, question) is None


@pytest.mark.parametrize('question', ['Tháng 4/2026 có bao nhiêu đơn?', 'Trung bình một đơn tháng 4/2026 bao nhiêu tiền?'])
def test_missing_order_identity_is_proven_unanswerable_not_a_guessed_count(sales, question):
    result = solve(sales, question)
    assert result['type'] == 'unanswerable'


def test_line_ledger_requires_source_declaration_and_private_rules(sales):
    sales.readme = sales.readme.replace('không ghi mã đơn', 'có mã đơn')
    assert LineLedger.from_shop(sales) is None
    sales.readme = sales.readme.replace('có mã đơn', 'không ghi mã đơn')
    v = LineLedger.from_shop(sales, '| Đơn tính vào doanh thu | khác |')
    assert v.answer('Doanh thu tháng 4/2026?', DAY, Periods(DAY)) is None


def test_line_ledger_comparison_and_month_series(sales):
    assert solve(sales, 'Doanh thu tháng 4/2026 tăng bao nhiêu % so với tháng 3/2026?') == {'type': 'percent', 'answer': 20.0}
    assert solve(sales, 'Doanh thu tháng 3/2026 và tháng 4/2026 lần lượt bao nhiêu?') == {'type': 'list', 'answer': [100, 120]}


@pytest.fixture
def repair():
    rows = pd.DataFrame({'Ngày': pd.to_datetime(['2026-04-01', '2026-04-01', '2026-04-02', '2026-04-22']),
        'Tiền': [120, 120, 0, 900], 'Thợ': ['An', 'An', 'Bình', 'Bình'],
        'Xe': ['Wave 59A-12345'] * 4, 'Việc': ['thay nhớt'] * 4, 'Note': ['', '', 'Bảo hành', '']})
    people = pd.DataFrame({'Tên': ['Nguyễn An', 'Lê Bình'], 'Rate': [25, 20]})
    readme = '''Mỗi dòng một lượt xe vào sửa. Phiếu bảo hành không thu tiền.
## Cột trong `visits.csv`
- **Ngày**: ngày sửa
- **Tiền**: tổng tiền cả phiếu
- **Thợ**: thợ làm phiếu
- **Xe**: dòng xe và biển số
- **Việc**: các việc đã làm
- **Note**: ghi chú
## Cột trong `staff.csv`
- **Tên**: họ tên thợ
- **Rate**: phần trăm hoa hồng
'''
    return Shop('synthetic', readme, {'visits': rows, 'staff': people}, {t: {'sources': [f'{t}.csv']} for t in ['visits', 'staff']})


def test_service_visit_count_includes_warranty_average_excludes_it(repair):
    assert solve(repair, 'Tháng 4/2026 có bao nhiêu lượt xe?') == {'type': 'number', 'answer': 3}
    assert solve(repair, 'Tháng 4/2026 có bao nhiêu phiếu bảo hành?') == {'type': 'number', 'answer': 1}
    assert solve(repair, 'Bình quân một phiếu tháng 4/2026 bao nhiêu tiền?') == {'type': 'number', 'answer': 120}
    assert solve(repair, 'Tỉ lệ bảo hành tháng 4/2026 là bao nhiêu %?') == {'type': 'percent', 'answer': 50.0}


def test_service_commission_matches_unambiguous_registry(repair):
    assert solve(repair, 'Thợ Nguyễn An tháng 4/2026 hoa hồng bao nhiêu tiền?') == {'type': 'number', 'answer': 60}
    assert solve(repair, 'Thợ nào hoa hồng cao nhất tháng 4/2026?') == {'type': 'string', 'answer': 'Nguyễn An'}
    repair.tables['staff'].loc[2] = ['Trần An', 50]
    assert solve(repair, 'Thợ An tháng 4/2026 hoa hồng bao nhiêu tiền?') is None


def test_service_does_not_guess_missing_warranty_or_conflate_sales(repair):
    repair.tables['visits'].loc[2, 'Tiền'] = 5
    assert ServiceLedger.from_shop(repair) is None
