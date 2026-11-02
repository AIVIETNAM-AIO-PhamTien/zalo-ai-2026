import datetime as dt
import json
import pickle

import pandas as pd
import pytest

from datamate.agent import Task, ShopContext, solve_once, solve
from datamate.agent_evidence import output_candidate, output_disagreement, ResultChannel
from datamate.answer import answer_key
from datamate.executor import Executor
from datamate.llm import ScriptedLLM
from datamate.loader import Shop


@pytest.fixture(scope='module')
def context(tmp_path_factory):
    frame = pd.DataFrame({'id': ['a','b','c','d','e'], 'date': pd.to_datetime(['2026-04-01','2026-04-02','2026-04-03','2026-04-04','2026-04-25']),
        'status': ['Hoàn thành','Hoàn thành','Huỷ','Trả hàng','Hoàn thành'], 'amount': [40,60,999,999,999]})
    readme = '## Cột trong `sales.csv`\n- **id**: mã đơn\n- **date**: ngày bán\n- **status**: trạng thái đơn\n- **amount**: thành tiền sau giảm\n'
    path = tmp_path_factory.mktemp('reliability')/'shop.pkl'
    path.write_bytes(pickle.dumps(Shop('synthetic', readme, {'sales': frame}, {'sales': {'sources': ['sales.csv']}})))
    return ShopContext(path, None, readme, 'brief', '', max_turns=4)


TASK = Task('synthetic-q', 'synthetic', 'Doanh thu tháng 4/2026 là bao nhiêu?', dt.date(2026,4,20))


def test_original_question_recall_is_executed_and_certified(context):
    llm = ScriptedLLM(['```python\nSEM.answer()\n```'])
    result, trace = solve_once(TASK, context, llm, 1000)
    assert result == {'type':'number', 'answer':100}
    assert trace[-1]['execution_result_source'] == 'deterministic'
    assert len(llm.calls) == 1
    assert trace[-1]['sem_calls'][-1] == {'method':'answer','status':'solved','original_question':True}


def test_last_rescue_sample_can_recall_rules_after_two_exhausted_samples(context):
    llm = ScriptedLLM({1000:['Chưa xong'],1001:['Chưa xong'],1002:['```python\nSEM.answer()\n```']})
    result, info = solve(TASK, context, llm, n_samples=2, max_samples=3)
    assert result == {'type':'number','answer':100}
    assert info['n_samples'] == 3
    assert info['sample_outcomes'][:2] == ['turn_budget_exhausted']*2


def test_verified_zero_is_not_abstained(context):
    task = Task('zero', 'synthetic', 'Doanh thu ngày 03/04/2026 là bao nhiêu?', TASK.as_of)
    llm = ScriptedLLM({1000:['```python\nSEM.answer()\n```'],1001:['Chưa xong']})
    result, info = solve(task, context, llm, n_samples=2, max_samples=2)
    assert result == {'type':'number','answer':0}
    assert info['decision'] == 'original_question_deterministic_result'


def test_executor_preserves_runner_schema_card(context):
    from datamate.executor import _build_namespace
    card = {'sales': {'amount': 'thành tiền sau giảm'}}
    namespace = _build_namespace(context.cache_path, TASK.as_of, None, schema_card=card)
    assert namespace['SEM']._shop.card == card


def test_agent_working_tables_do_not_mutate_sem_source(context, tmp_path):
    from datamate.executor import _build_namespace
    shop = Shop('synthetic', '', {'ledger':pd.DataFrame({'amount':[3,7]})}, {})
    path = tmp_path/'independent.pkl'
    path.write_bytes(pickle.dumps(shop))
    namespace = _build_namespace(path, TASK.as_of, None)
    namespace['T']['ledger'].loc[0,'amount'] = 999
    assert namespace['SEM']._shop.tables['ledger'].amount.tolist() == [3,7]


def test_certificate_resolves_wrong_votes_without_oracle(context):
    wrong = '<answer>{"type":"number","answer":999}</answer>'
    llm = ScriptedLLM({1000:[wrong],1001:['```python\nSEM.answer()\n```']})
    result, info = solve(TASK, context, llm, n_samples=2, max_samples=2)
    assert result == {'type':'number','answer':100}
    assert info['decision'] == 'original_question_deterministic_result'


def test_alternate_question_cannot_certify_dropped_conditions(context):
    with Executor(context.cache_path, TASK.as_of, None, question='Doanh thu tháng 4/2026 của khách Lan?', rulebook='') as ex:
        out, err = ex.run("SEM.answer('Doanh thu tháng 4/2026 là bao nhiêu?')")
        assert err is None and '100' in out
        assert ex.last_result is None


def test_failed_cell_discards_emission_and_next_cell_has_no_stale_result(context):
    with Executor(context.cache_path, TASK.as_of, None) as ex:
        _, err = ex.run("emit_result('number', T['sales']['amount'].sum())\nraise ValueError('stop')")
        assert err and ex.last_result is None
        out, err = ex.run("print(7)")
        assert err is None and out.strip() == '7' and ex.last_result is None
        _, err = ex.run("emit_result('number', 1)\nemit_result('number', 2)")
        assert err and ex.last_result is None


def test_printing_json_is_not_a_result_commitment(context):
    with Executor(context.cache_path, TASK.as_of, None) as ex:
        _, err = ex.run("print({'type':'number','answer':123})")
        assert err is None and ex.last_result is None


def test_emit_accepts_numpy_calculation_without_copying_number(context):
    llm = ScriptedLLM(["```python\nemit_result('number', T['sales'].loc[T['sales']['id'].isin(['a','b']), 'amount'].sum())\n```"])
    result, trace = solve_once(TASK, context, llm, 1000)
    assert result == {'type':'number','answer':100}
    assert trace[-1]['execution_result_source'] == 'code'


def test_context_rulebook_is_used_even_without_calendar(context):
    with Executor(context.cache_path, TASK.as_of, None, question=TASK.question,
                  rulebook='| Đơn tính vào doanh thu | định nghĩa khác |') as ex:
        out, err = ex.run('print(SEM.revenue(P.month(4,2026)))\nprint(SEM.answer())')
        assert err is None and out.strip() == 'None\nNone' and ex.last_result is None


@pytest.mark.parametrize('rule,method,args', [
    ('Đơn tính vào doanh thu', 'revenue', ()),
    ('Xác định khách', 'customers_min_orders', (2,)),
    ('Doanh thu theo hình thức thanh toán', 'payment_revenue', ('cash',)),
])
def test_legacy_sem_entrypoints_honor_the_public_policy_name(context, rule, method, args):
    from types import SimpleNamespace
    from datamate.sem_api import SemAPI
    shop = pickle.loads(context.cache_path.read_bytes())
    sem = SemAPI(shop, TASK.as_of, f'| {rule} | định nghĩa mới |')
    sem._built = True
    sem._view = SimpleNamespace(covers=lambda p:True, revenue=lambda *a:100,
        customers_with_orders=lambda *a:(1,2), attr_labels=lambda:{'pay':{'cash'}}, attr_revenue=lambda *a:50)
    assert getattr(sem, method)((dt.date(2026,4,1), TASK.as_of), *args) is None


def test_debt_rule_and_import_asof_are_enforced(context):
    from types import SimpleNamespace
    from datamate.sem_api import SemAPI
    shop = pickle.loads(context.cache_path.read_bytes())
    sem = SemAPI(shop, TASK.as_of, '| Đơn được tính vào nợ | định nghĩa mới |')
    sem._built, sem._view = True, SimpleNamespace(debt=SimpleNamespace(total_owed=lambda day:999))
    assert sem.debt_total(TASK.as_of) is None
    sem = SemAPI(shop, TASK.as_of)
    sem._built, sem._view = True, SimpleNamespace(imports=SimpleNamespace(avg_price=lambda item,unit,period:period[1].day))
    assert sem.import_avg_price((dt.date(2026,4,1),dt.date(2026,4,30)), 'A','hop') == 20


def test_table_vote_key_preserves_measure_identity():
    a = {'type':'table','answer':{'columns':['item','quantity','money'],'rows':[['A',2,100]]}}
    b = {'type':'table','answer':{'columns':['item','money','quantity'],'rows':[['A',2,100]]}}
    c = {'type':'table','answer':{'columns':['money','item','quantity'],'rows':[[100,'A',2]]}}
    assert answer_key(a) != answer_key(b)
    assert answer_key(a) == answer_key(c)


def test_printed_literals_are_only_diagnostics_and_never_execute_code(tmp_path):
    assert output_candidate('np.int64(42)', 'number') == {'type':'number','answer':42}
    assert output_disagreement({'type':'number','answer':9}, [{'out':'42','err':None}]) is True
    assert output_disagreement({'type':'number','answer':21}, [{'out':'20.6','err':None}]) is None
    assert output_candidate("__import__('os').system('echo bad')", 'number') is None
    assert output_candidate('debug: 42', 'number') is None


def test_namespace_gold_matches_sem_revenue(context):
    from datamate.executor import _build_namespace
    namespace = _build_namespace(context.cache_path, TASK.as_of, None)
    G = namespace['G']
    if 'don' not in G:  # shop tổng hợp không dựng được OrderView → agent tự tính từ T, có lý do
        assert namespace['SEM'].view is None and G['_bao_cao']['why']
        return
    o = G['don']
    apr = o[(o.date.dt.month == 4) & (o.date.dt.year == 2026) & o.counts_revenue]
    assert int(apr.revenue.sum()) == namespace['SEM'].revenue(('2026-04-01', '2026-04-30'))
    assert o.order_id.is_unique and (o.date <= pd.Timestamp(TASK.as_of)).all()
    if 'dong' in G:
        assert {'bqty', 'counts_revenue', 'date'} <= set(G['dong'].columns)


def test_namespace_gold_empty_when_no_orders(tmp_path):
    from datamate.executor import _build_namespace
    shop = Shop('synthetic', '', {'ledger':pd.DataFrame({'amount':[3,7]})}, {})
    path = tmp_path/'noorders.pkl'
    path.write_bytes(pickle.dumps(shop))
    G = _build_namespace(path, TASK.as_of, None)['G']
    assert set(G) == {'_bao_cao'} and G['_bao_cao']['why'].startswith('không dựng được')
