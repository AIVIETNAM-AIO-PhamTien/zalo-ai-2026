import datetime as dt
import pickle
import threading

import pandas as pd
import pytest

from datamate.agent import ShopContext, Task, extract_answer, extract_code, solve, solve_once
from datamate.llm import ScriptedLLM
from datamate.loader import Shop


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    shop = Shop("biz_t", "README", {"ban_hang": pd.DataFrame({"sp": ["A", "B"], "tien": [100, 250]})},
                {"ban_hang": {"rows": 2, "types": {"sp": "text", "tien": "int"}, "unparsed": {},
                              "subtotal_rows_dropped": 0, "continuation_rows": 0, "sources": ["x.csv"]}})
    p = tmp_path_factory.mktemp("c") / "biz_t.pkl"
    p.write_bytes(pickle.dumps(shop))
    return ShopContext(cache_path=p, calendar_csv=None, readme="README", brief="BRIEF", rulebook="RULES",
                       exec_timeout=60, max_turns=4)


TASK = Task("q1", "biz_t", "Tổng tiền là bao nhiêu?", dt.date(2026, 3, 4))
CODE = "Mình tính:\n```python\nprint(T['ban_hang']['tien'].sum())\n```"
ANS = '<answer>{"type": "number", "answer": 350}</answer>'


def test_extractors():
    assert extract_code(CODE).strip() == "print(T['ban_hang']['tien'].sum())"
    assert extract_answer(ANS) == {"type": "number", "answer": 350}
    assert extract_answer("no answer") is None
    assert extract_answer("<answer>{bad json}</answer>") == "INVALID"


def test_solve_once_runs_code_then_answers(ctx):
    llm = ScriptedLLM([CODE, ANS])
    ans, trace = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}
    # kết quả chạy code được gửi lại cho model
    assert "350" in llm.calls[1][-1]["content"]


def test_answer_in_same_message_as_code_is_not_trusted(ctx):
    llm = ScriptedLLM([CODE + "\n" + '<answer>{"type": "number", "answer": 999}</answer>', ANS])
    ans, _ = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}


def test_bad_format_gets_retry(ctx):
    llm = ScriptedLLM([CODE, '<answer>{"type": "number", "answer": "350 đồng"}</answer>', ANS])
    ans, _ = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}


ZERO = '<answer>{"type": "number", "answer": 0}</answer>'
UNANS = '<answer>{"type": "unanswerable", "reason": "không có cột"}</answer>'


def test_zero_answer_is_rechecked_once(ctx):
    llm = ScriptedLLM([CODE, ZERO, CODE, ANS])
    ans, trace = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}
    assert any("rất hiếm" in m["content"] for m in llm.calls[-1] if m["role"] == "user")
    assert trace[1]["recheck"] == {"type": "number", "answer": 0}


@pytest.mark.parametrize("reply", [ZERO, UNANS])
def test_suspicious_answer_kept_when_model_insists(ctx, reply):
    llm = ScriptedLLM([CODE, reply, reply])
    ans, _ = solve_once(TASK, ctx, llm, seed=0)
    assert ans["type"] == extract_answer(reply)["type"]
    assert len(llm.calls) == 3  # chỉ hỏi lại một lần


def test_missing_column_gets_column_hint(ctx):
    bad = "```python\nprint(T['ban_hang']['Tiền'].sum())\n```"
    llm = ScriptedLLM([bad, CODE, ANS])
    ans, trace = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}
    fb = " ".join(m["content"] for m in llm.calls[-1] if m["role"] == "user")
    assert "Không có cột 'Tiền'" in fb and "T['ban_hang']['tien']" in fb and trace[0]["col_hint"]


def test_unanswerable_name_gets_value_candidates(ctx):
    unans = '<answer>{"type": "unanswerable", "reason": "Không tìm thấy mặt hàng \'san pham AA\'"}</answer>'
    llm = ScriptedLLM([CODE, unans, ANS])
    solve_once(TASK, ctx, llm, seed=0)
    fb = " ".join(m["content"] for m in llm.calls[-1] if m["role"] == "user")
    assert "Ứng viên gần giống" in fb and "ban_hang" in fb


def test_no_recheck_without_turns_left(ctx):
    llm = ScriptedLLM([CODE, CODE, ZERO])
    ans, _ = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 0}


def test_no_answer_within_turns(ctx):
    llm = ScriptedLLM([CODE])
    ans, _ = solve_once(TASK, ctx, llm, seed=0)
    assert ans is None


def test_vote_agree(ctx):
    llm = ScriptedLLM({1000: [CODE, ANS], 1001: [ANS]})
    final, info = solve(TASK, ctx, llm, n_samples=2, max_samples=3, base_seed=1000)
    assert final == {"type": "number", "answer": 350}
    assert info["n_samples"] == 2


def test_vote_disagree_then_tiebreak(ctx):
    other = '<answer>{"type": "number", "answer": 351}</answer>'
    llm = ScriptedLLM({1000: [ANS], 1001: [other], 1002: [other]})
    final, info = solve(TASK, ctx, llm, n_samples=2, max_samples=3, base_seed=1000)
    assert final == {"type": "number", "answer": 351}
    assert info["n_samples"] == 3


class BarrierLLM(ScriptedLLM):
    """Lượt đầu của mỗi mẫu chờ đủ `parties` mẫu cùng gọi: nếu các mẫu chạy nối tiếp thì barrier hết hạn."""

    def __init__(self, replies, parties):
        super().__init__(replies)
        self.barrier = threading.Barrier(parties, timeout=20)

    def chat(self, messages, seed=None):
        if not any(m["role"] == "assistant" for m in messages):
            self.barrier.wait()
        return super().chat(messages, seed)


def test_first_samples_run_concurrently(ctx):
    llm = BarrierLLM({1000: [CODE, ANS], 1001: [ANS]}, parties=2)
    final, info = solve(TASK, ctx, llm, n_samples=2, max_samples=3, base_seed=1000)
    assert final == {"type": "number", "answer": 350}
    assert info["n_samples"] == 2
    # thứ tự mẫu theo seed, không theo thứ tự xong: mẫu seed 1000 có lượt chạy code
    assert len(info["traces"][0]) == 2 and len(info["traces"][1]) == 1


def test_vote_no_majority_abstains(ctx):
    llm = ScriptedLLM({1000: [ANS], 1001: ['<answer>{"type": "number", "answer": 1}</answer>'],
                       1002: ['<answer>{"type": "unanswerable", "reason": "x"}</answer>']})
    final, _ = solve(TASK, ctx, llm, n_samples=2, max_samples=3, base_seed=1000)
    assert final == {"type": "abstain"}


def test_answer_of_unexpected_type_is_rechecked_once(ctx):
    task = Task("q2", "biz_t", "Tháng nào cửa hàng bán được nhiều tiền nhất?", dt.date(2026, 3, 4))
    as_string = '<answer>{"type": "string", "answer": "Tháng 5"}</answer>'
    as_date = '<answer>{"type": "date", "answer": "2025-05"}</answer>'
    llm = ScriptedLLM([CODE, as_string, as_date])
    ans, trace = solve_once(task, ctx, llm, seed=0)
    assert ans == {"type": "date", "answer": "2025-05"}
    assert "Kiểu đáp án dự kiến: date" in llm.calls[0][1]["content"]
    assert trace[1]["type_recheck"]["type"] == "string"


def test_reconcile_sample_sees_conflicting_answers(ctx):
    one = '<answer>{"type": "number", "answer": 1}</answer>'
    llm = ScriptedLLM({1000: [CODE, ANS], 1001: [CODE, one], 1002: [CODE, ANS]})
    final, info = solve(TASK, ctx, llm, reconcile=True)
    assert final == {"type": "number", "answer": 350}
    third = [m for m in llm.calls if any("KHÁC NHAU" in x["content"] for x in m if x["role"] == "user")]
    assert third, "mẫu thứ ba phải nhận phần hoà giải"


def test_diverse_second_sample_has_no_shots(ctx):
    from dataclasses import replace

    class Bank:
        def find(self, *a, **k):
            return [{"question": "Q mẫu", "code": "print(1)", "answer": {"type": "number", "answer": 1}}]

    llm = ScriptedLLM({1000: [ANS], 1001: [ANS]})
    solve(TASK, replace(ctx, shots=Bank()), llm, diverse=True)
    with_shots = [m for m in llm.calls if "Q mẫu" in m[1]["content"]]
    assert len(with_shots) == 1


def test_answer_logprob_from_token_stream():
    from datamate.llm import answer_logprob
    toks = [{"token": "xong ", "logprob": -5.0}, {"token": "<answer>", "logprob": -0.1},
            {"token": '{"type"', "logprob": -0.3}, {"token": "</answer>", "logprob": -9.0}]
    assert answer_logprob(toks, 'xong <answer>{"type"</answer>') == -0.2
    assert answer_logprob(None, "x") is None


def test_think_block_is_stripped_but_kept_for_logs(monkeypatch):
    import io
    import json

    from datamate import llm as llm_mod
    body = {"choices": [{"message": {"content": "<think>\ncột ngày là NgayLap\n</think>\n\n```python\nx\n```"}}]}
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", lambda req, timeout: io.BytesIO(json.dumps(body).encode()))
    m = llm_mod.LLM("http://x/v1", "m")
    assert m.chat([{"role": "user", "content": "q"}]) == "```python\nx\n```"
    assert m.last_think == "cột ngày là NgayLap"


def test_verifier_feedback_lets_sample_fix_answer(ctx):
    from dataclasses import replace

    class Verifying(ScriptedLLM):
        def chat(self, messages, seed=None):
            if messages[0]["content"].startswith("Bạn là người kiểm tra"):
                return "SAI: chưa lọc đơn huỷ" if "999" in messages[-1]["content"] else "OK"
            return super().chat(messages, seed)

    wrong = '<answer>{"type": "number", "answer": 999}</answer>'
    llm = Verifying([CODE, wrong, CODE, ANS])
    ans, trace = solve_once(TASK, replace(ctx, verify=True, max_turns=6), llm, seed=0)
    assert ans == {"type": "number", "answer": 350}
    assert trace[1]["verifier"] == "chưa lọc đơn huỷ"


def test_guessed_answer_next_to_code_is_hidden_from_history(ctx):
    llm = ScriptedLLM([CODE + "\n" + '<answer>{"type": "number", "answer": 999}</answer>', ANS])
    solve_once(TASK, ctx, llm, seed=0)
    assert not any("999" in m["content"] for m in llm.calls[-1] if m["role"] == "assistant")


def test_abbreviations_explained_with_dates():
    from datamate.agent import abbr_hint
    h = abbr_hint("Cho tôi biết số đơn hqua, dthu bn", dt.date(2026, 2, 10))
    assert "'hqua' = hôm qua (2026-02-09)" in h and "'dthu' = doanh thu" in h and "'bn' = bao nhiêu" in h
    assert abbr_hint("Savi Losartan 50mg hộp 3 vỉ x 10 viên", dt.date(2026, 2, 10)) == ""


def test_month_shorthand_explained():
    from datamate.agent import month_hint
    h = month_hint("Máy cũ 8/25 bán đc mấy chiếc vậy?")
    assert "'8/25' = tháng 8/2025, P.month(8, 2025)" in h and "không phải ngày 25 tháng 8" in h
    assert "tháng 8/2026" in month_hint("8/26 tiền bán ca chiều là bnhiu?")
    for q in ["Quý 3/26 lượt bảo hành", "tháng 6/2025 bán", "Tổng tiền 02/2026?", "Từ 1/1 đến hết hnay", "Den ngay 13/03/2026 co may khach",
              "ngày 3/12 bán bao nhiêu", "hết ngày 8/2/2026 shop"]:
        assert month_hint(q) == "", q


def test_answer_written_inside_code_gets_told_off(ctx):
    bad = "```python\nif True:\n    <answer>{\"type\": \"number\", \"answer\": 1}</answer>\n```"
    llm = ScriptedLLM([bad, CODE, ANS])
    ans, _ = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}
    assert any("Không viết <answer> bên trong code" in m["content"] for m in llm.calls[-1] if m["role"] == "user")


def test_unanswerable_before_any_successful_run_is_sent_back(ctx):
    bad = "```python\nprint(T['khong_co'])\n```"
    llm = ScriptedLLM([bad, UNANS, CODE, ANS])
    ans, trace = solve_once(TASK, ctx, llm, seed=0)
    assert ans == {"type": "number", "answer": 350}
    assert trace[1]["recheck_no_run"]["type"] == "unanswerable"


def test_second_missing_column_gets_short_hint(ctx):
    bad1 = "```python\nprint(T['ban_hang']['Tiền'].sum())\n```"
    bad2 = "```python\nprint(T['ban_hang']['SP'].nunique())\n```"
    llm = ScriptedLLM([bad1, bad2, CODE, ANS])
    solve_once(TASK, ctx, llm, seed=0)
    fb = [m["content"] for m in llm.calls[-1] if m["role"] == "user"]
    assert "Cột thật của các bảng" in fb[1] and "Không có cột 'SP'. Cột gần tên nhất: T['ban_hang']['sp']" in fb[2]
    assert "Cột thật của các bảng" not in fb[2]


def test_period_hint_resolves_dates_and_skips_ambiguous():
    from datamate.agent import period_hint
    d = dt.date(2026, 9, 19)
    assert "tháng: 2026-08-01 → 2026-08-31" in period_hint("8/26 tiền bán ca chiều là bnhiu?", d, None)
    assert "tuần trước: 2026-09-07 → 2026-09-13" in period_hint("tuần trước bán bao nhiêu", d, None)
    assert period_hint("Tính đến hết 10/10/2025, nợ quá 7 ngày còn bnhiu?", dt.date(2025, 10, 10), None) == ""
    assert period_hint("doanh thu từ tháng 2/2025 đến tháng 4/2025", d, None) == ""
    assert period_hint("hôm nay bán bao nhiêu", d, None) == ""


def test_unanswerable_after_successful_run_is_abstained(ctx):
    llm = ScriptedLLM({1000: [CODE, UNANS, UNANS], 1001: [CODE, UNANS, UNANS]})
    final, info = solve(TASK, ctx, llm, n_samples=2, max_samples=3, base_seed=1000)
    assert final == {"type": "abstain"} and info["abstained_unanswerable_after_run"]["type"] == "unanswerable"


def test_plan_parse_execute_and_vote():
    from datamate.plan import execute, parse_plan, solve_plan

    class Sem:
        def revenue(self, ky):
            return {(dt.date(2026, 8, 1), dt.date(2026, 8, 31)): 1200, (dt.date(2026, 7, 1), dt.date(2026, 7, 31)): 1000}[ky]

    two = ('{"op": "pct_change", "calls": [{"fn": "revenue", "args": {"ky": {"tu": "2026-08-01", "den": "2026-08-31"}}},'
           '{"fn": "revenue", "args": {"ky": {"tu": "2026-07-01", "den": "2026-07-31"}}}], "bo_sot": []}')
    assert execute(parse_plan(two), Sem()) == {"type": "percent", "answer": 20.0}
    assert parse_plan('{"op": "value", "calls": [{"fn": "revenue", "args": {}}], "bo_sot": []}') is None
    assert parse_plan('{"op": "value", "calls": [{"fn": "revenue", "args": {"ky": {}}}], "bo_sot": ["khách A"]}') is None
    q = "Doanh thu tháng 8/2026 tăng bao nhiêu % so với tháng 7/2026?"
    ans, _ = solve_plan(q, dt.date(2026, 9, 19), Sem(), ScriptedLLM({2000: [two], 2001: [two]}))
    assert ans == {"type": "percent", "answer": 20.0}
    none = '{"op": "none", "calls": [], "bo_sot": []}'
    ans, _ = solve_plan(q, dt.date(2026, 9, 19), Sem(), ScriptedLLM({2000: [two], 2001: [none]}), samples=2)
    assert ans is None


def test_zero_answer_is_abstained(ctx):
    zero = '<answer>{"type": "number", "answer": 0}</answer>'
    llm = ScriptedLLM({1000: [CODE, zero], 1001: [CODE, zero]})
    final, info = solve(TASK, ctx, llm, n_samples=2, max_samples=3, base_seed=1000)
    assert final == {"type": "abstain"} and info["abstained_zero"]["answer"] == 0


def test_is_zero_answer():
    from datamate.agent import is_zero_answer
    assert is_zero_answer({"type": "percent", "answer": 0.0})
    assert is_zero_answer({"type": "list", "answer": [3, 0]})
    assert not is_zero_answer({"type": "boolean", "answer": False})
    assert not is_zero_answer({"type": "list", "answer": [False, 2]})
    assert not is_zero_answer({"type": "number", "answer": 12})


class _FakeSem:
    def cancel_rate(self, ky):
        return 1.02

    def cancelled_count(self, ky):
        return 7

    def shift_order_count(self, ky, ca):
        return 169 if ca == "chieu" else None


def test_sem_facts_maps_question_to_one_metric_with_period():
    from datamate.agent import sem_facts
    f = sem_facts("Tháng 8/2025 tiệm bị huỷ mấy % đơn?", dt.date(2026, 1, 10), _FakeSem())
    assert f == [("SEM.cancel_rate(ky={'tu': '2025-08-01', 'den': '2025-08-31'})", "percent", 1.02)]
    f = sem_facts("02/2025 ca chieu len duoc bao nhieu don?", dt.date(2026, 1, 10), _FakeSem())
    assert [x[2] for x in f] == [169]
    assert sem_facts("doanh thu tháng 8/2025", dt.date(2026, 1, 10), _FakeSem()) == []


def test_fact_conflict_only_for_numbers_dates():
    from datamate.agent import fact_conflict
    facts = [("SEM.cancel_rate(...)", "percent", 1.02), ("SEM.big_spender(...)", "string", "KH0060")]
    assert fact_conflict({"type": "percent", "answer": 1.15}, facts)
    assert not fact_conflict({"type": "percent", "answer": 1.02}, facts)
    assert not fact_conflict({"type": "string", "answer": "Hồ Cường Hùng"}, facts)
    assert not fact_conflict({"type": "number", "answer": 5}, [])
