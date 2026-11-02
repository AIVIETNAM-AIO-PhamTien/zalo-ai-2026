"""Chạy agent trên một split (train / public_test / private) và ghi submission.jsonl + log từng câu.

    python -m datamate.run --data data/public_test --out outputs/public/submission.jsonl \
        --llm-url http://localhost:8000/v1 --model Qwen/Qwen3-14B

Cấu trúc --data: questions.jsonl, rulebook.md, calendar.csv, data/<business_id>/…
Ghi từng dòng ngay khi xong (chạy lại sẽ bỏ qua câu đã có); câu chưa xong khi hết giờ ghi abstain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import hashlib
import pickle
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from datamate.agent import ShopContext, Task, solve
from datamate.baseline import answer as baseline_answer
from datamate import auxiliary
from datamate.baseline import debt_any, delivery_answer, lot_answer, prepare_sem
from datamate.brief import build_brief
from datamate.shots import ShotBank
from datamate.helpers import Periods
from datamate.linking import column_index, rule_rows
from datamate.lint import build_hints
from datamate.unanswerable import ShopProfile
from datamate.debt import DebtView
from datamate.delivery import DeliveryView
from datamate.lots import LotView
from datamate.unanswerable import check as check_unanswerable
from datamate.semantic import view_report, OrderView, parse_shifts
from datamate.llm import LLM
from datamate.loader import CACHE_VERSION, load_shop
from datamate.schema_card import load_card, make_card
from datamate.plan import solve_plan
from datamate import selector
from datamate.shop_type import shop_type
from datamate.rule_guard import Guard
from datamate.sem_api import SemAPI
from datamate.agent import abbr_hint, period_hint, sem_facts


def find_split_files(root: Path) -> dict:
    root = Path(root)
    q = root / "questions.jsonl"
    if not q.exists():
        found = sorted(root.rglob("questions.jsonl"))
        if not found:
            raise FileNotFoundError(f"không thấy questions.jsonl trong {root}")
        q = found[0]
    base = q.parent
    data_dir = base / "data" if (base / "data").is_dir() else base
    return {"questions": q, "rulebook": base / "rulebook.md", "calendar": base / "calendar.csv", "data": data_dir}


def _ingest(biz_dir: str, cache_dir: str) -> tuple[str, str, float]:
    t0 = time.time()
    biz = Path(biz_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    pkl = Path(cache_dir) / f"{biz.name}.v{CACHE_VERSION}.pkl"
    brief_path = Path(cache_dir) / f"{biz.name}.v{CACHE_VERSION}.brief.md"
    if not (pkl.exists() and brief_path.exists()):
        shop = load_shop(biz)
        pkl.write_bytes(pickle.dumps(shop, protocol=pickle.HIGHEST_PROTOCOL))
        brief_path.write_text(build_brief(shop), encoding="utf-8")
    return biz.name, str(pkl), time.time() - t0


def _prebuild_gold(b: str, cache_path: str, rulebook: str, card) -> tuple[str, float, str, dict]:
    """Dựng sẵn phần nặng của G (lưu cạnh cache) để tiến trình chạy code của agent chỉ đọc file; trả mô tả G cho
    prompt và báo cáo G (bảng có / thiếu / lý do) cho log."""
    from datamate.gold import agent_tables, gold_brief, gold_store_path
    t0 = time.time()
    shop = pickle.loads(Path(cache_path).read_bytes())
    if card is not None:
        shop.card = card
    G = agent_tables(shop, None, rulebook, gold_store_path(cache_path, rulebook, card))
    return b, time.time() - t0, gold_brief(G), G["_bao_cao"]


def ingest_all(data_dir: Path, biz_ids: list[str], cache_dir: Path, workers: int) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_ingest, str(data_dir / b), str(cache_dir)) for b in biz_ids]
        for f in as_completed(futs):
            name, pkl, sec = f.result()
            out[name] = Path(pkl)
            print(f"[ingest] {name} {sec:.1f}s", flush=True)
    return out


def wait_llm(url: str, model: str, wait: float) -> None:
    """Chờ tới khi model trả lời được một câu ngắn (không chỉ cổng mở); quá `wait` giây thì dừng chương trình.
    Log Colab B9: vLLM chưa lên mà vẫn chạy → 192/192 câu ghi abstain (Connection refused), lượt chạy lại bỏ qua."""
    t0, last = time.time(), None
    probe = LLM(url, model, temperature=0, max_tokens=4, enable_thinking=False, timeout=60)
    while True:
        try:
            probe.chat([{"role": "user", "content": "hi"}])
            print(f"[run] model sẵn sàng ({time.time() - t0:.0f}s)", flush=True)
            return
        except Exception as e:  # noqa: BLE001 — mọi lỗi kết nối / chưa nạp xong đều chờ tiếp
            last = e
        if time.time() - t0 > wait:
            sys.exit(f"[run] DỪNG: model chưa sẵn sàng sau {wait:.0f}s ({last!r}) — không ghi câu nào")
        time.sleep(10)


# Cờ nền của lượt Bước 9 (notebooks/colab_buoc9_s2pub_v4.ipynb). Cơ chế agent (shots, lint, verify, col_hints,
# reconcile, diverse) KHÔNG nằm ở đây: bật riêng từng cái để tách tác dụng; dòng "[run] cờ agent đang TẮT" nhắc.
PROFILES = {
    "b9": {"plan": True, "plan_all": True, "plan_shots": "datamate/plan_shots.jsonl",
           "selector": "datamate/selector_b3.json", "logprobs": True, "exec_timeout": 60},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--cache", default="/tmp/datamate_cache")
    ap.add_argument("--llm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=3)
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--ingest-workers", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--llm-timeout", type=float, default=600, help="giây mỗi lượt gọi model (T4 chậm: đặt cao hơn)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--llm-wait", type=float, default=1800,
                    help="giây chờ model sẵn sàng trước khi chạy; hết hạn thì dừng, không ghi câu trống; 0 = không chờ")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--sem-facts", action="store_true",
                    help="máy gọi sẵn hàm SEM khớp câu hỏi, đưa giá trị vào prompt agent, đáp án lệch → bỏ trống")
    ap.add_argument("--plan", action="store_true",
                    help="trước agent: LLM chuyển câu thành kế hoạch gọi SEM, máy tính (datamate/plan.py)")
    ap.add_argument("--plan-shots", default=None,
                    help="kho ví dụ kế hoạch đúng (datamate/plan_shots.jsonl, scripts/build_plan_shots.py), bỏ ví dụ cùng ngành")
    ap.add_argument("--plan-all", action="store_true",
                    help="chạy cả kế hoạch trên câu tầng tất định đã trả lời, chỉ ghi plan_check vào log (không đổi đáp án)")
    ap.add_argument("--cards", default=None, help="thư mục <biz>.card.json: vai trò cột LLM gán (schema_card.py); 'auto' = sinh lúc chạy")
    ap.add_argument("--no-baseline", action="store_true", help="bỏ tầng giải tất định, mọi câu đi qua agent")
    ap.add_argument("--agent-only", action="store_true", help="đo agent: tắt baseline, plan và sem-facts; vẫn đọc rulebook")
    ap.add_argument("--checked-joins", action="store_true", help="thử nghiệm: hướng agent dùng phép nối có kiểm quan hệ")
    ap.add_argument("--selector", default=None,
                    help="file hệ số bộ chọn (scripts/calibrate_select.py --save): chọn giữa tất định / kế hoạch / agent "
                         "theo P(đúng), dưới ngưỡng thì bỏ trống; nên kèm --plan-all để có ứng viên kế hoạch")
    ap.add_argument("--exec-timeout", type=float, default=120)
    ap.add_argument("--time-budget", type=float, default=5.5 * 3600, help="giây; quá hạn thì ngừng nhận câu mới")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--biz", nargs="*", default=None)
    ap.add_argument("--ids", default=None, help="file chứa danh sách id cần chạy, mỗi dòng một id")
    ap.add_argument("--shots", default=None, help="kho ví dụ few-shot (jsonl, scripts/build_shots.py)")
    # phát lại trên log train cũ: mẫu bị nhắc đúng 48%, không bị nhắc 53% → chưa rõ lợi, mặc định tắt, đo bằng GPU
    ap.add_argument("--lint", action="store_true", help="bật nhắc lỗi code hay gặp (datamate/lint.py)")
    ap.add_argument("--diverse", action="store_true", help="mẫu thứ hai không few-shot (lời giải đa dạng)")
    ap.add_argument("--no-rule-hints", action="store_true", help="tắt gợi ý luật rulebook liên quan dưới câu hỏi")
    ap.add_argument("--logprobs", action="store_true", help="ghi logprob của khối <answer> vào log (để hiệu chỉnh)")
    ap.add_argument("--verify", action="store_true", help="verifier LLM kiểm đáp án mỗi mẫu một lần (DS-STAR)")
    ap.add_argument("--col-hints", action="store_true", help="bật gợi ý cột ứng viên (còn nhiễu, mặc định tắt)")
    ap.add_argument("--reconcile", action="store_true",
                    help="mẫu thêm khi bất đồng được xem các đáp án + code đang mâu thuẫn để tự kiểm chứng (ReFoRCE)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None,
                    help="bộ cờ nền đã chốt (xem PROFILES); cờ gõ tay vẫn bổ sung được. Chống sót cờ khi dựng notebook")
    args = ap.parse_args(argv)
    for k, v in PROFILES.get(args.profile, {}).items():
        if getattr(args, k) == ap.get_default(k):
            setattr(args, k, v)
    off = [f for f in ("shots", "lint", "verify", "col_hints", "reconcile", "diverse") if not getattr(args, f)]
    print(f"[run] cờ agent đang TẮT: {', '.join(off) or 'không'}" + (f" (profile {args.profile})" if args.profile else ""),
          flush=True)
    if args.agent_only:
        args.no_baseline, args.plan, args.sem_facts = True, False, False

    t_start = time.time()
    files = find_split_files(Path(args.data))
    questions = [json.loads(l) for l in files["questions"].read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.biz:
        questions = [q for q in questions if q["business_id"] in set(args.biz)]
    if args.ids:
        keep = set(Path(args.ids).read_text(encoding="utf-8").split())
        questions = [q for q in questions if q["id"] in keep]
    if args.limit:
        questions = questions[:args.limit]
    questions.sort(key=lambda q: (q["business_id"], q["id"]))  # cùng shop liền nhau → tận dụng prefix cache

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs) if args.logs else out_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [q for q in questions if q["id"] not in done]
    manifest_path = out_path.parent / "run_manifest.json"
    mode = {"agent_only": args.agent_only, "baseline": not args.no_baseline,
            "plan": args.plan, "sem_facts": args.sem_facts, "checked_joins": args.checked_joins,
            "model": args.model, "samples": args.samples, "max_samples": args.max_samples,
            "min_agree": args.min_agree, "verify": args.verify,
            "diverse": args.diverse, "reconcile": args.reconcile, "max_turns": args.max_turns}
    mode.update(temperature=args.temperature, max_tokens=args.max_tokens, no_thinking=args.no_thinking,
                lint=args.lint, col_hints=args.col_hints,
                rule_hints=not args.no_rule_hints, cards=args.cards, shots=args.shots,
                agent_sem_recall=True, execution_result_protocol=True)
    if done and args.agent_only:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if previous.get("mode") != mode:
            ap.error("Agent-only cần thư mục output mới hoặc manifest cùng cấu hình; không trộn log với lượt cũ")
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    manifest = {"mode": mode, "code_sha256": digest.hexdigest(),
                "questions_sha256": hashlib.sha256(json.dumps(questions, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "question_count": len(questions), "resumed_rows": len(done),
                "rulebook_sha256": hashlib.sha256(files["rulebook"].read_bytes()).hexdigest()}
    if done and args.agent_only and any(previous.get(k) != manifest[k] for k in
                                        ("code_sha256", "questions_sha256", "rulebook_sha256")):
        ap.error("Code hoặc bộ câu/rulebook đã đổi; dùng output mới để đo agent-only")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[run] {len(questions)} câu, đã có {len(questions) - len(todo)}, cần chạy {len(todo)}", flush=True)

    if todo and args.llm_wait > 0:
        wait_llm(args.llm_url, args.model, args.llm_wait)
    biz_ids = sorted({q["business_id"] for q in todo})
    caches = ingest_all(files["data"], biz_ids, Path(args.cache), args.ingest_workers)
    rulebook = files["rulebook"].read_text(encoding="utf-8")
    guard = Guard(rulebook)  # luật đã cài mà rulebook split viết khác → tắt phép tính dựa trên luật đó
    print(f"[run] canh luật: {guard.summary()}", flush=True)
    calendar = files["calendar"] if files["calendar"].exists() else None
    shots = ShotBank(args.shots) if args.shots else None
    plan_shots = ShotBank(args.plan_shots) if args.plan_shots else None
    sel_cfg = selector.load_cfg(args.selector) if args.selector else None
    sel_ctx = {}  # Bước 9: đặc trưng shop cho bộ chọn (chỉ khi cấu hình học với --ctx)
    if sel_cfg is not None and sel_cfg.get("ctx"):
        rb_text = files["rulebook"].read_text(encoding="utf-8") if files["rulebook"].exists() else None
        for b, path in caches.items():
            sel_ctx[b] = selector.shop_context(pickle.loads(Path(path).read_bytes()), rb_text)
    shop_types = {}
    if shots:
        print(f"[run] few-shot: {len(shots.shots)} ví dụ", flush=True)
    auto_cards = {}
    if args.cards == "auto":  # shop lạ: model gán vai trò cột ngay lúc chạy, song song mọi shop (một lượt / shop)
        # không thinking: chỉ gán vai trò cột → JSON; Kaggle T4 có thinking 11/14 thẻ lỗi (hết giờ chờ)
        card_llm = LLM(args.llm_url, args.model, temperature=0, max_tokens=2048, enable_thinking=False,
                       timeout=args.llm_timeout)

        def card_job(b):
            brief = (Path(args.cache) / f"{b}.v{CACHE_VERSION}.brief.md").read_text(encoding="utf-8")
            return make_card(pickle.loads(caches[b].read_bytes()), brief, card_llm, Path(args.cache) / "cards", b)

        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(biz_ids)))) as pool:
            futs = {pool.submit(card_job, b): b for b in biz_ids}
            for f in as_completed(futs):
                try:
                    auto_cards[futs[f]] = f.result()
                except Exception as e:
                    print(f"[run] {futs[f]}: không sinh được thẻ schema ({e!r})", flush=True)
        print(f"[run] thẻ schema: {sum(bool(c) for c in auto_cards.values())}/{len(biz_ids)} shop có vai trò cột",
              flush=True)
    contexts = {}
    views, lots, debts, deliveries, ledgers = {}, {}, {}, {}, {}
    profiles = {}
    shops = {}
    for b in biz_ids:
        readme_p = files["data"] / b / "README.md"
        shop = pickle.loads(caches[b].read_bytes())
        if args.cards == "auto":
            shop.card = auto_cards.get(b) or None
        else:
            shop.card = load_card(Path(args.cards), b) if args.cards else None
        hints = None
        if args.lint:
            try:
                hints = build_hints(shop)
            except Exception as e:
                print(f"[run] {b}: không dựng được gợi ý lint ({e!r})", flush=True)
        contexts[b] = ShopContext(
            cache_path=caches[b], calendar_csv=calendar,
            readme=readme_p.read_text(encoding="utf-8") if readme_p.exists() else "",
            brief=(Path(args.cache) / f"{b}.v{CACHE_VERSION}.brief.md").read_text(encoding="utf-8"),
            rulebook=rulebook, exec_timeout=args.exec_timeout, max_turns=args.max_turns, shots=shots,
            lint_hints=hints, rule_rows=None if args.no_rule_hints else rule_rows(rulebook),
            col_index=column_index(shop) if args.col_hints else None, verify=args.verify,
            checked_joins=args.checked_joins, schema_card=shop.card)
        if not args.no_baseline:
            ledgers[b] = auxiliary.prepare(shop, rulebook)
            try:
                views[b] = OrderView.from_shop(shop)
                if views[b] is not None:
                    views[b].shifts = parse_shifts(rulebook)
                contexts[b].brief += "\n\n" + view_report(views[b])
            except Exception as e:
                print(f"[run] {b}: không dựng được bảng đơn hàng chuẩn ({e!r})", flush=True)
            try:
                deliveries[b] = DeliveryView.from_shop(shop)
            except Exception as e:
                print(f"[run] {b}: không dựng được sổ giao hàng ({e!r})", flush=True)
            try:
                debts[b] = DebtView.from_shop(shop)
            except Exception as e:
                print(f"[run] {b}: không dựng được bảng công nợ ({e!r})", flush=True)
            try:
                lots[b] = LotView.from_shop(shop)
            except Exception as e:
                print(f"[run] {b}: không dựng được bảng lô ({e!r})", flush=True)
            try:
                v = views.get(b)
                profiles[b] = ShopProfile.from_shop(shop, v.date_range()[0] if v is not None else None)
            except Exception as e:
                print(f"[run] {b}: không dựng được hồ sơ shop ({e!r})", flush=True)
        if not args.no_baseline and views.get(b) is not None:
            try:
                prepare_sem(views[b], shop, rulebook, calendar)
            except Exception as e:
                print(f"[run] {b}: không dựng được SEM tất định ({e!r})", flush=True)
        if args.plan or args.sem_facts:
            shops[b] = shop  # SEM dựng lười các bảng phụ (công nợ, nhập, lô) từ shop
        del shop
    gold_reports = {}
    with ProcessPoolExecutor(max_workers=args.ingest_workers) as pool:
        futs = [pool.submit(_prebuild_gold, b, str(contexts[b].cache_path), rulebook, contexts[b].schema_card)
                for b in biz_ids]
        for f in as_completed(futs):
            try:
                b, sec, brief, rep = f.result()
                contexts[b].gold_brief, gold_reports[b] = brief, rep
                if contexts[b].lint_hints is not None:
                    contexts[b].lint_hints.gold_tables = set(rep.get("tables", []))
                print(f"[gold] {b} {sec:.1f}s, bảng: {', '.join(rep.get('tables', [])) or rep.get('why', '-')}",
                      flush=True)
            except Exception as e:
                print(f"[run] không dựng sẵn được G ({e!r}); agent sẽ tự dựng", flush=True)
    if not args.no_baseline:
        print(f"[run] bộ giải tất định dùng được cho {sum(v is not None for v in views.values())}/{len(biz_ids)} shop",
              flush=True)

    llm = LLM(args.llm_url, args.model, temperature=args.temperature, max_tokens=args.max_tokens,
              enable_thinking=not args.no_thinking, logprobs=args.logprobs,
              timeout=args.llm_timeout)
    # kế hoạch: tham lam, không thinking (câu ngắn; tái lập được khi BTC chạy lại)
    plan_llm = LLM(args.llm_url, args.model, temperature=0, max_tokens=2048, enable_thinking=False,
                   timeout=args.llm_timeout)
    lock = threading.Lock()
    deadline = t_start + args.time_budget
    stats = {"done": 0, "abstain": 0, "error": 0, "backend_streak": 0}
    abort = threading.Event()

    def write(rec: dict):
        with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def job(q: dict):
        if time.time() > deadline or abort.is_set():
            return
        t0 = time.time()
        task = Task(q["id"], q["business_id"], q["question"], dt.date.fromisoformat(q["as_of_date"]))
        final = None
        info = {}
        prof = profiles.get(q["business_id"])
        if prof is not None:  # tầng 0: luật chắc chắn không trả lời được (train: bắt 30/70, nhầm 0)
            try:
                final = check_unanswerable(task.question, task.as_of, prof, calendar)
                info = {"source": "unanswerable_rule"}
            except Exception:
                final = None
        lv = lots.get(q["business_id"])
        if final is None and lv is not None and not guard.lot:  # hạn dùng theo lô (không cần bảng đơn chuẩn)
            try:
                final = lot_answer(task.question, task.as_of, lv)
                info = {"source": "baseline"}
            except Exception:
                final = None
        dl = deliveries.get(q["business_id"])
        if final is None and dl is not None and not guard.delivery:  # số đợt giao (sổ giao hàng nhiều đợt)
            try:
                final = delivery_answer(task.question, task.as_of, dl, Periods(task.as_of, calendar))
                info = {"source": "baseline"}
            except Exception:
                final = None
        dv = debts.get(q["business_id"])
        if final is None and dv is not None and not guard.debt:  # công nợ (bảng thanh toán + đơn, không cần bảng đơn chuẩn)
            try:
                final = debt_any(task.question, task.as_of, dv, Periods(task.as_of, calendar))
                info = {"source": "baseline"}
            except Exception:
                final = None
        view = views.get(q["business_id"])
        if final is None and view is not None and not guard.view:  # tầng 1: bộ giải tất định (chỉ trả lời khi nhận diện trọn câu hỏi)
            try:
                final = baseline_answer(task.question, task.as_of, view, Periods(task.as_of, calendar))
                info = {"source": "baseline"}
            except Exception:
                final = None
        if final is None and not args.no_baseline:
            final = auxiliary.answer(task.question, task.as_of, ledgers.get(q['business_id'], []), Periods(task.as_of, calendar))
            if final is not None:
                info = {'source': 'baseline'}
        checking = final is not None
        if (final is None or args.plan_all) and args.plan and guard.plan and q["business_id"] in shops:  # tầng 1b: kế hoạch → SEM
            try:
                sem = SemAPI(shops[q["business_id"]], task.as_of, rulebook)
                sem._view, sem._built = views.get(q["business_id"]), True
                sem._calendar = calendar  # kiểm kỳ Tết của kế hoạch (plan.check_plan)
                hints = "\n".join(x for x in (abbr_hint(task.question, task.as_of),
                                              period_hint(task.question, task.as_of, calendar)) if x)
                pshots = None
                if plan_shots:
                    b = q["business_id"]
                    if b not in shop_types:
                        shop_types[b] = shop_type(shops[b])
                    pshots = plan_shots.find(task.question, b, exclude_id=q["id"], exclude_type=shop_types[b])
                pfinal, pinfo = solve_plan(task.question, task.as_of, sem, plan_llm, hints,
                                           blocked=guard.plan_blocked, shots=pshots)
                if checking:  # đáp án tất định giữ nguyên; kế hoạch chỉ để đối chiếu
                    info["plan_check"] = {"answer": pfinal, "agree": pfinal == final, "plan": pinfo}
                else:
                    final, info = pfinal, {"source": "plan", "plan": pinfo}
            except Exception:
                if checking:
                    info["plan_check"] = {"error": traceback.format_exc()[-1500:]}
                else:
                    final, info = None, {"plan_error": traceback.format_exc()[-1500:]}
        if final is None and args.sem_facts and not guard.core and q["business_id"] in shops:
            try:  # máy gọi sẵn hàm SEM khớp câu → đưa vào prompt agent + chặn đáp án mâu thuẫn
                sem = SemAPI(shops[q["business_id"]], task.as_of, rulebook)
                sem._view, sem._built = views.get(q["business_id"]), True
                task.facts = [f for f in sem_facts(task.question, task.as_of, sem, calendar)
                              if f[0][4:].split("(")[0] not in guard.plan_blocked]
            except Exception:
                task.facts = []
        if final is None:  # tầng 2: agent LLM viết code
            tried = {k: info[k] for k in ("plan", "plan_error") if k in info}  # giữ log kế hoạch không được nhận
            try:
                final, info = solve(task, contexts[q["business_id"]], llm, n_samples=args.samples,
                                    max_samples=args.max_samples, min_agree=args.min_agree,
                                    base_seed=1000, diverse=args.diverse, reconcile=args.reconcile)
                info["source"] = "agent"
            except Exception:
                final, info = {"type": "abstain"}, {"error": traceback.format_exc(), "source": "agent"}
                stats["error"] += 1
            info.update(tried)
        # luật tầng 0 (train: bắt 30/70, nhầm 0) không là ứng viên của bộ chọn: trước đây bị bộ chọn đổi thành abstain
        # (log Colab B9 v4 public: 10/10 câu như 'khách nữ', 'khách thân thiện nhất' ra abstain thay vì unanswerable)
        if sel_cfg is not None and info.get("source") != "unanswerable_rule":  # bộ chọn hiệu chỉnh (datamate/selector.py)
            pc = info.get("plan_check") or {}
            cand = {"question": q["question"], "ctx": sel_ctx.get(q["business_id"]) if sel_cfg.get("ctx") else None,
                    "det": final if info.get("source") == "baseline" else None,
                    "dsl": selector._plans(pc.get("plan") or info.get("plan")),
                    "agent": info if info.get("source") == "agent" else None}
            src, ans, p = selector.choose(cand, sel_cfg)
            # Đã thử giữ đáp án đồng thuận khi agent là ứng viên duy nhất ("agent_only"): public 0.616 < 0.633 (bản
            # safe không có nhánh này) → bỏ, để bộ chọn quyết định.
            info["selector"] = {"source": src, "p": round(p, 4), "before": final}
            final = ans or {"type": "abstain"}
        outcomes = info.get("sample_outcomes") or []
        if info.get("source") == "agent" and outcomes and all(o == "backend_error" for o in outcomes):
            with lock:
                stats["backend_streak"] += 1
                if stats["backend_streak"] >= 5 and not abort.is_set():
                    abort.set()
                    print("[run] DỪNG: 5 câu liền model không trả lời (lỗi kết nối) — câu chưa xong không ghi, "
                          "khởi động lại model rồi chạy lại lệnh này để làm tiếp", flush=True)
            return  # không ghi: chạy lại sẽ làm lại câu này
        with lock:
            stats["backend_streak"] = 0
        rec = {"id": q["id"], **final}
        write(rec)
        elapsed = time.time() - t0
        log = {"id": q["id"], "business_id": q["business_id"], "question": q["question"],
               "gold": gold_reports.get(q["business_id"]),
               "as_of_date": q["as_of_date"], "final": final, "elapsed_sec": round(elapsed, 1), **info}
        (logs_dir / f"{q['id']}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
        with lock:
            stats["done"] += 1
            stats["abstain"] += final["type"] == "abstain"
            n = stats["done"]
        if n % 10 == 0 or n == len(todo):
            rate = (time.time() - t_start) / n
            print(f"[run] {n}/{len(todo)} xong, abstain={stats['abstain']}, lỗi={stats['error']}, "
                  f"{rate:.1f}s/câu (tổng), câu này {elapsed:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(job, todo))

    if abort.is_set():
        sys.exit("[run] dừng vì model không trả lời — chưa điền abstain cho câu còn thiếu")
    # câu chưa chạy (hết giờ) → abstain để file đủ dòng
    have = set()
    for l in out_path.read_text(encoding="utf-8").splitlines():
        try:
            have.add(json.loads(l)["id"])
        except Exception:
            pass
    for q in questions:
        if q["id"] not in have:
            write({"id": q["id"], "type": "abstain"})
    print(f"[run] xong sau {(time.time() - t_start) / 60:.1f} phút → {out_path}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
