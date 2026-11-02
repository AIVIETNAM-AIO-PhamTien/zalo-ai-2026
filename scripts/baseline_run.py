"""Chạy bộ giải tất định (không LLM) trên một split; câu không giải được ghi abstain.

    python scripts/baseline_run.py data/train --cache outputs/cache --out outputs/baseline_train.jsonl --score
    python scripts/baseline_run.py data/public_test --cache outputs/cache --out outputs/baseline_public/submission.jsonl
"""
import argparse
import datetime as dt
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.schema_card import load_card  # noqa: E402
from datamate.baseline import answer, debt_any, delivery_answer, lot_answer, stock_answer  # noqa: E402
from datamate.stock import StockView, base_units  # noqa: E402
from datamate.delivery import DeliveryView  # noqa: E402
from datamate.debt import DebtView  # noqa: E402
from datamate.helpers import Periods  # noqa: E402
from datamate.loader import CACHE_VERSION  # noqa: E402
from datamate.lots import LotView  # noqa: E402
from datamate.run import _ingest, find_split_files  # noqa: E402
from datamate.scoring import score_item  # noqa: E402
from datamate.semantic import OrderView, parse_shifts  # noqa: E402
from datamate.unanswerable import ShopProfile  # noqa: E402
from datamate.unanswerable import check as check_unanswerable  # noqa: E402
from datamate.baseline import prepare_sem  # noqa: E402
from datamate import auxiliary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--cache", default="outputs/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cards", default=None, help="thư mục <biz>.card.json (datamate/schema_card.py)")
    ap.add_argument("--show-wrong", type=int, default=40)
    ap.add_argument("--no-sem-baseline", action="store_true", help="tắt các SEM mới để so A/B")
    ap.add_argument("--no-public-rules", action="store_true", help="đo A/B: tắt các nhánh public mới, giữ cùng schema")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8, help="số shop chạy song song")
    a = ap.parse_args()

    files = find_split_files(Path(a.data))
    qs = [json.loads(l) for l in files["questions"].read_text(encoding="utf-8").splitlines() if l.strip()]
    cal = files["calendar"] if files["calendar"].exists() else None
    rulebook = files["rulebook"].read_text(encoding="utf-8")
    by_biz = {}
    for q in qs:
        by_biz.setdefault(q["business_id"], []).append(q)
    # mỗi shop một việc (dựng view + trả lời độc lập giữa các shop); shop lớn chạy trước
    jobs = [(b, items, str(files["data"] / b), a.cache, a.cards, rulebook, cal, a.no_sem_baseline, a.no_public_rules)
            for b, items in sorted(by_biz.items(), key=lambda kv: -len(kv[1]))]
    got = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for rows in pool.map(solve_shop, *zip(*jobs)):
            got.update((r["id"], r) for r in rows)
    out = [got[q["id"]] for q in qs]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"[baseline] {len(out)} câu, trả lời {sum(r['type'] != 'abstain' for r in out)} → {a.out}")
    print("  type:", dict(Counter(r["type"] for r in out)))

    if a.score:
        refs = {}
        for l in (files["questions"].parent / "answers.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                refs[r["id"]] = r
        qmap = {q["id"]: q for q in qs}
        s = Counter()
        wrong = []
        for r in out:
            sc = score_item(r if r["type"] != "abstain" else None, refs[r["id"]])
            s[sc] += 1
            if sc == -1:
                wrong.append((qmap[r["id"]], r, refs[r["id"]]))
        n = len(out)
        print(f"PA={(s[1] - s[-1]) / n:.4f} correct={s[1]} wrong={s[-1]} abstain={s[0]} "
              f"precision={s[1] / max(1, s[1] + s[-1]):.3f}")
        for q, r, ref in wrong[:a.show_wrong]:
            print(f"  ✗ {q['business_id']} {q['as_of_date']} | {q['question']} | got {r.get('answer')} "
                  f"({r['type']}) | ref {ref['answer']} ({ref['type']})")


def solve_shop(b, qs, data_dir, cache, cards, rulebook, cal, no_sem_baseline=False, no_public_rules=False):
    """Dựng các view của một shop rồi trả lời câu của shop đó → [{id, type, …}] (abstain khi không giải được)."""
    _ingest(data_dir, cache)
    shop = pickle.loads((Path(cache) / f"{b}.v{CACHE_VERSION}.pkl").read_bytes())
    if cards:
        shop.card = load_card(Path(cards), b)
    try:
        v = OrderView.from_shop(shop)
        if v is not None:
            v.shifts = parse_shifts(rulebook)
    except Exception as e:  # shop lạ: bỏ qua, abstain
        print(f"[baseline] {b}: lỗi dựng OrderView {e!r}")
        v = None

    def build(make):
        try:
            return make()
        except Exception:
            return None
    lot = build(lambda: LotView.from_shop(shop))
    debt = build(lambda: DebtView.from_shop(shop))
    dlv = build(lambda: DeliveryView.from_shop(shop))
    stock = build(lambda: StockView.from_shop(shop, v.date_range()[1] if v is not None else None, base_units(v)))
    ledger = auxiliary.prepare(shop, rulebook)
    profile = ShopProfile.from_shop(shop, v.date_range()[0] if v is not None else None)
    if v is not None and not no_sem_baseline:
        try:
            prepare_sem(v, shop, rulebook, cal, public_rules=not no_public_rules)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            print(f"[baseline] {b}: không chuẩn bị được luật SEM {e!r}")
    out = []
    for q in qs:
        as_of = dt.date.fromisoformat(q["as_of_date"])
        rec = check_unanswerable(q["question"], as_of, profile, cal)
        if rec is None and lot is not None:
            rec = lot_answer(q['question'], as_of, lot)
        if rec is None and stock is not None:
            rec = stock_answer(q['question'], as_of, stock)
        if rec is None and dlv is not None:
            rec = delivery_answer(q['question'], as_of, dlv, Periods(as_of, cal))
        if rec is None and debt is not None:
            rec = debt_any(q['question'], as_of, debt, Periods(as_of, cal))
        if rec is None and v is not None:
            try:
                rec = answer(q["question"], as_of, v, Periods(as_of, cal))
            except Exception as e:
                print(f"[baseline] {q['id']}: {e!r}")
        if rec is None:
            rec = auxiliary.answer(q['question'], as_of, ledger, Periods(as_of, cal))
        out.append({"id": q["id"], **(rec or {"type": "abstain"})})
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
