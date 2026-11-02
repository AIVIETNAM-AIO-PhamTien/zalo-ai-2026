"""Chạy code Python do LLM sinh trong process con riêng, namespace giữ qua các lượt.

Process con nạp shop từ file pickle (bảng đã gán kiểu) và các helper; mỗi `run` có timeout.
Quá giờ: giết process, khởi động lại và chạy lại các cell đã thành công trước đó.
"""
from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
import traceback
from pathlib import Path

MAX_OUT = 4000  # ký tự output mỗi lượt gửi lại cho LLM (giữ prompt trong giới hạn context)


def _rulebook_next_to(calendar_csv) -> str:
    p = Path(calendar_csv).with_name("rulebook.md") if calendar_csv else None
    return p.read_text(encoding="utf-8") if p is not None and p.exists() else ""


def _build_namespace(cache_path, as_of, calendar_csv, question=None, rulebook=None, schema_card=None):
    import datetime as dt
    import pickle
    import re

    import numpy as np
    import pandas as pd

    from datamate import helpers, parsing
    from datamate.sem_api import SemAPI
    from datamate.semantic import clean_order_rows, order_id_columns
    from datamate.agent_tools import checked_merge
    from datamate.agent_evidence import ResultChannel
    from datamate.gold import agent_tables, gold_store_path

    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    with open(cache_path, "rb") as fh:
        shop = pickle.load(fh)
    if schema_card is not None:
        shop.card = schema_card
    # dòng tiếp nối (mã đơn / ngày / trạng thái chỉ ghi ở dòng đầu mỗi đơn) kế thừa dòng đầu nhóm (rulebook):
    # để trống thì lọc theo ngày/trạng thái sẽ đánh rơi cả nửa số dòng hàng
    # rulebook §8: hai dòng của cùng một đơn giống nhau ở mọi ô là lỗi xuất file → tính một lần;
    # bảng không ghi theo đơn thì giữ cả hai
    from datamate.rule_guard import Guard
    shop.rule_policy = Guard(_rulebook_next_to(calendar_csv) if rulebook is None else rulebook, policy_aware=True).policy
    with_order_id = order_id_columns(shop)
    tables = {}
    for name, df in shop.tables.items():
        # Agent working frames must not mutate the source later used by SEM to verify.
        if name in with_order_id:
            df = helpers.fill_order_status(clean_order_rows(shop, name), with_order_id[name])
        else:
            df = df.copy(deep=True)
            cols = [c for c in shop.meta.get(name, {}).get("continuation_cols", []) if c in df.columns]
            if cols and shop.meta[name].get("continuation_rows"):
                df[cols] = df[cols].ffill()
        tables[name] = df
    channel = ResultChannel()
    sem = SemAPI(shop, as_of, rulebook=_rulebook_next_to(calendar_csv) if rulebook is None else rulebook,
                 question=question, on_answer=channel.certify)
    sem._calendar = calendar_csv
    return {
        "pd": pd, "np": np, "dt": dt, "re": re,
        "T": tables, "G": agent_tables(shop, as_of, sem._rulebook, gold_store_path(cache_path, sem._rulebook, schema_card)), "README": shop.readme,
        "checked_merge": checked_merge,
        "emit_result": channel.emit, '__result_channel__': channel,
        # Timestamp chứ không phải date: so `date` với cột datetime64 luôn ra False → tổng 0
        "TODAY": pd.Timestamp(as_of), "P": helpers.TsPeriods(helpers.Periods(as_of, calendar_csv)),
        "round_money": helpers.round_money, "top_k": helpers.top_k, "round2": helpers.round2,
        "pct_change": helpers.pct_change, "share": helpers.share,
        "strip_phone": helpers.strip_phone, "find_names": helpers.find_names,
        "find_values": lambda query, limit=8, min_score=60: helpers.find_values(tables, query, limit, min_score),
        "order_status": helpers.order_status, "valid_order": helpers.valid_order, "name_mask": helpers.name_mask,
        "fold": parsing.fold, "parse_number": parsing.parse_number, "parse_date": parsing.parse_date,
        "parse_quantity": parsing.parse_quantity, "is_walkin": parsing.is_walkin,
        # khung giờ ca nằm trong rulebook của split (cùng thư mục với calendar.csv)
        "SEM": sem,
    }


def _key_hint(e: Exception, tables: dict) -> str:
    """KeyError hay gặp (log public 29/10: 1344 lần trong 4190 lượt code): s[-1] trên Series chỉ số nguyên, hoặc tên cột
    của bảng khác / gõ gần đúng → chỉ chỗ có cột đó để lượt sau sửa ngay."""
    if not isinstance(e, KeyError) or not e.args:
        return ""
    k = e.args[0]
    if isinstance(k, int):
        return f"\nGợi ý: KeyError {k} là nhãn chỉ số, không phải vị trí: dùng .iloc[{k}] (hoặc .iat[{k}])."
    if not isinstance(k, str):
        return ""
    import difflib
    where = [f"T[{t!r}]" for t, df in tables.items() if k in getattr(df, "columns", [])]
    allcols = {str(c): t for t, df in tables.items() for c in getattr(df, "columns", [])}
    near = difflib.get_close_matches(k, list(allcols), n=4, cutoff=0.6)
    parts = []
    if where:
        parts.append(f"cột {k!r} có ở " + ", ".join(where[:4]) + " (DataFrame đang dùng không có cột này)")
    if near and not where:
        parts.append("tên cột gần giống: " + ", ".join(f"T[{allcols[c]!r}][{c!r}]" for c in near))
    return ("\nGợi ý: " + "; ".join(parts) + ".") if parts else ""


def _exec_cell(code: str, ns: dict) -> tuple[str, str | None]:
    buf = io.StringIO()
    err = None
    try:
        tree = ast.parse(code)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        if last is not None:
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
                if val is not None:
                    print(val.to_string() if hasattr(val, "to_string") and getattr(val, "size", 0) <= 3000 else repr(val))
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        err = tb[-2500:] + _key_hint(e, ns.get("T") or {})
    out = buf.getvalue()
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT // 2] + f"\n... [cắt bớt {len(out) - MAX_OUT} ký tự] ...\n" + out[-MAX_OUT // 2:]
    return out, err


def _worker(conn, cache_path, as_of, calendar_csv, question=None, rulebook=None, schema_card=None):
    ns = _build_namespace(cache_path, as_of, calendar_csv, question, rulebook, schema_card)
    channel = ns.pop('__result_channel__')
    sem = ns['SEM']
    conn.send(("ready", None))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        channel.reset()
        sem._calls.clear()
        out, err = _exec_cell(msg, ns)
        meta = {'answer': channel.value if err is None else None, 'source': channel.source if err is None else None,
                'sem_calls': list(sem._calls)}
        conn.send((out, err, meta))


class Executor:
    def __init__(self, cache_path, as_of, calendar_csv, timeout: float = 120, *, question=None, rulebook=None, schema_card=None):
        self.args = (str(cache_path), as_of, str(calendar_csv) if calendar_csv else None, question, rulebook, schema_card)
        self.timeout = timeout
        self.cells: list[str] = []
        self.proc = None
        self.last_result = self.last_result_source = None
        self.last_sem_calls = []
        self._start()

    def _start(self):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, *self.args), daemon=True)
        self.proc.start()
        for _ in range(300):
            if self.conn.poll(1):
                try:
                    self.conn.recv()
                    return
                except EOFError:
                    break
            if not self.proc.is_alive():
                break
        self.close()
        raise RuntimeError(f"executor không khởi động được (cache {self.args[0]})")

    def _restart(self):
        self.close()
        self._start()
        for c in self.cells:
            self.conn.send(c)
            if self.conn.poll(self.timeout):
                self.conn.recv()

    def run(self, code: str) -> tuple[str, str | None]:
        self.last_result = self.last_result_source = None
        self.last_sem_calls = []
        try:
            self.conn.send(code)
            if self.conn.poll(self.timeout):
                out, err, meta = self.conn.recv()
                self.last_sem_calls = meta.get('sem_calls', []) if meta else []
                if err is None and meta is not None:
                    self.last_result, self.last_result_source = meta['answer'], meta['source']
            else:
                self._restart()
                return "", f"Timeout: code chạy quá {self.timeout:.0f}s, đã huỷ. Viết lại cách nhanh hơn (vector hoá pandas)."
        except (EOFError, BrokenPipeError, OSError):
            self._restart()
            return "", "Process chạy code bị dừng bất thường (hết bộ nhớ?), đã khởi động lại."
        if err is None:
            self.cells.append(code)
        return out, err

    def close(self):
        if self.proc is not None:
            try:
                self.conn.send(None)
            except Exception:
                pass
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=2)
            self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
