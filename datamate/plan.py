"""Kế hoạch có kiểu: LLM đọc câu hỏi → JSON gọi các phép tính chuẩn của semantic layer (SEM), máy thực thi.

Tầng tất định đọc câu bằng regex chặt nên chỉ phủ 31% câu public ở các shop dựng được bảng chuẩn (train 56%): cách
hỏi mới thì regex không nhận ra. Ở đây LLM chỉ HIỂU câu (chọn phép tính, kỳ, tham số); con số do SEM tính (sai 0
trên train), LLM không viết SQL / pandas (Bounded Semantic Planning, SemPlan 2026; ARCANE R2AI: luật / cấu trúc
quyết định phép tính, model chỉ làm phần hiểu). Câu có điều kiện mà SEM không biểu diễn được → model trả 'none' hoặc
liệt kê điều kiện bỏ sót → để agent làm. Nhận đáp án khi hai mẫu ra cùng kế hoạch và kiểu đáp án hợp câu hỏi.
"""
from __future__ import annotations

import datetime as dt
import json
import re

from datamate.answer_type import expected_type
from datamate.helpers import pct_change, round2, share

# phép tính → (tham số, kiểu kết quả, mô tả)
FUNCS = {
    "revenue": (["ky"], "number", "doanh thu (đơn hợp lệ theo rulebook)"),
    "order_count": (["ky"], "number", "số đơn hợp lệ"),
    "cancelled_count": (["ky"], "number", "số đơn huỷ"),
    "cancel_rate": (["ky"], "percent", "tỉ lệ đơn huỷ (%)"),
    "avg_order_value": (["ky"], "number", "giá trị trung bình một đơn"),
    "qty_sold": (["ky", "don_vi", "mat_hang?", "size?"], "number",
                 "số lượng bán quy về đơn vị cơ bản 'don_vi' ('ly', 'kg', 'hộp'…), của một mặt hàng nếu có; size 'm'/'l'"),
    "group_revenue": (["ky", "nhom"], "number", "doanh thu một nhóm hàng (tên nhóm như trong dữ liệu)"),
    "payment_revenue": (["ky", "hinh_thuc"], "number",
                        "doanh thu theo hình thức thanh toán: 'cod' / 'cash' / 'transfer' / 'ewallet' / 'debt'"),
    "channel_revenue": (["ky", "kenh"], "number", "doanh thu theo kênh bán: 'store' / 'phone' / 'zalo' / 'market'"),
    "shift_revenue": (["ky", "ca"], "number", "doanh thu một ca: 'sang' / 'chieu' / 'toi'"),
    "shift_order_count": (["ky", "ca"], "number", "số đơn một ca: 'sang' / 'chieu' / 'toi'"),
    "rx_revenue": (["ky", "ke_don?"], "number", "doanh thu thuốc kê đơn (ke_don=false: hàng không kê đơn)"),
    "ingredient_revenue": (["ky", "hoat_chat"], "number", "doanh thu các thuốc có đúng hoạt chất đó"),
    "debt_total": (["ngay"], "number", "tổng tiền khách còn nợ tính đến hết ngày"),
    "debtor_count": (["ngay"], "number", "số khách còn nợ tính đến hết ngày"),
    "debt_added": (["ky"], "number", "tiền nợ mới phát sinh trong kỳ"),
    "debt_repaid": (["ky"], "number", "tiền khách trả nợ trong kỳ"),
    "debt_by_age": (["ngay", "tu_ngay", "den_ngay?"], "number",
                    "tổng còn nợ của các đơn có tuổi nợ (ngày) trong khoảng [tu_ngay, den_ngay]; 'quá N ngày' = tu_ngay N+1, 'trong N ngày' = 0 → N"),
    "import_avg_price": (["ky", "mat_hang", "don_vi"], "number", "giá nhập bình quân một đơn vị nhập của mặt hàng"),
    "delivery_count": (["ky"], "number", "số đợt giao trong kỳ"),
    "best_revenue_day": (["ky"], "date", "ngày doanh thu cao nhất"),
    "customer_most_orders": (["ky"], "string", "khách có nhiều đơn nhất"),
    "loyal_customer": (["ky"], "string", "khách trung thành nhất (nhiều ngày có mua nhất)"),
    "big_spender": (["ky"], "string", "khách sộp nhất (giá trị trung bình một đơn cao nhất, từ 3 đơn)"),
    "top_spender": (["ky"], "string", "khách mua nhiều tiền nhất (tổng tiền các đơn trong kỳ)"),
    "fickle_customer": (["ky"], "string", "khách hay đổi ý nhất (nhiều đơn huỷ nhất)"),
    "popular_item": (["ky"], "string", "mặt hàng được ưa chuộng nhất (có trong nhiều đơn nhất)"),
    "busiest_day": (["ky"], "date", "ngày đắt khách nhất (nhiều đơn nhất)"),
    "peak_hour": (["ky"], "number", "giờ cao điểm: giờ bắt đầu khung một giờ có nhiều đơn nhất"),
    "item_revenue": (["ky", "mat_hang"], "number", "doanh thu một mặt hàng (tên như trong câu hỏi)"),
    "best_item": (["ky"], "string", "mặt hàng bán chạy nhất (theo số lượng quy về đơn vị cơ bản)"),
    "top_items": (["ky", "k"], "list", "k mặt hàng bán chạy nhất, theo thứ tự"),
    "group_order_count": (["ky", "nhom"], "number", "số đơn có ít nhất một mặt hàng thuộc nhóm"),
    "orders_min": (["ky", "nguong"], "number", "số đơn có doanh thu từ ngưỡng (đồng) trở lên"),
    "days_revenue_min": (["ky", "nguong"], "number", "số ngày có doanh thu từ ngưỡng (đồng) trở lên"),
    "customers_min_orders": (["ky", "n", "ti_le?"], "number",
                             "số khách có từ n đơn trở lên; ti_le=true → % trên số khách đã mua trong kỳ"),
    "weekday_avg": (["ky", "thu", "chi_so"], "number",
                    "trung bình mỗi ngày thứ X trong kỳ: thu 2..7 (Chủ nhật = 8), chi_so 'doanh_thu' / 'so_don'"),
    # các phép SEM đã kiểm, mở cho kế hoạch theo độ phủ (scripts/program_search.py, train 30/10)
    "returned_count": (["ky"], "number", "số đơn trả hàng"),
    "customer_count": (["ky"], "number", "số khách đã mua (có đơn hợp lệ) trong kỳ, không tính khách lẻ"),
    "customer_revenue": (["ky", "khach"], "number", "doanh thu của một khách (tên như trong dữ liệu)"),
    "customer_order_count": (["ky", "khach"], "number", "số đơn hợp lệ của một khách"),
    "customer_avg_order_value": (["ky", "khach"], "number", "giá trị trung bình một đơn của một khách"),
    "item_order_count": (["ky", "mat_hang"], "number", "số đơn hợp lệ có mặt hàng"),
    "active_days": (["ky"], "number", "số ngày có đơn hợp lệ"),
    "avg_daily_revenue": (["ky"], "number", "doanh thu trung bình mỗi ngày của kỳ"),
    "avg_daily_orders": (["ky"], "number", "số đơn trung bình mỗi ngày của kỳ"),
    "payment_order_count": (["ky", "hinh_thuc"], "number", "số đơn theo hình thức: 'cod' / 'cash' / 'transfer' / "
                                                          "'ewallet' / 'debt'"),
    "channel_order_count": (["ky", "kenh"], "number", "số đơn theo kênh: 'store' / 'phone' / 'zalo' / 'market'"),
    "top_customers": (["ky", "k"], "list", "k khách mua nhiều tiền nhất, theo thứ tự"),
    "debt_customer": (["ky", "ten"], "list", "[nợ thêm, trả nợ] trong kỳ của một khách (tên như trong dữ liệu)"),
    "price_effect": (["mat_hang", "ngay", "n_ngay"], "percent",
                     "% thay đổi số lượng bán N ngày kể từ ngày đổi giá so với N ngày ngay trước"),
    "lot_nearest_expiry": (["mat_hang"], "date", "hạn dùng của lô còn tồn gần hết hạn nhất của mặt hàng"),
    "unit_price": (["ngay", "mat_hang"], "number",
                   "giá bán một đơn vị cơ bản của mặt hàng ngày D (đơn giá trên dòng bán, trước chiết khấu dòng)"),
    "import_spend": (["ky"], "number", "tổng tiền nhập hàng trong kỳ (sổ nhập shop tự nhận diện, bỏ phiếu huỷ)"),
    "top_supplier": (["ky", "theo_so_phieu?"], "string",
                     "nhà cung cấp nhập nhiều tiền nhất trong kỳ (theo_so_phieu=true: nhiều phiếu nhập nhất)"),
    # phép cần chỉ bảng / cột của shop (tên cột lấy đúng như trong dữ liệu)
    "import_total": (["ky", "bang", "cot_phieu", "cot_ngay", "cot_tong_phieu", "cot_trang_thai"], "number",
                     "tổng tiền nhập hàng trong kỳ (bảng phiếu nhập; bỏ phiếu huỷ)"),
    "main_supplier": (["ky", "bang", "cot_phieu", "cot_ncc", "cot_ngay", "cot_trang_thai"], "string",
                      "nhà cung cấp nhập nhiều tiền nhất trong kỳ"),
    "expense_total": (["ky", "bang", "cot_ngay", "cot_tien"], "number", "tổng chi phí trong kỳ (sổ chi phí)"),
    "gross_profit": (["ky", "cot_sl", "cot_gia_von", "cot_hang?", "mat_hang?"], "number",
                     "lãi gộp trong kỳ = doanh thu − giá vốn hàng bán (của cả shop hoặc một mặt hàng)"),
    "net_profit": (["ky", "cot_sl", "cot_gia_von", "bang_chi_phi", "cot_ngay_chi", "cot_tien_chi"], "number",
                   "lãi ròng trong kỳ = lãi gộp − chi phí"),
    "best_profit_item": (["ky", "cot_sl", "cot_gia_von", "cot_ten_hang", "cot_ma_hang"], "string",
                         "mặt hàng lời (lãi gộp) nhiều nhất trong kỳ"),
    "unsold_items": (["ky", "cot_ten_hang", "cot_ma_hang"], "list",
                     "mã các mặt hàng vẫn đang bán (đã bán trước kỳ) mà trong kỳ không bán được cái nào"),
    "unsold_count": (["ky", "cot_ten_hang", "cot_ma_hang"], "number", "số mặt hàng như unsold_items"),
    "staff_revenue": (["ky", "bang", "cot_don", "cot_nhan_vien", "nhan_vien"], "number",
                      "doanh thu các đơn của một nhân viên trong kỳ"),
    "staff_best": (["ky", "bang", "cot_don", "cot_nhan_vien"], "string", "nhân viên có doanh thu cao nhất trong kỳ"),
    "customer_group_revenue": (["ky", "bang", "cot_don", "cot_nhom", "nhom"], "number",
                               "doanh thu các đơn của một nhóm khách trong kỳ (khách vãng lai không thuộc nhóm nào)"),
    "new_members": (["ky", "bang", "cot_khach", "cot_dang_ky"], "number", "số khách đăng ký thành viên mới trong kỳ"),
    "order_adjustment_total": (["ky", "bang", "cot_don", "cot_tien"], "number",
                               "tổng một khoản cấp đơn trong kỳ (giảm voucher, giảm nhờ điểm, phí ship, thu máy cũ)"),
}
COLUMN_ARGS = {"bang", "bang_chi_phi"} | {a.rstrip("?") for spec in FUNCS.values() for a in spec[0]
                                          if a.startswith("cot_")}
OPS = {"value": "một phép tính", "pct_change": "% thay đổi của phép 1 so với phép 2",
       "share": "phép 1 chiếm bao nhiêu % phép 2", "diff": "phép 1 trừ phép 2", "list": "nhiều số theo thứ tự câu hỏi",
       "argmax_period": "cùng một phép qua nhiều kỳ → kỳ có giá trị lớn nhất (tháng: YYYY-MM, ngày: YYYY-MM-DD)",
       "argmin_period": "như argmax_period nhưng nhỏ nhất",
       "increasing": "cùng một phép qua các kỳ theo thứ tự thời gian → true nếu kỳ sau luôn lớn hơn kỳ trước",
       "decreasing": "như increasing nhưng kỳ sau luôn nhỏ hơn"}

PROMPT = """Chuyển câu hỏi của chủ cửa hàng thành kế hoạch gọi các phép tính chuẩn (đã đúng luật nghiệp vụ). Bạn KHÔNG tính số.

Phép tính (tham số có ? là tuỳ chọn; ky = {{"tu": "YYYY-MM-DD", "den": "YYYY-MM-DD"}}, đầu và cuối đều tính; ngay = "YYYY-MM-DD"):
{funcs}

Cách ghép (op): {ops}

Hôm nay: {today} ({weekday}). Kỳ tương đối tính theo hôm nay: tuần từ thứ Hai đến Chủ nhật; "tháng này" = ngày 1 → hôm nay.
{hints}
Câu hỏi: {question}

Trả lời đúng một khối JSON:
{{"op": "value|pct_change|share|diff|list|argmax_period|argmin_period|increasing|decreasing|none", "calls": [{{"fn": "...", "args": {{...}}}}], "bo_sot": ["điều kiện trong câu mà các phép tính trên KHÔNG thể hiện"]}}
Ví dụ (cửa hàng giả định):
- "dthu T10/2025 bnhiu" → {{"op": "value", "calls": [{{"fn": "revenue", "args": {{"ky": {{"tu": "2025-10-01", "den": "2025-10-31"}}}}}}], "bo_sot": []}}
- "quý 2/2025 chốt được mấy o đờ" → {{"op": "value", "calls": [{{"fn": "order_count", "args": {{"ky": {{"tu": "2025-04-01", "den": "2025-06-30"}}}}}}], "bo_sot": []}}
- "th5/2025 dthu tăng giảm mấy % so tháng trước" → {{"op": "pct_change", "calls": [{{"fn": "revenue", "args": {{"ky": {{"tu": "2025-05-01", "den": "2025-05-31"}}}}}}, {{"fn": "revenue", "args": {{"ky": {{"tu": "2025-04-01", "den": "2025-04-30"}}}}}}], "bo_sot": []}}
- "doanh thu lần lượt T2, T3/2026" → {{"op": "list", "calls": [{{"fn": "revenue", "args": {{"ky": {{"tu": "2026-02-01", "den": "2026-02-28"}}}}}}, {{"fn": "revenue", "args": {{"ky": {{"tu": "2026-03-01", "den": "2026-03-31"}}}}}}], "bo_sot": []}}
- "th6/2025 mấy % khách mua từ 4 đơn trở lên" → {{"op": "value", "calls": [{{"fn": "customers_min_orders", "args": {{"ky": {{"tu": "2025-06-01", "den": "2025-06-30"}}, "n": 4, "ti_le": true}}}}], "bo_sot": []}}
- "nhóm bánh chiếm bao nhiêu % doanh thu tháng 3/2026" → {{"op": "share", "calls": [{{"fn": "group_revenue", "args": {{"ky": {{"tu": "2026-03-01", "den": "2026-03-31"}}, "nhom": "bánh"}}}}, {{"fn": "revenue", "args": {{"ky": {{"tu": "2026-03-01", "den": "2026-03-31"}}}}}}], "bo_sot": []}}
- "doanh thu của nhân viên NV002 tháng 3" → {{"op": "none", "calls": [], "bo_sot": ["lọc theo nhân viên"]}}
Câu cần điều kiện mà không phép tính nào biểu diễn được (lọc theo khách, nhân viên, mặt hàng khi phép tính không có
tham số mặt hàng, xếp hạng, so sánh nhiều nhóm…) → op "none". Không ép câu hỏi vào phép tính gần giống."""

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]


def build_prompt(question: str, as_of: dt.date, hints: str = "") -> str:
    # tham số bảng / cột không đưa vào prompt: fill_columns gắn cột tầng tất định đã nhận diện cho shop
    funcs = "\n".join(f"- {n}({', '.join(x for x in a if x.rstrip('?') not in COLUMN_ARGS)}): {d}"
                      for n, (a, _, d) in FUNCS.items())
    ops = "; ".join(f"{k} = {v}" for k, v in OPS.items())
    return PROMPT.format(funcs=funcs, ops=ops, today=as_of.isoformat(), weekday=WEEKDAYS[as_of.weekday()],
                         hints=(hints + "\n") if hints else "", question=question)


def parse_plan(text: str) -> dict | None:
    """JSON của model → kế hoạch hợp lệ (đúng tên phép, đủ tham số bắt buộc, không bỏ sót điều kiện) hoặc None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        plan = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(plan, dict) or plan.get("op") not in OPS or plan.get("bo_sot"):
        return None
    calls = plan.get("calls")
    if not isinstance(calls, list) or not calls:
        return None
    need = {"value": 1, "pct_change": 2, "share": 2, "diff": 2}.get(plan["op"])
    if need is not None and len(calls) != need:
        return None
    for c in calls:
        spec = FUNCS.get(c.get("fn")) if isinstance(c, dict) else None
        if spec is None or not isinstance(c.get("args"), dict):
            return None
        # tham số bảng / cột được phép bỏ: fill_columns lấy cột tầng tất định đã gắn cho shop
        if any(a not in c["args"] for a in spec[0] if not a.endswith("?") and a not in COLUMN_ARGS):
            return None
    return plan


def _arg(name: str, v):
    if name == "ky":
        if not isinstance(v, dict):
            raise ValueError("ky")
        return dt.date.fromisoformat(v["tu"]), dt.date.fromisoformat(v["den"])
    if name in ("ngay",):
        return dt.date.fromisoformat(str(v))
    return v


def execute(plan: dict, sem, blocked=frozenset()) -> dict | None:
    """Chạy kế hoạch trên SEM → đáp án {type, answer}; None khi một phép tính không chạy được / không chắc."""
    vals, kinds = [], []
    for c in plan["calls"]:
        if c["fn"] in blocked:  # luật của phép này đổi trong rulebook split (datamate/rule_guard.py)
            return None
        names, kind, _ = FUNCS[c["fn"]]
        args = {}
        try:  # ngày không tồn tại ('2026-02-29'), kỳ thiếu khoá… → kế hoạch hỏng
            for a in names:
                key = a.rstrip("?")
                if key in c["args"] and c["args"][key] is not None:
                    args[key] = _arg(key, c["args"][key])
        except (ValueError, KeyError, TypeError):
            return None
        try:
            v = getattr(sem, c["fn"])(**args)
        except Exception:
            return None
        if v is None or isinstance(v, dict) or (isinstance(v, (list, tuple)) and kind != "list"):
            return None
        if kind == "list" and (plan["op"] != "value" or not v):
            return None
        vals.append(list(v) if kind == "list" else v)
        kinds.append(kind)
    op = plan["op"]
    if op == "value":
        if kinds[0] == "percent" or (plan["calls"][0]["fn"] == "customers_min_orders"
                                     and plan["calls"][0]["args"].get("ti_le")):
            return {"type": "percent", "answer": vals[0]}
        return {"type": kinds[0], "answer": vals[0]}
    if op == "list":
        return {"type": "list", "answer": vals}
    if op in PERIOD_OPS:
        return _period_op(op, plan["calls"], vals)
    # ghép số (%, tỉ trọng, chênh) chỉ nhận hai giá trị số
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return None
    if op == "diff":
        return {"type": kinds[0], "answer": round2(vals[0] - vals[1]) if kinds[0] == "percent" else vals[0] - vals[1]}
    r = pct_change(vals[0], vals[1]) if op == "pct_change" else share(vals[0], vals[1])
    return None if r is None else {"type": "percent", "answer": r}


PERIOD_OPS = ("argmax_period", "argmin_period", "increasing", "decreasing")


def _period_label(ky) -> str | None:
    """Kỳ là trọn một tháng → 'YYYY-MM'; một ngày → 'YYYY-MM-DD'; khác → None (không gọi tên được)."""
    try:
        a, b = dt.date.fromisoformat(ky["tu"]), dt.date.fromisoformat(ky["den"])
    except (KeyError, TypeError, ValueError):
        return None
    if a == b:
        return a.isoformat()
    if a.day == 1 and a.month == b.month and a.year == b.year and (b + dt.timedelta(days=1)).day == 1:
        return a.strftime("%Y-%m")
    return None


def _period_op(op: str, calls: list, vals: list) -> dict | None:
    """Cùng một phép (cùng tham số ngoài kỳ) qua ≥ 2 kỳ khác nhau: kỳ lớn/nhỏ nhất, hoặc tăng/giảm ngặt theo thời gian."""
    if len(calls) < 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return None
    rest = {json.dumps({"fn": c["fn"], **{k: v for k, v in c["args"].items() if k != "ky"}}, sort_keys=True)
            for c in calls}
    kys = [c["args"].get("ky") for c in calls]
    if len(rest) != 1 or any(k is None for k in kys) or len({json.dumps(k, sort_keys=True) for k in kys}) != len(kys):
        return None
    order = sorted(range(len(kys)), key=lambda i: kys[i]["tu"])  # bằng nhau: kỳ sớm hơn đứng trước
    if op in ("increasing", "decreasing"):
        seq = [vals[i] for i in order]
        up = all(b > a for a, b in zip(seq, seq[1:]))
        down = all(b < a for a, b in zip(seq, seq[1:]))
        return {"type": "boolean", "answer": up if op == "increasing" else down}
    best = (max if op == "argmax_period" else min)(order, key=lambda i: vals[i])
    label = _period_label(kys[best])
    return None if label is None else {"type": "date", "answer": label}


def format_plan_shots(shots: list[dict]) -> str:
    """Ví dụ kế hoạch đúng ở cửa hàng khác (kho datamate/plan_shots.jsonl, scripts/build_plan_shots.py)."""
    if not shots:
        return ""
    lines = ["Ví dụ kế hoạch đúng ở cửa hàng khác (ngày tính theo hôm nay của ví dụ; tên hàng, nhóm là của cửa hàng đó):"]
    for x in shots:
        lines.append(f'- Hôm nay {x["as_of"]}: "{x["question"]}" → '
                     + json.dumps({**x["plan"], "bo_sot": []}, ensure_ascii=False))
    return "\n".join(lines)


def _name_values(shop) -> list[str]:
    """Giá trị chữ có thể là tên (hàng, khách, nhóm, nhân viên, NCC) của mọi bảng; cache trên shop."""
    import pandas as pd
    from datamate.parsing import fold
    cached = getattr(shop, "_plan_name_values", None)
    if cached is not None:
        return cached
    vals: dict[str, str] = {}
    for df in shop.tables.values():
        for c in df.columns:
            s = df[c]
            if c == "_sheet" or not (s.dtype == object or pd.api.types.is_string_dtype(s)):
                continue
            u = [v for v in s.dropna().unique() if isinstance(v, str)]
            if not u or len(u) > 20000:
                continue
            for v in u:
                k = " ".join(fold(v).split())
                # tên có chữ, không phải mã / ngày / số thuần
                if 4 <= len(k) <= 80 and re.search(r"[a-z]{2}", k) and not re.fullmatch(r"[a-z]{0,4}[\d\W_]+", k):
                    vals.setdefault(k, v.strip())
    shop._plan_name_values = list(vals.items())
    return shop._plan_name_values


def value_hint(question: str, shop, limit: int = 6, cutoff: float = 85) -> str:
    """Liên kết giá trị kiểu CHESS: tên trong câu (gõ sai, không dấu) → giá trị thật trong dữ liệu, để model không đoán
    tên (log Colab 31/10: 'gạnbá chiên' → model đặt 'Gà chiên nước mắm', SEM khớp đúng tên nên ra None).
    Khớp n-gram của câu với giá trị chữ của mọi bảng bằng rapidfuzz, không dùng embedding model."""
    from rapidfuzz import fuzz, process
    from datamate.parsing import fold
    vals = _name_values(shop) if shop is not None else []
    if not vals:
        return ""
    words = re.sub(r"[^\w\s.,/%-]", " ", fold(question)).split()
    grams = list({" ".join(words[i:i + n]) for n in range(1, 9) for i in range(len(words) - n + 1)
                  if len(" ".join(words[i:i + n])) >= 4})
    if not grams:
        return ""
    keys = [k for k, _ in vals]
    m = process.cdist(grams, keys, scorer=fuzz.ratio, score_cutoff=cutoff, workers=1)
    best: dict[int, float] = {}
    for gi, ki in zip(*m.nonzero()):
        sc = float(m[gi, ki]) + min(len(grams[gi]), 30) / 30  # hoà điểm: ưu tiên khớp dài
        if sc > best.get(ki, 0):
            best[ki] = sc
    if not best:
        return ""
    top = sorted(best, key=lambda k: (-best[k], keys[k]))[:limit]
    names = "; ".join(f'"{vals[k][1]}"' for k in top)
    return ("Giá trị có thật trong dữ liệu gần với chữ trong câu hỏi (mat_hang / khach / ten / nhom phải chép đúng một "
            f"giá trị này nếu câu nói tới nó; không có giá trị nào đúng thì op \"none\"): {names}.")


def solve_plan(question: str, as_of: dt.date, sem, llm, hints: str = "", samples: int = 1,
               blocked=frozenset(), shots: list[dict] | None = None) -> tuple[dict | None, dict]:
    """Kế hoạch giải mã tham lam (llm temperature 0: tái lập được); samples > 1 thì các mẫu phải ra cùng đáp án.
    Kiểu đáp án phải khớp kiểu dự kiến của câu (nếu đoán được). shots: ví dụ kế hoạch đúng ở cửa hàng khác."""
    vh = value_hint(question, getattr(sem, "_shop", None))
    hints = "\n".join(x for x in (hints, vh, format_plan_shots(shots or [])) if x)
    prompt = [{"role": "user", "content": build_prompt(question, as_of, hints)}]
    info = {"plans": []}
    answers = []
    for i in range(samples):
        try:
            text = llm.chat(prompt, seed=2000 + i)
        except RuntimeError as e:
            info["error"] = str(e)
            return None, info
        ans = answer_from(text, question, sem, blocked)
        info["plans"].append({"raw": text[-1500:], "answer": ans})
        answers.append(ans)
    if any(a is None for a in answers) or any(a != answers[0] for a in answers):
        return None, info
    return answers[0], info


# khái niệm mà chưa phép tính nào biểu diễn: câu có các từ này mà kế hoạch không gọi phép tương ứng → không nhận
# (log kế hoạch train: 'lãi gộp tháng 7' → model gọi revenue)
# phần tử "@<op>" nghĩa là chấp nhận theo phép ghép (khái niệm nằm ở cách ghép, không ở phép tính)
_CONCEPTS = [(r"\blai\b|loi nhuan|gia von|\bloi\b", {"gross_profit", "net_profit", "best_profit_item"}),
             (r"nhan vien|\btho\b|\bnv\d", {"staff_revenue", "staff_best"}),
             (r"\bton\b|het hang|dut hang", set()),
             (r"\bnhap\b|nha cung cap", {"import_avg_price", "import_spend", "top_supplier", "import_total",
                                         "main_supplier"}),
             (r"\bno\b|mua chiu|ban chiu|tra gop", {"debt_total", "debtor_count", "debt_added", "debt_repaid",
                                                  "debt_by_age", "debt_customer"}),
             (r"voucher|phi ship|dung (\d+ )?diem|tich diem", {"order_adjustment_total"}),
             (r"(khong|ko|chua) ban (ra |duoc |dc )*(duoc|dc|cai|mon|ly|sp|san pham)(?!\w)", {"unsold_items", "unsold_count"}),
             # log public 31/10 (không nhãn, đối chiếu tầng tất định): 'dthu ca sáng' → revenue cả ngày, 'mấy đh' →
             # revenue, 'dthu bình quân một đh' → revenue, 'ncc uy tín' (chủ quan) → top_supplier
             (r"\bca (sang|chieu|toi|dem|trua)\b", {"shift_revenue", "shift_order_count"}),
             (r"\b(baonhieu|bao nhieu|bnhiu|may|so|tong) (dh|don|don hang|hoa don|order)\b",
              {"order_count", "cancelled_count", "shift_order_count", "group_order_count", "returned_count",
               "customer_order_count", "item_order_count", "avg_daily_orders", "payment_order_count",
               "channel_order_count", "customers_min_orders", "orders_min", "customer_most_orders"}),
             (r"binh quan|trung binh", {"avg_order_value", "import_avg_price", "weekday_avg", "customer_avg_order_value",
                                       "avg_daily_revenue", "avg_daily_orders", "big_spender"}),
             (r"uy tin|le phep|chu dao", set()),
             # 'hqua bán được mấy gói' → order_count
             (r"\b(may|bao nhieu|baonhieu|bnhiu|bn) (goi|hop|chai|ly|vi|vien|thung|bao|kg|lon|chiec|dia|to|suat|tui|lit)\b",
              {"qty_sold", "price_effect", "unit_price", "import_avg_price"}),
             # 'nhóm vitamin … mặt hàng nào thu nhiều nhất' → best_item cả cửa hàng (bỏ lọc nhóm)
             (r"\bnhom\b", {"group_revenue", "group_order_count", "customer_group_revenue", "debt_by_age"}),
             (r"bao hanh", set()), (r"may cu", {"order_adjustment_total"}),
             (r"tang gia|giam gia|doi gia|len gia", set()),
             (r"don gia|gia ban|\bgia (baonhieu|bao nhieu|bn|bnhiu|may)|gia .{0,20}(mot|1) (ly|chai|cai|bo|goi|hop|vi)",
              {"unit_price"}),
             (r"(thang|th) nao .{0,15}(lon|cao|nhieu|it|thap|nhat)", {"@argmax_period", "@argmin_period"})]
# 'bán <mặt hàng> được bao nhiêu tiền' → không phải doanh thu cả cửa hàng
_SOLD_OBJ = re.compile(r"\bban (?!(duoc|dc|ra|bao|baonhieu|bn|bnhiu|nhieu|het|hang|cho|tai|qua|online|o|nhiu|nhieu|"
                       r"them|lai|chay|tien)\b)\w+")

# phép → (câu phải có, câu không được có): log kế hoạch train 29/10 — 'khách mua nhiều tiền nhất' gán big_spender (sộp
# = TB một đơn), 'bao nhiêu gói / phần' gán doanh thu, 'mặt hàng mua ở N đơn' gán orders_min, 'mấy khách nợ' gán debt_total
_MONEY = r"tien|doanh thu|dthu|\bdt\b|\bdong\b|gia tri|\bd\b|trieu|nghin|\bk\b|\bcu\b"
_COUNT = (r"(baonhieu|bao nhieu|may|mi|bn|so|tong) (goi|phan|ly|chai|hop|cai|vien|lon|bich|cay|kg|thung|chiec|don|khach|"
          r"ngay|mat hang|lan|luot|phieu|to|dia|suat|tui|cuon|bao|vi|tuyp|lo|ong|met|m)\b")
_MONEY_FNS = {"revenue", "item_revenue", "group_revenue", "shift_revenue", "payment_revenue", "channel_revenue",
              "rx_revenue", "ingredient_revenue"}
_FN_RULES = {
    "big_spender": (r"\bsop\b", None), "top_spender": (r"(nhieu|nhiu) tien|chi tieu|tong tien|chi nhieu", r"\bsop\b"),
    "orders_min": (r"\bdon\b", r"mat hang|\bsp\b|san pham|khach"), "customers_min_orders": (r"khach", r"mat hang"),
    "days_revenue_min": (r"ngay|bua|hom", None),
    "top_items": (None, r"khach"), "best_item": (None, r"khach"), "debt_total": (None, r"(baonhieu|may|bn|so|mi) khach"),
    "loyal_customer": (r"trung thanh", None), "fickle_customer": (r"doi y", None),
}


def _rule_ok(plan: dict, s: str) -> bool:
    fns = [c["fn"] for c in plan["calls"]]
    if re.search(_COUNT, s) and not re.search(_MONEY, s) and set(fns) & _MONEY_FNS:
        return False
    for f in fns:
        need, forbid = _FN_RULES.get(f, (None, None))
        if (need and not re.search(need, s)) or (forbid and re.search(forbid, s)):
            return False
    calls = [json.dumps(c, sort_keys=True, ensure_ascii=False) for c in plan["calls"]]
    if plan["op"] == "list" and len(set(calls)) < len(calls):
        return False
    m = re.search(r"\b(\d+) thang (lien|gan|lien tiep)", s)
    if plan["op"] == "list" and m and int(m.group(1)) != len(calls):
        return False
    if "revenue" in fns and _SOLD_OBJ.search(s):
        return False
    for c in plan["calls"]:
        a = c["args"]
        if c["fn"] == "top_items" and not (isinstance(a.get("k"), int) and 1 <= a["k"] <= 20):
            return False
        if c["fn"] == "debt_by_age" and a.get("den_ngay") is not None and re.search(r"(qua|tren|hon) \d+ ngay", s):
            return False
    return True


def _months_apart(p0, p1) -> int:
    return (p0[0].year * 12 + p0[0].month) - (p1[0].year * 12 + p1[0].month)


def _periods_of(plan: dict) -> set:
    out = set()
    for c in plan["calls"]:
        k = c["args"].get("ky")
        try:
            if c["args"].get("ngay") is not None:
                d = dt.date.fromisoformat(str(c["args"]["ngay"]))
                out.add((d, d))
        except ValueError:
            pass
        if isinstance(k, dict):
            try:
                out.add((dt.date.fromisoformat(k["tu"]), dt.date.fromisoformat(k["den"])))
            except (KeyError, ValueError, TypeError):
                pass
    return out

_SHORT_MONTH = re.compile(r"\b(?:t|th)(\d{1,2})\s*/\s*(20\d\d)\b")

def concepts_ok(plan: dict, s: str) -> bool:
    """Câu nhắc khái niệm có phép riêng (lãi, nợ, hàng ế…) thì kế hoạch phải dùng phép đó. s: câu đã norm + lower."""
    fns = {c["fn"] for c in plan["calls"]} | {"@" + plan["op"]}
    return all(not re.search(rx, s) or fns & ok for rx, ok in _CONCEPTS)

def check_plan(plan: dict, question: str, as_of: dt.date | None, calendar_csv=None) -> dict | None:
    """Đối chiếu kế hoạch với những gì đọc tất định được từ câu (sai 0 trên train): kỳ đọc được phải có trong kế
    hoạch; khái niệm chưa có phép tính thì không nhận; % thay đổi lấy kỳ sau so với kỳ trước (rulebook)."""
    from datamate.baseline import extract_periods, norm
    from datamate.helpers import Periods, anchored_windows
    s = norm(question).lower()
    win = anchored_windows(s)
    # 'so số lượng N ngày kể từ ngày tăng giá D với N ngày trước đó' = % thay đổi qty_sold trên hai khoảng tính bằng
    # code (kiểm khoảng ở dưới); chỉ khi đó mới bỏ qua khái niệm 'tăng giá' (chưa có phép riêng ngoài price_effect)
    qty_windows = win is not None and plan["op"] == "pct_change" and {c["fn"] for c in plan["calls"]} == {"qty_sold"}
    if not concepts_ok(plan, re.sub(r"tang gia|len gia|doi gia", " ", s) if qty_windows else s):
        return None
    if not _rule_ok(plan, s):
        return None
    if as_of is None:
        return plan
    try:
        found, _ = extract_periods(norm(question), Periods(as_of, calendar_csv))
    except Exception:
        found = []
    have = _periods_of(plan)
    if re.search(r"\b(th|thang) (nay|ni)\b", s) and (as_of.replace(day=1), as_of) not in have:
        return None
    for m, y in _SHORT_MONTH.findall(s):  # 'dthu T4/2026': extract_periods chưa đọc dạng này
        if 1 <= int(m) <= 12 and not any(a.year == int(y) and a.month == int(m) for a, _ in have):
            return None
    labels = {lab for lab, _ in found}
    # 'N ngày kể từ ngày D … với N ngày ngay trước đó': hai khoảng tính bằng code, kế hoạch phải dùng đúng hai khoảng đó
    if win and plan["op"] in ("pct_change", "diff", "list"):
        if have != set(win):
            return None
        found = [(lab, p) for lab, p in found if not (lab == "day" and p is not None and p[0] == win[0][0])]
    for lab, p in found:
        if lab == "BAD" or p is None:
            continue
        # 'đến hết hôm nay' là mốc cuối của kỳ khác
        if lab == "today" and (len(labels) > 1 or (re.search(r"\b(tu|den|toi|tinh)\b", s)
                                                    and any(e == p[1] for _, e in have))):
            continue
        if tuple(p) not in have:
            return None
    # 'cùng kỳ' = cùng số ngày (rulebook): hai kỳ so sánh phải dài bằng nhau
    if re.search(r"cung ky", s) and plan["op"] in ("pct_change", "diff") and len(have) == 2:
        (a0, b0), (a1, b1) = sorted(have)
        if (b0 - a0) != (b1 - a1):
            return None
    # 'so với tháng trước / cùng kỳ tháng trước' → hai kỳ cách nhau đúng 1 tháng (không phải cùng kỳ năm trước)
    if plan["op"] in ("pct_change", "diff") and len(have) == 2 and re.search(
            r"(thang|th) (truoc|trc|roi|lien truoc)", s) and not re.search(r"nam (truoc|ngoai|trc)", s):
        (p0, p1) = sorted(have, reverse=True)
        if _months_apart(p0, p1) != 1:
            return None
    # Tết theo lịch âm (rulebook): kỳ phải đúng dịp Tết / trước Tết tra từ calendar.csv; không có lịch → không nhận
    if re.search(r"\btet\b", s):
        if calendar_csv is None:
            return None
        P = Periods(as_of, calendar_csv)
        ok = set()
        for y in range(as_of.year - 2, as_of.year + 2):
            try:
                t = P.tet(y)
            except Exception:
                continue
            ok.add(tuple(t))
            for (a, b) in have:
                try:
                    ok.add(tuple(P.before_tet(y, (b - a).days + 1)))
                except Exception:
                    pass
        if not have or not have <= ok:
            return None
    if plan["op"] == "pct_change" and len(have) == 2:
        k0, k1 = (_periods_of({"calls": [c]}) for c in plan["calls"])
        if k0 and k1 and next(iter(k0))[1] < next(iter(k1))[0]:  # kỳ hỏi (sau) đứng trước kỳ so
            plan = {**plan, "calls": plan["calls"][::-1]}
    return plan


def bound_columns(rules) -> dict[str, list[dict]]:
    """Cột tầng tất định (SemRules) đã gắn từ README cho từng phép cần cột: dùng khi kế hoạch không ghi cột, và làm
    ứng viên đầu tiên của bộ tìm chương trình (scripts/program_search.py)."""
    if rules is None:
        return {}
    out = {}
    cost, exp, item = getattr(rules, "cost", None), getattr(rules, "expenses", None), getattr(rules, "item", None)
    if cost:
        c = {"cot_sl": cost[0], "cot_gia_von": cost[1]}
        out.setdefault("gross_profit", []).append(c)
        if exp:
            out.setdefault("net_profit", []).append({**c, "bang_chi_phi": exp[0], "cot_ngay_chi": exp[1],
                                                      "cot_tien_chi": exp[2]})
        if item:
            out.setdefault("best_profit_item", []).append({**c, "cot_ten_hang": item[0], "cot_ma_hang": item[1]})
    if item:
        for fn in ("unsold_items", "unsold_count"):
            out.setdefault(fn, []).append({"cot_ten_hang": item[0], "cot_ma_hang": item[1]})
    if exp:
        out.setdefault("expense_total", []).append({"bang": exp[0], "cot_ngay": exp[1], "cot_tien": exp[2]})
    for fn, attr, col in (("staff_revenue", "staff", "cot_nhan_vien"), ("staff_best", "staff", "cot_nhan_vien"),
                          ("customer_group_revenue", "groups", "cot_nhom")):
        b = getattr(rules, attr, None)
        if b:
            out.setdefault(fn, []).append({"bang": b[0], "cot_don": b[1], col: b[2]})
    m = getattr(rules, "members", None)
    if m:
        out.setdefault("new_members", []).append({"bang": m[0], "cot_khach": m[1], "cot_dang_ky": m[2]})
    for t, oid, col in (getattr(rules, "adjustments", None) or {}).values():
        out.setdefault("order_adjustment_total", []).append({"bang": t, "cot_don": oid, "cot_tien": col})
    return out


def fill_columns(plan: dict, sem) -> dict | None:
    """Lời gọi phép cần cột mà kế hoạch không ghi cột → cột tầng tất định đã gắn cho shop (đúng một cách gắn; nhiều
    cách, như các khoản cấp đơn, thì kế hoạch phải tự ghi). Không gắn được → None."""
    need = [c for c in plan["calls"] if any(a.rstrip("?") in COLUMN_ARGS and not a.endswith("?")
                                              and a.rstrip("?") not in c["args"] for a in FUNCS[c["fn"]][0])]
    if not need:
        return plan
    view = getattr(sem, "view", None)
    if view is None:
        return None
    if not hasattr(view, "_sem_rules"):
        from datamate.baseline import prepare_sem
        try:
            prepare_sem(view, sem._shop, sem._rulebook, getattr(sem, "_calendar", None))
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
    bound = bound_columns(getattr(view, "_sem_rules", None))
    calls = []
    for c in plan["calls"]:
        if c in need:
            opts = bound.get(c["fn"], [])
            if len(opts) != 1:
                return None
            c = {"fn": c["fn"], "args": {**opts[0], **c["args"]}}
        calls.append(c)
    return {**plan, "calls": calls}


def answer_from(text: str, question: str, sem, blocked=frozenset()) -> dict | None:
    """Câu trả lời của model → đáp án đã thực thi, hoặc None (kế hoạch không hợp lệ / không chạy / sai kiểu dự kiến)."""
    plan = parse_plan(text)
    if plan is not None:
        plan = check_plan(plan, question, getattr(sem, "_as_of", None), getattr(sem, "_calendar", None))
    if plan is not None:
        plan = fill_columns(plan, sem)
    ans = execute(plan, sem, blocked) if plan else None
    want = expected_type(question)
    if ans is None or (want and want != ans["type"]):
        return None
    # số / % bằng 0 thường là lọc rỗng (sai tên hàng, sai size) — cùng luật bỏ trống của agent
    if ans["type"] in ("number", "percent") and ans["answer"] == 0:
        return None
    return ans
