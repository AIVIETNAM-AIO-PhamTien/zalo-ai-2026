"""Agent: LLM viết code pandas, executor chạy, lặp tới khi có <answer>; nhiều mẫu rồi vote, không đủ đồng thuận thì abstain."""
from __future__ import annotations

import datetime as dt
import difflib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from datamate.gold import gold_dictionary
from datamate.rule_policy import policy_hint
from datamate.answer import answer_key, normalize_answer
from datamate.arbiter import veto
from datamate.telex import telex_fix
from datamate.answer_type import expected_type
from datamate.executor import Executor
from datamate.linking import hints as linking_hints
from datamate.lint import LintHints, lint
from datamate.parsing import fold
from datamate.shots import ShotBank, format_shots

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích dữ liệu bán hàng của tiểu thương Việt Nam. Bạn trả lời câu hỏi của chủ tiệm bằng cách viết code Python (pandas) chạy trên dữ liệu thật của cửa hàng.

## Cách làm việc
- Mỗi lượt viết ĐÚNG MỘT khối ```python ... ```. Hệ thống chạy code và gửi lại kết quả (stdout và giá trị biểu thức cuối). Biến được giữ giữa các lượt.
- Khám phá trước khi tính: xem cột, các giá trị trạng thái/nhóm/hình thức thanh toán, vài dòng mẫu; in kết quả trung gian để tự kiểm tra.
- Khi đã có kết quả từ code và chắc chắn, trả lời ở một lượt riêng (không kèm code) bằng đúng một dòng:
<answer>{{"type": "...", "answer": ...}}</answer>
- Tối đa {max_turns} lượt.

## Môi trường có sẵn
- `T`: dict tên bảng → DataFrame (số tiền, số lượng, ngày bẩn đã đổi sang kiểu chuẩn; xem mô tả bảng). `README`: README của cửa hàng.
- `TODAY`: hôm nay (pd.Timestamp 00:00) = as_of_date của câu hỏi. Lọc một ngày hay một kỳ luôn dùng `P.mask(cot_ngay, P.today())`, `P.mask(cot_ngay, P.yesterday())`… (cột ngày có thể kèm giờ nên `==` dễ ra rỗng).
- `P`: kỳ thời gian theo rulebook, mỗi hàm trả (pd.Timestamp đầu kỳ, pd.Timestamp cuối ngày cuối kỳ), so được trực tiếp với cột ngày: P.today(), P.yesterday(), P.day(d), P.this_week(), P.last_week(), P.this_month(), P.prev_month_same_days(), P.month(m, y=None), P.prev_month(m, y=None), P.quarter(q, y=None), P.year(y), P.year_to_date(), P.last_n_days(n), P.tet(y), P.before_tet(y, n_days=None); P.n_days(ky). Lọc theo kỳ: `df[P.mask(df['Ngày'], ky)]`.
- `valid_order(series_trang_thai)` → mask đơn hợp lệ (không huỷ, không trả; đang giao vẫn tính); `order_status(series)` → completed / cancelled / returned / delivering. Hiểu mọi cách ghi trạng thái; ô trống là đơn bình thường (completed).
- `find_values(query, limit=8)` → [(giá trị, bảng, cột, điểm)]: tìm tên hàng / khách / nhân viên / nhóm gần giống trong MỌI cột chữ của mọi bảng (không dấu, sai chính tả, thiếu chữ).
- `name_mask(series, ten)` → mask mọi dòng ghi cùng tên (khác hoa/thường, dấu, khoảng trắng thừa vẫn tính là một).
- `top_k(df, so_do, ma, k=None, ascending=False)`: xếp hạng đúng rulebook (số đo giảm, hoà thì mã tăng) — mọi câu 'nhiều nhất / top k / ít nhất' dùng hàm này.
- `round_money(x)` làm tròn về đồng (nửa đồng lên), `round2(x)`, `pct_change(moi, cu)`, `share(phan, tong)` (đều trả số phần trăm làm tròn 2 chữ số).
- `find_names(query, candidates, limit=5)` khớp tên mờ không dấu → [(tên, điểm 0–100)]; `fold(s)` bỏ dấu + lowercase; `strip_phone(ten)`; `is_walkin(ten)` nhận diện khách lẻ; `parse_number`, `parse_date`, `parse_quantity`.
- `pd`, `np`, `dt` (module datetime), `re`.
- `SEM`: phép tính chuẩn theo rulebook đã kiểm chứng (doanh thu, số đơn, đơn huỷ, tỉ lệ huỷ, giá trị trung bình đơn, số lượng bán quy về đơn vị cơ bản, top bán chạy, công nợ; `SEM.help()` xem danh sách). Trả None khi shop không dựng được bảng chuẩn → tự tính bằng pandas. Câu hỏi đúng dạng đó thì dùng SEM; câu phức tạp hơn thì dùng để đối chiếu kết quả của bạn.
- `SEM.answer()`: yêu cầu tầng tất định giải lại TOÀN BỘ câu hỏi gốc. Gọi trước khi tự tính; trả dict đáp án hoặc None. `SEM.explain()` cho biết view và trạng thái chưa hỗ trợ. None không có nghĩa là dữ liệu không đủ. `SEM.answer(cau_viet_lai)` chỉ là phép thử: phải giữ mọi điều kiện của câu gốc.
- `G`: bảng chuẩn đã áp sẵn rulebook (từ điển dưới đây). Bảng nào shop không dựng được thì không có trong `G`; `G['_bao_cao']` cho biết lý do.
{gold_dict}
- THỨ TỰ: (1) `SEM.answer()` / hàm `SEM` đúng dạng câu; (2) không có thì tự tính trên `G` (doanh thu = cộng `revenue` của `G['don']` có counts_revenue; doanh thu một mặt hàng / nhóm = cộng `amount` của `G['tien_mat_hang']` / `G['tien_nhom']` có counts_revenue; số lượng bán = cộng `bqty` của `G['dong']` có counts_revenue; đếm đơn = nunique order_id; xếp hạng dùng `top_k`) — không tự lọc trạng thái hay quy đổi lại; (3) chỉ dùng `T` cho thông tin `G` không có (nhân viên, kho, chi phí…). Ô NaN trong `G` là dữ liệu thiếu: tổng có NaN thì không trả số (không coi là 0).
- Khi tự tính xong, dùng `emit_result('number', ket_qua_da_tinh)` (hoặc percent/string/date/list/table/boolean) ở cuối code. Máy nhận giá trị đã thực thi, không cần chép số sang một lượt trả lời khác. Không truyền số đoán/gõ tay; lỗi trong cell làm kết quả bị huỷ.

## Nguyên tắc
1. Tuân thủ tuyệt đối RULEBOOK bên dưới (doanh thu, đơn hợp lệ, cách hiểu kỳ, xếp hạng, hoà, làm tròn, format). Câu hỏi nói khác thì theo câu hỏi.
2. Bản ghi có ngày sau hôm nay coi như chưa xảy ra (riêng trạng thái huỷ/trả hàng lấy theo trạng thái ghi trên đơn).
3. Tên trong câu hỏi có thể không dấu, viết tắt, sai chính tả, thiếu chữ (xem 7). Đáp án tên (string/list/table) chép NGUYÊN tên như trong dữ liệu.
4. Không trả lời được (rulebook §7) khi dữ liệu không có thông tin câu hỏi cần, kỳ hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu, hoặc câu hỏi chủ quan / cần hiểu biết bên ngoài → <answer>{{"type": "unanswerable", "reason": "..."}}</answer>. Không có dữ liệu khác với bằng 0. Câu không trả lời được rất hiếm: trước khi kết luận, xem cột của MỌI bảng (kể cả bảng danh mục, bảng phụ) — tên cột có thể viết liền không dấu, viết tắt hay tiếng Anh — và tìm tên bằng `find_values`.
5. Type đáp án: number (tiền là số nguyên VND, đếm, số lượng), percent (số phần trăm: x% ghi là x, không chia 100), string, date ("YYYY-MM-DD"; hỏi theo tháng thì "YYYY-MM"), boolean (true/false), list (câu hỏi nhiều giá trị hoặc top-k, đúng thứ tự), table ({{"columns": [...], "rows": [[...], ...]}}, cột đầu là tên đối tượng, các cột sau là số đo theo thứ tự câu hỏi). Một mặt hàng "nhất" → string; "k mặt hàng" → list.
6. Không đoán: mọi con số phải đến từ code đã chạy; không gõ tay con số vào code rồi trả lời.
7. Tên hàng / khách / nhân viên trong câu hỏi: LUÔN gọi `find_values("...")` trước rồi lọc bằng `name_mask(df[cot], gia_tri_tim_duoc)` để gom mọi cách ghi của cùng tên (không dùng `==` hay `str.contains` với chữ trong câu hỏi). Một mặt hàng có thể có cả tên và mã ở bảng khác: kiểm tra kết quả find_values. Chỉ kết luận không có / bằng 0 khi find_values không ra ứng viên nào gần nghĩa.
8. Đơn hợp lệ: dùng `valid_order(...)` trên cột trạng thái, không tự liệt kê các giá trị trạng thái. Câu hỏi về doanh thu / số lượng bán / số đơn đều phải lọc đơn hợp lệ.
9. Làm tròn bằng `round2` / `round_money` / `pct_change` / `share`, không dùng `round()`.
10. "tháng này" = P.this_month(), "hôm nay" = P.today(): đừng nhầm. "nhiều nhất / cao nhất / top k" = xếp giảm dần theo số đo rồi theo mã tăng dần khi bằng nhau (rulebook): `sort_values([so_do, ma], ascending=[False, True])`, không dùng `idxmax` trần. Sau khi lọc mà không còn dòng nào (hoặc đối tượng không phát sinh trong kỳ) thì không lấy phần tử của bảng khác / kỳ khác để bù: kiểm `len(df)` trước khi lấy phần tử đầu.
11. Câu hỏi đòi nhiều con số ("mỗi loại", "trả lời hai số", "A và B") → list các số theo đúng thứ tự câu hỏi, không dùng table. Hỏi "ai / khách nào / mặt hàng nào / thợ nào…" → trả TÊN (tra mã sang tên ở bảng danh mục); chỉ trả mã khi câu hỏi hỏi "mã", hoặc dữ liệu chỉ có mã mà không có tên (khi đó không trả unanswerable).
12. Số lượng bán (bao nhiêu kg / lon / ly / hộp…, bán chạy, so sánh số lượng) luôn quy về ĐƠN VỊ CƠ BẢN và cộng MỌI dòng của mặt hàng, kể cả dòng bán theo đơn vị lớn (thùng, bao, lốc…): KHÔNG lọc theo cột đơn vị của dòng. Bảng có sẵn cột số lượng đã quy đổi (vd 'Số lượng quy đổi', 'SLQuyDoi', 'SL cơ bản', 'Quy ra ĐV lẻ') thì dùng cột đó; không có thì nhân số lượng với hệ số trong bảng quy đổi / README (đơn vị không có dòng quy đổi là đơn vị cơ bản, hệ số 1). Giá bán một đơn vị = đơn giá dòng chia hệ số của đơn vị dòng. "Bao nhiêu ly size M" = cộng mọi món có tên kết thúc bằng (M).
13. Khách hàng: shop có mã khách thì mỗi mã là một khách; không có mã thì gom tên theo `name_mask`. Đơn khách lẻ (`is_walkin(ten)`: trống, "khách lẻ", "kl", "vãng lai"…) KHÔNG phải khách: bỏ ra trước khi đếm số khách hay xếp hạng khách.
14. Đếm đơn / đợt giao / khách / lô = đếm số MÃ khác nhau (`nunique`, hoặc cặp (mã đơn, số đợt)), KHÔNG đếm số dòng: bảng chi tiết thường có nhiều dòng (nhiều mặt hàng) cho một đơn, một đợt. In thử số dòng và số mã để đối chiếu.
15. Trước khi trả lời, kiểm độ lớn: phần trăm thay đổi doanh thu tháng thường vài chục % trở xuống (đừng nhân 100 hai lần); giá nhập / giá bán một đơn vị phải là giá MỘT đơn vị được hỏi (hộp, thùng…), không phải tổng tiền dòng hay giá nhân hệ số quy đổi; tổng tiền trong kỳ phải cộng đủ mọi dòng của kỳ.
16. Lô hàng / hạn dùng / sắp hết hạn: chỉ tính lô còn tồn (số lượng còn lại > 0) tại hôm nay, trừ khi câu hỏi hỏi về lịch sử lô; dùng `SEM.lots_expiring` / `SEM.lot_nearest_expiry` khi có.

# RULEBOOK
{rulebook}"""

USER_PROMPT = """# README của cửa hàng
{readme}

{gold}{brief}

{shots}# Câu hỏi
Hôm nay (as_of_date): {as_of} ({weekday})
Câu hỏi: {question}{type_hint}"""

GOLD_BRIEF = ("# Bảng G của cửa hàng này (đã áp rulebook: trạng thái, dòng lặp, quy đổi, khách lẻ)\n"
              "Doanh thu / số lượng / số đơn / tiền theo mặt hàng, nhóm: tính trên G trước (G đã lọc đơn sau hôm nay). "
              "Bảng thô T bên dưới chỉ dùng cho thông tin G không có.\n{g}\n\n")
TYPE_HINT = "\nKiểu đáp án dự kiến: {t} (nếu dữ liệu không đủ để trả lời thì vẫn là unanswerable)."
RECHECK_TYPE = ("Câu hỏi này thường có đáp án kiểu \"{t}\", bạn đang trả lời kiểu \"{got}\". Kiểm tra lại type theo "
                "nguyên tắc 5 (vd hỏi tháng nào → date \"YYYY-MM\"; nhiều số → list; phần trăm → percent) rồi trả lời lại; "
                "chắc chắn kiểu của bạn đúng thì giữ nguyên.")

# đáp án đáng ngờ → hỏi lại một lần trước khi nhận (tự sửa kiểu Self-Debugging / Reflexion).
# Train: đáp án 0 chỉ 4/3000 câu, unanswerable 2.3%; còn agent hay ra 0 / bỏ cuộc vì lọc sai, tìm sót cột.
RECHECK_ZERO = ("Kết quả bằng 0 / rỗng rất hiếm khi đúng: thường do một điều kiện lọc loại hết dòng. "
                "Kiểm tra lại: in số dòng còn lại sau TỪNG điều kiện (kỳ, trạng thái, tên, nhóm…), xem các giá trị thật "
                "của cột đang lọc (`value_counts()`), kiểm tra cột ngày có kèm giờ, tên có đúng như trong dữ liệu. "
                "Nếu kiểm tra xong vẫn đúng là 0 thì trả lời lại như cũ.")
RECHECK_UNANSWERABLE = ("Trước khi kết luận không trả lời được: in tên cột của MỌI bảng "
                        "(`{t: list(df.columns) for t, df in T.items()}`) và đọc lại README; tên cột có thể viết liền, "
                        "không dấu, viết tắt hay tiếng Anh. Tên trong câu hỏi có thể sai chính tả: dùng `find_values`. "
                        "Nếu vẫn không có thông tin thì trả lời lại unanswerable.")
# log public v3: 68/822 câu có mẫu kết luận unanswerable khi chưa chạy thành công đoạn code nào (thường vì viết
# <answer> trong code → SyntaxError); train: mẫu như vậy chỉ đúng 35%
RECHECK_NO_RUN = ("Bạn chưa chạy thành công đoạn code nào nên chưa thấy dữ liệu thật: chưa thể kết luận không trả "
                  "lời được. Viết lại code chỉ gồm pandas, in kết quả bằng print(), rồi trả lời ở lượt sau.")
ANSWER_IN_CODE = ("Không viết <answer> bên trong code (không phải cú pháp Python). Code chỉ tính và print() kết quả; "
                  "thấy kết quả rồi mới trả lời <answer> ở một lượt riêng.")
MONTH_END = ("Đừng gõ tay ngày cuối tháng (tháng 2 có 28 hoặc 29 ngày): lọc theo tháng bằng "
             "`P.mask(cot_ngay, P.month(m, y))`.")


# schema / value linking theo lỗi (RSL-SQL, CHESS): lượt GPU 2 có ~15 lần KeyError tên cột đoán sai
# ('Thời gian', 'Mã KH', 'Product') và ~7 câu bỏ cuộc vì không thấy tên hàng gõ sai / thiếu chữ trong câu hỏi
_MISSING_COL = re.compile(r"Column not found: ([^'\"\n]+)|KeyError: \"?\[?'([^']+)'|'([^']+)'\]? not in index")
_COLS_PROBE = "import json as _j\nprint(_j.dumps({t: [str(c) for c in df.columns] for t, df in T.items()}, ensure_ascii=False))"
_QUOTED = re.compile(r"['\"‘“]([^'\"’”]{3,80})['\"’”]")


def column_hint(ex, err: str, full: bool = True) -> str | None:
    """Lỗi do gọi cột không có → cột gần tên nhất (không dấu) + danh sách cột thật của mọi bảng (full=False: chỉ
    cột gần tên nhất; log public v3: KeyError lần hai trong một mẫu không còn được gợi ý, vd 'Giờ bán' 18 lần)."""
    m = _MISSING_COL.search(err or "")
    if not m:
        return None
    missing = next(g for g in m.groups() if g).strip()
    out, e = ex.run(_COLS_PROBE)
    try:
        cols = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    if e or not isinstance(cols, dict):
        return None
    flat = {fold(c): f"T[{t!r}][{c!r}]" for t, cs in cols.items() for c in cs}
    near = [flat[k] for k in difflib.get_close_matches(fold(missing), list(flat), n=4, cutoff=0.4)]
    first = f"Không có cột {missing!r}." + (f" Cột gần tên nhất: {', '.join(near)}." if near else "")
    if not full:
        return first if near else None
    lines = [first, "Cột thật của các bảng:"] + [f"- {t}: {cs}" for t, cs in cols.items()]
    return "\n".join(lines)[:3000]


def value_hint(ex, ans: dict) -> str | None:
    """Đáp án unanswerable vì 'không tìm thấy <tên>' → chạy find_values (ngưỡng thấp) với tên đó, đưa ứng viên."""
    names = _QUOTED.findall(str(ans.get("reason") or ""))
    if not names:
        return None
    code = "\n".join(f"print({n!r}, '→', find_values({n!r}, limit=5, min_score=45))" for n in names[:2])
    out, err = ex.run(code)
    if err or "(" not in (out or ""):
        return None
    return ("Ứng viên gần giống tên bạn không tìm thấy (giá trị, bảng, cột, điểm):\n" + out.strip()[:1500] +
            "\nNếu một ứng viên đúng là thứ câu hỏi nói tới (gõ sai, thiếu chữ, viết tắt) thì dùng nó và tính lại.")


def recheck_prompt(ans: dict) -> str | None:
    """Lời nhắc kiểm tra lại cho đáp án đáng ngờ; None nếu đáp án bình thường."""
    t, a = ans.get("type"), ans.get("answer")
    if t == "unanswerable":
        return RECHECK_UNANSWERABLE
    if (t in ("number", "percent") and a == 0) or (t in ("list", "string") and not a):
        return RECHECK_ZERO
    if t == "table" and isinstance(a, dict) and not a.get("rows"):
        return RECHECK_ZERO
    return None


VERIFY_SYSTEM = """Bạn là người kiểm tra lời giải phân tích dữ liệu bán hàng (DS-STAR verifier). Bạn KHÔNG viết code.
Đọc câu hỏi, luật liên quan, code cuối, kết quả chạy và đáp án; kiểm tra: đúng kỳ (theo hôm nay), đúng bảng/cột,
lọc đơn hợp lệ theo rulebook, đúng đơn vị / làm tròn / type đáp án, trả lời đúng điều câu hỏi hỏi.
Trả lời đúng một dòng: "OK" nếu không thấy lỗi, hoặc "SAI: <lỗi cụ thể, ngắn gọn>"."""
VERIFY_USER = """Hôm nay: {as_of}
Câu hỏi: {question}{hints}

Code cuối:
```python
{code}
```
Kết quả: {out}
Đáp án: {answer}"""
VERIFY_FEEDBACK = ("Người kiểm tra thấy có thể sai: {reason}\nKiểm tra lại điểm đó trên dữ liệu rồi trả lời lại; "
                   "nếu chắc chắn lời giải đúng thì giữ đáp án.")


def verify(task: Task, ctx: ShopContext, llm, trace: list, ans: dict, seed: int) -> str | None:
    """Lý do sai theo verifier, hoặc None nếu OK / không kiểm được."""
    code, out = _last_code(trace)
    if not code:
        return None
    hints = linking_hints(task.question, ctx.rule_rows, None)
    try:
        reply = llm.chat([{"role": "system", "content": VERIFY_SYSTEM},
                          {"role": "user", "content": VERIFY_USER.format(
                              as_of=task.as_of.isoformat(), question=task.question,
                              hints=f"\n\n{hints}" if hints else "", code=code[-2500:], out=out[-800:],
                              answer=json.dumps(ans, ensure_ascii=False))}], seed=seed + 7)
    except RuntimeError:
        return None
    m = re.search(r"\bSAI\s*:\s*(.+)", reply, re.S)
    return m.group(1).strip()[:400] if m else None


_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


@dataclass
class Task:
    qid: str
    biz_id: str
    question: str
    as_of: dt.date
    facts: list = field(default_factory=list)  # [(lời gọi SEM, kiểu, giá trị)] máy gọi sẵn (sem_facts)


@dataclass
class ShopContext:
    cache_path: Path
    calendar_csv: Path | None
    readme: str
    brief: str
    rulebook: str
    exec_timeout: float = 120
    max_turns: int = 8
    shots: ShotBank | None = None  # few-shot động; None → không chèn ví dụ
    lint_hints: LintHints | None = None  # bắt lỗi code hay gặp (datamate/lint.py); None → tắt
    rule_rows: list | None = None  # dòng luật rulebook để gợi ý luật liên quan (datamate/linking.py); None → tắt
    col_index: list | None = None  # chỉ mục cột để gợi ý cột ứng viên; None → tắt
    verify: bool = False  # verifier LLM riêng kiểm đáp án trước khi nhận (DS-STAR)
    checked_joins: bool = False  # thử nghiệm: công cụ kiểm quan hệ nối, chưa đo PA sinh mới
    schema_card: dict | None = None  # preserve the runner's validated column bindings inside SEM
    gold_brief: str = ""  # G thật của shop (gold.gold_brief): bảng, cột, dòng mẫu, bảng thiếu


def extract_code(text: str) -> str | None:
    blocks = _CODE.findall(text)
    return blocks[0] if blocks else None


def extract_answer(text: str):
    m = _ANSWER.findall(text)
    if not m:
        return None
    raw = m[-1].strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return "INVALID"


# câu khớp một hàm SEM → nhắc gọi thử trước (log lượt GPU 2: nhóm công nợ PA −0.54, agent không gọi SEM.debt_* lần nào)
_SEM_ROUTES = [
    (r"\bno\b|cong no|mua chiu", "SEM.debt_total / debtor_count / debt_added / debt_repaid / debt_top / debt_by_age / "
                                 "debt_customer"),
    (r"\bgia nhap\b|\bnhap\b.*\bbinh quan\b", "SEM.import_avg_price"),
    (r"\bhet han\b|\bhan dung\b", "SEM.lot_nearest_expiry / lots_expiring"),
    (r"\bhuy\b", "SEM.cancelled_count / cancel_rate"),
    (r"\b(tang gia|doi gia|len gia)\b", "SEM.price_effect"),
    (r"\bdot\b", "SEM.delivery_count"),
    (r"\bca (sang|chieu|toi)\b", "SEM.shift_revenue / shift_order_count"),
    (r"\bhoat chat\b", "SEM.ingredient_revenue"),
    (r"\bke\s+don\b|\bdon bac si\b", "SEM.rx_revenue"),
    (r"\bcod\b|chuyen khoan|vi dien tu|tien mat", "SEM.payment_revenue"),
    # rulebook public: các câu 'nhất' có định nghĩa riêng (trung thành = nhiều ngày mua, sộp = TB đơn từ 3 đơn…)
    (r"trung thanh|\bsop\b|doi y|khach nao.*(nhieu|nhiu) (don|lan)",
     "SEM.loyal_customer / big_spender / fickle_customer / customer_most_orders"),
    (r"ua chuong|dat khach|cao diem", "SEM.popular_item / busiest_day / peak_hour"),
    (r"trung binh.*(moi )?(ngay )?(thu (hai|ba|tu|nam|sau|bay|[2-7])|chu nhat)", "SEM.weekday_avg"),
]
SEM_HINT = ("Gợi ý: câu này khớp hàm chuẩn {f} (đã kiểm chứng theo rulebook). Gọi thử trước: có số thì dùng (hoặc đối "
            "chiếu với pandas của bạn), None thì tự tính bằng pandas.")


# viết tắt hay gặp trong câu hỏi (log lượt GPU 2: 'số đơn hqua' bị hiểu thành 'số đơn huỷ hôm nay')
_ABBR = [(r"hqua|bua qua", "hôm qua"), (r"hnay|bua nay", "hôm nay"), (r"dthu", "doanh thu"),
         (r"bn|bnhiu|bnhieu|bao nhiu|nhiu", "bao nhiêu"), (r"trc", "trước"), (r"dc|đc", "được"), (r"ko", "không"),
         (r"sl", "số lượng"), (r"kh", "khách hàng"), (r"nv", "nhân viên"), (r"sp", "sản phẩm"), (r"z", "vậy")]


def abbr_hint(question: str, as_of: dt.date) -> str:
    """Giải nghĩa viết tắt có trong câu; 'hôm qua' / 'hôm nay' kèm ngày cụ thể."""
    q = question.lower()
    found = []
    for pat, full in _ABBR:
        m = re.search(rf"(?<![\wÀ-ỹ])({pat})(?![\wÀ-ỹ])", q)
        if m:
            day = {"hôm qua": as_of - dt.timedelta(days=1), "hôm nay": as_of}.get(full)
            found.append(f"'{m.group(1)}' = {full}" + (f" ({day.isoformat()})" if day else ""))
    hint = ("Viết tắt trong câu hỏi: " + "; ".join(found) + ".") if found else ""
    typos = telex_fix(question)[1]
    if typos:
        hint = "\n".join(x for x in (hint, "Lỗi gõ trong câu hỏi: " + "; ".join(f"'{a}' = {b}" for a, b in typos)
                                     + ".") if x)
    return "\n".join(x for x in (hint, month_hint(question)) if x)


# tháng viết tắt kiểu '8/25' (log public v3: 76 câu, có câu '8/25' bị hiểu thành ngày 25/8; train không có dạng này,
# dạng 'tháng 6/2025' agent đã hiểu đúng nên không nhắc)
_MONTH = re.compile(r"(?<![\d/.,])(\d{1,2})/(\d{2})(?![\d/])")


def month_hint(question: str) -> str:
    """'m/yy' với yy > 12 (không thể là ngày/tháng) → tháng m năm 20yy; bỏ qua 'quý 3/26'."""
    found = []
    for m in _MONTH.finditer(question):
        mo, yy = int(m.group(1)), int(m.group(2))
        before = fold(question[max(0, m.start() - 4):m.start()]).rstrip()
        if not 1 <= mo <= 12 or yy <= 12 or re.search(r"(quy|q)$", before):
            continue
        found.append(f"'{m.group(0)}' = tháng {mo}/{2000 + yy}, P.month({mo}, {2000 + yy}) "
                     f"(không phải ngày {yy} tháng {mo})")
    return ("Tháng viết tắt trong câu hỏi: " + "; ".join(found) + ".") if found else ""


# hàm SEM chỉ cần kỳ (và ca) → máy gọi sẵn khi câu khớp route và đọc ra đúng một kỳ (agent hầu như không tự gọi SEM:
# log public v3 5/822 lượt; "Library Learning Doesn't", 2024: thư viện hàm gần như không được tái dùng nếu để model tự
# quyết; semantic layer làm phép tính thành tra cứu có ràng buộc)
# (mẫu câu đã norm, hàm, kiểu): mỗi mẫu chỉ ra đúng MỘT chỉ số (route rộng gọi cả debt_added lẫn debt_repaid cho câu
# 'còn nợ' → đối chiếu nhầm; đo trên log public v3)
_FACTS = [
    (r"\bhuy\b.*%|%.*\bhuy\b|ti le.*\bhuy\b|\bhuy\b.*phan tram", "cancel_rate", "percent"),
    (r"(may|mi|BAONHIEU|bn|so) don.*\bhuy\b|\bhuy\b.*(may|mi|BAONHIEU|bn) don", "cancelled_count", "number"),
    (r"trung thanh", "loyal_customer", "string"),
    (r"\bsop\b", "big_spender", "string"),
    (r"doi y", "fickle_customer", "string"),
    (r"khach nao.*(nhieu|nhiu) (don|lan)", "customer_most_orders", "string"),
    (r"ua chuong", "popular_item", "string"),
    (r"dat khach", "busiest_day", "date"),
    (r"cao diem", "peak_hour", "number"),
    (r"(may|mi|BAONHIEU|bn|so) dot giao", "delivery_count", "number"),
    (r"(tra no|thu no|thu duoc.*\bno\b|doi duoc.*\bno\b)", "debt_repaid", "number"),
    (r"(no them|no moi|phat sinh.*\bno\b|ghi no them)", "debt_added", "number"),
]
_CA = {"sang": "sang", "chieu": "chieu", "toi": "toi"}


def sem_facts(question: str, as_of: dt.date, sem, calendar_csv=None) -> list[tuple[str, str, object]]:
    """[(lời gọi, kiểu, giá trị)] của các hàm SEM khớp câu hỏi, máy gọi sẵn với kỳ đọc từ câu."""
    from datamate.baseline import extract_periods, norm
    from datamate.helpers import Periods
    s = norm(question)
    calls = [(fn, {}, kind) for rx, fn, kind in _FACTS if re.search(rx, s)]
    m = re.search(r"\bca (sang|chieu|toi)\b", s)
    if m:
        ca = {"ca": _CA[m.group(1)]}
        calls.append(("shift_order_count", ca, "number") if re.search(r"(may|mi|BAONHIEU|bn|so) don", s)
                     else ("shift_revenue", ca, "number"))
    if not calls or sem is None:
        return []
    try:
        found, _ = extract_periods(s, Periods(as_of, calendar_csv))
    except Exception:
        return []
    if any(label == "BAD" or p is None for label, p in found):
        return []
    kys = {tuple(p) for _, p in found}
    if len(kys) != 1:
        return []
    a, b = kys.pop()
    ky = {"tu": a.isoformat(), "den": b.isoformat()}
    out = []
    for fn, extra, kind in calls:
        try:
            v = getattr(sem, fn)((a, b), **extra)
        except Exception:
            continue
        if v is None or isinstance(v, (list, tuple, dict)):
            continue
        args = ", ".join([f"ky={ky}"] + [f"{k}={v2!r}" for k, v2 in extra.items()])
        out.append((f"SEM.{fn}({args})", kind, v.isoformat() if isinstance(v, dt.date) else v))
    return out


FACTS_HINT = ("Máy đã gọi sẵn hàm chuẩn theo rulebook với kỳ trong câu (đúng luật, đã kiểm chứng):\n{facts}\n"
              "Câu hỏi đúng là chỉ số này thì trả đúng giá trị đó; câu có thêm điều kiện thì tự tính và đối chiếu.")


def sem_hint(question: str) -> str:
    from datamate.baseline import norm
    s = norm(question)
    fs = [f for rx, f in _SEM_ROUTES if re.search(rx, s)]
    return SEM_HINT.format(f=" / ".join(fs)) if fs else ""


# kỳ trong câu đã quy ra ngày theo rulebook §2 (bộ đọc kỳ của tầng tất định, sai 0 trên train). Không gợi ý:
# 'hôm nay' / 'hôm qua' / 'ngày X' (thường là mốc 'tính đến hết …' của kỳ khác; abbr_hint đã ghi ngày hôm nay),
# câu 'từ kỳ A đến kỳ B' (bộ đọc chỉ ra hai đầu mút), kỳ lệch xa hôm nay (đọc nhầm, vd '10/10/2025' → tháng 10/2010)
_PERIOD_NAMES = {"month": "tháng", "quarter": "quý", "year": "năm", "this_month": "tháng này",
                 "prev_month_same_days": "cùng kỳ tháng trước", "this_week": "tuần này", "last_week": "tuần trước",
                 "last_n": "N ngày gần nhất", "ytd": "từ đầu năm đến hết hôm nay"}


def tet_hint(question: str, as_of: dt.date, calendar_csv) -> str:
    """'Tết năm Y' quy ra ngày dương theo rulebook (23 tháng Chạp → hết mùng 10 tháng Giêng, tra calendar.csv);
    log kế hoạch public: model trả 'none' vì không biết ngày Tết."""
    from datamate.baseline import norm
    from datamate.helpers import Periods
    s = norm(question).lower()
    if calendar_csv is None or not re.search(r"\btet\b", s):
        return ""
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", s)}
    # 'Tết năm nay so với Tết năm ngoái': log Colab 31/10 — model lấy Tết năm ngoái trùng ngày dương với năm nay
    if re.search(r"\bnam (nay|ni)\b", s):
        years.add(as_of.year)
    if re.search(r"\bnam (ngoai|truoc|trc|roi)\b", s):
        years.add(as_of.year - 1)
    years = sorted(years) or [as_of.year]
    P, parts = Periods(as_of, calendar_csv), []
    for y in years:
        try:
            a, b = P.tet(y)
        except Exception:
            continue
        parts.append(f"Tết {y}: {a.isoformat()} → {b.isoformat()} ({(b - a).days + 1} ngày)")
        m = re.search(r"(\d{1,3}) ngay (ngay )?truoc tet|truoc tet (\d{1,3}) ngay", s, re.I)
        n = int(m.group(1) or m.group(3)) if m else None
        c, d = P.before_tet(y, n)
        parts.append(f"{n or 'cùng số'} ngày trước Tết {y}: {c.isoformat()} → {d.isoformat()}")
    return ("Dịp Tết theo rulebook (đầu và cuối đều tính): " + "; ".join(parts) + ".") if parts else ""


def period_hint(question: str, as_of: dt.date, calendar_csv) -> str:
    tet = tet_hint(question, as_of, calendar_csv)
    base = _period_hint(question, as_of, calendar_csv)
    return "\n".join(x for x in (base, compare_hint(question, as_of, calendar_csv), tet) if x)


_PREV = r"(truoc|trc|roi|lien truoc)"


def compare_hint(question: str, as_of: dt.date, calendar_csv) -> str:
    """Kỳ đối chiếu quy ra ngày bằng code (rulebook §2) để model không tự tính: log Colab 31/10, kế hoạch bị
    check_plan chặn mà vẫn chạy được thì sai chủ yếu ở kỳ đối chiếu ('cùng kỳ tháng trước' lấy thành năm trước,
    'N ngày ngay trước ngày đổi giá' lệch ngày)."""
    from datamate.baseline import extract_periods, norm
    from datamate.helpers import Periods, anchored_windows
    s = norm(question).lower()
    P, parts = Periods(as_of, calendar_csv), []

    def fmt(name, p):
        return f"{name}: {p[0].isoformat()} → {p[1].isoformat()}"
    if re.search(rf"cung ky (thang|th) {_PREV}", s) and re.search(r"\b(thang|th) (nay|ni)\b", s):
        parts += [fmt("từ đầu tháng này đến hết hôm nay", P.this_month()),
                  fmt("cùng kỳ tháng trước (cùng số ngày)", P.prev_month_same_days())]
    win = anchored_windows(s)
    if win:
        n = (win[0][1] - win[0][0]).days + 1
        parts += [fmt(f"{n} ngày kể từ ngày {win[0][0].isoformat()}", win[0]), fmt(f"{n} ngày ngay trước đó", win[1])]
    try:
        found, _ = extract_periods(norm(question), P)
    except Exception:
        found = []
    months = sorted({tuple(p) for label, p in found if label == "month" and p is not None})
    if len(months) == 1:
        a = months[0][0]
        if re.search(rf"\b(thang|th) {_PREV}( do)?\b", s) and not re.search(rf"\bnam ({_PREV[1:-1]}|ngoai)\b", s):
            parts.append(fmt("tháng liền trước tháng hỏi", P.prev_month(a.month, a.year)))
        elif re.search(rf"\bnam ({_PREV[1:-1]}|ngoai)\b", s):
            parts.append(fmt("cùng tháng năm trước", P.month(a.month, a.year - 1)))
    return ("Kỳ đối chiếu, đã quy ra ngày theo rulebook (đầu và cuối đều tính): " + "; ".join(parts) + ".") \
        if parts else ""


def _period_hint(question: str, as_of: dt.date, calendar_csv) -> str:
    from datamate.baseline import extract_periods, norm
    from datamate.helpers import Periods
    try:
        found, _ = extract_periods(norm(question), Periods(as_of, calendar_csv))
    except Exception:
        return ""
    if any(label == "BAD" for label, _ in found) or (len(found) > 1 and re.search(r"\b(den|toi)\b", norm(question))):
        return ""
    if any(abs(a.year - as_of.year) > 2 for _, (a, _b) in found):
        return ""
    parts = [f"{_PERIOD_NAMES[label]}: {a.isoformat()} → {b.isoformat()}" for label, (a, b) in found
             if label in _PERIOD_NAMES]
    return ("Kỳ trong câu hỏi, đã quy ra ngày theo rulebook (đầu và cuối đều tính): " + "; ".join(parts) + ".") \
        if parts else ""


def build_messages(task: Task, ctx: ShopContext) -> list[dict]:
    shots = ""
    if ctx.shots is not None:
        shots = format_shots(ctx.shots.find(task.question, task.biz_id, exclude_id=task.qid))
        shots = shots + "\n\n" if shots else ""
    t = expected_type(task.question)
    extra = "\n\n".join(x for x in (abbr_hint(task.question, task.as_of),
                                     period_hint(task.question, task.as_of, ctx.calendar_csv),
                                     linking_hints(task.question, ctx.rule_rows, ctx.col_index),
                                     policy_hint(ctx.rulebook) if ctx.rulebook else "",
                                     sem_hint(task.question),
                                     ("Khi nối bảng, dùng checked_merge(left, right, on=..., validate='many_to_one') "
                                      "(hoặc left_on/right_on khi tên khoá khác). Khai báo đúng quan hệ; "
                                      "in result.attrs['join_audit'] để kiểm số dòng. Nối danh mục có thể dùng "
                                      "require_match=True để bắt thiếu mã. Nếu báo khoá trùng, kiểm độ hạt và "
                                      "khoá ghép; không xoá dòng trùng để che lỗi nối.") if ctx.checked_joins else "",
                                     FACTS_HINT.format(facts="\n".join(f"- {c} = {v!r}" for c, _, v in task.facts))
                                     if task.facts else "") if x)
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=ctx.max_turns, rulebook=ctx.rulebook, gold_dict=gold_dictionary())},
        {"role": "user", "content": USER_PROMPT.format(
            readme=ctx.readme, brief=ctx.brief, shots=shots,
            gold=(GOLD_BRIEF.format(g=ctx.gold_brief) if ctx.gold_brief else ""), as_of=task.as_of.isoformat(),
            weekday=WEEKDAYS[task.as_of.weekday()], question=task.question,
            type_hint=(TYPE_HINT.format(t=t) if t else "") + (f"\n\n{extra}" if extra else ""))},
    ]


def solve_once(task: Task, ctx: ShopContext, llm, seed: int, extra: str | None = None) -> tuple[dict | None, list]:
    messages = build_messages(task, ctx)
    if extra:
        messages[-1]["content"] += "\n\n" + extra
    trace: list[dict] = []
    rechecked = type_checked = verified = col_hinted = no_run_checked = False
    linted: set[str] = set()
    want = expected_type(task.question)
    with Executor(ctx.cache_path, task.as_of, ctx.calendar_csv, timeout=ctx.exec_timeout,
                  question=task.question, rulebook=ctx.rulebook, schema_card=ctx.schema_card) as ex:
        for turn in range(ctx.max_turns):
            try:
                reply = llm.chat(messages, seed=seed)
            except RuntimeError as e:  # lỗi server / vượt context → mẫu này thất bại, không làm hỏng câu
                trace.append({"turn": turn, "llm_error": str(e)})
                return None, trace
            # đáp án viết cùng lượt với code là đoán trước khi chạy: không giữ trong lịch sử, kẻo model coi con số
            # đoán đó là "đáp án mong đợi" (log lượt GPU 2: 'the system's expected answer was 1,428,571,428')
            shown = _ANSWER.sub("", reply).strip() if extract_code(reply) is not None else reply
            messages.append({"role": "assistant", "content": shown})
            step = {"turn": turn, "reply": reply}
            if getattr(llm, "last_think", None):
                step["think"] = llm.last_think
            trace.append(step)
            code = extract_code(reply)
            if code is not None:
                out, err = ex.run(code)
                step.update(out=out, err=err)
                if ex.last_sem_calls:
                    step['sem_calls'] = ex.last_sem_calls
                if err is None and ex.last_result is not None:
                    step['execution_result'] = ex.last_result
                    step['execution_result_source'] = ex.last_result_source
                    if ex.last_result_source != 'deterministic' and want and ex.last_result['type'] not in (want, 'unanswerable'):
                        step['execution_type_conflict'] = True
                        messages.append({'role': 'user', 'content': RECHECK_TYPE.format(t=want, got=ex.last_result['type']) +
                                         ' Hãy chạy lại emit_result với đúng kiểu sau khi kiểm tra.'})
                        continue
                    if ctx.verify and ex.last_result_source != 'deterministic' and not verified and turn < ctx.max_turns - 2:
                        verified = True
                        reason = verify(task, ctx, llm, trace, ex.last_result, seed)
                        if reason:
                            step['verifier'] = reason
                            messages.append({'role': 'user', 'content': VERIFY_FEEDBACK.format(reason=reason)})
                            continue
                    return ex.last_result, trace
                feedback = f"Kết quả chạy:\n```\n{out or '(không in gì)'}\n```"
                if err:
                    feedback += f"\nLỗi:\n```\n{err}\n```"
                    if hint := column_hint(ex, err, full=not col_hinted):
                        col_hinted = True  # danh sách cột dài: mỗi mẫu một lần, sau đó chỉ cột gần tên nhất
                        step["col_hint"] = True
                        feedback += "\n" + hint
                    if "<answer>" in code:
                        feedback += "\n" + ANSWER_IN_CODE
                    elif "day is out of range for month" in err:
                        feedback += "\n" + MONTH_END
                if ctx.lint_hints is not None:
                    notes = [n for n in lint(code, ctx.lint_hints) if n not in linted]
                    if notes:
                        linted.update(notes)  # mỗi lời nhắc chỉ một lần mỗi mẫu
                        step["lint"] = notes
                        feedback += "\n" + "\n".join(notes)
                if turn == ctx.max_turns - 2:
                    feedback += "\nĐây là lượt cuối: hãy trả lời ngay bằng <answer>…</answer>."
                messages.append({"role": "user", "content": feedback})
                continue
            raw = extract_answer(reply)
            if raw is None:
                messages.append({"role": "user", "content":
                                 "Hãy viết một khối ```python``` để tính, hoặc trả lời bằng <answer>{…}</answer>."})
                continue
            ans = normalize_answer(raw) if raw != "INVALID" else None
            if ans is None or ans["type"] == "abstain":
                step["invalid_answer"] = True
                messages.append({"role": "user", "content":
                                 "Đáp án sai format. Ghi lại đúng một dòng <answer>{\"type\": ..., \"answer\": ...}</answer> "
                                 "theo format của rulebook §9 (số là JSON number, không đơn vị, không dấu phân cách)."})
                continue
            if (ans["type"] == "unanswerable" and not no_run_checked and turn < ctx.max_turns - 2
                    and not any("out" in s and not s.get("err") for s in trace)):
                no_run_checked = True  # chỉ nhắc một lần
                step["recheck_no_run"] = ans
                messages.append({"role": "user", "content": RECHECK_NO_RUN})
                continue
            nudge = recheck_prompt(ans)
            # hỏi lại một lần mỗi mẫu, chỉ khi còn ít nhất hai lượt (một lượt code, một lượt trả lời)
            if nudge and not rechecked and turn < ctx.max_turns - 2:
                rechecked = True
                step["recheck"] = ans
                if ans["type"] == "unanswerable" and (hint := value_hint(ex, ans)):
                    step["value_hint"] = True
                    nudge += "\n" + hint
                messages.append({"role": "user", "content": nudge})
                continue
            # khác kiểu dự kiến (đoán đúng 99.8% trên train): hỏi lại một lần, chỉ cần một lượt trả lời
            if want and ans["type"] not in (want, "unanswerable") and not type_checked and turn < ctx.max_turns - 1:
                type_checked = True
                step["type_recheck"] = ans
                messages.append({"role": "user", "content": RECHECK_TYPE.format(t=want, got=ans["type"])})
                continue
            step["answer_logprob"] = getattr(llm, "last_logprob", None)
            if ctx.verify and not verified and turn < ctx.max_turns - 2:
                verified = True
                reason = verify(task, ctx, llm, trace, ans, seed)
                if reason:
                    step["verifier"] = reason
                    messages.append({"role": "user", "content": VERIFY_FEEDBACK.format(reason=reason)})
                    continue
            return ans, trace
    return None, trace


RECONCILE = """# Các lời giải trước đã ra đáp án KHÁC NHAU
{candidates}
Đừng tin lời giải nào. Tự kiểm tra trên dữ liệu đúng những chỗ các lời giải làm khác nhau (bảng, cột, điều kiện lọc,
kỳ, đơn hợp lệ, đơn vị, luật rulebook), in kết quả trung gian để đối chiếu, rồi trả lời đáp án đúng
(có thể trùng một lời giải ở trên hoặc khác hẳn)."""


def _last_code(trace: list[dict]) -> tuple[str, str]:
    for step in reversed(trace):
        code = extract_code(step.get("reply", "") or "")
        if code and 'out' in step:
            out = str(step.get('out') or '')
            if step.get('err'):
                out += '\nEXECUTION ERROR: ' + str(step['err'])[-500:]
            return code, out
    return "", ""


def reconcile_prompt(samples: list) -> str | None:
    """Tóm tắt các đáp án khác nhau kèm code cuối của chúng (ReFoRCE: ca bất đồng → khám phá có hướng)."""
    seen, blocks = set(), []
    for ans, trace in samples:
        if ans is None or answer_key(ans) in seen:
            continue
        seen.add(answer_key(ans))
        code, out = _last_code(trace)
        from datamate.agent_evidence import output_disagreement
        note = ('\nCảnh báo kiểm tra: literal cuối output khác đáp án mẫu; có thể là giá trị trung gian. '
                'Đối chiếu phép tính trước khi chọn.') if output_disagreement(ans, trace) is True else ''
        blocks.append(f"## Lời giải {len(blocks) + 1}: đáp án {json.dumps(ans, ensure_ascii=False)}\n"
                      f"```python\n{code[-1500:]}\n```\nKết quả: {out[-300:]}{note}")
        if len(blocks) == 3:
            break
    return RECONCILE.format(candidates="\n\n".join(blocks)) if len(blocks) >= 2 else None


def solve(task: Task, ctx: ShopContext, llm, n_samples: int = 2, max_samples: int = 3,
          min_agree: int = 2, base_seed: int = 1000, diverse: bool = False,
          reconcile: bool = False) -> tuple[dict, dict]:
    """Chạy n_samples mẫu song song; chưa đủ min_agree mẫu trùng đáp án thì chạy thêm từng mẫu tới max_samples;
    vẫn không đủ → abstain.
    diverse: mẫu thứ hai không có few-shot (Agentar-Scale-SQL: sinh lời giải đa dạng để các mẫu không cùng sai
    một kiểu chép từ ví dụ). reconcile: mẫu thêm được xem các đáp án đang bất đồng kèm code để tự kiểm chứng."""
    plain = replace(ctx, shots=None) if diverse and ctx.shots is not None else ctx

    def run(i, extra=None):
        return solve_once(task, plain if i % 2 == 1 else ctx, llm, seed=base_seed + i, extra=extra)

    # các mẫu đầu độc lập nhau: chạy chồng lên nhau để câu xong sớm hơn và vLLM gộp được lô lớn hơn
    with ThreadPoolExecutor(max_workers=max(1, n_samples)) as pool:
        samples = list(pool.map(run, range(n_samples)))
    reconciled = False
    for i in range(n_samples, max_samples):
        keys = Counter(answer_key(a) for a, _ in samples if a is not None)
        best = keys.most_common(1)[0][1] if keys else 0
        # đã đồng thuận, hoặc chạy hết số mẫu còn lại cũng không đủ
        if best >= min_agree:
            break
        extra = reconcile_prompt(samples) if reconcile else None
        reconciled = reconciled or extra is not None
        samples.append(run(i, extra))
    keys = Counter(answer_key(a) for a, _ in samples if a is not None)
    final = {"type": "abstain"}
    confidence = None
    if keys:
        key, cnt = keys.most_common(1)[0]
        if cnt >= min_agree:
            final = next(a for a, _ in samples if a is not None and answer_key(a) == key)
            lps = [t[-1].get("answer_logprob") for a, t in samples
                   if a is not None and answer_key(a) == key and t and t[-1].get("answer_logprob") is not None]
            confidence = max(lps) if lps else None
    info = {"n_samples": len(samples), "answers": [a for a, _ in samples], "traces": [t for _, t in samples],
            "reconciled": reconciled, "confidence": confidence}
    info['sample_outcomes'] = [
        'backend_error' if any('llm_error' in s for s in t) else
        'execution_error' if a is None and t and t[-1].get('err') else
        'turn_budget_exhausted' if a is None else 'candidate'
        for a,t in samples]
    from datamate.agent_evidence import committed_answer, last_execution
    certified = [a for a, t in samples if a is not None and committed_answer(t) is not None
                 and last_execution(t).get('execution_result_source') == 'deterministic'
                 and answer_key(a) == answer_key(committed_answer(t))]
    certified_keys = {answer_key(a) for a in certified}
    if len(certified_keys) == 1:
        final = certified[0]
        info['decision'] = 'original_question_deterministic_result'
    elif len(certified_keys) > 1:
        info['decision'] = 'conflicting_deterministic_results'
        return {'type': 'abstain'}, info
    else:
        info['decision'] = 'agreement' if final['type'] != 'abstain' else 'insufficient_agreement'
    if certified:
        return final, info
    # unanswerable dù các mẫu đó đã chạy được code (đã thấy dữ liệu): log train lượt GPU 2 đúng 1/15 câu → bỏ trống
    if final.get("type") == "unanswerable" and all(
            any("out" in s and not s.get("err") for s in t)
            for a, t in samples if a is not None and answer_key(a) == answer_key(final)):
        info["abstained_unanswerable_after_run"] = final
        info['decision'] = 'unanswerable_without_independent_confirmation'
        final = {"type": "abstain"}
    # đáp án 0 thường là lọc rỗng (sai kỳ / sai tên cột / sai giá trị lọc): log agent train+strange 10 lượt GPU,
    # số / % bằng 0 đúng 2 sai 41, list có số 0 đúng 0 sai 5 → bỏ trống
    elif is_zero_answer(final):
        info["abstained_zero"] = final
        info['decision'] = 'zero_without_independent_confirmation'
        final = {"type": "abstain"}
    elif implausible(task.question, final):
        info["abstained_implausible"] = final
        info['decision'] = 'implausible'
        final = {"type": "abstain"}
    elif fact_conflict(final, task.facts):
        info["abstained_fact_conflict"] = final
        info['decision'] = 'partial_fact_conflict'
        final = {"type": "abstain"}
    elif final.get("type") not in ("abstain", "unanswerable"):
        vetoed = veto(task.question, final, [(a, _last_code(t)[0]) for a, t in samples])
        if vetoed:  # vd câu lô / hạn dùng mà không mẫu nào lọc lô còn tồn (datamate/arbiter.py)
            info["abstained_guard"] = {"answer": final, "guards": sorted(vetoed)}
            info['decision'] = 'guard_veto'
            final = {"type": "abstain"}
    return final, info


_SIGNED = re.compile(r"\b(lai|loi|lo|tru|chenh|thay doi|tang|giam|bien dong|so voi|vs|ton)\b")


def _placeholder(x) -> bool:
    """Số 'giữ chỗ' model bịa khi không tính được: dãy chữ số liên tiếp (1234567, 890123, 9876543) hoặc một chữ số lặp
    (11111). Log Colab B9 v4: public [1234567, 890123], 12345678; strange2 [1234567, 2345678, 3456789], 12345."""
    d = str(int(abs(x)))
    if len(d) < 5:
        return False
    steps = {(int(b) - int(a)) % 10 for a, b in zip(d, d[1:])}
    return steps in ({1}, {9}, {0})


def implausible(question: str, final: dict) -> bool:
    """Đáp án số không thể đúng: số giữ chỗ, hoặc số âm khi câu hỏi không về lãi / chênh lệch / thay đổi / tồn (doanh
    thu, nợ còn, số lượng không thể âm; đáp án train + strange2: 4/6000 số âm, đều lãi ròng hoặc 'nhập trừ bán'; log
    Colab B9 v4: 'khách còn nợ' ra -222.727.216)."""
    from datamate.parsing import fold
    v = final.get("answer")
    vals = [x for x in (v if final.get("type") == "list" and isinstance(v, list) else
                        [v] if final.get("type") == "number" else [])
            if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if any(_placeholder(x) for x in vals):
        return True
    return any(x < 0 for x in vals) and not _SIGNED.search(fold(question))


def fact_conflict(final: dict, facts: list) -> bool:
    """Đáp án số / % / ngày lệch giá trị hàm chuẩn cùng kiểu mà máy đã gọi cho câu này → bỏ trống (kiểm độc lập song
    song bộ giải, mâu thuẫn thì không trả lời: 'Solving versus Verifying', 2026). Log agent cũ, câu tầng 1 không trả
    lời: train lệch 3 / agent sai 3; tên khách (mã vs tên) không so."""
    from datamate.scoring import score_item
    same = [(k, v) for _, k, v in facts if k == final.get("type") and k in ("number", "percent", "date")]
    return bool(same) and not any(score_item(final, {"type": k, "answer": v}) == 1 for k, v in same)


CALIB_TYPES = ["number", "percent", "list", "boolean", "unanswerable"]
# hệ số logistic (scripts/calibrate_abstain.py, log agent train + strange: 2173 câu / 60 shop). Out-of-fold theo shop:
# PA 1044 so với 961 của luật min_agree=2 (1005 khi thêm bỏ trống đáp án 0); 24 shop tốt hơn, 1 kém hơn
CALIB_W = [-0.552, 0.634, 1.667, 0.553, -0.87, 0.807, -0.517, -0.259, 0.534, -3.185,
           -0.346, -0.568, -0.333, 0.767, -2.376]


def calib_features(answers: list, traces: list, reconciled: bool) -> tuple[dict | None, list[float]]:
    """Các mẫu agent → (đáp án đa số, đặc trưng cho hiệu chỉnh P(đúng)). Không dùng logprob: chỉ có ở vài lượt log."""
    pairs = [(a, t) for a, t in zip(answers, traces) if a and a.get("type") not in (None, "abstain")]
    if not pairs:
        return None, []
    key, cnt = Counter(answer_key(a) for a, _ in pairs).most_common(1)[0]
    cand = next(a for a, _ in pairs if answer_key(a) == key)
    agree = [t for a, t in pairs if answer_key(a) == key]
    steps = [s for t in agree for s in t]
    first2 = len(answers) >= 2 and answers[0] and answers[1] and answer_key(answers[0]) == answer_key(answers[1])
    return cand, [1.0, cnt / max(len(answers), 1), float(cnt >= 2), float(bool(first2)), float(len(answers) >= 3),
                  float(reconciled), len(steps) / len(agree) / 4, sum(1 for s in steps if s.get("err")) / len(agree),
                  float(not any("out" in s for s in steps)), float(is_zero_answer(cand)),
                  *[float(cand["type"] == t) for t in CALIB_TYPES]]


def _zero(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0


def is_zero_answer(ans: dict) -> bool:
    if ans.get("type") in ("number", "percent"):
        return _zero(ans.get("answer"))
    if ans.get("type") == "list":
        return any(_zero(v) for v in ans.get("answer") or [])
    return False
