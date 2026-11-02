"""Thẻ schema: LLM đọc README + hồ sơ cột của một shop, gán vai trò chuẩn cho cột (mã đơn, ngày bán, tiền dòng…).

Semantic layer đọc nghĩa cột từ mục '## Cột trong `file`' của README — train có, public không (log public v3:
6/14 shop không dựng được bảng đơn chuẩn vì không nhận ra cột tiền / mã đơn, vd 'Thành tiền', 'Đơn số', 'HĐ').
Thẻ chỉ lấp vai trò còn thiếu; mọi kiểm chéo của OrderView (tiền dòng = SL × giá, fan-out, khớp tiền hàng của đơn,
phủ trạng thái) vẫn chạy, nên thẻ sai thì view bị loại và câu hỏi về lại agent. Một lượt gọi model mỗi shop.

    python -m datamate.schema_card prompts --cache outputs/cache --out cards_prompts.jsonl biz_101 dev_003 …
    python -m datamate.schema_card parse --responses responses.jsonl --cache outputs/cache   # ghi <biz>.card.json
"""
from __future__ import annotations

import argparse
import json
import hashlib
import pickle
import re
from pathlib import Path

import pandas as pd

from datamate.loader import CACHE_VERSION, Shop
from datamate.parsing import fold

# vai trò → (nghĩa chuẩn mà semantic layer tìm bằng _find, mô tả cho model). Thứ tự quan trọng: _find lấy cột đầu
# tiên có nghĩa bắt đầu bằng cụm cần tìm ('số lượng' khớp cả 'số lượng đã đổi…'), nên số lượng dòng đứng trước.
ROLES = {
    "order_id": ("mã đơn", "mã / số hoá đơn, đơn hàng, chứng từ bán; các dòng hàng của cùng một đơn có cùng giá trị"),
    "order_date": ("ngày bán", "ngày hoặc ngày giờ bán / lập hoá đơn / đặt đơn"),
    "order_status": ("trạng thái đơn", "trạng thái đơn (hoàn thành, huỷ, trả hàng, đang giao); cả cột ghi chú chỉ ghi "
                                         "khi đơn huỷ / trả"),
    "line_amount": ("thành tiền sau giảm", "tiền của MỘT dòng hàng sau giảm giá dòng (= số lượng × đơn giá − giảm giá "
                                            "dòng); không phải tổng cả đơn, chưa trừ voucher đơn, chưa VAT / phí giao"),
    "order_goods": ("tiền hàng của đơn", "tổng tiền hàng của CẢ đơn (cộng các dòng), lặp lại trên mọi dòng của đơn"),
    "qty": ("số lượng", "số lượng bán theo đơn vị của dòng"),
    "base_qty": ("số lượng đã đổi ra đơn vị cơ bản", "số lượng đã quy ra đơn vị cơ bản / đơn vị lẻ"),
    "return_qty": ("số lượng khách trả lại", "số lượng khách trả lại của dòng"),
    "unit": ("đơn vị tính", "đơn vị tính của dòng bán (hộp, thùng, kg…)"),
    "item_name": ("tên hàng", "tên mặt hàng / sản phẩm / món / dịch vụ"),
    "item_code": ("mã hàng", "mã mặt hàng / SKU"),
    "group": ("nhóm hàng", "nhóm / loại / danh mục hàng"),
}
PROFILE = False  # hồ sơ cột chi tiết trong prompt (bật sau khi đo A/B trên train)
_NUMERIC = {"line_amount", "order_goods", "qty", "base_qty", "return_qty"}

PROMPT = """Bạn gán vai trò cho các cột dữ liệu bán hàng của một cửa hàng Việt Nam. Chỉ dùng README và hồ sơ cột bên dưới.

Vai trò (mỗi bảng mỗi vai trò tối đa một cột; cột không thuộc vai trò nào thì bỏ qua, KHÔNG đoán):
{roles}

# README
{readme}

# Hồ sơ cột (tên bảng T['…'], kiểu, giá trị mẫu)
{brief}

Trả lời đúng một khối JSON, khoá là tên bảng và tên cột chép NGUYÊN như trong hồ sơ cột:
{{"<tên bảng>": {{"<tên cột>": "<vai trò>", ...}}, ...}}"""


def _profile(shop: Shop, table: str, col: str, budget: int) -> str:
    """Hồ sơ cột kiểu AILS-NTUA (SemEval-2025 Task 8): kiểu loader đọc được (cả 'int(qty)' = số kèm đơn vị như
    '2 ly'), số ô trống / không đọc được, phân bố giá trị phân loại có tần suất, mẫu chữ trong nháy."""
    from datamate.brief import describe_column  # tránh import vòng
    m = shop.meta.get(table, {})
    kind = m.get("types", {}).get(col, "text")
    text = describe_column(shop.tables[table][col], kind)
    bad = m.get("unparsed", {}).get(col)
    if bad:
        ex = ", ".join(f'"{v}"' for v in m.get("unparsed_examples", {}).get(col, [])[:3])
        text += f"; {bad} ô gốc không đọc được ({ex})"
    return "- " + text[:budget + len(col) + 20]


def build_prompt(shop: Shop, brief: str, readme_chars: int = 6000, brief_chars: int = 14000,
                 profile: bool = PROFILE) -> str:
    roles = "\n".join(f"- {r}: {desc}" for r, (_, desc) in ROLES.items())
    # Không cắt prefix của brief: bảng bán/danh mục ở cuối có thể mất hoàn toàn.
    # Giữ toàn bộ tên bảng/cột và kiểu; ngân sách còn lại chia đều cho ví dụ.
    blocks = []
    count = 0
    for table, df in shop.tables.items():
        cols = [c for c in df if c != "_sheet"]
        count += len(cols)
        blocks.append((table, df, cols))
    base_size = sum(len(f"## T[{t!r}] — {len(df)} dòng\n") +
                    sum(len(f"- {c!r} ({df[c].dtype})\n") for c in cols) for t, df, cols in blocks)
    sample_budget = min(120, max(0, (brief_chars - base_size) // max(count, 1) - 3))
    lines = []
    for table, df, cols in blocks:
        lines.append(f"## T[{table!r}] — {len(df)} dòng")
        for col in cols:
            values = df[col].dropna()
            detail = ""
            if sample_budget >= 20 and not values.empty:
                if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
                    detail = f"; min={values.min()}, max={values.max()}, trống={int(df[col].isna().sum())}"
                else:
                    samples = list(values.astype(str).drop_duplicates().head(3))
                    detail = f"; mẫu={samples!r}, trống={int(df[col].isna().sum())}"
                detail = detail[:sample_budget]
            if profile and table in shop.meta:
                lines.append(_profile(shop, table, col, max(sample_budget, 40)))
                continue
            lines.append(f"- {col!r} ({df[col].dtype}){detail}")
    compact = "\n".join(lines) if blocks else brief[:brief_chars]
    return PROMPT.format(roles=roles, readme=shop.readme[:readme_chars], brief=compact)


def parse_card(text: str, shop: Shop) -> dict[str, dict[str, str]]:
    """JSON của model → {bảng: {cột: nghĩa chuẩn}}; bỏ bảng / cột không có thật, vai trò lạ, kiểu dữ liệu không hợp."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # suy luận có thể chứa '{' lạc
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    card: dict[str, dict[str, str]] = {}
    for table, cols in raw.items() if isinstance(raw, dict) else []:
        # model hay chép nguyên cách hồ sơ cột ghi tên bảng: T['BanHang'] → BanHang
        m = re.fullmatch(r"\s*T\[\s*['\"](.+?)['\"]\s*\]\s*", str(table))
        df = shop.tables.get(m.group(1) if m else table)
        table = m.group(1) if m else table
        if df is None or not isinstance(cols, dict):
            continue
        used, picked = set(), {}
        for col, role in cols.items():
            if role not in ROLES or role in used or col not in df.columns or not _type_ok(df[col], role):
                continue
            used.add(role)
            picked[col] = role
        picked = _check_amounts(df, picked)
        if picked:  # theo thứ tự ROLES (xem ghi chú ở trên)
            order = list(ROLES)
            card[table] = {c: ROLES[r][0] for c, r in sorted(picked.items(), key=lambda kv: order.index(kv[1]))}
    return card


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _check_amounts(df: pd.DataFrame, picked: dict[str, str]) -> dict[str, str]:
    """Kiểm vai trò tiền bằng số liệu (tên cột dễ nhầm: 'Giá sản phẩm' / 'Tổng tiền'):
    - tiền dòng L: nếu có cột C khác với C ≈ số lượng × L ở ≥ 90% dòng thì L là đơn giá → tiền dòng là C;
    - tiền cả đơn: phải giống nhau trên mọi dòng của một đơn (≥ 95% đơn), không thì bỏ vai trò."""
    role_of = {r: c for c, r in picked.items()}
    out = dict(picked)
    L, Q = role_of.get("line_amount"), role_of.get("qty")
    if L and Q:
        prod = _num(df[Q]) * _num(df[L])
        ok = prod.notna() & (prod != 0) & _num(df[Q]).ne(1)
        for c in df.columns:
            if c in (L, Q) or not ok.any():
                continue
            v = _num(df[c])
            if v.notna().mean() < 0.9:
                continue
            if ((v[ok] - prod[ok]).abs() <= 0.01 * prod[ok].abs() + 1).mean() >= 0.9:
                out.pop(L, None)
                out = {k: r for k, r in out.items() if k != c}
                out[c] = "line_amount"
                break
    O = role_of.get("order_id")
    # cột kiểu số thứ tự dòng trong đơn (1, 2, 3…) không phải tiền / số lượng
    if O:
        for c, r in list(out.items()):
            ordinal_name = re.sub(r"[^a-z0-9]", "", fold(c)) in {"stt", "sothutu", "sodong", "dong", "no", "lineno", "rownumber"}
            # SL=1 ở mọi dòng của đơn một dòng không phải bằng chứng đó là STT.
            if r in _NUMERIC and ordinal_name:
                v = _num(df[c])
                rank = df.groupby(O).cumcount() + 1
                if v.notna().any() and (v == rank).mean() >= 0.8:
                    out.pop(c)
    rc = {r: c for c, r in out.items()}
    Q = rc.get("qty")
    # SL quy đổi / SL bán là hệ số dương, ít giá trị; SL trả lại ≤ SL bán và phần lớn bằng 0
    if rc.get("base_qty") and Q:
        ratio = (_num(df[rc["base_qty"]]) / _num(df[Q])).replace([float("inf"), -float("inf")], pd.NA).dropna()
        if ratio.empty or (ratio <= 0).any() or ratio.round(6).nunique() > max(20, 0.05 * len(ratio)):
            out.pop(rc["base_qty"])
    elif rc.get("base_qty"):
        out.pop(rc["base_qty"])
    if rc.get("return_qty"):
        ret = _num(df[rc["return_qty"]]).fillna(0)
        if not Q or (ret > _num(df[Q]).fillna(0)).mean() > 0.05 or (ret == 0).mean() < 0.5:
            out.pop(rc["return_qty"])
    G = {r: c for c, r in out.items()}.get("order_goods")
    if G and O:
        per = pd.DataFrame({"o": df[O], "g": _num(df[G])}).dropna().groupby("o")["g"].nunique()
        if len(per) and (per <= 1).mean() < 0.95:
            out.pop(G, None)
    return out


def _type_ok(s: pd.Series, role: str) -> bool:
    if not s.notna().any():
        return False
    if role == "order_date":
        return pd.api.types.is_datetime64_any_dtype(s)
    if role in _NUMERIC:
        return pd.to_numeric(s, errors="coerce").notna().mean() >= 0.9 * s.notna().mean()
    if role == "order_id":
        return not pd.api.types.is_float_dtype(s) and s.nunique() > 1
    return True


REFINE = """Kiểm bằng số liệu thấy bảng đơn hàng chưa dựng được từ cách gán trên: {why}.
Xem lại hồ sơ cột: cột mã đơn (các dòng cùng đơn cùng giá trị), ngày bán, trạng thái đơn (hoặc cột ghi chú chỉ ghi khi
đơn huỷ / trả), tiền MỘT dòng hàng (= số lượng × đơn giá, không phải đơn giá, không phải tiền cả đơn). Trả lại đúng
một khối JSON như trên; cột không chắc thì bỏ."""


def _view_fails(shop: Shop, card: dict) -> str | None:
    """Lý do bảng đơn chuẩn không dựng được với thẻ này; None khi dựng được (hoặc shop vốn không có bảng bán)."""
    from datamate.semantic import OrderView  # tránh import vòng
    old = getattr(shop, "card", None)
    shop.card = card
    try:
        OrderView.why = None
        ok = OrderView.from_shop(shop) is not None
        return None if ok else (OrderView.why or "không rõ")
    except Exception as e:
        return f"lỗi khi dựng: {e!r}"[:200]
    finally:
        shop.card = old


def load_card(cache_dir: Path, biz: str) -> dict | None:
    p = Path(cache_dir) / f"{biz}.card.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _card_fingerprint(shop: Shop, prompt: str, model: str = "") -> str:
    digest = hashlib.sha256(f"schema-card-v2:{CACHE_VERSION}:{model}:{prompt}".encode("utf-8"))
    digest.update(shop.readme.encode("utf-8"))  # cả phần README nằm ngoài ngân sách prompt
    for name, frame in sorted(shop.tables.items()):
        digest.update(name.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def make_card(shop: Shop, brief: str, llm, out_dir: Path, biz: str) -> dict:
    """Một lượt gọi + tối đa một lượt sửa; dùng cache khi input/model/phiên bản khớp.
    Lỗi model / JSON hỏng → thẻ rỗng, không lưu lỗi thành cache vĩnh viễn."""
    p = Path(out_dir) / f"{biz}.card.json"
    meta = p.with_suffix(".meta.json")
    prompt = build_prompt(shop, brief, profile=PROFILE)
    fingerprint = _card_fingerprint(shop, prompt, str(getattr(llm, "model", type(llm).__name__)))
    if p.exists() and meta.exists():
        try:
            saved = meta.read_text(encoding="utf-8")
            contents = p.read_text(encoding="utf-8")
            if json.loads(saved).get("fingerprint") == fingerprint and json.loads(saved).get("card_hash") == hashlib.sha256(contents.encode("utf-8")).hexdigest():
                return json.loads(contents)
        except (OSError, ValueError):
            pass  # cache hỏng/cũ: tạo lại, không dùng nhầm vai trò từ input khác
    msgs = [{"role": "user", "content": prompt}]
    try:
        first = llm.chat(msgs, seed=0)
    except RuntimeError:
        return {}  # không cache lỗi server thành thẻ rỗng vĩnh viễn
    card = parse_card(first, shop)
    why = _view_fails(shop, card)
    if why:  # tự sửa có phản hồi (Korini & Bizer 2025): báo lý do kiểm số liệu không qua, hỏi lại đúng một lượt
        try:
            again = parse_card(llm.chat(msgs + [{"role": "assistant", "content": first},
                                                {"role": "user", "content": REFINE.format(why=why)}], seed=1), shop)
        except Exception:
            again = {}
        if again and not _view_fails(shop, again):  # chỉ nhận thẻ mới khi bảng đơn dựng được, qua mọi kiểm chéo
            card = again
    if card:
        p.parent.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(card, ensure_ascii=False, indent=1)
        p.write_text(contents, encoding="utf-8")
        meta.write_text(json.dumps({"fingerprint": fingerprint,
                                   "card_hash": hashlib.sha256(contents.encode("utf-8")).hexdigest()}, indent=2), encoding="utf-8")
    return card


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("biz", nargs="+")
    q = sub.add_parser("parse")
    q.add_argument("--responses", required=True)
    q.add_argument("--cache", required=True)
    q.add_argument("--out", default=None, help="thư mục ghi <biz>.card.json (mặc định: --cache)")
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    if a.cmd == "prompts":
        with open(a.out, "w", encoding="utf-8") as fh:
            for b in a.biz:
                shop = pickle.loads((cache / f"{b}.v{CACHE_VERSION}.pkl").read_bytes())
                brief = (cache / f"{b}.v{CACHE_VERSION}.brief.md").read_text(encoding="utf-8")
                fh.write(json.dumps({"id": b, "prompt": build_prompt(shop, brief, profile=PROFILE)}, ensure_ascii=False) + "\n")
        return
    out = Path(a.out or a.cache)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(a.responses).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        shop = pickle.loads((cache / f"{r['id']}.v{CACHE_VERSION}.pkl").read_bytes())
        card = parse_card(r.get("response", ""), shop)
        (out / f"{r['id']}.card.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        print(r["id"], {t: len(c) for t, c in card.items()})


if __name__ == "__main__":
    main()
