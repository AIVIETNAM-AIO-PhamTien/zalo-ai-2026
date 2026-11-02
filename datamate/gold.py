"""Tầng gold: bảng sự kiện đã áp sẵn luật rulebook, cùng khuôn cho mọi shop, để agent chỉ còn chọn công thức.

Dựng trên OrderView (semantic.py) — nơi các luật đã được cài — nhưng xuất thành bảng phẳng có cờ tường minh:
- fact_order: order_id, date, ts, shift, status (gốc), cls (§8: completed / cancelled / returned / delivering),
  counts_revenue (§1: không huỷ, không trả; đang giao tính theo policy), revenue (tổng thành tiền dòng, §1),
  customer_key (§3: mã khách hoặc tên chỉ khác cách viết; khách lẻ = None).
- fact_line: order_id, item_key, name, bqty (số lượng quy về đơn vị cơ bản, §3), bunit, base_price.
gold_report kiểm các bất biến rulebook trên gold; vi phạm = dữ liệu chưa đủ sạch để công thức cho đáp án chắc chắn.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import threading
from pathlib import Path

import pandas as pd

from datamate.semantic import OrderView

SHIFTS = {"sang": (6, 13), "chieu": (14, 21)}  # rulebook §2: ca sáng 06:00–13:59, ca chiều 14:00–21:59


def _shift(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    h = pd.Timestamp(ts).hour
    return next((k for k, (a, b) in SHIFTS.items() if a <= h <= b), None)


def build_gold(shop, view: OrderView | None = None) -> dict | None:
    view = view if view is not None else OrderView.from_shop(shop)
    if view is None:
        return None
    o = view.orders.copy()
    valid = set(view._valid(o)["order_id"])
    money = view.lines.groupby("order_id")["amount"].sum(min_count=1)
    o["counts_revenue"] = o["order_id"].isin(valid)
    o["revenue"] = o["order_id"].map(money)
    o["shift"] = o["ts"].map(_shift) if "ts" in o else None
    if view.customers is not None:
        o["customer_key"] = o["order_id"].map(view.customers.drop_duplicates("order_id")
                                               .set_index("order_id")["customer"])
    else:
        o["customer_key"] = None
    lines = None
    if view.items is not None:
        lines = view.items.rename(columns={"item": "item_key"}).reindex(
            columns=["order_id", "item_key", "name", "bqty", "bunit", "base_price"])  # shop thiếu cột → NaN
    from datamate.semantic import _conversions, shop_columns
    # mặt hàng CÓ hệ số quy đổi trong dữ liệu; mặt hàng chỉ biết đơn vị cơ bản mà bán theo đơn vị lớn không có hệ số
    # ('bó' ống PVC) là thiếu thông tin
    convertible = {k for k, (_, facs) in _conversions(shop, shop_columns(shop)).items() if facs}
    return {"fact_order": o, "fact_line": lines, "convertible": convertible}


# bảng tiền phụ của OrderView cùng khuôn (order_id, amount, group, gname) → tên bảng trong G
GOLD_VERSION = 2  # tăng khi đổi cách dựng G: file lưu cũ không dùng lại
_MONEY_FRAMES = {"tien_mat_hang": "item_lines", "tien_nhom": "groups", "tien_ke_don": "rx", "tien_hoat_chat": "ingredients"}
_ORDER_COLS = ["order_id", "date", "ts", "cls", "counts_revenue", "customer_key"]


def _ledgers(built: dict, out: dict, rep: dict, as_of) -> None:
    """Sổ ngoài đơn / dòng (B1), xuất phẳng như các *View đã đọc (đã nhận diện cột, chưa cộng): công nợ, nhập hàng,
    lô. Bản ghi sau hôm nay bỏ. Không đọc được → ghi lý do vào báo cáo."""
    cut = pd.Timestamp(as_of) if as_of is not None else None

    def until(df, col="date"):
        return df if cut is None or col not in df.columns else df[~(df[col] > cut)]
    if built["debt"] is not None:
        out["cong_no_don"] = until(built["debt"][0]).reset_index(drop=True)
        out["cong_no_tra"] = until(built["debt"][1]).reset_index(drop=True)
    else:
        rep["missing"]["cong_no_don"] = rep["missing"]["cong_no_tra"] = "shop không có sổ nợ đọc được"
    if built["imports"] is not None:
        out["nhap_hang"] = until(built["imports"]).reset_index(drop=True)
    else:
        rep["missing"]["nhap_hang"] = "shop không có phiếu nhập đọc được"
    if built["lots"] is not None:
        out["lo_hang"] = until(built["lots"][0]).reset_index(drop=True)
        out["lo_xuat"] = built["lots"][1].reset_index(drop=True)
    else:
        rep["missing"]["lo_hang"] = "shop không có sổ lô / hạn dùng đọc được"


def _build(shop, rulebook) -> dict:
    """Phần nặng của G, không phụ thuộc ngày hỏi: view đơn + gold + sổ (vài giây đến vài chục giây một shop).
    Chỉ giữ các bảng phẳng để lưu ra đĩa được."""
    try:
        guard = None
        if rulebook:
            from datamate.rule_guard import Guard
            guard = Guard(rulebook, policy_aware=True)
            if guard.view:
                return {"why": "rulebook đổi luật đơn / tiền mà policy chưa diễn đạt được: " + ", ".join(guard.changed)}
            shop.rule_policy = guard.policy  # bỏ trùng §8 / gộp tên khách §3 theo luật của split, trước khi dựng view
        view = OrderView.from_shop(shop)
        if view is None:
            return {"why": "không dựng được bảng đơn chuẩn: " + str(OrderView.why or "không rõ")}
        if guard is not None:
            view.apply_policy(guard.policy)
        g = build_gold(shop, view)
    except Exception as e:  # noqa: BLE001 — báo lý do thay vì im lặng
        return {"why": f"lỗi dựng gold: {type(e).__name__}: {e}"[:300]}
    try:
        from datamate.lots import LotView
        lv = LotView.from_shop(shop)
    except Exception:
        lv = None
    d, im = getattr(view, "debt", None), getattr(view, "imports", None)
    return {"order": g["fact_order"], "line": g["fact_line"], "attrs": getattr(view, "attrs", None),
            "money": {name: getattr(view, attr, None) for name, attr in _MONEY_FRAMES.items()},
            "debt": (d.orders, d.pays) if d is not None else None,
            "imports": im.lines if im is not None else None,
            "lots": (lv.lots, lv.moves) if lv is not None else None}


_BUILT: dict = {}
_BUILT_LOCK = threading.Lock()
_KEY_LOCKS: dict = {}


def _built(shop, rulebook, store=None) -> dict:
    """_build nhớ trong tiến trình theo (shop, rulebook), và trên đĩa ở `store` (file cạnh cache shop): executor dựng
    namespace trong tiến trình con mới cho mỗi phiên chạy code, trước đây mỗi phiên dựng lại G (4–26 s một shop)."""
    key = (id(shop), hash(rulebook or ""))
    with _BUILT_LOCK:
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    with lock:
        hit = _BUILT.get(key)
        if hit is not None and hit[0] is shop:
            return hit[1]
        res = None
        if store is not None and Path(store).is_file():
            try:
                res = pickle.loads(Path(store).read_bytes())
            except Exception:  # file hỏng / đang ghi dở → dựng lại
                res = None
        if res is None:
            res = _build(shop, rulebook)
            if store is not None:
                tmp = Path(f"{store}.{os.getpid()}.{threading.get_ident()}.tmp")
                try:
                    tmp.write_bytes(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
                    os.replace(tmp, store)  # nhiều tiến trình ghi cùng lúc: bản nào cũng như nhau
                except OSError:
                    tmp.unlink(missing_ok=True)
        with _BUILT_LOCK:
            while len(_BUILT) >= 64:
                _BUILT.pop(next(iter(_BUILT)))
            _BUILT[key] = (shop, res)
        return res


def gold_store_path(cache_path, rulebook, card=None) -> Path:
    """File lưu phần nặng của G cạnh cache shop; đổi khi rulebook, thẻ cột hoặc bản code gold / loader đổi."""
    import json
    from datamate.loader import CACHE_VERSION
    key = f"{GOLD_VERSION}|{CACHE_VERSION}|{rulebook or ''}|{json.dumps(card, sort_keys=True, default=str)}"
    tag = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return Path(f"{cache_path}.gold_{tag}.pkl")


def agent_tables(shop, as_of=None, rulebook: str | None = None, store=None) -> dict:
    """G cho namespace agent, luật rulebook áp sẵn:
    - 'don': mỗi đơn một dòng (fact_order) + hinh_thuc_tt / kenh khi shop có;
    - 'dong': dòng hàng, bqty đã quy đổi (fact_line) + cột của đơn;
    - 'tien_mat_hang' / 'tien_nhom' / 'tien_ke_don' / 'tien_hoat_chat': tiền dòng theo mặt hàng / nhóm / kê đơn
      ('rx' / 'otc') / hoạt chất — cột order_id, key (khoá so khớp), name (tên gốc), amount + cột của đơn;
    - '_bao_cao': bảng nào có, bảng nào không và vì sao, số ô thiếu (B5: không im lặng).
    as_of: bỏ đơn đặt sau hôm nay (rulebook: chưa xảy ra; trạng thái huỷ / trả đã nằm sẵn trong cls của đơn).
    rulebook: Policy diễn đạt được thì áp lên view; luật đơn / tiền đổi mà Policy không diễn đạt được → không có bảng
    (như SEM chặn), không để agent tin cờ theo luật cũ.
    store: file lưu phần nặng (gold_store_path) để tiến trình con không dựng lại."""
    rep = {"tables": [], "missing": {}}
    out = {"_bao_cao": rep}
    built = _built(shop, rulebook, store)
    if "why" in built:
        rep["why"] = built["why"]
        return out
    o = built["order"].drop_duplicates("order_id")
    if as_of is not None:
        o = o[~(o["date"] > pd.Timestamp(as_of))]  # đơn không có ngày giữ lại (không xếp kỳ được, không coi là tương lai)
    attrs = built["attrs"]
    if attrs is not None:
        keep = {"pay": "hinh_thuc_tt", "channel": "kenh"}
        a = attrs[["order_id", *[c for c in keep if c in attrs.columns]]].drop_duplicates("order_id").rename(columns=keep)
        o = o.merge(a, on="order_id", how="left")
    out["don"] = o.reset_index(drop=True)
    ids, flags = set(o["order_id"]), o[_ORDER_COLS]
    if built["line"] is not None:
        out["dong"] = built["line"][built["line"]["order_id"].isin(ids)].merge(flags, on="order_id", how="left")
    for name, f in built["money"].items():
        if f is None or not {"order_id", "amount", "group"} <= set(f.columns):
            rep["missing"][name] = "shop không có cột / bảng tương ứng"
            continue
        f = f[f["order_id"].isin(ids)].rename(columns={"group": "key", "gname": "name"})
        out[name] = f.reindex(columns=["order_id", "key", "name", "amount"]).merge(flags, on="order_id", how="left")
    _ledgers(built, out, rep, as_of)
    for name, df in out.items():
        if name == "_bao_cao":
            continue
        out[name] = df = df.copy()  # code agent sửa bảng không đụng bản đã nhớ
        if "counts_revenue" in df.columns:
            # đơn không tính doanh thu (huỷ / trả... theo rulebook split): tiền và số lượng bán = 0, bản gốc ở cột _goc.
            # Log Colab B9 v4: mẫu dùng G nhưng quên lọc counts_revenue đúng 51 / sai 18, có lọc đúng 196 / sai 14.
            off = df["counts_revenue"].eq(False)
            for c in ("revenue", "bqty", "amount"):
                if c in df.columns:
                    df[c + "_goc"] = df[c]
                    df.loc[off & df[c].notna(), c] = 0
        rep["tables"].append(name)
        for c in ("revenue", "bqty", "amount", "date"):
            if c in df.columns and df[c].isna().any():
                rep.setdefault("nan_cells", {})[f"{name}.{c}"] = int(df[c].isna().sum())
    return out


# B2 — từ điển dữ liệu gold cho prompt agent: nghĩa, đơn vị, luật nguồn (§ rulebook train), dùng cho khái niệm nào
GOLD_DICT = {
    "don": ("mỗi đơn một dòng, chỉ đơn có ngày ≤ hôm nay", {
        "order_id": "mã đơn (đếm đơn = nunique)",
        "date": "ngày đặt (kỳ của đơn theo ngày đặt, §1)", "ts": "ngày giờ đặt (giờ cao điểm, ca)",
        "status": "trạng thái gốc như trong dữ liệu",
        "cls": "completed / cancelled / returned / delivering (§8; ô trống = completed)",
        "counts_revenue": "đơn tính doanh thu / số đơn / số lượng bán theo rulebook của split (§1); đếm đơn vẫn phải lọc cột này",
        "revenue": "doanh thu đơn, đồng (§1; theo policy: giảm giá cả đơn, phí giao nếu luật tính); NaN = thiếu tiền; đã = 0 ở đơn counts_revenue=False (giá trị gốc: revenue_goc)",
        "shift": "ca theo khung giờ rulebook ('sang' / 'chieu'), None ngoài ca hoặc không có giờ",
        "customer_key": "khách (mã khách, hoặc tên chỉ khác cách viết, §3); None = khách lẻ, không đếm là khách",
        "hinh_thuc_tt": "hình thức thanh toán ghi trên đơn: cod / cash / transfer / ewallet / debt (chỉ khi shop có)",
        "kenh": "kênh bán: store / phone / zalo / market (chỉ khi shop có)"}),
    "dong": ("dòng hàng, dòng lặp §8 đã bỏ, dòng tiếp nối đã điền", {
        "item_key": "khoá mặt hàng (một mặt hàng dù ghi khác hoa / thường / dấu, §3)",
        "name": "tên như trong dữ liệu (đáp án tên chép cột này)",
        "bqty": "số lượng ĐÃ quy về đơn vị cơ bản (§3 bán chạy / bán được bao nhiêu); NaN = thiếu hệ số, không đoán; = 0 ở đơn counts_revenue=False (gốc: bqty_goc)",
        "bunit": "đơn vị cơ bản", "base_price": "giá một đơn vị cơ bản trên dòng",
        "date / cls / counts_revenue / customer_key": "lấy từ đơn"}),
    "tien_mat_hang / tien_nhom / tien_ke_don / tien_hoat_chat": (
        "tiền dòng theo mặt hàng / nhóm hàng / kê đơn ('rx' / 'otc') / hoạt chất (doanh thu nhóm chỉ cộng dòng của "
        "nhóm, §1)", {
            "key": "khoá so khớp (so với fold của tên trong câu)", "name": "tên gốc", "amount": "thành tiền dòng, đồng; = 0 ở đơn counts_revenue=False (gốc: amount_goc)",
            "date / cls / counts_revenue / customer_key": "lấy từ đơn"}),
    "cong_no_don": ("đơn trong sổ công nợ (§ công nợ)", {
        "order_id": "mã đơn", "date": "ngày", "cust": "khoá khách (None = khách lẻ)", "name": "tên khách gốc",
        "total": "tổng khách phải trả của đơn", "debt": "đơn ghi nợ", "counted": "đơn tính vào công nợ theo rulebook"}),
    "cong_no_tra": ("các lần khách trả nợ", {"order_id": "mã đơn", "date": "ngày trả", "amount": "số tiền trả"}),
    "nhap_hang": ("dòng phiếu nhập hàng", {
        "receipt": "mã phiếu", "date": "ngày nhập", "item / name": "khoá / tên mặt hàng", "unit": "đơn vị nhập (khoá)",
        "qty": "số lượng nhập", "price": "giá nhập một đơn vị nhập", "cancelled": "phiếu huỷ (bỏ khi tính)"}),
    "lo_hang / lo_xuat": ("lô hàng (lot, item, name, expiry, qty_in, date, cancelled) và xuất theo lô (lot, order_id, "
                          "out, ret): tồn lô = qty_in − xuất + trả, chỉ lô còn tồn > 0 mới tính hạn dùng", {}),
    "_bao_cao": ("bảng nào có, bảng nào không và vì sao, số ô NaN", {}),
}


def gold_dictionary() -> str:
    """Từ điển dữ liệu gold dạng văn bản cho prompt (không chứa ngoặc nhọn: prompt đi qua str.format)."""
    out = []
    for t, (desc, cols) in GOLD_DICT.items():
        out.append(f"  - G['{t}']: {desc}." + ("" if not cols else " Cột: " + "; ".join(f"{c} = {d}" for c, d in cols.items()) + "."))
    return "\n".join(out)


def gold_brief(G: dict, sample_rows: int = 2) -> str:
    """G thật của shop cho tin nhắn đầu: bảng, số dòng, cột, vài dòng mẫu, bảng thiếu và lý do. Prompt chỉ có từ điển
    chung thì agent (log Colab B9: 63/67 câu) đọc bảng thô T đã liệt kê chi tiết, bỏ qua G."""
    rep = G.get("_bao_cao", {})
    if rep.get("why"):
        return f"G của cửa hàng này rỗng: {rep['why']}. Tự tính trên T theo rulebook."
    out = []
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 30):
        for name, df in G.items():
            if name == "_bao_cao":
                continue
            out.append(f"G['{name}']: {len(df)} dòng; cột: {', '.join(map(str, df.columns))}")
            if sample_rows and len(df):
                out.append(df.head(sample_rows).to_string(index=False))
    for name, why in rep.get("missing", {}).items():
        out.append(f"G không có '{name}': {why}")
    return "\n".join(out).replace("{", "(").replace("}", ")")


def gold_report(gold: dict | None) -> dict:
    """Bất biến rulebook trên gold. Mỗi mục là số vi phạm (0 = sạch)."""
    if gold is None:
        return {"available": False}
    o, ln = gold["fact_order"], gold["fact_line"]
    multi_status = o.groupby("order_id")["cls"].nunique()
    rep = {
        "available": True, "orders": int(o["order_id"].nunique()),
        # §8: mỗi đơn có đúng một trạng thái
        "orders_multi_status": int((multi_status > 1).sum()),
        "orders_no_status": int(o["cls"].isna().sum()),
        # §1: đơn thuộc kỳ theo ngày đặt — đơn không có ngày thì không xếp kỳ được
        "orders_no_date": int(o["date"].isna().sum()),
        # §1: doanh thu đơn = tổng thành tiền dòng — đơn tính doanh thu mà không có tiền dòng
        "revenue_orders_no_amount": int((o["counts_revenue"] & o["revenue"].isna()).sum()),
    }
    if ln is not None:
        ids = set(o["order_id"])
        rep.update({
            "lines": len(ln),
            # toàn vẹn tham chiếu: dòng hàng phải thuộc một đơn đã biết
            "lines_orphan": int((~ln["order_id"].isin(ids)).sum()),
            # §3: bán chạy / bán được bao nhiêu theo số lượng quy về đơn vị cơ bản
            "lines_no_base_qty": int(ln["bqty"].isna().sum()),
            # mặt hàng không có quy đổi nào trong dữ liệu (không có ở danh mục / bảng quy đổi): thiếu thông tin, không
            # phải dữ liệu bẩn — công thức phải bỏ trống, không đoán hệ số
            "lines_no_conversion_in_data": int((ln["bqty"].isna() & ln["item_key"].notna() & ~ln["item_key"].isin(gold.get("convertible", set()))).sum()),
            "lines_no_item": int(ln["item_key"].isna().sum()),
        })
    gaps = rep.get("lines_no_conversion_in_data", 0)
    rep["data_gaps"] = gaps
    rep["violations"] = sum(v for k, v in rep.items()
                            if k not in ("available", "orders", "lines", "lines_no_conversion_in_data", "data_gaps")
                            and isinstance(v, int)) - gaps
    return rep
