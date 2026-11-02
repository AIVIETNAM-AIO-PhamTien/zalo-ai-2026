"""Xuất World ra thư mục shop giống POS thật, với kiểu xuất bốc ngẫu nhiên theo nhiều trục (giả lập cửa hàng chưa gặp):
tên cột (có dấu / không dấu / snake / camel / tiếng Anh), bố cục (bảng phẳng, có thể kèm dòng nối tiếp để trống cột
của đơn; hoặc tách đơn + dòng hàng), định dạng (csv / xlsx nhiều file / xlsx nhiều sheet / sqlite), cách ghi trạng
thái, định dạng ngày, README (đủ nghĩa cột / chỉ quy ước / trống). Trả về Style để bộ tính đáp án biết shop xuất
những gì (vd. có mã khách hay chỉ có tên khách)."""
from __future__ import annotations

import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from datamate.synth.simulate import World

# trường chuẩn → (nghĩa trong README, tên cột tiếng Việt, tên tiếng Anh)
FIELDS = {
    "order_id": ("mã đơn", ["Mã hóa đơn", "Số HĐ", "Mã đơn hàng", "Số chứng từ"], ["Order ID", "Invoice No"]),
    "ts": ("ngày bán", ["Ngày bán", "Thời gian", "Ngày tạo", "Ngày giờ bán"], ["Order Date", "Created At"]),
    "status": ("trạng thái đơn", ["Trạng thái", "Tình trạng đơn", "Kết quả"], ["Status", "Order Status"]),
    "cust_id": ("mã khách", ["Mã KH", "Mã khách hàng"], ["Customer ID", "Customer Code"]),
    "cust_name": ("tên khách (Khách lẻ nếu không lưu)", ["Tên khách", "Khách hàng", "Người mua"], ["Customer", "Customer Name"]),
    "channel": ("kênh bán", ["Kênh bán", "Nguồn đơn"], ["Channel", "Sales Channel"]),
    "payment": ("hình thức thanh toán", ["Hình thức TT", "Thanh toán", "PT thanh toán"], ["Payment Method", "Payment"]),
    "order_discount": ("giảm giá trên cả đơn", ["Giảm giá đơn", "Chiết khấu đơn"], ["Order Discount"]),
    "ship_fee": ("phí giao hàng", ["Phí ship", "Phí giao hàng"], ["Shipping Fee"]),
    "total_due": ("tổng tiền đơn (khách phải trả)", ["Khách cần trả", "Tổng thanh toán", "Phải thu"], ["Total Due", "Grand Total"]),
    "item_id": ("mã hàng", ["Mã hàng", "Mã SP", "Mã hàng hóa"], ["SKU", "Item Code"]),
    "item_name": ("tên hàng", ["Tên hàng", "Tên sản phẩm", "Sản phẩm"], ["Item Name", "Product"]),
    "unit": ("đơn vị tính", ["ĐVT", "Đơn vị tính", "Đơn vị"], ["Unit", "UOM"]),
    "qty": ("số lượng", ["Số lượng", "SL"], ["Qty", "Quantity"]),
    "unit_price": ("đơn giá", ["Đơn giá", "Giá bán"], ["Unit Price", "Price"]),
    "line_discount": ("giảm giá dòng", ["Giảm giá", "Chiết khấu"], ["Discount", "Line Discount"]),
    "amount": ("thành tiền sau giảm", ["Thành tiền", "Tiền hàng dòng", "Thành tiền sau CK"], ["Line Total", "Amount"]),
    "bqty": ("số lượng đã đổi ra đơn vị cơ bản", ["SL quy đổi", "Số lượng quy đổi"], ["Base Qty", "Qty In Base Unit"]),
    "ret_qty": ("số lượng khách trả lại", ["SL trả lại", "Số lượng trả"], ["Returned Qty"]),
    "group": ("nhóm hàng", ["Nhóm hàng", "Danh mục", "Loại hàng"], ["Category", "Item Group"]),
    "bunit": ("đơn vị cơ bản", ["ĐVT cơ bản", "Đơn vị cơ bản"], ["Base Unit"]),
    "big_unit": ("đơn vị lớn", ["Đơn vị quy đổi", "ĐVT lớn"], ["Pack Unit"]),
    "factor": ("hệ số", ["Hệ số", "Tỉ lệ quy đổi"], ["Factor", "Conversion Rate"]),
    "cost": ("giá vốn một đơn vị cơ bản", ["Giá vốn", "Giá nhập chuẩn"], ["Cost"]),
    "pay_date": ("ngày khách trả", ["Ngày thu", "Ngày trả"], ["Paid Date", "Payment Date"]),
    "pay_amount": ("số tiền khách trả lần đó", ["Số tiền thu", "Tiền trả"], ["Paid Amount"]),
    "receipt_id": ("mã phiếu nhập", ["Mã phiếu nhập", "Số phiếu", "Mã nhập hàng"], ["Receipt ID", "PO Number"]),
    "rdate": ("ngày nhập", ["Ngày nhập", "Ngày phiếu"], ["Receipt Date"]),
    "rstatus": ("trạng thái phiếu nhập", ["Trạng thái phiếu", "Tình trạng"], ["Receipt Status"]),
    "supplier": ("nhà cung cấp", ["Nhà cung cấp", "NCC"], ["Supplier"]),
    "r_unit": ("đơn vị nhập", ["ĐVT nhập", "Đơn vị nhập"], ["Purchase Unit"]),
    "r_qty": ("số lượng nhập", ["SL nhập", "Số lượng nhập"], ["Purchased Qty"]),
    "r_cost": ("giá nhập một đơn vị (theo đơn vị nhập của dòng)", ["Giá nhập", "Đơn giá nhập"], ["Unit Cost"]),
    "note": ("ghi chú", ["Ghi chú", "Diễn giải"], ["Note"]),
}
STATUS_VOCAB = [
    {"completed": "Hoàn thành", "cancelled": "Đã hủy", "returned": "Trả hàng", "delivering": "Đang giao"},
    {"completed": "Đã giao", "cancelled": "Hủy", "returned": "Khách trả hàng", "delivering": "Đang vận chuyển"},
    {"completed": "completed", "cancelled": "cancelled", "returned": "returned", "delivering": "shipping"},
    {"completed": "", "cancelled": "Huỷ đơn", "returned": "Hoàn trả", "delivering": "Chờ giao"},  # chỉ ghi khi có chuyện
]
RSTATUS_VOCAB = [{"done": "Đã nhập hàng", "cancelled": "Đã hủy"}, {"done": "Hoàn thành", "cancelled": "Hủy phiếu"},
                 {"done": "received", "cancelled": "cancelled"}]
WALKIN = ["Khách lẻ", "", "kl", "Khách vãng lai"]


def _ascii(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _style_name(vi: str, en: str, naming: str) -> str:
    if naming == "vi":
        return vi
    if naming == "vi_ascii":
        return _ascii(vi)
    if naming == "snake":
        return "_".join(_ascii(vi).lower().replace("/", " ").split())
    if naming == "camel":
        return "".join(w[:1].upper() + w[1:] for w in _ascii(vi).split())
    return en


@dataclass
class Style:
    naming: str
    layout: str  # flat | split
    continuation: bool
    fmt: str  # csv | xlsx_files | xlsx_sheets | sqlite
    status_vocab: int
    date_text: bool
    readme: str  # columns | notes | bare
    line_code_only: bool  # dòng hàng chỉ ghi mã hàng (tên tra ở danh mục)
    has_cust_code: bool
    has_bqty: bool
    has_ret_qty: bool
    walkin: str
    names: dict = field(default_factory=dict)  # trường → tên cột đã chọn

    @classmethod
    def sample(cls, rng: np.random.Generator, w: World) -> "Style":
        layout = "split" if w.arch.debt or rng.random() < 0.5 else "flat"
        st = cls(naming=str(rng.choice(["vi", "vi", "vi_ascii", "snake", "camel", "en"])), layout=layout,
                 continuation=layout == "flat" and rng.random() < 0.3,
                 fmt=str(rng.choice(["csv", "xlsx_files", "xlsx_sheets", "sqlite"])),
                 status_vocab=int(rng.integers(len(STATUS_VOCAB))), date_text=bool(rng.random() < 0.35),
                 readme=str(rng.choice(["columns", "columns", "notes", "bare"])),
                 line_code_only=layout == "split" and rng.random() < 0.4,
                 has_cust_code=w.arch.debt or rng.random() < 0.5, has_bqty=rng.random() < 0.4,
                 has_ret_qty=rng.random() < 0.4, walkin=str(rng.choice(WALKIN)))
        for f, (_, vi, en) in FIELDS.items():
            st.names[f] = _style_name(str(rng.choice(vi)), str(rng.choice(en)), st.naming)
        return st


def _fmt_ts(s: pd.Series, text: bool) -> pd.Series:
    return s.map(lambda t: t.strftime("%d/%m/%Y %H:%M:%S")) if text else pd.to_datetime(s)


def build_tables(w: World, st: Style) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Bảng xuất (tên bảng → DataFrame với tên cột theo style) + bảng → các trường chuẩn đã dùng (để viết README)."""
    vocab = STATUS_VOCAB[st.status_vocab]
    names = dict(zip(w.customers["cust_id"], w.customers["name"]))
    o = w.orders.copy()
    o["status"] = o["status"].map(vocab)
    o["ts"] = _fmt_ts(o["ts"], st.date_text)
    o["cust_name"] = o["cust_id"].map(lambda c: names[c] if isinstance(c, str) else st.walkin)
    goods = w.lines.groupby("order_id")["amount"].sum()
    o["total_due"] = o["order_id"].map(goods) - o["order_discount"] + o["ship_fee"]
    it = w.items.set_index("item_id")
    ln = w.lines.copy()
    ln["item_name"] = ln["item_id"].map(it["name"])
    ln["group"] = ln["item_id"].map(it["group"])

    order_f = ["order_id", "ts", "status"] + (["cust_id"] if st.has_cust_code else []) + ["cust_name"]
    order_f += (["channel"] if len(w.arch.channels) > 1 else []) + ["payment"]
    line_f = ["item_id"] if st.line_code_only else ["item_id", "item_name"]
    line_f += ["unit", "qty", "unit_price", "line_discount", "amount"] + (["bqty"] if st.has_bqty else [])
    line_f += ["ret_qty"] if st.has_ret_qty else []
    tables, used = {}, {}
    if st.layout == "flat":
        # khoản cấp đơn (giảm giá cả đơn, phí giao) lặp trên mọi dòng của đơn như file POS xuất phẳng
        flat_f = order_f + ["order_discount", "ship_fee"]
        df = ln.merge(o, on="order_id")
        df = df[flat_f + line_f + ["group"]]
        if st.continuation:  # dòng sau của cùng đơn để trống các cột của đơn (rulebook: ô trống lấy theo dòng đầu)
            later = df["order_id"].duplicated()
            df.loc[later, [c for c in flat_f if c != "order_id"]] = None
        tables["ban_hang"], used["ban_hang"] = df, flat_f + line_f + ["group"]
    else:
        of = order_f + ["order_discount", "ship_fee", "total_due"]
        tables["don_hang"], used["don_hang"] = o[of], of
        tables["chi_tiet_don"], used["chi_tiet_don"] = ln[["order_id"] + line_f], ["order_id"] + line_f
    cat = w.items.rename(columns={"name": "item_name"})
    cf = ["item_id", "item_name", "group", "bunit", "cost"]
    tables["hang_hoa"], used["hang_hoa"] = cat[cf], cf
    conv = [(r.item_name, b, f, r.bunit) for r in cat.itertuples() for b, f in r.conv.items()]
    if conv:
        tables["quy_doi"] = pd.DataFrame(conv, columns=["item_name", "big_unit", "factor", "bunit"])
        used["quy_doi"] = ["item_name", "big_unit", "factor", "bunit"]
    if st.has_cust_code:
        tables["khach_hang"] = w.customers.rename(columns={"name": "cust_name"})
        used["khach_hang"] = ["cust_id", "cust_name"]
    if len(w.payments):
        tables["thu_no"] = w.payments.rename(columns={"date": "pay_date", "amount": "pay_amount"})
        tables["thu_no"]["pay_date"] = pd.to_datetime(tables["thu_no"]["pay_date"])
        used["thu_no"] = ["order_id", "pay_date", "pay_amount"]
    if len(w.receipts):
        rv = RSTATUS_VOCAB[st.status_vocab % len(RSTATUS_VOCAB)]
        r = w.receipts.assign(rstatus=w.receipts["status"].map(rv), rdate=pd.to_datetime(w.receipts["date"]),
                              note=w.receipts["opening"].map(lambda x: "nhập đầu kỳ" if x else None))
        tables["phieu_nhap"] = r
        used["phieu_nhap"] = ["receipt_id", "rdate", "rstatus", "supplier", "note"]
        rl = w.receipt_lines.rename(columns={"unit": "r_unit", "qty": "r_qty", "unit_cost": "r_cost"})
        tables["chi_tiet_nhap"], used["chi_tiet_nhap"] = rl, ["receipt_id", "item_id", "r_unit", "r_qty", "r_cost"]
    for t in tables:
        tables[t] = tables[t][used[t]].rename(columns=st.names)
    return tables, used


TABLE_TITLES = {"ban_hang": "BanHang", "don_hang": "DonHang", "chi_tiet_don": "ChiTietDonHang", "hang_hoa": "DanhMucHang",
                "quy_doi": "QuyDoiDonVi", "khach_hang": "KhachHang", "thu_no": "ThuNo", "phieu_nhap": "PhieuNhap",
                "chi_tiet_nhap": "ChiTietNhap"}


def write_shop(w: World, st: Style, out: Path) -> None:
    tables, used = build_tables(w, st)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}  # bảng → "file" hoặc "file:sheet" (để viết README)
    if st.fmt == "sqlite":
        with sqlite3.connect(out / "pos.sqlite") as con:
            for t, df in tables.items():
                df2 = df.copy()
                for c in df2.columns:
                    if pd.api.types.is_datetime64_any_dtype(df2[c]):
                        df2[c] = df2[c].dt.strftime("%Y-%m-%d %H:%M:%S")
                df2.to_sql(t, con, index=False)
                files[t] = f"pos.sqlite:{t}"
    elif st.fmt == "xlsx_sheets":
        with pd.ExcelWriter(out / "DuLieuXuat.xlsx") as xw:
            for t, df in tables.items():
                df.to_excel(xw, sheet_name=TABLE_TITLES[t], index=False)
                files[t] = f"DuLieuXuat.xlsx:{TABLE_TITLES[t]}"
    else:
        for t, df in tables.items():
            fn = TABLE_TITLES[t] + (".csv" if st.fmt == "csv" else ".xlsx")
            if st.fmt == "csv":
                df.to_csv(out / fn, index=False, encoding="utf-8-sig")
            else:
                df.to_excel(out / fn, index=False)
            files[t] = fn
    (out / "README.md").write_text(readme(w, st, tables, used, files), encoding="utf-8")


def readme(w: World, st: Style, tables, used, files) -> str:
    fnames = sorted({f.split(":")[0] for f in files.values()})
    kind = {".csv": "csv", ".xlsx": "xlsx", ".sqlite": "sqlite"}
    out = [f"# Dữ liệu {w.biz_id}", "", f"{w.arch.label} File máy POS xuất ra.", "", "## File"]
    out += [f"- `{f}` ({kind[Path(f).suffix]})" for f in fnames]
    if st.fmt == "sqlite":
        out += ["", "## Bảng", ", ".join(f"`{t}`" for t in tables)]
    if st.readme == "columns":
        for t, fs in used.items():
            fn, _, sheet = files[t].partition(":")
            head = f"## Cột trong `{fn}`" + (f", sheet `{sheet}`" if sheet and st.fmt != "sqlite" else "")
            if st.fmt == "sqlite":
                head = f"## Cột trong bảng `{t}`"
            out += ["", head] + [f"- **{st.names[f]}**: {FIELDS[f][0]}" for f in fs]
    out += ["", "## Quy ước", f"- Dữ liệu có đến hết {w.end.strftime('%d/%m/%Y')}."]
    if st.readme != "bare":
        n = st.names
        if st.layout == "flat":
            out.append("- Mỗi dòng là một dòng hàng trong đơn." + (
                " Dòng sau của cùng một đơn để trống các cột của đơn." if st.continuation else
                f" **{n['order_discount']}**, **{n['ship_fee']}** là của cả đơn, ghi lặp trên mọi dòng của đơn."))
        out.append(f"- **Dòng hàng**: **{n['amount']}** là thành tiền của dòng (đã trừ giảm giá dòng).")
        if st.status_vocab == 3:
            out.append(f"- **{n['status']}** để trống là đơn bình thường (đã hoàn thành).")
        if "thu_no" in tables:
            out.append(f"- **Công nợ**: mỗi lần khách trả tiền cho một đơn là một dòng, **{n['pay_amount']}** là số tiền "
                       f"trả lần đó; **{n['total_due']}** là số khách phải trả cho cả đơn.")
        if "phieu_nhap" in tables:
            out.append(f"- **Nhập hàng**: **{n['r_cost']}** là giá nhập một đơn vị, theo đơn vị nhập của dòng.")
        if "quy_doi" in tables:
            out.append("- **Quy đổi đơn vị**: số lượng ghi theo đơn vị của dòng, hệ số quy đổi ở bảng quy đổi.")
    return "\n".join(out) + "\n"
