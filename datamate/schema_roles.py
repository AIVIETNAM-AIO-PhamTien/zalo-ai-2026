"""Conservative lexical business roles, independent of shops and questions.

Aliases propose meanings, not joins. Consumers still validate keys, coverage,
amount arithmetic and cardinality. Ambiguous aliases for a role are rejected.
"""
from __future__ import annotations

import re
from datamate.parsing import fold


def column_key(value):
    return re.sub(r'[^a-z0-9]', '', fold(str(value)))


def bilingual_key(value):
    """Match documented identifiers after spelling/translation changes only.

    Preserve token order: order_id and item_id must never collapse together.
    This does not use table position, values, question words or fuzzy distance.
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(value))
    text = re.sub(r'[^a-z0-9]', ' ', fold(text))
    text = ' '.join(text.split())
    phrases = [('so luong da doi', 'base qty'), ('so luong', 'qty'), ('thanh tien', 'amount'),
               ('trang thai', 'status'), ('tinh trang', 'state'), ('ghi chu', 'note'),
               ('don vi', 'unit'), ('nhan vien', 'staff'), ('thanh toan', 'payment'),
               ('khach hang', 'customer'), ('khach', 'customer'), ('thoi gian', 'time')]
    for before, after in phrases:
        text = re.sub(r'\b' + before + r'\b', after, text)
    words = {'ma': 'id', 'ngay': 'date', 'don': 'order', 'hoa': '', 'hang': 'item',
             'ten': 'name', 'gia': 'price', 'nhom': 'group', 'tong': 'total',
             'tien': 'money', 'kenh': 'channel', 'so': 'no'}
    return ''.join(words.get(w, w) for w in text.split())


def align_declared_columns(frame, declared):
    """Rebind stale README column names only on a unique bilingual equivalent."""
    available = {}
    for column in frame:
        available.setdefault(bilingual_key(column), []).append(column)
    result = {c: meaning for c, meaning in declared.items() if c in frame}
    proposals = {}
    for column, meaning in declared.items():
        if column in frame:
            continue
        candidates = available.get(bilingual_key(column), [])
        if len(candidates) == 1 and candidates[0] not in result:
            proposals.setdefault(candidates[0], []).append(meaning)
    result.update({c: meanings[0] for c, meanings in proposals.items() if len(meanings) == 1})
    return result


ALIASES = {
    'mã hàng': 'mã hàng|mã hàng hóa|mã mặt hàng|mã sản phẩm|mã sp|mã hh|mã h.hóa|mã món|sku|item code|item id|product id|product code|id sp',
    'tên hàng': 'tên hàng|tên hàng hóa|tên mặt hàng|tên sản phẩm|tên sp|mặt hàng|sản phẩm|hàng hóa|chi tiết hàng|item name|product name|product|item',
    'nhóm hàng': ('nhóm hàng|nhóm sản phẩm|nhóm sp|loại hàng|loại sản phẩm|loại sp|danh mục|danh mục hàng|'
                   'phân loại hàng|phân loại sản phẩm|category|item group|product category'),
    'đơn vị tính': 'đvt|đ.vị|đv|đơn vị|đơn vị tính|unit|uom',
    'đơn vị cơ bản': 'đơn vị cơ bản|đvt cơ bản|đơn vị cơ sở|đơn vị lẻ|base unit|basic unit',
    'đơn vị lớn': 'đơn vị lớn|đvt lớn|đơn vị quy đổi|pack unit',
    'hệ số': 'hệ số|hệ số quy đổi|tỉ lệ quy đổi|tỷ lệ quy đổi|conversion rate|conversion factor|factor',
    'hoạt chất': 'hoạt chất|thành phần chính|active ingredient|ingredient',
    'thuốc kê đơn': 'kê đơn|thuốc kê đơn|bán theo đơn|prescription|prescription required|rx',
    'mã khách': 'mã khách|mã khách hàng|mã kh|mã kh mua|customer id|customer code',
    'tên khách': 'tên khách|tên khách hàng|tên kh|khách hàng tên|customer name',
    'nhóm khách': 'nhóm khách|nhóm khách hàng|loại khách hàng|đối tượng kh|customer group',
    'mã nhân viên bán': 'mã nhân viên|mã nv|mã nv bán|staff id|employee id|salesperson id',
    'đơn giá': 'đơn giá|giá đơn vị|unit price',
    'số lượng': 'số lượng|sl|s.l|slg|qty|quantity',
    'số lượng đã đổi ra đơn vị cơ bản': 'sl quy đổi|số lượng quy đổi|sl cơ bản|sl đv cơ bản|sl quy ra lẻ|base qty|qty in base unit',
    'số lượng khách trả lại': 'sl trả|sl trả lại|số lượng trả|số lượng trả lại|returned qty',
}
BILINGUAL = {meaning: {bilingual_key(x) for x in names.split('|')} for meaning, names in ALIASES.items()}
ALIASES = {meaning: {column_key(x) for x in names.split('|')} for meaning, names in ALIASES.items()}


def business_roles(frame):
    names = {c: column_key(c) for c in frame.columns if c != '_sheet'}
    out = {}
    for meaning, aliases in ALIASES.items():
        # A bare "category"/"loại" is not necessarily a product group. It
        # becomes one only after the table is evidenced as item-scoped below.
        if meaning == 'nhóm hàng':
            continue
        matches = [c for c, name in names.items() if name in aliases]
        if not matches:
            # cột đổi cách viết / dịch từng chữ ('name_item' = Tên hàng, 'qty_quy_doi' = SL quy đổi)
            matches = [c for c in names if bilingual_key(c) in BILINGUAL[meaning]]
        if len(matches) == 1:
            out[matches[0]] = meaning
    # "Tên gọi" and "Nhóm" only have meaning in the context of one entity.
    entities = [role for role in ('mã hàng', 'mã khách', 'mã nhân viên bán') if role in out.values()]
    if len(entities) == 1:
        name_role = {'mã hàng': 'tên hàng', 'mã khách': 'tên khách', 'mã nhân viên bán': 'tên nhân viên'}[entities[0]]
        generic = [c for c, name in names.items() if name in {'tengoi', 'name'}]
        if name_role not in out.values() and len(generic) == 1:
            out[generic[0]] = name_role
    if 'tên hàng' in out.values() or entities == ['mã hàng']:
        generic = [c for c, name in names.items()
                   if name in ALIASES['nhóm hàng'] or name in {'nhom', 'phanloai'}]
        if 'nhóm hàng' not in out.values() and len(generic) == 1 and not any(x in out.values() for x in ('mã khách', 'mã nhân viên bán')):
            out[generic[0]] = 'nhóm hàng'
    return out
