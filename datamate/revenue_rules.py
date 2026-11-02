"""A tested adapter for the supported train revenue policy and its relational core.

Policy recognition is deliberately complete-row matching. Source role binding and
status vocabulary must be grounded separately before this core is used by QA.
"""
from dataclasses import dataclass
import hashlib
import json
import re

from datamate.ir import PlanError
from datamate.parsing import fold
from datamate.semantics import Evidence, SemanticContract
from datamate.adapters import register_contract
from datamate.temporal import reference_date, resolve_window
from datamate.temporal_rules import grounded_window
from datamate.question_text import normalize_question
from datamate.entities import resolve_entity, entity_key


SUPPORTED = {
    "Đơn tính vào doanh thu": "đơn không huỷ và không trả hàng. Đơn có trả hàng, kể cả chỉ trả một phần, là đơn trả hàng: bỏ cả đơn. Đơn đang giao vẫn tính",
    "Đơn thuộc kỳ nào": "theo ngày đặt, không theo ngày giao hay ngày thu tiền",
    "Doanh thu một đơn": "tổng thành tiền các dòng hàng (đã trừ giảm giá trên từng dòng). Không trừ giảm giá trên cả đơn (khuyến mãi theo tổng tiền đơn, voucher shop), không cộng thuế VAT, không gồm phí giao hàng. Tiền hàng, tiền bán của đơn bán cùng nghĩa với doanh thu",
}


@dataclass(frozen=True)
class RevenuePolicy:
    evidence: tuple[Evidence, ...]
    amount_basis: str = "line_amount_after_line_discount"
    time_basis: str = "order_placed"
    returns: str = "exclude_entire_order_including_partial_returns"


def revenue_policy(documents):
    text = documents.get("$rulebook", "")
    found, offset = {}, 0
    expected = {fold(k): fold(v) for k, v in SUPPORTED.items()}
    for line in text.splitlines(keepends=True):
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and parts and fold(parts[0]) in expected:
            key = fold(parts[0])
            if key in found or len(parts) != 2 or fold(parts[1]) != expected[key]:
                raise PlanError("revenue_policy", "revenue rule changed, duplicated or unsupported: " + parts[0])
            found[key] = Evidence("$rulebook", offset, offset + len(line), line)
        offset += len(line)
    if set(found) != set(expected):
        raise PlanError("revenue_policy", "incomplete supported revenue rulebook")
    return RevenuePolicy(tuple(found.values()))


def revenue_relation(policy, table, frame, columns, valid_states, excluded_states, window, line_predicate=None):
    """Build a grounded core once a caller has verified column/status bindings.

    Classify across the entire source before date filtering so a returned line
    outside the requested period still excludes its whole order.
    """
    if not isinstance(policy, RevenuePolicy):
        raise PlanError("revenue_policy", "supported revenue policy required")
    if set(columns) != {"order", "amount", "placed", "status"} or len(set(columns.values())) != 4:
        raise PlanError("revenue_binding", "four distinct source roles required")
    if any(c not in frame for c in columns.values()):
        raise PlanError("revenue_binding", "source role references missing column")
    if not (valid_states or excluded_states) or set(valid_states) & set(excluded_states):
        raise PlanError("revenue_status", "grounded disjoint status vocabularies required")
    order, amount, placed, status = (columns[k] for k in ("order", "amount", "placed", "status"))
    if frame[[order, placed, status]].isna().any().any() or (frame[order] == "").any():
        raise PlanError("revenue_source", "missing order/date/status evidence")
    if not set(frame[status]).issubset(set(valid_states) | set(excluded_states)):
        raise PlanError("revenue_status", "unknown status cannot be treated as completed")
    if (frame.groupby(order)[placed].nunique() > 1).any():
        raise PlanError("revenue_grain", "conflicting order placement dates")
    rejected = set(frame.loc[frame[status].isin(excluded_states), order])
    if frame.loc[~frame[order].isin(rejected), amount].isna().any():
        raise PlanError("revenue_source", "missing valid-order line amount")
    scan = {"kind": "scan", "table": table}
    conditions = [{"kind": "binary", "op": "eq", "left": {"kind": "column", "name": status},
                   "right": {"kind": "literal", "type": "text", "value": value}} for value in sorted(excluded_states)]
    predicate = conditions[0] if conditions else {"kind": "literal", "type": "boolean", "value": False}
    for condition in conditions[1:]:
        predicate = {"kind": "binary", "op": "or", "left": predicate, "right": condition}
    bad = {"kind": "distinct", "input": {"kind": "project", "input": {
        "kind": "filter", "input": scan, "predicate": predicate}, "columns": [
            {"name": "excluded_order", "expression": {"kind": "column", "name": order}}]}}
    eligible = {"kind": "join", "how": "anti", "left": scan, "right": bad,
                "keys": [{"left": order, "right": "excluded_order"}], "cardinality": "many-to-one"}
    eligible = window.apply(eligible, placed)
    if line_predicate is not None:
        eligible = {"kind": "filter", "input": eligible, "predicate": line_predicate}
    total = {"kind": "aggregate", "op": "sum", "value": {"kind": "column", "name": amount}}
    count = {"kind": "aggregate", "op": "count", "value": {"kind": "literal", "type": "number", "value": 1}}
    zero = {"kind": "literal", "type": "number", "value": 0}
    expression = {"kind": "case", "when": {"kind": "binary", "op": "eq", "left": count, "right": zero},
                  "then": zero, "else": total}
    return {"kind": "aggregate", "input": eligible, "groups": [], "metrics": [
        {"name": "revenue", "expression": expression}]}


def revenue_requirements(question, prepared):
    canonical = normalize_question(question.get("question", "")).canonical.strip()
    product_match = re.fullmatch(r"(?:(?:cho tôi biết|xem giúp)\s+)?(?:tổng )?doanh thu của (.+?) trong (.+?)(?:\s+là bao nhiêu)?[?.]?", canonical, re.IGNORECASE)
    match = re.fullmatch(r"(?:(?:cho tôi biết|xem giúp)\s+)?(?:tổng )?doanh thu (?:trong )?(.+?)(?:\s+của cửa hàng)?(?:\s+là bao nhiêu)?[?.]?", canonical, re.IGNORECASE)
    if not match:
        return None
    if product_match and fold(product_match[1]).startswith(("khach ", "khach hang ", "nhan vien ", "nhom ", "cua hang ")):
        return None
    # An entity qualifier is outside the whole-business metric contract.
    # Do not misreport it as a missing calendar rule or discard the qualifier.
    if fold(match[1]).startswith("cua ") and product_match is None:
        return None
    warehouse = prepared.warehouse
    policy = revenue_policy(warehouse.catalog.foundation.documents)
    # This version handles a documented complete, single source only. A second
    # table may contain returns or another partition; it must not be ignored.
    if len(warehouse.catalog.foundation.tables) != 1 or len(prepared.frames) != 1:
        raise PlanError("revenue_completeness", "single complete line source required; multi-source binding pending")
    table = next(iter(prepared.frames))
    descriptions = {"order": {"ma don", "ma don hang"},
                    "amount": {"thanh tien dong hang sau giam", "thanh tien sau giam"},
                    "placed": {"ngay dat don", "ngay dat hang", "ngay ban"}, "status": {"trang thai don"}}
    columns, evidence = {}, list(policy.evidence)
    for role, meanings in descriptions.items():
        bindings = [b for b in warehouse.bindings if b.table == table and fold(b.description) in meanings]
        if len(bindings) != 1:
            raise PlanError("revenue_binding", "one documented unambiguous source role required: " + role)
        columns[role] = bindings[0].column
        evidence.append(bindings[0].evidence)
    required_assertions = {"moi dong la mot dong hang trong don",
                           "trang thai don da phan anh moi huy tra hang"}
    witnessed = set()
    for document, text in warehouse.catalog.foundation.documents.items():
        if document == "$rulebook" or document.startswith("$"):
            continue
        offset = 0
        for line in text.splitlines(keepends=True):
            normalized = fold(line)
            if normalized == "mot file phang moi dong mot ban ghi mot dong hang trong don":
                normalized = "moi dong la mot dong hang trong don"
            if normalized in required_assertions:
                witnessed.add(normalized)
                evidence.append(Evidence(document, offset, offset + len(line), line))
            offset += len(line)
    status_rule = "Riêng trạng thái huỷ, trả hàng của đơn lấy theo trạng thái ghi trên đơn, kể cả khi phiếu trả ghi ngày sau hôm nay."
    rulebook = warehouse.catalog.foundation.documents.get("$rulebook", "")
    if rulebook.count(status_rule) == 1:
        start = rulebook.index(status_rule)
        witnessed.add("trang thai don da phan anh moi huy tra hang")
        evidence.append(Evidence("$rulebook", start, start + len(status_rule), status_rule))
    if witnessed != required_assertions:
        raise PlanError("revenue_completeness", "line grain and complete order-status coverage need source documentation")
    schema, frame = prepared.schemas[table], prepared.frames[table]
    if not set(columns.values()).issubset(schema):
        raise PlanError("revenue_type", "one or more bound columns have no usable typed projection")
    if schema[columns["amount"]] != "number" or schema[columns["placed"]] != "date" or schema[columns["status"]] != "text":
        raise PlanError("revenue_type", "numeric/date/status projections required")
    accepted = {"hoan tat", "hoan thanh", "da lay", "dang giao", "xong"}
    excluded = {"huy", "da huy", "tra hang", "tra mot phan", "doi tra"}
    valid_states, excluded_states = set(), set()
    for value in frame[columns["status"]].unique():
        normalized = fold(value)
        if normalized in accepted:
            valid_states.add(value)
        elif normalized in excluded:
            excluded_states.add(value)
        else:
            raise PlanError("revenue_status", "status vocabulary has an unresolved value")
    reference = reference_date(question=question["as_of_date"]) if question.get("as_of_date") else None
    try:
        period = product_match[2] if product_match else match[1]
        window, time_evidence = grounded_window(period.strip(), reference, warehouse.catalog.foundation.documents)
    except PlanError as error:
        if error.code == "temporal_unsupported":
            return None
        raise
    evidence.extend(time_evidence)
    entity, line_predicate = None, None
    extra_sources = []
    if product_match:
        products = [b for b in warehouse.bindings if b.table == table and fold(b.description) in {"ten hang", "ten san pham"}]
        identifiers = [b for b in warehouse.bindings if b.table == table and fold(b.description) in {"ma hang", "ma san pham"}]
        if len(products) != 1 or len(identifiers) > 1 or prepared.schemas[table].get(products[0].column) != "text":
            raise PlanError("entity_binding", "one documented product-name column and unambiguous identifier mapping required")
        for binding in warehouse.bindings:
            if (binding.table == table and fold(binding.description).startswith(("ten khach", "ten nhan vien"))
                    and binding.column in frame
                    and any(isinstance(v, str) and entity_key(v) == entity_key(product_match[1]) for v in frame[binding.column].dropna().unique())):
                raise PlanError("entity_kind_ambiguity", "requested label also names a customer/employee; entity role needs clarification")
        rejected = set(frame.loc[frame[columns["status"]].isin(excluded_states), columns["order"]])
        entity = resolve_entity(frame, products[0].column, product_match[1],
            identifiers[0].column if identifiers else None, ~frame[columns["order"]].isin(rejected))
        extra_sources = [(table, b.column) for b in products + identifiers]
        evidence.extend(b.evidence for b in products + identifiers)
        line_predicate = {"kind": "binary", "op": "eq", "left": {"kind": "column", "name": entity.column},
                          "right": {"kind": "literal", "type": "text", "value": entity.value}}
    relation = revenue_relation(policy, table, frame, columns, valid_states, excluded_states, window, line_predicate)
    digest = hashlib.sha256((question["question"] + json.dumps(relation, sort_keys=True)).encode()).hexdigest()
    contract = SemanticContract("revenue", [(table, c) for c in columns.values()] + extra_sources, [(table, "$physical_row")],
        relation, status_policy=[{"eligible": sorted(valid_states), "excluded_entire_orders": sorted(excluded_states)}],
        time_column=(table, columns["placed"]), units="source monetary unit",
        evidence=evidence, applicability={"question": question["question"], "source_scope": "complete single line source",
            "entity": {"column": entity.column, "value": entity.value} if entity else None},
        rule_ids=["revenue:" + digest])
    view_id = register_contract(warehouse, contract, policy.evidence[0])
    return {"relation": relation, "answer_type": "number", "contract_ids": [view_id]}
