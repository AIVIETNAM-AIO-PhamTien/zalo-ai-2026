"""SQLGlot AST -> closed relational IR. Original SQL is never executed.

Uses the parser employed by inspected CHESS; independently implemented lowering.
Unsupported clauses fail closed. This first bridge supports single-table SELECT.
"""
import sqlglot
from sqlglot import exp

from datamate.engine import Compiler
from datamate.ir import PlanError
from datamate.parsing import fold


def sql_to_plan(sql, schemas, answer_type="table", contract_ids=()):
    if not isinstance(sql, str) or len(sql.encode()) > 65536:
        raise PlanError("sql_budget", "SQL must be bounded text")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except (sqlglot.errors.SqlglotError, RecursionError) as error:
        raise PlanError("sql_parse", str(error)[:1000]) from error
    if len(statements) != 1 or type(statements[0]) is not exp.Select:
        raise PlanError("sql_unsupported", "one SELECT statement required")
    tree = statements[0]
    if sum(1 for _ in tree.walk()) > 256:
        raise PlanError("sql_budget", "SQL AST exceeds node budget")
    allowed = {"expressions", "from", "from_", "where", "group", "order", "limit", "distinct"}
    if any(v is not None and v != [] and k not in allowed for k, v in tree.args.items()):
        raise PlanError("sql_unsupported", "unsupported SELECT clause")
    source = tree.args.get("from_") or tree.args.get("from")
    if source is None or type(source.this) is not exp.Table or source.expressions:
        raise PlanError("sql_unsupported", "one physical table required")
    table = source.this
    if any(v not in (None, False, []) for k, v in table.args.items() if k not in {"this", "alias", "db", "catalog"}):
        raise PlanError("sql_unsupported", "table modifiers are unsupported")
    if table.db or table.catalog or not isinstance(table.this, exp.Identifier) or table.name not in schemas:
        raise PlanError("unknown_table", "table is outside the source allowlist")
    alias = table.alias_or_name
    if table.args.get("alias") and table.args["alias"].args.get("columns"):
        raise PlanError("sql_unsupported", "table alias column renaming is unsupported")
    def expression(node):
        if type(node) is exp.Paren:
            return expression(node.this)
        if type(node) is exp.Column:
            if not isinstance(node.this, exp.Identifier):
                raise PlanError("sql_unsupported", "explicit column identifier required")
            if node.db or node.catalog or node.table and node.table not in {alias, table.name}:
                raise PlanError("unknown_column", "unbound column qualifier")
            column = node.name
            if column not in schemas[table.name]:
                matches = [candidate for candidate in schemas[table.name]
                           if fold(candidate) == fold(column)]
                if len(matches) != 1:
                    raise PlanError("unknown_column", column)
                column = matches[0]
            return {"kind": "column", "name": column}
        if type(node) is exp.Literal:
            return {"kind": "literal", "type": "text" if node.is_string else "number", "value": node.this}
        if type(node) is exp.Null:
            return {"kind": "literal", "type": "null", "value": None}
        if type(node) is exp.Boolean:
            return {"kind": "literal", "type": "boolean", "value": node.this}
        binary = {exp.Add: "add", exp.Sub: "subtract", exp.Mul: "multiply", exp.Div: "divide", exp.EQ: "eq",
                  exp.NEQ: "ne", exp.LT: "lt", exp.LTE: "le", exp.GT: "gt", exp.GTE: "ge", exp.And: "and", exp.Or: "or"}
        if type(node) in binary:
            return {"kind": "binary", "op": binary[type(node)], "left": expression(node.this), "right": expression(node.expression)}
        if type(node) is exp.Between:
            if node.args.get("symmetric"):
                raise PlanError("sql_unsupported", "symmetric BETWEEN is unsupported")
            value = expression(node.this)
            return {"kind": "binary", "op": "and",
                    "left": {"kind": "binary", "op": "ge", "left": value, "right": expression(node.args["low"])},
                    "right": {"kind": "binary", "op": "le", "left": value, "right": expression(node.args["high"])}}
        if type(node) is exp.Is and type(node.expression) is exp.Null:
            return {"kind": "is_null", "value": expression(node.this)}
        if type(node) is exp.Not:
            return {"kind": "not", "value": expression(node.this)}
        aggregates = {exp.Sum: "sum", exp.Count: "count", exp.Avg: "avg", exp.Min: "min", exp.Max: "max"}
        if type(node) in aggregates:
            if any(v not in (None, False, []) for k, v in node.args.items()
                   if k != "this" and not (type(node) is exp.Count and k == "big_int" and v is True)):
                raise PlanError("sql_unsupported", "aggregate modifiers are unsupported")
            value = node.this
            if type(value) is exp.Star and type(node) is exp.Count:
                arg = {"kind": "literal", "type": "number", "value": 1}
            elif type(value) is exp.Distinct and type(node) is exp.Count:
                if len(value.expressions) != 1 or any(
                        v not in (None, False, []) for k, v in value.args.items() if k != "expressions"):
                    raise PlanError("sql_unsupported", "COUNT DISTINCT requires one plain expression")
                arg = expression(value.expressions[0])
                return {"kind": "aggregate", "op": "count_distinct", "value": arg}
            else:
                arg = expression(value)
            return {"kind": "aggregate", "op": aggregates[type(node)], "value": arg}
        raise PlanError("sql_unsupported", "unsupported expression: " + type(node).__name__)
    relation = {"kind": "scan", "table": table.name}
    if tree.args.get("where"):
        relation = {"kind": "filter", "input": relation, "predicate": expression(tree.args["where"].this)}
    selected = []
    group_nodes = tree.args["group"].expressions if tree.args.get("group") else []
    if tree.args.get("group") and any(v not in (None, False, []) for k, v in tree.args["group"].args.items() if k != "expressions"):
        raise PlanError("sql_unsupported", "grouping modifiers are unsupported")
    for item in tree.expressions:
        value = item.this if type(item) is exp.Alias else item
        if type(value) is exp.Star:
            raise PlanError("sql_unsupported", "explicit output columns required")
        name = item.alias or (value.name if type(value) is exp.Column else None)
        if not name:
            raise PlanError("sql_unsupported", "computed outputs require explicit aliases")
        selected.append((name, value, expression(value)))
    if group_nodes or any(value.find(exp.AggFunc) for _, value, _ in selected):
        groups = [{"name": name, "expression": ir} for name, value, ir in selected if value in group_nodes]
        if len(groups) != len(group_nodes):
            raise PlanError("sql_unsupported", "all group expressions must appear in output")
        metrics = [{"name": name, "expression": ir} for name, value, ir in selected if value not in group_nodes]
        relation = {"kind": "aggregate", "input": relation, "groups": groups, "metrics": metrics}
        relation = {"kind": "project", "input": relation, "columns": [{"name": n, "expression": {"kind": "column", "name": n}} for n, _, _ in selected]}
    else:
        relation = {"kind": "project", "input": relation, "columns": [{"name": n, "expression": ir} for n, _, ir in selected]}
    if tree.args.get("distinct"):
        if tree.args["distinct"].args.get("on"):
            raise PlanError("sql_unsupported", "DISTINCT ON unsupported")
        relation = {"kind": "distinct", "input": relation}
    if tree.args.get("order"):
        order = []
        for item in tree.args["order"].expressions:
            if type(item.this) is not exp.Column or item.this.table or item.this.name not in [n for n, _, _ in selected]:
                raise PlanError("sql_unsupported", "ORDER BY requires unqualified output names")
            if item.args.get("nulls_first"):
                raise PlanError("sql_unsupported", "NULLS FIRST cannot be lowered to current IR")
            order.append({"column": item.this.name, "descending": bool(item.args.get("desc"))})
        relation = {"kind": "sort", "input": relation, "by": order}
    if tree.args.get("limit"):
        if any(v not in (None, False, []) for k, v in tree.args["limit"].args.items() if k != "expression"):
            raise PlanError("sql_unsupported", "LIMIT modifiers unsupported")
        value = tree.args["limit"].expression
        if type(value) is not exp.Literal or value.is_string or not value.this.isdigit():
            raise PlanError("sql_unsupported", "literal integer LIMIT required")
        relation = {"kind": "limit", "input": relation, "count": int(value.this)}
    plan = {"relation": relation, "answer_type": answer_type, "contract_ids": list(contract_ids), "assumptions": []}
    Compiler(schemas).compile(plan)
    return plan


def decode_plan_reply(payload, schemas):
    """Optional structured SQL envelope; existing native IR stays unchanged."""
    if isinstance(payload, dict) and "sql" in payload:
        if set(payload) != {"sql", "answer_type", "contract_ids"} or not isinstance(payload["contract_ids"], list):
            raise PlanError("sql_envelope", "SQL envelope requires only sql/answer_type/contract_ids")
        return sql_to_plan(payload["sql"], schemas, payload["answer_type"], payload["contract_ids"])
    return payload
