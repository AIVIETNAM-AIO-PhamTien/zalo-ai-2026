"""Shared semantic candidates, executable obligations, and grounded virtual views.

Document spans prove that text exists, not that a model interpreted it correctly.
Only structured rules supplied by a trusted rule adapter can authorize a view.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

from datamate.catalog import DataCatalog
from datamate.foundation import Foundation
from datamate.parsing import fold
from datamate.value_context import ValueIndex


@dataclass(frozen=True)
class Evidence:
    document: str
    start: int
    end: int
    quote: str

    def valid(self, documents: dict[str, str]) -> bool:
        text = documents.get(self.document)
        return (text is not None and 0 <= self.start < self.end <= len(text)
                and text[self.start:self.end] == self.quote)


@dataclass(frozen=True)
class Binding:
    table: str
    column: str
    description: str
    evidence: Evidence


@dataclass
class SemanticContract:
    concept: str
    sources: list[tuple[str, str]]
    grain: list[tuple[str, str]]
    expression: dict
    joins: list[dict] = field(default_factory=list)
    status_policy: list[dict] = field(default_factory=list)
    time_column: tuple[str, str] | None = None
    units: str | None = None
    transformations: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    applicability: dict = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class GroundedRule:
    """Created by deterministic rule adapters, never accepted from planner JSON.

    Expected contract fields are executable obligations, not a boolean 'verified'.
    A future natural-language rule adapter must have independently tested semantics.
    """
    rule_id: str
    evidence: Evidence
    expected: dict


@dataclass
class VirtualView:
    name: str
    contract: SemanticContract
    fingerprint: str
    lineage: list[dict]


def document_bindings(foundation: Foundation) -> list[Binding]:
    """Extract explicit Markdown column descriptions; retain competing descriptions."""
    bindings = []
    available_labels = {name: set(raw.columns) if raw.columns is not None else
                        {label for option in raw.header_options() for label in raw.header_labels(option)}
                        for name, raw in foundation.tables.items()}
    for document, text in foundation.documents.items():
        current_tables = []
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("#"):
                names = re.findall(r"`([^`]+)`", line)
                current_tables = [name for name in foundation.tables
                                  if any(name == n or name.startswith(n + "::") for n in names)]
            match = re.match(r"\s*[-*]\s+\*\*(.+?)\*\*\s*:\s*(.+?)\s*$", line)
            if match:
                column, description = match.groups()
                evidence = Evidence(document, offset + match.start(), offset + match.end(), match.group())
                for table in current_tables:
                    if column in available_labels[table]:
                        bindings.append(Binding(table, column, description, evidence))
            offset += len(line)
    return bindings


def ground_headers(foundation: Foundation, bindings: list[Binding]) -> tuple[dict, list[str]]:
    headers, ambiguities = {}, []
    for name, raw in foundation.tables.items():
        if raw.columns is not None:
            headers[name] = None
            continue
        documented = {b.column for b in bindings if b.table == name}
        matches = [(i, len(set(raw.header_labels(i)) & documented)) for i in raw.header_options()]
        best = max((n for _, n in matches), default=0)
        selected = [i for i, n in matches if n == best]
        if best >= 2 and len(selected) == 1:
            labels = [v for v in raw.header_labels(selected[0]) if v]
            if len(labels) != len(set(labels)):
                ambiguities.append(f"{name}: duplicate column labels cannot be bound unambiguously")
            else:
                headers[name] = selected[0]
        else:
            ambiguities.append(f"{name}: cannot ground a unique header from documentation")
    return headers, ambiguities


@dataclass
class SemanticWarehouse:
    catalog: DataCatalog
    bindings: list[Binding]
    roles: dict[str, list[dict]]
    graph: list[dict]
    contracts: dict[str, list[SemanticContract]] = field(default_factory=dict)
    views: dict[str, VirtualView] = field(default_factory=dict)
    rules: dict[str, GroundedRule] = field(default_factory=dict)
    value_index: ValueIndex | None = field(default=None, repr=False)

    @classmethod
    def build(cls, foundation: Foundation):
        bindings = document_bindings(foundation)
        headers, ambiguities = ground_headers(foundation, bindings)
        catalog = DataCatalog.build(foundation, headers)
        catalog.ambiguities.extend(ambiguities)
        roles, graph = {}, []
        # Role names are retrieval hints only, never canonical schemas or metric bindings.
        role_words = {"transaction_line": ("dong hang", "so luong", "thanh tien"),
                      "payment": ("thanh toan", "thu tien"), "return": ("tra hang", "hoan tien"),
                      "inventory": ("ton kho", "nhap kho", "xuat kho"),
                      "service": ("sua chua", "dich vu"), "employee": ("nhan vien", "nhan su"),
                      "ledger": ("tai khoan", "but toan")}
        for name in catalog.frames:
            relevant = [b for b in bindings if b.table == name]
            descriptions = " ".join(fold(b.description) for b in relevant)
            roles[name] = [{"role": role, "status": "candidate", "evidence": [asdict(b.evidence) for b in relevant]}
                           for role, words in role_words.items() if any(word in descriptions for word in words)]
            if not roles[name]:
                roles[name] = [{"role": "unknown", "status": "candidate", "evidence": []}]
            graph.append({"kind": "entity", "id": name, "grain": "unknown"})
            for b in relevant:
                graph.append({"kind": "attribute", "table": name, "column": b.column,
                              "concept": b.description, "evidence": asdict(b.evidence)})
        graph.extend({"kind": "relation", **asdict(r)} for r in catalog.relations)
        graph.extend({"kind": "partition_candidate", **asdict(candidate)}
                     for candidate in catalog.partition_candidates)
        graph.extend({"kind": "duplicate_source", "paths": list(paths)}
                     for paths in catalog.duplicate_sources)
        warehouse = cls(catalog, bindings, roles, graph)
        for binding in bindings:
            warehouse.add_candidate(SemanticContract(
                concept=binding.description, sources=[(binding.table, binding.column)], grain=[],
                expression={"kind": "column", "table": binding.table, "column": binding.column},
                evidence=[binding.evidence],
                prerequisites=["metric aggregation, grain, units and applicability require grounding"]))
        return warehouse

    def add_candidate(self, contract: SemanticContract):
        self.contracts.setdefault(contract.concept, []).append(contract)
        self.graph.append({"kind": "metric", "concept": contract.concept,
                           "fingerprint": contract.fingerprint(), "status": "candidate"})

    def validate_contract(self, contract: SemanticContract) -> list[str]:
        errors = list(contract.ambiguities) + list(contract.prerequisites)
        documents = self.catalog.foundation.documents
        if not contract.sources or not contract.grain or not contract.units:
            errors.append("sources, grain and units required")
        for table, column in contract.sources + contract.grain + ([contract.time_column] if contract.time_column else []):
            physical_grain = (column == "$physical_row" and (table, column) in contract.grain
                              and (table, column) not in contract.sources and contract.time_column != (table, column))
            if table not in self.catalog.frames or (column not in self.catalog.frames[table] and not physical_grain):
                errors.append(f"unknown source {table}.{column}")
            elif self.catalog.foundation.tables[table].columns is None and self.catalog.frames[table].attrs["header"] is None:
                errors.append(f"unresolved header {table}")
        if not contract.evidence or any(not e.valid(documents) for e in contract.evidence):
            errors.append("missing or invalid source evidence")
        if not contract.rule_ids:
            errors.append("no executable semantic obligations")
        actual = asdict(contract)
        covered = set()
        for rule_id in contract.rule_ids:
            rule = self.rules.get(rule_id)
            if rule is None or not rule.evidence.valid(documents):
                errors.append(f"unknown or stale grounded rule {rule_id}")
                continue
            for field_name, expected in rule.expected.items():
                covered.add(field_name)
                if field_name not in actual or actual[field_name] != expected:
                    errors.append(f"rule {rule_id}: violated {field_name}")
        required = {"concept", "sources", "grain", "expression", "joins", "status_policy",
                    "time_column", "units", "transformations", "applicability"}
        if required - covered:
            errors.append("ungrounded contract fields: " + ", ".join(sorted(required - covered)))
        for join in contract.joins:
            matching = [r for r in self.catalog.relations
                        if (r.left_table, r.left_column, r.right_table, r.right_column) ==
                        (join.get("left_table"), join.get("left_column"), join.get("right_table"), join.get("right_column"))]
            if len(matching) != 1 or matching[0].cardinality != join.get("cardinality"):
                errors.append("join cardinality mismatch")
            elif matching[0].cardinality == "many-to-many":
                errors.append("many-to-many aggregation requires preaggregation")
        return errors

    def create_view(self, name: str, contract: SemanticContract) -> VirtualView:
        errors = self.validate_contract(contract)
        if errors:
            raise ValueError("; ".join(errors))
        if name in self.views:
            raise ValueError("view already exists")
        lineage = [{"table": t, "column": c,
                    "source_sha256": self.catalog.foundation.tables[t].source.sha256,
                    "physical_rows": self.catalog.frames[t].attrs["physical_rows"]}
                   for t, c in contract.sources]
        view = VirtualView(name, contract, contract.fingerprint(), lineage)
        self.views[name] = view
        return view

    def retrieve(self, question: str, limit: int = 12) -> list[dict]:
        tokens = set(re.findall(r"\w+", fold(question)))
        scored = []
        for binding in self.bindings:
            text = f"{binding.column} {binding.description}"
            overlap = len(tokens & set(re.findall(r"\w+", fold(text))))
            if overlap:
                scored.append((overlap, asdict(binding)))
        return [b for _, b in sorted(scored, key=lambda pair: -pair[0])[:limit]]

    def retrieve_values(self, question: str, *, tables=None, limit: int = 12) -> dict:
        if self.value_index is None:
            self.value_index = ValueIndex(self.catalog.frames, profiles=self.catalog.profiles)
        return self.value_index.search(question, tables=tables, limit=limit)
