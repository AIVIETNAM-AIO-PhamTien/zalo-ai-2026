"""Data evidence, candidate relations and conservative partition checks."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any

import pandas as pd

from datamate.foundation import Foundation


def key(value: Any) -> tuple:
    return type(value).__name__, repr(value)


@dataclass
class Relation:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    cardinality: str
    left_coverage: float
    right_coverage: float
    name_similarity: float
    evidence: list[str] = field(default_factory=list)
    verified: bool = False  # Data overlap alone never establishes business semantics.


@dataclass(frozen=True)
class PartitionCandidate:
    left_table: str
    right_table: str
    left_label: str
    right_label: str
    schema_compatible: bool
    overlap_rows: int
    duplicate_source: bool
    evidence: tuple[str, ...]
    verified: bool = False


def partition_hint(table: str) -> tuple[str, str] | None:
    """Use source naming only for candidate discovery, never union authority."""
    file_name, _, sheet = table.partition("::")
    path = Path(file_name.replace("\\", "/"))
    stem = path.stem
    match = re.fullmatch(r"(.+?)[_-](20\d{2})(?:[_-](0[1-9]|1[0-2]))?", stem)
    if match:
        base, year, month = match.groups()
        return base.casefold(), year + (f"-{month}" if month else "")
    partitions = re.findall(r"(?:^|/)(year|month|branch|shift)=([^/]+)", file_name, re.I)
    if partitions:
        base = stem.casefold()
        return base, "/".join(f"{key.lower()}={value}" for key, value in partitions)
    if sheet:
        return stem.casefold(), sheet
    return None


@dataclass
class DataCatalog:
    foundation: Foundation
    frames: dict[str, pd.DataFrame]
    profiles: dict[str, dict]
    relations: list[Relation]
    ambiguities: list[str]
    partition_candidates: list[PartitionCandidate] = field(default_factory=list)
    duplicate_sources: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def build(cls, foundation: Foundation, headers: dict[str, int | tuple[int, ...] | None] | None = None):
        headers = headers or {}
        frames, profiles = {}, {}
        ambiguities = list(foundation.ambiguities)
        for name, raw in foundation.tables.items():
            frame = raw.project(headers.get(name))
            frames[name] = frame
            if raw.columns is None and name not in headers:
                ambiguities.append(f"{name}: unresolved header candidates {raw.header_candidates}")
            columns = {}
            for col in frame.columns:
                vals = frame[col].tolist()
                present = [v for v in vals if v is not None]
                counts = Counter(key(v) for v in present)
                columns[col] = {"types": dict(Counter(type(v).__name__ for v in present)),
                                "nulls": len(vals) - len(present), "empty_strings": sum(v == "" for v in present),
                                "distinct": len(counts), "candidate_key": bool(vals) and len(counts) == len(vals),
                                "unit": None, "temporal_coverage": None}
                dates = [v.isoformat() for v in present if hasattr(v, "isoformat")]
                if dates and len(dates) == len(present):
                    columns[col]["temporal_coverage"] = [min(dates), max(dates)]
            profiles[name] = {"rows": len(frame), "columns": columns,
                              "duplicate_rows": len(frame) - len({tuple(key(v) for v in r)
                                                                  for r in frame.itertuples(index=False, name=None)}),
                              "grain": "unknown", "source": raw.source}
        catalog = cls(foundation, frames, profiles, [], ambiguities)
        catalog.relations = catalog.find_relations()
        catalog.duplicate_sources = catalog.find_duplicate_sources()
        catalog.partition_candidates = catalog.find_partitions()
        return catalog

    def find_duplicate_sources(self) -> list[tuple[str, str]]:
        paths = {}
        for table, raw in self.foundation.tables.items():
            paths[raw.source.path] = raw.source.sha256
        by_hash = {}
        for path, digest in sorted(paths.items()):
            by_hash.setdefault(digest, []).append(path)
        return [(left, right) for group in by_hash.values()
                for i, left in enumerate(group) for right in group[i + 1:]]

    def find_partitions(self, max_pairs: int = 200) -> list[PartitionCandidate]:
        groups = {}
        for name in self.frames:
            hint = partition_hint(name)
            if hint:
                groups.setdefault(hint[0], []).append((name, hint[1]))
        candidates = []
        row_sets = {}
        for group in groups.values():
            for i, (left, left_label) in enumerate(group):
                for right, right_label in group[i + 1:]:
                    if len(candidates) >= max_pairs:
                        self.ambiguities.append("partition candidate search truncated at pair budget")
                        return candidates
                    left_frame, right_frame = self.frames[left], self.frames[right]
                    columns_equal = list(left_frame.columns) == list(right_frame.columns)
                    headers_grounded = all(
                        self.foundation.tables[name].columns is not None or
                        self.frames[name].attrs.get("header") is not None
                        for name in (left, right))
                    compatible = headers_grounded and columns_equal and all(
                        set(self.profiles[left]["columns"][c]["types"]) ==
                        set(self.profiles[right]["columns"][c]["types"])
                        for c in left_frame.columns)
                    for name, frame in ((left, left_frame), (right, right_frame)):
                        if name not in row_sets:
                            row_sets[name] = {tuple(key(v) for v in row) for row in
                                              frame.itertuples(index=False, name=None)}
                    duplicate = (self.foundation.tables[left].source.sha256 ==
                                 self.foundation.tables[right].source.sha256 and
                                 self.foundation.tables[left].source.path !=
                                 self.foundation.tables[right].source.path)
                    evidence = (f"source labels {left_label!r} and {right_label!r}",
                                "schema and row overlap profiled; coverage, units and grain unverified")
                    candidates.append(PartitionCandidate(left, right, left_label, right_label,
                                                          compatible, len(row_sets[left] & row_sets[right]),
                                                          duplicate, evidence))
        return candidates

    def find_relations(self, max_pairs: int = 10000) -> list[Relation]:
        relations, checked = [], 0
        items = [(t, c, Counter(key(v) for v in frame[c] if v is not None and v != ""))
                 for t, frame in self.frames.items() for c in frame.columns]
        for i, (lt, lc, left) in enumerate(items):
            for rt, rc, right in items[i + 1:]:
                if lt == rt or not left or not right:
                    continue
                checked += 1
                if checked > max_pairs:
                    self.ambiguities.append("relation search truncated at pair budget")
                    return relations
                overlap = left.keys() & right.keys()
                if not overlap:
                    continue
                lu, ru = max(left.values()) == 1, max(right.values()) == 1
                cardinality = ("one" if lu else "many") + "-to-" + ("one" if ru else "many")
                relations.append(Relation(lt, lc, rt, rc, cardinality, len(overlap) / len(left),
                                          len(overlap) / len(right), SequenceMatcher(None, lc, rc).ratio(),
                                          [f"{len(overlap)} overlapping distinct typed values"]))
        return relations

    def check_union(self, names: list[str], *, units: dict[str, str], grain: str | None,
                    partition_evidence: str | None) -> list[str]:
        problems = []
        if len(names) < 2 or len(set(names)) != len(names):
            return ["at least two distinct partitions required"]
        if any(n not in self.frames for n in names):
            return ["unknown partition"]
        first = self.frames[names[0]]
        if not grain or not partition_evidence:
            problems.append("partition coverage and grain evidence required")
        if set(units) != set(first.columns):
            problems.append("explicit units (including unitless) required for every column")
        seen = set()
        types = self.profiles[names[0]]["columns"]
        for name in names:
            frame = self.frames[name]
            if list(frame.columns) != list(first.columns):
                problems.append(f"{name}: incompatible schema")
            elif any(set(self.profiles[name]["columns"][c]["types"]) != set(types[c]["types"]) for c in first.columns):
                problems.append(f"{name}: incompatible raw types")
            rows = {tuple(key(v) for v in row) for row in frame.itertuples(index=False, name=None)}
            if seen & rows:
                problems.append(f"{name}: partition overlap; do not deduplicate automatically")
            seen |= rows
        return problems
