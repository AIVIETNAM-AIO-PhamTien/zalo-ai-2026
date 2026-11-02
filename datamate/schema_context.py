"""Bounded schema context preserving candidate join paths.

Design adaptation after inspecting CHESS SelectTables/add_connections at commit
3d6e835f858d26885d21d4bc0215aeecf855efbe (Apache-2.0).
No upstream implementation copied; no gold/corrects mode or external model call.
Candidate connections remain unverified evidence, never authorized joins.
"""
from collections import deque
from dataclasses import asdict
from itertools import combinations

from datamate.ir import PlanError


def schema_context(schemas, selected_tables, relations, max_tables=12, max_hops=3, max_paths=16, max_edges=128):
    seeds = list(dict.fromkeys(selected_tables))
    if any(t not in schemas for t in seeds):
        raise PlanError("schema_context", "selected table is unavailable")
    if len(seeds) > max_tables:
        raise PlanError("schema_context_budget", "too many selected tables")
    adjacency = {table: set() for table in schemas}
    usable = []
    for relation in relations:
        left, right = relation.left_table, relation.right_table
        if (left in schemas and right in schemas and relation.left_column in schemas[left]
                and relation.right_column in schemas[right] and left != right):
            adjacency[left].add(right)
            adjacency[right].add(left)
            usable.append(relation)
    included, paths, disconnected = set(seeds), [], []
    for start, end in combinations(seeds, 2):
        queue, found, length = deque([(start,)]), [], None
        work = 0
        while queue:
            path = queue.popleft()
            work += 1
            if work > 4096:
                raise PlanError("schema_context_budget", "join-path search work budget exceeded")
            hops = len(path) - 1
            if length is not None and hops > length:
                break
            if path[-1] == end:
                length = hops
                found.append(path)
                if len(found) > max_paths:
                    raise PlanError("schema_context_budget", "too many equally short candidate paths")
                continue
            if hops < max_hops:
                queue.extend(path + (neighbor,) for neighbor in sorted(adjacency[path[-1]]) if neighbor not in path)
        if not found:
            disconnected.append([start, end])
        for path in found:
            included.update(path)
            paths.append(list(path))
        if len(included) > max_tables:
            raise PlanError("schema_context_budget", "bridge tables exceed context budget")
    edges = [asdict(r) for r in usable if r.left_table in included and r.right_table in included]
    if len(edges) > max_edges:
        raise PlanError("schema_context_budget", "candidate relations exceed context budget")
    # Retain complete selected table schemas in this first adaptation. Column
    # pruning must later preserve contract/status/time fields as well as keys.
    ordered = seeds + sorted(included - set(seeds))
    return {"schema": {table: schemas[table] for table in ordered},
            "candidate_relations": edges, "candidate_join_paths": paths,
            "unconnected_table_pairs": disconnected,
            "join_evidence_status": "candidates only; business semantics/cardinality still require verification"}
