from datamate.catalog import DataCatalog
from datamate.foundation import load_foundation
from datamate.semantics import SemanticWarehouse


def test_partition_candidates_keep_overlap_and_duplicate_evidence(tmp_path):
    (tmp_path / "orders_2025_01.csv").write_text("id,amount\na,1\nb,2\n", encoding="utf-8")
    (tmp_path / "orders_2025_02.csv").write_text("id,amount\nb,2\nc,3\n", encoding="utf-8")
    (tmp_path / "orders_2025_03.csv").write_text("id,amount\na,1\nb,2\n", encoding="utf-8")
    (tmp_path / "orders_2025_04.csv").write_text("id,total\nd,4\n", encoding="utf-8")
    foundation = load_foundation(tmp_path)
    catalog = DataCatalog.build(foundation, {name: 0 for name in foundation.tables})
    candidates = {(c.left_label, c.right_label): c for c in catalog.partition_candidates}
    january_february = candidates["2025-01", "2025-02"]
    assert january_february.schema_compatible
    assert january_february.overlap_rows == 1
    assert not january_february.verified
    assert candidates["2025-01", "2025-03"].duplicate_source
    assert not candidates["2025-01", "2025-04"].schema_compatible
    assert len(catalog.duplicate_sources) == 1
    assert all("coverage" in " ".join(c.evidence) for c in catalog.partition_candidates)


def test_partition_hints_enter_semantic_graph_as_unverified_candidates(tmp_path):
    (tmp_path / "events_2024.csv").write_text("id,value\na,1\n", encoding="utf-8")
    (tmp_path / "events_2025.csv").write_text("id,value\nb,2\n", encoding="utf-8")
    warehouse = SemanticWarehouse.build(load_foundation(tmp_path))
    partition_edges = [edge for edge in warehouse.graph if edge["kind"] == "partition_candidate"]
    assert len(partition_edges) == 1
    assert partition_edges[0]["verified"] is False
    assert partition_edges[0]["schema_compatible"] is False  # positional headers are unresolved
    assert not warehouse.views
