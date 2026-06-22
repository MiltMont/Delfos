from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from delfos.schema import (
    ContentKind,
    ContentNode,
    CueNode,
    CueType,
    Direction,
    Edge,
    EdgeType,
    MemoryLayer,
    NodeType,
    TagCategory,
    TagNode,
)
from delfos.store.base import IndexedFile
from delfos.store.duckdb_store import DuckDBGraphStore

EMBEDDING_DIM = 8
EMBEDDING_MODEL = "fake-v1"
NOW = datetime(2026, 6, 22, 12, 0, 0)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[DuckDBGraphStore]:
    s = DuckDBGraphStore(
        tmp_path / "t.duckdb",
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    s.initialize()
    yield s
    s.close()


def test_initialize_is_idempotent(store: DuckDBGraphStore) -> None:
    # Second call must not raise.
    store.initialize()
    tables = {
        row[0]
        for row in store._con.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert {"nodes", "edges", "indexed_files"} <= tables


def test_constructor_stores_config(store: DuckDBGraphStore) -> None:
    assert store.embedding_dim == EMBEDDING_DIM
    assert store.embedding_model == EMBEDDING_MODEL


def make_cue(node_id: str = "cue-1", embedding: list[float] | None = None) -> CueNode:
    return CueNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="load_config",
        embedding=embedding,
        embedding_model=EMBEDDING_MODEL if embedding is not None else None,
    )


def make_tag(node_id: str = "tag-1") -> TagNode:
    return TagNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        category=TagCategory.MODULE_PATH,
        value="delfos.config",
    )


def make_content(node_id: str = "content-1", embedding: list[float] | None = None) -> ContentNode:
    return ContentNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="load_config",
        body="def load_config(): ...",
        embedding=embedding,
        embedding_model=EMBEDDING_MODEL if embedding is not None else None,
    )


def vec(seed: float) -> list[float]:
    return [seed + i for i in range(EMBEDDING_DIM)]


def test_roundtrip_cue(store: DuckDBGraphStore) -> None:
    node = make_cue(embedding=vec(0.1))
    store.upsert_node(node)
    assert store.get_node("cue-1") == node


def test_roundtrip_tag(store: DuckDBGraphStore) -> None:
    node = make_tag()
    store.upsert_node(node)
    assert store.get_node("tag-1") == node


def test_roundtrip_content(store: DuckDBGraphStore) -> None:
    node = make_content(embedding=vec(0.2))
    store.upsert_node(node)
    assert store.get_node("content-1") == node


def test_get_node_missing_returns_none(store: DuckDBGraphStore) -> None:
    assert store.get_node("nope") is None


def test_upsert_node_replaces(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue(embedding=vec(0.1)))
    updated = make_cue(embedding=vec(0.9))
    store.upsert_node(updated)
    assert store.get_node("cue-1") == updated


def test_upsert_rejects_wrong_embedding_model(store: DuckDBGraphStore) -> None:
    bad = CueNode(
        id="cue-x",
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="x",
        embedding=vec(0.1),
        embedding_model="other-model",
    )
    with pytest.raises(ValueError, match="does not match store model"):
        store.upsert_node(bad)


def test_upsert_rejects_wrong_embedding_dim(store: DuckDBGraphStore) -> None:
    bad = make_cue(embedding=[0.1, 0.2, 0.3])  # dim 3, store expects 8
    with pytest.raises(ValueError, match="!= store dim"):
        store.upsert_node(bad)


def _edge(src: str, tgt: str, etype: EdgeType = EdgeType.CUE_OF) -> Edge:
    return Edge(source_id=src, target_id=tgt, edge_type=etype, source_file="a.py")


def test_neighbors_outgoing(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    result = store.neighbors("cue-1", direction=Direction.OUTGOING)
    assert [n.id for n in result] == ["content-1"]


def test_neighbors_incoming(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    result = store.neighbors("content-1", direction=Direction.INCOMING)
    assert [n.id for n in result] == ["cue-1"]


def test_neighbors_filters_by_edge_type(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_node(make_tag("tag-1"))
    store.upsert_edge(_edge("cue-1", "content-1", EdgeType.CUE_OF))
    store.upsert_edge(_edge("cue-1", "tag-1", EdgeType.TAGGED_WITH))
    result = store.neighbors("cue-1", edge_type=EdgeType.TAGGED_WITH)
    assert [n.id for n in result] == ["tag-1"]


def test_upsert_edge_replaces(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))  # same triple
    count = store._con.execute("SELECT count(*) FROM edges").fetchone()  # pyright: ignore[reportPrivateUsage]
    assert count is not None and count[0] == 1


def test_delete_node_removes_node_and_incident_edges(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(_edge("cue-1", "content-1"))
    store.delete_node("cue-1")
    assert store.get_node("cue-1") is None
    assert store.get_node("content-1") is not None  # non-target node survives
    edge_count = store._con.execute("SELECT count(*) FROM edges").fetchone()  # pyright: ignore[reportPrivateUsage]
    assert edge_count is not None and edge_count[0] == 0


def test_delete_nodes_for_file_removes_nodes_and_edges(store: DuckDBGraphStore) -> None:
    # Two nodes from a.py, one cross-file REDIRECTS_TO edge into a.py from b.py.
    store.upsert_node(make_cue("cue-1"))  # source_file a.py
    store.upsert_node(make_content("content-1"))  # source_file a.py
    keep = make_tag("tag-keep")
    keep_in_b = keep.model_copy(update={"source_file": "b.py"})
    store.upsert_node(keep_in_b)
    store.upsert_edge(_edge("cue-1", "content-1"))  # file-scoped, a.py
    cross = Edge(
        source_id="tag-keep",
        target_id="cue-1",
        edge_type=EdgeType.REDIRECTS_TO,
        source_file="b.py",
    )
    store.upsert_edge(cross)  # provenance b.py but touches a node in a.py

    store.delete_nodes_for_file("a.py")

    assert store.get_node("cue-1") is None
    assert store.get_node("content-1") is None
    assert store.get_node("tag-keep") is not None  # b.py node survives
    edge_count = store._con.execute("SELECT count(*) FROM edges").fetchone()  # pyright: ignore[reportPrivateUsage]
    assert edge_count is not None and edge_count[0] == 0  # both edges gone


def test_delete_nodes_for_file_clears_null_provenance_edge(store: DuckDBGraphStore) -> None:
    # An edge with source_file=None is removed via the source_id/target_id
    # fallback, not the source_file clause.
    store.upsert_node(make_cue("cue-1"))
    store.upsert_node(make_content("content-1"))
    store.upsert_edge(
        Edge(source_id="cue-1", target_id="content-1", edge_type=EdgeType.CUE_OF)
    )  # source_file defaults to None
    store.delete_nodes_for_file("a.py")
    edge_count = store._con.execute("SELECT count(*) FROM edges").fetchone()  # pyright: ignore[reportPrivateUsage]
    assert edge_count is not None and edge_count[0] == 0


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


def test_vector_search_orders_by_similarity(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-a", embedding=_unit(0)))
    store.upsert_node(make_cue("cue-b", embedding=_unit(1)))
    results = store.vector_search(_unit(0), k=2)
    assert [r.node_id for r in results] == ["cue-a", "cue-b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].node is None


def test_vector_search_respects_k(store: DuckDBGraphStore) -> None:
    for i in range(5):
        store.upsert_node(make_cue(f"cue-{i}", embedding=_unit(i % EMBEDDING_DIM)))
    assert len(store.vector_search(_unit(0), k=2)) == 2


def test_vector_search_filters_by_node_type(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_cue("cue-1", embedding=_unit(0)))
    store.upsert_node(make_content("content-1", embedding=_unit(0)))
    results = store.vector_search(_unit(0), k=5, node_type=NodeType.CUE)
    assert [r.node_id for r in results] == ["cue-1"]


def test_vector_search_skips_null_embeddings(store: DuckDBGraphStore) -> None:
    store.upsert_node(make_tag("tag-1"))  # no embedding
    store.upsert_node(make_cue("cue-1", embedding=_unit(0)))
    results = store.vector_search(_unit(0), k=5)
    assert [r.node_id for r in results] == ["cue-1"]


def test_manifest_record_and_read_sha(store: DuckDBGraphStore) -> None:
    store.record_indexed_file("a.py", "sha1", NOW)
    assert store.indexed_file_sha("a.py") == "sha1"
    assert store.indexed_file_sha("missing.py") is None


def test_manifest_record_replaces_sha(store: DuckDBGraphStore) -> None:
    store.record_indexed_file("a.py", "sha1", NOW)
    store.record_indexed_file("a.py", "sha2", NOW)
    assert store.indexed_file_sha("a.py") == "sha2"


def test_manifest_list(store: DuckDBGraphStore) -> None:
    store.record_indexed_file("a.py", "sha1", NOW)
    store.record_indexed_file("b.py", "sha2", NOW)
    listed = store.list_indexed_files()
    assert {f.file_path for f in listed} == {"a.py", "b.py"}
    assert all(isinstance(f, IndexedFile) for f in listed)


def test_commit_persists(store: DuckDBGraphStore) -> None:
    with store.transaction():
        store.upsert_node(make_cue("cue-1"))
    assert store.get_node("cue-1") is not None


def test_rollback_discards(store: DuckDBGraphStore) -> None:
    try:
        with store.transaction():
            store.upsert_node(make_cue("cue-1"))
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert store.get_node("cue-1") is None


def test_nested_transaction_raises(store: DuckDBGraphStore) -> None:
    store.begin_transaction()
    with pytest.raises(RuntimeError, match="transaction already open"):
        store.begin_transaction()
    store.rollback()


def test_committed_state_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.duckdb"
    s1 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s1.initialize()
    with s1.transaction():
        s1.upsert_node(make_cue("cue-1"))
    s1.close()

    s2 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s2.initialize()
    assert s2.get_node("cue-1") is not None
    s2.close()


def test_uncommitted_state_lost_on_reopen(tmp_path: Path) -> None:
    path = tmp_path / "crash.duckdb"
    s1 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s1.initialize()
    s1.begin_transaction()
    s1.upsert_node(make_cue("cue-1"))
    s1.close()  # die mid-file: never committed

    s2 = DuckDBGraphStore(path, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    s2.initialize()
    assert s2.get_node("cue-1") is None
    s2.close()
