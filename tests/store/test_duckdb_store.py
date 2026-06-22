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
    TagCategory,
    TagNode,
)
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
        for row in store._con.execute(  # type: ignore[reportPrivateUsage]
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
    count = store._con.execute("SELECT count(*) FROM edges").fetchone()  # type: ignore[reportPrivateUsage]
    assert count is not None and count[0] == 1
