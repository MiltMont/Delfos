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
