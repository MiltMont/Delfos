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
    Edge,
    EdgeType,
    MemoryLayer,
    Node,
    TagCategory,
    TagNode,
)
from delfos.store.native_store import NativeGraphStore

EMB_DIM = 8
EMB_MODEL = "fake-v1"
NOW = datetime(2026, 6, 23, 12, 0, 0)


class FakeEmbedder:
    """Embedder protocol double: maps known texts to fixed vectors."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    @property
    def model(self) -> str:
        return EMB_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return EMB_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[t] for t in texts]


def vec(seed: float) -> list[float]:
    return [seed + i for i in range(EMB_DIM)]


def make_cue(node_id: str, text: str, *, embedding: list[float] | None = None) -> CueNode:
    return CueNode(
        id=node_id,
        source_file="a.py",
        git_sha="s",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text=text,
        embedding=embedding,
        embedding_model=EMB_MODEL if embedding is not None else None,
    )


def make_content(node_id: str, symbol: str) -> ContentNode:
    return ContentNode(
        id=node_id,
        source_file="a.py",
        git_sha="s",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name=symbol,
        signature=f"def {symbol}()",
        docstring=None,
        body=f"def {symbol}(): ...",
    )


def make_tag(node_id: str, category: TagCategory, value: str) -> TagNode:
    return TagNode(id=node_id, indexed_at=NOW, category=category, value=value)


def edge(source: str, target: str, edge_type: EdgeType) -> Edge:
    return Edge(source_id=source, target_id=target, edge_type=edge_type)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    yield s
    s.close()


def load(store: NativeGraphStore, nodes: list[Node], edges: list[Edge]) -> None:
    """Persist a fixture graph in one transaction."""
    with store.transaction():
        for node in nodes:
            store.upsert_node(node)
        for e in edges:
            store.upsert_edge(e)
