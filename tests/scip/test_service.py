"""Tests for the SCIP read-path bridge (delfos.scip.service.ScipService)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from delfos.schema import ContentKind, ContentNode, MemoryLayer
from delfos.scip.reader import ScipIndex
from delfos.scip.service import ScipService
from delfos.store.native_store import NativeGraphStore

from .builders import (
    document,
    occurrence,
    relationship,
    symbol_information,
    write_index,
)

EMBEDDING_DIM = 8
EMBEDDING_MODEL = "fake-v1"
NOW = datetime(2026, 6, 22, 12, 0, 0)

SYM = "scip-python python . a/foo()."


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(
        tmp_path / "snap", embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    s.initialize()
    yield s
    s.close()


def _content(node_id: str, scip_symbol: str | None) -> ContentNode:
    return ContentNode(
        id=node_id,
        source_file="a.py",
        git_sha="sha1",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="foo",
        scip_symbol=scip_symbol,
        body="def foo(): ...",
    )


def _index(path: Path) -> ScipIndex:
    write_index(
        path,
        documents=[
            document(
                "a.py",
                occurrences=[
                    occurrence(SYM, 0, definition=True),
                    occurrence(SYM, 20),
                ],
                symbols=[
                    symbol_information(
                        SYM,
                        [
                            relationship("iface#", is_implementation=True),
                            relationship("Type#", is_type_definition=True),
                        ],
                    )
                ],
            ),
            document("b.py", occurrences=[occurrence(SYM, 5)]),
        ],
    )
    return ScipIndex(path)


def test_references_resolves_symbol_and_excludes_definition(
    store: NativeGraphStore, tmp_path: Path
) -> None:
    store.upsert_node(_content("content:a.py::foo", SYM))
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    refs = svc.references("content:a.py::foo")
    assert {(path, occ.start_line) for path, occ in refs} == {("a.py", 20), ("b.py", 5)}


def test_implementations_and_type_definition(store: NativeGraphStore, tmp_path: Path) -> None:
    store.upsert_node(_content("content:a.py::foo", SYM))
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    assert [r.symbol for r in svc.implementations("content:a.py::foo")] == ["iface#"]
    assert [r.symbol for r in svc.type_definition("content:a.py::foo")] == ["Type#"]


def test_node_without_scip_symbol_returns_empty(store: NativeGraphStore, tmp_path: Path) -> None:
    store.upsert_node(_content("content:a.py::foo", None))
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    assert svc.references("content:a.py::foo") == []
    assert svc.implementations("content:a.py::foo") == []
    assert svc.type_definition("content:a.py::foo") == []


def test_unknown_content_id_raises(store: NativeGraphStore, tmp_path: Path) -> None:
    svc = ScipService(store, _index(tmp_path / "index.scip"))
    with pytest.raises(ValueError, match="no content node"):
        svc.references("content:missing")
