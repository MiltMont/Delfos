"""Tests for SCIP wiring in the indexer pipeline.

Covers the positional (1-based tree-sitter ``lineno`` ↔ 0-based SCIP
``start_line``) join, graceful degradation when generation fails, and the
end-to-end population of ``ContentNode.scip_symbol`` at index time.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from delfos.indexer import Indexer
from delfos.indexer import pipeline as pipeline_mod
from delfos.schema import ContentNode
from delfos.scip.generate import ScipGenerationError
from delfos.scip.reader import ScipIndex
from delfos.store.native_store import NativeGraphStore
from tests.scip.builders import document, occurrence, write_index

EMBEDDING_DIM = 32
EMBEDDING_MODEL = "hash-sha256-d32"

SYM = "scip-python python . mod/foo()."


class _HashEmbedder:
    @property
    def model(self) -> str:
        return EMBEDDING_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(b) - 128.0 for b in digest]
            length = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / length for x in raw])
        return out


def test_scip_symbols_for_converts_zero_based_to_one_based(tmp_path: Path) -> None:
    write_index(
        tmp_path / "index.scip",
        documents=[
            document(
                "mod.py",
                occurrences=[
                    occurrence(SYM, 0, definition=True),  # 0-based SCIP line
                    occurrence("scip-python python . mod/bar().", 4, definition=True),
                    occurrence(SYM, 9),  # a non-definition usage is ignored here
                ],
            )
        ],
    )
    idx = ScipIndex(tmp_path / "index.scip")
    mapping = Indexer._scip_symbols_for("mod.py", idx)  # pyright: ignore[reportPrivateUsage]
    assert mapping == {1: SYM, 5: "scip-python python . mod/bar()."}


def test_scip_symbols_for_none_index_returns_none() -> None:
    assert Indexer._scip_symbols_for("mod.py", None) is None  # pyright: ignore[reportPrivateUsage]


def test_load_scip_index_degrades_when_generation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(_root: Path) -> Path:
        raise ScipGenerationError("scip-python not found on PATH")

    monkeypatch.setattr(pipeline_mod, "generate_scip_index", _raise)
    store = NativeGraphStore(
        tmp_path / "snap", embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    store.initialize()
    indexer = Indexer(store, _HashEmbedder())
    assert indexer._load_scip_index(tmp_path) is None  # pyright: ignore[reportPrivateUsage]
    store.close()


def test_index_populates_scip_symbol_by_lineno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def foo():\n    return 1\n")

    idx = write_index(
        tmp_path / "index.scip",
        documents=[document("mod.py", occurrences=[occurrence(SYM, 0, definition=True)])],
    )
    scip = ScipIndex(idx)

    def _fake_load(_self: Indexer, _root: Path) -> ScipIndex:
        return scip

    monkeypatch.setattr(Indexer, "_load_scip_index", _fake_load)

    store = NativeGraphStore(
        tmp_path / "snap", embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    store.initialize()
    indexer = Indexer(store, _HashEmbedder())
    stats = indexer.index(repo)
    assert stats.indexed_files == 1

    foo = store.get_node("content:mod.py::foo")
    assert isinstance(foo, ContentNode)
    assert foo.scip_symbol == SYM

    # The module node has no matching SCIP definition line → no symbol.
    module = store.get_node("content:mod.py::<module>")
    assert isinstance(module, ContentNode)
    assert module.scip_symbol is None

    store.close()
