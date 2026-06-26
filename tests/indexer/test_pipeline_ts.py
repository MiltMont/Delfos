from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from delfos.indexer import Embedder, Indexer
from delfos.indexer.pipeline import _module_path  # pyright: ignore[reportPrivateUsage]
from delfos.store import NativeGraphStore

# ── Deterministic embedder (no network) ──────────────────────────────────────

HASH_DIM = 32
HASH_MODEL = "hash-sha256-d32"


class HashEmbedder:
    @property
    def model(self) -> str:
        return HASH_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return HASH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(b) - 128.0 for b in digest]
            length = math.sqrt(sum(x * x for x in raw)) or 1.0
            results.append([x / length for x in raw])
        return results


assert isinstance(HashEmbedder(), Embedder)


# ── _module_path unit tests ────────────────────────────────────────────────────


def test_module_path_ts_regular() -> None:
    assert _module_path("src/utils.ts") == "src.utils"


def test_module_path_ts_index() -> None:
    assert _module_path("src/index.ts") == "src"


def test_module_path_tsx_regular() -> None:
    assert _module_path("src/App.tsx") == "src.App"


def test_module_path_tsx_index() -> None:
    assert _module_path("src/index.tsx") == "src"


def test_module_path_py_regular() -> None:
    assert _module_path("delfos/store/base.py") == "delfos.store.base"


def test_module_path_py_init() -> None:
    assert _module_path("delfos/__init__.py") == "delfos"


# ── Pipeline integration ───────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> NativeGraphStore:
    s = NativeGraphStore(tmp_path / "store", embedding_dim=HASH_DIM, embedding_model=HASH_MODEL)
    s.initialize()
    return s


def test_discovers_ts_files(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def hello(): pass")
    (tmp_path / "app.ts").write_text("function greet(): void {}")
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.indexed_files == 2
    assert len(stats.failed_files) == 0


def test_discovers_tsx_files(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "App.tsx").write_text("const App = (): JSX.Element => <div/>;")
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.indexed_files == 1


def test_ts_nodes_written_to_store(store: NativeGraphStore, tmp_path: Path) -> None:
    src = "export function greet(name: string): string { return name; }"
    (tmp_path / "greeter.ts").write_text(src)
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.nodes_written > 0
    cue = store.get_node("cue:symbol:greeter.ts::greet")
    assert cue is not None


def test_python_files_still_indexed(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "util.py").write_text("def helper(): pass")
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.indexed_files == 1
    cue = store.get_node("cue:symbol:util.py::helper")
    assert cue is not None


def test_ts_incremental_skip(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "app.ts").write_text("function foo(): void {}")
    indexer = Indexer(store, HashEmbedder())
    indexer.index(tmp_path)
    stats2 = indexer.index(tmp_path)
    assert stats2.indexed_files == 0
    assert stats2.skipped_files == 1
