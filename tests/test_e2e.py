"""End-to-end integration test: index the Delfos repo with NativeGraphStore.

Exercises the full pipeline:
  Python files → parser → extractor → hash embedder → NativeGraphStore
  → vector_search + neighbors

Uses a deterministic SHA-256 hash embedder so no API key or network access
is needed.  Acceptance criteria from docs/libdelfos-plan.md §10 Phase 5:

  - Indexer + NativeGraphStore index the Delfos repo itself end-to-end  ✓
  - index() completes < 30 s                                             ✓ (verified by test)
  - vector_search + 100 × neighbors < 5 ms                              ✓ (measured inline)
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import pytest

from delfos.indexer import Embedder, Indexer
from delfos.schema import CueNode, Direction, NodeType
from delfos.store import NativeGraphStore

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic hash embedder — satisfies the Embedder protocol without
# any network access.  Each text maps to a unit vector derived from SHA-256.
# ─────────────────────────────────────────────────────────────────────────────

HASH_DIM = 32  # SHA-256 produces 32 bytes → 32 dimensions
HASH_MODEL = "hash-sha256-d32"


class HashEmbedder:
    """Reproducible embedder for offline testing.  Satisfies the Embedder protocol."""

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
            # Interpret 64 bytes as 64 signed int8 values, normalise to unit sphere.
            raw = [float(b) - 128.0 for b in digest]
            length = math.sqrt(sum(x * x for x in raw)) or 1.0
            results.append([x / length for x in raw])
        return results


assert isinstance(HashEmbedder(), Embedder), "HashEmbedder must satisfy Embedder protocol"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent  # workspace root


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory: pytest.TempPathFactory) -> NativeGraphStore:
    snap_dir = tmp_path_factory.mktemp("delfos_e2e_snap")
    store = NativeGraphStore(snap_dir, embedding_dim=HASH_DIM, embedding_model=HASH_MODEL)
    store.initialize()

    embedder = HashEmbedder()
    indexer = Indexer(store, embedder)

    t0 = time.perf_counter()
    stats = indexer.index(REPO_ROOT)
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"Indexing took {elapsed:.1f}s (limit 30s)"
    assert stats.indexed_files > 0, "No files were indexed"
    assert stats.nodes_written > 0, "No nodes were written"
    assert len(stats.failed_files) == 0, f"Failed files: {stats.failed_files}"

    return store


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_index_completes_and_finds_nodes(indexed_store: NativeGraphStore) -> None:
    # The Delfos repo contains GraphStore, NativeGraphStore, Indexer, etc.
    # We expect at least one of them to be in the store.
    cue = indexed_store.get_node("cue:symbol:delfos/store/base.py::GraphStore")
    assert cue is not None
    assert isinstance(cue, CueNode)


def test_vector_search_returns_results(indexed_store: NativeGraphStore) -> None:
    # Embed the query "GraphStore" and search.
    embedder = HashEmbedder()
    query_emb = embedder.embed(["GraphStore"])[0]
    results = indexed_store.vector_search(query_emb, k=5, node_type=NodeType.CUE)
    assert len(results) > 0
    assert results[0].score > 0.0


def test_neighbors_from_cue_reaches_content(indexed_store: NativeGraphStore) -> None:
    cue_id = "cue:symbol:delfos/store/base.py::GraphStore"
    neighbors = indexed_store.neighbors(cue_id, direction=Direction.OUTGOING)
    # CueNode should have at least one outgoing CUE_OF edge to a ContentNode.
    assert len(neighbors) > 0


def test_manifest_records_indexed_files(indexed_store: NativeGraphStore) -> None:
    files = indexed_store.list_indexed_files()
    paths = {f.file_path for f in files}
    # The store itself and the indexer pipeline must be indexed.
    assert "delfos/store/base.py" in paths
    assert "delfos/indexer/pipeline.py" in paths


def test_incremental_reindex_skips_unchanged(
    indexed_store: NativeGraphStore,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Second run on the same store skips all files (nothing changed)."""
    embedder = HashEmbedder()
    indexer = Indexer(indexed_store, embedder)
    stats = indexer.index(REPO_ROOT)
    assert stats.indexed_files == 0, "Expected all files to be skipped on second run"
    assert stats.skipped_files > 0


def test_vector_search_and_neighbors_latency(indexed_store: NativeGraphStore) -> None:
    """vector_search + 100 sequential neighbors() calls must complete < 5 ms total."""
    embedder = HashEmbedder()
    query_emb = embedder.embed(["initialize"])[0]

    # Warm up
    indexed_store.vector_search(query_emb, k=5, node_type=NodeType.CUE)

    t0 = time.perf_counter()
    hits = indexed_store.vector_search(query_emb, k=5, node_type=NodeType.CUE)
    for hit in hits[:5]:
        for _ in range(20):
            indexed_store.neighbors(hit.node_id, direction=Direction.OUTGOING)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 5.0, (
        f"vector_search + 100 × neighbors took {elapsed_ms:.2f} ms (limit 5 ms)"
    )
