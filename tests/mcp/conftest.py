from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from delfos.reconstruct import ReconstructionService
from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import EMB_DIM, EMB_MODEL, FakeEmbedder, vec


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    yield s
    s.close()


def make_service(store: NativeGraphStore, query_vec: list[float]) -> ReconstructionService:
    """Service over the seeded store, with an embedder that maps 'q' -> query_vec."""
    return ReconstructionService(store, FakeEmbedder({"q": query_vec}))


__all__ = ["store", "make_service", "vec"]
