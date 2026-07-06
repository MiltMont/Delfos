from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import (
    EMB_DIM,
    EMB_MODEL,
    FakeEmbedder,
    load,
    make_content,
    vec,
)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[NativeGraphStore]:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    yield s
    s.close()


__all__ = ["EMB_DIM", "EMB_MODEL", "FakeEmbedder", "load", "make_content", "store", "vec"]
