from __future__ import annotations

import hashlib
import math
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


HASH_DIM = 32
HASH_MODEL = "hash-sha256-d32"


class HashEmbedder:
    """Deterministic embedder: same text -> same unit vector. No network."""

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
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(b) - 128.0 for b in digest]
            length = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / length for x in raw])
        return out


__all__ = [
    "EMB_DIM",
    "EMB_MODEL",
    "FakeEmbedder",
    "HASH_DIM",
    "HASH_MODEL",
    "HashEmbedder",
    "load",
    "make_content",
    "store",
    "vec",
]
