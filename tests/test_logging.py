"""Tests for CLI step logging."""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import pytest

from delfos._logging import configure_cli_logging
from delfos.indexer import Indexer
from delfos.store.native_store import NativeGraphStore
from delfos.workspace import Workspace

EMBEDDING_DIM = 32
EMBEDDING_MODEL = "hash-sha256-d32"


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


def test_index_emits_numbered_step_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def foo():\n    return 1\n")
    store = NativeGraphStore(
        tmp_path / "snap", embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    store.initialize()

    with caplog.at_level(logging.INFO, logger="delfos"):
        Indexer(store, _HashEmbedder()).index(repo, workspace=Workspace(tmp_path / "ws"))
    store.close()

    messages = [r.getMessage() for r in caplog.records]
    assert any("[1/4]" in m for m in messages)
    assert any("[2/4]" in m for m in messages)
    assert any("[3/4]" in m for m in messages)
    assert any("[4/4]" in m for m in messages)


def test_configure_cli_logging_is_idempotent() -> None:
    logger = logging.getLogger("delfos")
    before = list(logger.handlers)
    configure_cli_logging()
    configure_cli_logging()
    added = [h for h in logger.handlers if h not in before]
    # Exactly one stderr handler is installed regardless of call count.
    assert len([h for h in added if h.name == "delfos-cli-stderr"]) <= 1
    assert any(h.name == "delfos-cli-stderr" for h in logger.handlers)
