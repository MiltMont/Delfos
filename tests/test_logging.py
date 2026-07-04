"""Tests for CLI step logging."""

from __future__ import annotations

import hashlib
import logging
import math
import types
from pathlib import Path

import pytest

import delfos.mcp.__main__ as mcp_main
from delfos._logging import configure_cli_logging
from delfos.cli import app
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


def test_cli_serve_passes_verbose_and_repo_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_serve(repo_root: str | None = None, *, verbose: bool | None = None) -> None:
        captured["repo_root"] = repo_root
        captured["verbose"] = verbose

    monkeypatch.setattr(mcp_main, "main", _fake_serve)
    app.main(["-v", "serve", "--repo", str(tmp_path)])
    assert captured["repo_root"] == str(tmp_path)
    assert captured["verbose"] is True

    app.main(["serve", "--repo", str(tmp_path)])
    assert captured["verbose"] is False


def test_mcp_main_honors_verbose_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # `delfos -v serve` reaches mcp main via DELFOS_VERBOSE; it must keep DEBUG.
    monkeypatch.setenv("DELFOS_VERBOSE", "1")
    cfg = types.SimpleNamespace(index_path="x", embed_model="m", embed_dim=8)

    def _resolve_config(*_a: object, **_k: object) -> object:
        return cfg

    def _build_obj(*_a: object, **_k: object) -> object:
        return object()

    def _noop(*_a: object, **_k: object) -> None:
        return None

    def _build_server(*_a: object, **_k: object) -> object:
        return types.SimpleNamespace(run=_noop)

    monkeypatch.setattr(mcp_main, "resolve_config", _resolve_config)
    monkeypatch.setattr(mcp_main, "build_store", _build_obj)
    monkeypatch.setattr(mcp_main, "build_embedder", _build_obj)
    monkeypatch.setattr(mcp_main, "check_model_match", _noop)
    monkeypatch.setattr(mcp_main, "ReconstructionService", _build_obj)
    monkeypatch.setattr(mcp_main, "build_scip_service", _noop)
    monkeypatch.setattr(mcp_main, "build_server", _build_server)

    mcp_main.main()
    assert logging.getLogger("delfos").level == logging.DEBUG
