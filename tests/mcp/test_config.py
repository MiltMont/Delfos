from __future__ import annotations

from pathlib import Path

import pytest

from delfos.mcp.config import check_model_match, resolve_config
from delfos.store.native_store import NativeGraphStore


class _Embedder:
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


class _DimEmbedder:
    def __init__(self, model: str, dim: int) -> None:
        self._model = model
        self._dim = dim

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]


def test_resolve_config_uses_defaults_for_fresh_repo(tmp_path: Path) -> None:
    cfg = resolve_config({}, repo_root=tmp_path)
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.embed_dim == 768
    assert cfg.embed_base_url is None
    assert cfg.embed_api_key is None
    assert cfg.send_dimensions is False
    assert cfg.index_path == tmp_path / ".delfos" / "store"
    assert cfg.scip_index_path == tmp_path / ".delfos" / "index.scip"


def test_resolve_config_reads_overrides(tmp_path: Path) -> None:
    cfg = resolve_config(
        {
            "DELFOS_EMBED_MODEL": "text-embedding-3-small",
            "DELFOS_EMBED_DIM": "1536",
            "DELFOS_EMBED_BASE_URL": "http://localhost:11434/v1",
            "DELFOS_EMBED_API_KEY": "ollama",
            "DELFOS_EMBED_SEND_DIM": "1",
        },
        repo_root=tmp_path,
    )
    assert cfg.embed_model == "text-embedding-3-small"
    assert cfg.embed_dim == 1536
    assert cfg.embed_base_url == "http://localhost:11434/v1"
    assert cfg.embed_api_key == "ollama"
    assert cfg.send_dimensions is True


def test_check_model_match_passes_when_equal(tmp_path: Path) -> None:
    store = NativeGraphStore(tmp_path / "g", embedding_dim=8, embedding_model="fake-v1")
    store.initialize()
    check_model_match(store, _Embedder("fake-v1"))  # no raise
    store.close()


def test_check_model_match_raises_on_mismatch(tmp_path: Path) -> None:
    store = NativeGraphStore(tmp_path / "g", embedding_dim=8, embedding_model="fake-v1")
    store.initialize()
    with pytest.raises(RuntimeError, match="fake-v1"):
        check_model_match(store, _Embedder("other-model"))
    store.close()


def test_check_model_match_raises_on_dim_mismatch(tmp_path: Path) -> None:
    store = NativeGraphStore(tmp_path / "g", embedding_dim=8, embedding_model="fake-v1")
    store.initialize()
    with pytest.raises(RuntimeError, match="dimension"):
        check_model_match(store, _DimEmbedder("fake-v1", 16))
    store.close()
