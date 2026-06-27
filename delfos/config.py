"""Env-driven startup configuration shared by the MCP server and the CLI.

Reuses the smoke harness's ``DELFOS_EMBED_*`` convention. The embedder is the
read/write path's only model dependency; point ``DELFOS_EMBED_BASE_URL`` at a
local OpenAI-compatible endpoint or leave it unset for OpenAI-hosted. The
query-time embedding model must match the index-time one;
:func:`check_model_match` enforces this at startup.

``reconstruct`` additionally needs a chat model (the hop planner), resolved from
the ``DELFOS_LLM_*`` settings via :func:`planner_config_from_env` /
:func:`build_planner`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from delfos.indexer import OpenAIEmbedder
from delfos.indexer.embedder import Embedder
from delfos.reconstruct.planners.openai import OpenAIHopPlanner
from delfos.scip.reader import ScipIndex
from delfos.scip.service import ScipService
from delfos.store import GraphStore, NativeGraphStore

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerConfig:
    """Resolved server configuration."""

    index_path: Path
    scip_index_path: Path
    embed_model: str
    embed_dim: int
    embed_base_url: str | None
    embed_api_key: str | None
    send_dimensions: bool


def config_from_env(env: Mapping[str, str]) -> ServerConfig:
    """Build a :class:`ServerConfig` from environment variables (with defaults)."""
    return ServerConfig(
        index_path=Path(env.get("DELFOS_INDEX_PATH", "delfos/store")),
        scip_index_path=Path(env.get("DELFOS_SCIP_PATH", "index.scip")),
        embed_model=env.get("DELFOS_EMBED_MODEL", "nomic-embed-text"),
        embed_dim=int(env.get("DELFOS_EMBED_DIM", "768")),
        embed_base_url=env.get("DELFOS_EMBED_BASE_URL"),
        embed_api_key=env.get("DELFOS_EMBED_API_KEY"),
        send_dimensions=env.get("DELFOS_EMBED_SEND_DIM", "0").strip().lower() in _TRUTHY,
    )


def build_embedder(cfg: ServerConfig) -> OpenAIEmbedder:
    """Construct the OpenAI-compatible embedder from config."""
    client = OpenAI(base_url=cfg.embed_base_url, api_key=cfg.embed_api_key)
    return OpenAIEmbedder(
        cfg.embed_model,
        dimensions=cfg.embed_dim,
        send_dimensions=cfg.send_dimensions,
        client=client,
    )


def build_store(cfg: ServerConfig) -> NativeGraphStore:
    """Open the persisted graph store at the configured path."""
    store = NativeGraphStore(
        cfg.index_path, embedding_dim=cfg.embed_dim, embedding_model=cfg.embed_model
    )
    store.initialize()
    return store


def build_scip_service(cfg: ServerConfig, store: GraphStore) -> ScipService | None:
    """Load the SCIP index and wrap it in a :class:`ScipService`, if available.

    Returns ``None`` when no index file exists at ``cfg.scip_index_path`` or it
    fails to parse — the MCP SCIP tools then report an actionable error instead
    of failing the whole server.
    """
    if not cfg.scip_index_path.is_file():
        return None
    try:
        index = ScipIndex(cfg.scip_index_path)
    except Exception:
        return None
    return ScipService(store, index)


def check_model_match(store: NativeGraphStore, embedder: Embedder) -> None:
    """Fail fast unless the embedder's model matches the store's index model."""
    if store.embedding_model != embedder.model:
        raise RuntimeError(
            f"embedder model {embedder.model!r} does not match index model "
            f"{store.embedding_model!r}; queries must use the model the index "
            f"was built with"
        )


@dataclass(frozen=True)
class PlannerConfig:
    """Hop-planner (chat LLM) configuration for `reconstruct`."""

    llm_model: str | None
    llm_base_url: str | None
    llm_api_key: str | None


def planner_config_from_env(env: Mapping[str, str]) -> PlannerConfig:
    """Read the `DELFOS_LLM_*` chat-model settings (all optional)."""
    return PlannerConfig(
        llm_model=env.get("DELFOS_LLM_MODEL"),
        llm_base_url=env.get("DELFOS_LLM_BASE_URL"),
        llm_api_key=env.get("DELFOS_LLM_API_KEY"),
    )


def build_planner(cfg: PlannerConfig) -> OpenAIHopPlanner:
    """Construct the OpenAI-compatible hop planner; require a model name."""
    if cfg.llm_model is None:
        raise RuntimeError(
            "reconstruct needs a chat model; set DELFOS_LLM_MODEL "
            "(and DELFOS_LLM_BASE_URL for a local endpoint)"
        )
    client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    return OpenAIHopPlanner(cfg.llm_model, client=client)
