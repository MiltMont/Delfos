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
from delfos.workspace import Workspace

_TRUTHY = {"1", "true", "yes", "on"}

_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_EMBED_DIM = 768


@dataclass(frozen=True)
class ServerConfig:
    """Resolved server configuration, anchored on a repo's ``.delfos/`` workspace."""

    workspace: Workspace
    embed_model: str
    embed_dim: int
    embed_base_url: str | None
    embed_api_key: str | None
    send_dimensions: bool

    @property
    def index_path(self) -> Path:
        return self.workspace.store_path

    @property
    def scip_index_path(self) -> Path:
        return self.workspace.scip_path


def resolve_config(env: Mapping[str, str], *, repo_root: str | Path = ".") -> ServerConfig:
    """Resolve config for ``repo_root``'s workspace, merging env, ``config.toml``,
    and the manifest.

    Precedence (highest first): environment variables, ``.delfos/config.toml``,
    the manifest's recorded ``embed`` info (for ``model``/``dim`` only — these
    must match what the index was built with), then built-in defaults. This is
    why a query against an already-indexed repo needs only credentials in the
    environment.
    """
    workspace = Workspace(repo_root)
    merged: dict[str, str] = {**workspace.load_config(), **env}
    manifest = workspace.load_manifest()
    default_model = manifest.embed.model if manifest else _DEFAULT_EMBED_MODEL
    default_dim = manifest.embed.dim if manifest else _DEFAULT_EMBED_DIM
    return ServerConfig(
        workspace=workspace,
        embed_model=merged.get("DELFOS_EMBED_MODEL", default_model),
        embed_dim=int(merged.get("DELFOS_EMBED_DIM", str(default_dim))),
        embed_base_url=merged.get("DELFOS_EMBED_BASE_URL"),
        embed_api_key=merged.get("DELFOS_EMBED_API_KEY"),
        send_dimensions=merged.get("DELFOS_EMBED_SEND_DIM", "0").strip().lower() in _TRUTHY,
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
    """Open the persisted graph store in the workspace."""
    cfg.workspace.ensure_dirs()
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


def planner_config_from_merged(
    env: Mapping[str, str], *, repo_root: str | Path = "."
) -> PlannerConfig:
    """Like :func:`planner_config_from_env` but also reads ``.delfos/config.toml``."""
    merged: dict[str, str] = {**Workspace(repo_root).load_config(), **env}
    return planner_config_from_env(merged)


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
