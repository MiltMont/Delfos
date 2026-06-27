from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from delfos.config import (
    PlannerConfig,
    build_planner,
    planner_config_from_env,
    planner_config_from_merged,
    resolve_config,
)
from delfos.workspace import EmbedInfo, GraphInfo, Manifest, ScipInfo, ScipStatus, Workspace


def test_resolve_config_defaults_for_fresh_repo(tmp_path: Path) -> None:
    # No manifest yet: fall back to built-in embed defaults.
    cfg = resolve_config({}, repo_root=tmp_path)
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.embed_dim == 768
    assert cfg.index_path == tmp_path / ".delfos" / "store"
    assert cfg.scip_index_path == tmp_path / ".delfos" / "index.scip"


def test_resolve_config_reads_embed_defaults_from_manifest(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write_manifest(
        Manifest(
            repo_root=str(ws.root),
            embed=EmbedInfo(model="text-embedding-3-small", dim=1536),
            graph=GraphInfo(last_run_id="r1", updated_at=datetime.now(tz=UTC), files=0),
            scip=ScipInfo(status=ScipStatus.ABSENT),
        )
    )
    cfg = resolve_config({}, repo_root=tmp_path)
    assert cfg.embed_model == "text-embedding-3-small"
    assert cfg.embed_dim == 1536


def test_resolve_config_env_overrides_manifest_and_config_file(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.dir.mkdir(parents=True)
    ws.config_path.write_text('[embed]\nmodel = "from-config"\ndim = 256\n')
    # config.toml supplies a model; the environment overrides it.
    cfg = resolve_config({"DELFOS_EMBED_MODEL": "from-env"}, repo_root=tmp_path)
    assert cfg.embed_model == "from-env"
    # dim still comes from config.toml (env did not set it).
    assert cfg.embed_dim == 256


def test_planner_config_defaults_to_none_model() -> None:
    assert planner_config_from_env({}) == PlannerConfig(
        llm_model=None, llm_base_url=None, llm_api_key=None
    )


def test_planner_config_reads_overrides() -> None:
    cfg = planner_config_from_env(
        {
            "DELFOS_LLM_MODEL": "local-chat",
            "DELFOS_LLM_BASE_URL": "http://localhost:8080/v1",
            "DELFOS_LLM_API_KEY": "local",
        }
    )
    assert cfg == PlannerConfig(
        llm_model="local-chat",
        llm_base_url="http://localhost:8080/v1",
        llm_api_key="local",
    )


def test_planner_config_from_merged_reads_config_file(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.dir.mkdir(parents=True)
    ws.config_path.write_text('[llm]\nmodel = "config-chat"\n')
    cfg = planner_config_from_merged({}, repo_root=tmp_path)
    assert cfg.llm_model == "config-chat"


def test_build_planner_without_model_raises() -> None:
    with pytest.raises(RuntimeError, match="DELFOS_LLM_MODEL"):
        build_planner(PlannerConfig(llm_model=None, llm_base_url=None, llm_api_key=None))


def test_build_planner_returns_planner_for_model() -> None:
    planner = build_planner(
        PlannerConfig(llm_model="local-chat", llm_base_url="http://x/v1", llm_api_key="k")
    )
    assert planner.model == "local-chat"
