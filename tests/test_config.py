from __future__ import annotations

import pytest

from delfos.config import (
    PlannerConfig,
    build_planner,
    config_from_env,
    planner_config_from_env,
)


def test_config_from_env_still_importable_from_delfos_config() -> None:
    # The embed config moved here; the MCP shim must keep re-exporting it.
    cfg = config_from_env({})
    assert cfg.embed_model == "nomic-embed-text"


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


def test_build_planner_without_model_raises() -> None:
    with pytest.raises(RuntimeError, match="DELFOS_LLM_MODEL"):
        build_planner(PlannerConfig(llm_model=None, llm_base_url=None, llm_api_key=None))


def test_build_planner_returns_planner_for_model() -> None:
    planner = build_planner(
        PlannerConfig(llm_model="local-chat", llm_base_url="http://x/v1", llm_api_key="k")
    )
    assert planner.model == "local-chat"
