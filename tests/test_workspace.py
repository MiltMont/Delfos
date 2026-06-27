"""Tests for the ``.delfos/`` workspace: path resolution, manifest round-trip,
the graph↔SCIP consistency verdict, and ``config.toml`` parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from delfos.workspace import (
    EmbedInfo,
    GraphInfo,
    Manifest,
    ScipInfo,
    ScipStatus,
    Workspace,
)


def _manifest(*, graph_run: str, scip_run: str | None, scip_status: ScipStatus) -> Manifest:
    return Manifest(
        repo_root="/repo",
        embed=EmbedInfo(model="m", dim=8),
        graph=GraphInfo(last_run_id=graph_run, updated_at=datetime.now(tz=UTC), files=3),
        scip=ScipInfo(status=scip_status, run_id=scip_run),
    )


def test_paths_resolve_under_dot_delfos(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    assert ws.dir == tmp_path / ".delfos"
    assert ws.store_path == tmp_path / ".delfos" / "store"
    assert ws.scip_path == tmp_path / ".delfos" / "index.scip"
    assert ws.manifest_path == tmp_path / ".delfos" / "manifest.json"
    assert ws.config_path == tmp_path / ".delfos" / "config.toml"


def test_ensure_dirs_creates_store(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.ensure_dirs()
    assert ws.store_path.is_dir()


def test_manifest_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    manifest = _manifest(graph_run="r1", scip_run="r1", scip_status=ScipStatus.PRESENT)
    ws.write_manifest(manifest)
    loaded = ws.load_manifest()
    assert loaded == manifest


def test_load_manifest_absent_returns_none(tmp_path: Path) -> None:
    assert Workspace(tmp_path).load_manifest() is None


def test_load_manifest_corrupt_returns_none(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.dir.mkdir(parents=True)
    ws.manifest_path.write_text("{ not json")
    assert ws.load_manifest() is None


def test_is_consistent_when_runs_match_and_present() -> None:
    assert _manifest(graph_run="r1", scip_run="r1", scip_status=ScipStatus.PRESENT).is_consistent


def test_is_inconsistent_when_runs_differ() -> None:
    assert not _manifest(
        graph_run="r2", scip_run="r1", scip_status=ScipStatus.PRESENT
    ).is_consistent


def test_is_inconsistent_when_scip_absent() -> None:
    assert not _manifest(graph_run="r1", scip_run=None, scip_status=ScipStatus.ABSENT).is_consistent


def test_load_config_absent_returns_empty(tmp_path: Path) -> None:
    assert Workspace(tmp_path).load_config() == {}


def test_load_config_maps_toml_to_env_keys(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.dir.mkdir(parents=True)
    ws.config_path.write_text(
        "[embed]\n"
        'model = "nomic-embed-text"\n'
        "dim = 768\n"
        'base_url = "http://localhost:11434/v1"\n'
        "send_dimensions = true\n"
        "[llm]\n"
        'model = "qwen2.5"\n'
    )
    assert ws.load_config() == {
        "DELFOS_EMBED_MODEL": "nomic-embed-text",
        "DELFOS_EMBED_DIM": "768",
        "DELFOS_EMBED_BASE_URL": "http://localhost:11434/v1",
        "DELFOS_EMBED_SEND_DIM": "1",
        "DELFOS_LLM_MODEL": "qwen2.5",
    }
