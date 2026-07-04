"""End-to-end check that a `.env` file at the repo root actually affects CLI config."""

from __future__ import annotations

from pathlib import Path

import pytest

from delfos.cli import app


def test_env_file_at_repo_root_overrides_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".env").write_text("DELFOS_EMBED_MODEL=from-dotenv\nDELFOS_EMBED_DIM=8\n")
    app.main(["status", "--repo", str(tmp_path)])
    out = capsys.readouterr().out
    assert "from-dotenv" in out
    assert "dim 8" in out
