"""Tests for SCIP generation and its graceful-degradation contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from delfos.scip import generate
from delfos.scip.generate import (
    ScipGenerationError,
    generate_scip_index,
    scip_binary_available,
)


def test_scip_binary_available_reflects_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def _absent(_name: str) -> str | None:
        return None

    def _present(_name: str) -> str | None:
        return "/usr/bin/scip-python"

    monkeypatch.setattr(generate.shutil, "which", _absent)
    assert scip_binary_available() is False
    monkeypatch.setattr(generate.shutil, "which", _present)
    assert scip_binary_available() is True


def test_missing_binary_raises_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _which(_name: str) -> str | None:
        return None

    monkeypatch.setattr(generate.shutil, "which", _which)
    with pytest.raises(ScipGenerationError, match="not found on PATH"):
        generate_scip_index(tmp_path, tmp_path / "index.scip")


def test_nonzero_exit_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_name: str) -> str | None:
        return "/usr/bin/scip-python"

    def _run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(2, "scip-python", stderr="boom")

    monkeypatch.setattr(generate.shutil, "which", _which)
    monkeypatch.setattr(generate.subprocess, "run", _run)
    with pytest.raises(ScipGenerationError, match="boom"):
        generate_scip_index(tmp_path, tmp_path / "index.scip")


def test_missing_output_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(_name: str) -> str | None:
        return "/usr/bin/scip-python"

    def _run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        # "succeeds" but writes no index file.
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr(generate.shutil, "which", _which)
    monkeypatch.setattr(generate.subprocess, "run", _run)
    with pytest.raises(ScipGenerationError, match="did not produce"):
        generate_scip_index(tmp_path, tmp_path / "index.scip")
