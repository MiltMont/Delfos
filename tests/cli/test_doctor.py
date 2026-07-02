"""Tests for the `doctor` checks and their rendering."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from delfos.cli.app import run_doctor
from delfos.cli.render import Check, render_doctor
from delfos.scip import generate
from delfos.workspace import EmbedInfo, GraphInfo, Manifest, ScipInfo, ScipStatus, Workspace


def _no_scip(_name: str) -> str | None:
    return None


def _has_scip(_name: str) -> str | None:
    return "/usr/bin/scip-python"


def _manifest(root: str, *, scip_run: str | None, status: ScipStatus) -> Manifest:
    return Manifest(
        repo_root=root,
        embed=EmbedInfo(model="nomic-embed-text", dim=768),
        graph=GraphInfo(last_run_id="r1", updated_at=datetime.now(tz=UTC), files=1),
        scip=ScipInfo(status=status, run_id=scip_run),
    )


def test_doctor_reports_missing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate.shutil, "which", _no_scip)
    ws = Workspace(tmp_path)
    checks = run_doctor(ws, None, "nomic-embed-text")
    by_name = {c.name: c for c in checks}
    assert by_name["scip-python on PATH"].ok is False
    assert by_name["manifest"].ok is False


def test_doctor_flags_stale_scip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate.shutil, "which", _has_scip)
    ws = Workspace(tmp_path)
    ws.ensure_dirs()
    ws.scip_path.write_bytes(b"")
    manifest = _manifest(str(ws.root), scip_run="OLD", status=ScipStatus.PRESENT)
    checks = run_doctor(ws, manifest, "nomic-embed-text")
    by_name = {c.name: c for c in checks}
    assert by_name["scip-python on PATH"].ok is True
    assert by_name["graph/scip consistent"].ok is False


def test_doctor_all_green(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate.shutil, "which", _has_scip)
    ws = Workspace(tmp_path)
    ws.ensure_dirs()
    ws.scip_path.write_bytes(b"")
    manifest = _manifest(str(ws.root), scip_run="r1", status=ScipStatus.PRESENT)
    checks = run_doctor(ws, manifest, "nomic-embed-text")
    assert all(c.ok for c in checks)


def test_render_doctor_marks_pass_and_fail() -> None:
    out = render_doctor([Check("a", True, "ok detail"), Check("b", False, "fix me")])
    assert "[ok  ] a  (ok detail)" in out
    assert "[FAIL] b  (fix me)" in out
