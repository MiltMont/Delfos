from __future__ import annotations

from datetime import UTC, datetime

from delfos.cli.render import render_index_stats, render_status
from delfos.indexer import IndexStats
from delfos.store.base import IndexedFile


def test_render_index_stats_reports_all_counters() -> None:
    stats = IndexStats(
        indexed_files=3,
        skipped_files=1,
        failed_files=["bad.py"],
        nodes_written=42,
        edges_written=40,
    )
    out = render_index_stats(stats)
    assert "indexed: 3" in out
    assert "skipped: 1" in out
    assert "nodes: 42" in out
    assert "edges: 40" in out
    assert "bad.py" in out  # failures are named, not just counted


def test_render_status_lists_model_and_files() -> None:
    files = [
        IndexedFile(
            file_path="a.py",
            git_sha="abcdef123456",
            indexed_at=datetime(2026, 6, 25, tzinfo=UTC),
        ),
    ]
    out = render_status("nomic-embed-text", 768, files)
    assert "nomic-embed-text" in out
    assert "768" in out
    assert "a.py" in out
    assert "abcdef1" in out  # short sha
    assert "1 file" in out  # count summary


def test_render_status_handles_empty_store() -> None:
    out = render_status("m", 8, [])
    assert "0 files" in out or "empty" in out.lower()
