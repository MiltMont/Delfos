"""Pure stdout formatters for CLI commands. No I/O, no graph access — trivially testable."""

from __future__ import annotations

from delfos.indexer import IndexStats
from delfos.store.base import IndexedFile


def render_index_stats(stats: IndexStats) -> str:
    lines = [
        f"indexed: {stats.indexed_files}  skipped: {stats.skipped_files}  "
        f"failed: {len(stats.failed_files)}",
        f"nodes: {stats.nodes_written}  edges: {stats.edges_written}",
    ]
    if stats.failed_files:
        lines.append("failed files:")
        lines.extend(f"  {path}" for path in stats.failed_files)
    return "\n".join(lines)


def render_status(embed_model: str, embed_dim: int, files: list[IndexedFile]) -> str:
    header = f"embedding model: {embed_model} (dim {embed_dim})"
    count = f"{len(files)} file{'s' if len(files) != 1 else ''} indexed"
    if not files:
        return f"{header}\n{count} (empty store)"
    rows = [
        f"  {f.git_sha[:7]}  {f.indexed_at.isoformat()}  {f.file_path}"
        for f in sorted(files, key=lambda f: f.file_path)
    ]
    return "\n".join([header, count, *rows])
