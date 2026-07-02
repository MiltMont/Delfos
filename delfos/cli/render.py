"""Pure stdout formatters for CLI commands. No I/O, no graph access — trivially testable."""

from __future__ import annotations

from dataclasses import dataclass

from delfos.indexer import IndexStats
from delfos.schema import ContentNode, CueNode
from delfos.store.base import IndexedFile
from delfos.workspace import Manifest


@dataclass(frozen=True)
class Check:
    """One ``doctor`` verification result."""

    name: str
    ok: bool
    detail: str = ""


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


def render_status(
    embed_model: str, embed_dim: int, files: list[IndexedFile], manifest: Manifest | None = None
) -> str:
    header = f"embedding model: {embed_model} (dim {embed_dim})"
    lines = [header, *_manifest_lines(manifest)]
    count = f"{len(files)} file{'s' if len(files) != 1 else ''} indexed"
    if not files:
        lines.append(f"{count} (empty store)")
        return "\n".join(lines)
    lines.append(count)
    lines.extend(
        f"  {f.git_sha[:7]}  {f.indexed_at.isoformat()}  {f.file_path}"
        for f in sorted(files, key=lambda f: f.file_path)
    )
    return "\n".join(lines)


def _manifest_lines(manifest: Manifest | None) -> list[str]:
    if manifest is None:
        return ["no manifest (run `delfos index`)"]
    verdict = "consistent" if manifest.is_consistent else "STALE (graph/scip run mismatch)"
    return [
        f"scip: {manifest.scip.status.value}  graph/scip: {verdict}",
    ]


def render_doctor(checks: list[Check]) -> str:
    lines: list[str] = []
    for check in checks:
        mark = "ok  " if check.ok else "FAIL"
        suffix = f"  ({check.detail})" if check.detail else ""
        lines.append(f"[{mark}] {check.name}{suffix}")
    return "\n".join(lines)


def render_search(cues: list[CueNode]) -> str:
    if not cues:
        return "no matching cues"
    return "\n".join(f"  {c.id}  {c.text}" for c in cues)


def render_reconstruct(contents: list[ContentNode]) -> str:
    if not contents:
        return "no content reconstructed"
    lines: list[str] = []
    for c in contents:
        label = c.signature or c.symbol_name or c.kind.value
        lines.append(f"  {c.id}  [{c.memory_layer.value}] {c.source_file}: {label}")
    return "\n".join(lines)
