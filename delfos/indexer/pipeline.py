"""The indexing pipeline: walk a repo, extract a graph, persist it atomically.

The :class:`Indexer` ties the pieces together. For each Python file it computes
a per-file content SHA (the git *blob* object id), skips the file when that SHA
already matches the checkpoint manifest, and otherwise re-indexes it inside a
single transaction (``decisions.md`` section 6: one file per transaction is the
atomic unit for crash recovery). Stale handling is delete-and-reindex
(``decisions.md`` section 4): the file's prior nodes are dropped before the new
ones are written, in the same transaction.

Cue embeddings are generated per-file, *inside* the transaction
(``decisions.md`` section 5), so a crash mid-file leaves the store untouched and
the file is retried cleanly on the next run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from delfos.schema import CueNode, Node
from delfos.scip.generate import ScipGenerationError, generate_scip_index
from delfos.scip.reader import ScipIndex
from delfos.store import GraphStore
from delfos.workspace import (
    EmbedInfo,
    GraphInfo,
    Manifest,
    ScipInfo,
    ScipStatus,
    Workspace,
)

from .embedder import Embedder
from .extractor import extract
from .parser import parse_module

logger = logging.getLogger(__name__)

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def _empty_str_list() -> list[str]:
    return []


@dataclass
class IndexStats:
    """Outcome of an :meth:`Indexer.index` run."""

    indexed_files: int = 0
    skipped_files: int = 0
    failed_files: list[str] = field(default_factory=_empty_str_list)
    nodes_written: int = 0
    edges_written: int = 0


def _git_blob_sha(data: bytes) -> str:
    """Return the git blob object id for ``data`` (``sha1("blob <len>\\0" + data)``).

    This matches ``git hash-object`` exactly but needs no git invocation and
    works on a working-tree file regardless of commit state, which is what the
    per-file stale check wants.
    """
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _module_path(relative_path: str) -> str:
    """Convert a posix relative path to a dotted module path.

    ``delfos/schema/nodes.py`` -> ``delfos.schema.nodes``;
    ``delfos/__init__.py`` -> ``delfos``.
    """
    parts = relative_path[: -len(".py")].split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class Indexer:
    """Drives the construction pipeline over a repository.

    The ``store`` and ``embedder`` must agree on the embedding model: the store
    rejects any node whose ``embedding_model`` differs from its configured one.
    """

    def __init__(self, store: GraphStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def index(self, repo_path: str | Path, *, workspace: Workspace | None = None) -> IndexStats:
        """Index every Python file under ``repo_path`` and return run stats.

        Files whose content SHA is unchanged since the last run are skipped.
        Files that fail to parse (syntax errors, non-UTF-8) are recorded in
        :attr:`IndexStats.failed_files` and skipped without aborting the run.

        The ``.delfos/`` workspace is the source of truth: the SCIP index is
        regenerated into it and a :class:`~delfos.workspace.Manifest` stamps the
        run id on both the graph and (when it regenerated) the SCIP index so
        their consistency is detectable afterwards. ``workspace`` defaults to
        ``Workspace(repo_path)`` — the production layout where artifacts live in
        the indexed repo; callers that keep the store elsewhere (tests, ad-hoc
        harnesses) can pass an explicit one to avoid touching the repo.
        """
        root = Path(repo_path).resolve()
        workspace = workspace or Workspace(root)
        workspace.ensure_dirs()
        run_id = uuid.uuid4().hex
        started_at = datetime.now(tz=UTC)

        stats = IndexStats()
        scip, scip_status = self._load_scip_index(root, workspace)
        for path in self._discover(root):
            relative_path = path.relative_to(root).as_posix()
            data = path.read_bytes()
            sha = _git_blob_sha(data)
            if self._store.indexed_file_sha(relative_path) == sha:
                stats.skipped_files += 1
                continue
            if not self._index_file(relative_path, data, sha, stats, scip):
                stats.failed_files.append(relative_path)

        self._write_manifest(workspace, run_id, started_at, scip_status)
        return stats

    def _load_scip_index(
        self, root: Path, workspace: Workspace
    ) -> tuple[ScipIndex | None, ScipStatus]:
        """Regenerate and load the repo's SCIP index plus its status.

        SCIP is a best-effort enrichment: a missing ``scip-python`` binary or a
        generation/parse failure degrades to "no SCIP" (the ``scip_symbol``
        foreign key is left empty) rather than aborting the index run. The
        returned :class:`~delfos.workspace.ScipStatus` records which happened so
        the manifest reflects it.
        """
        try:
            index_path = generate_scip_index(root, workspace.scip_path)
        except ScipGenerationError as exc:
            logger.warning("SCIP generation skipped: %s", exc)
            return None, ScipStatus.ABSENT
        try:
            return ScipIndex(index_path), ScipStatus.PRESENT
        except Exception:
            logger.warning("failed to load SCIP index at %s", index_path, exc_info=True)
            return None, ScipStatus.FAILED

    def _write_manifest(
        self,
        workspace: Workspace,
        run_id: str,
        run_at: datetime,
        scip_status: ScipStatus,
    ) -> None:
        """Persist provenance for this run, preserving prior SCIP run id if stale.

        ``graph.last_run_id`` is always this run. ``scip.run_id`` is this run
        only when SCIP regenerated successfully; otherwise the previous run id
        is kept so :attr:`Manifest.is_consistent` reports the divergence.
        """
        previous = workspace.load_manifest()
        if scip_status is ScipStatus.PRESENT:
            scip = ScipInfo(status=scip_status, run_id=run_id, generated_at=run_at)
        elif previous is not None:
            scip = previous.scip.model_copy(update={"status": scip_status})
        else:
            scip = ScipInfo(status=scip_status)
        manifest = Manifest(
            repo_root=str(workspace.root),
            embed=EmbedInfo(
                model=self._embedder.model,
                dim=self._embedder.dimensions,
                model_version=self._embedder.model_version,
            ),
            graph=GraphInfo(
                last_run_id=run_id,
                updated_at=run_at,
                files=len(self._store.list_indexed_files()),
            ),
            scip=scip,
        )
        workspace.write_manifest(manifest)

    def _index_file(
        self,
        relative_path: str,
        data: bytes,
        sha: str,
        stats: IndexStats,
        scip: ScipIndex | None = None,
    ) -> bool:
        try:
            source = data.decode("utf-8")
            module = parse_module(
                source,
                source_file=relative_path,
                module_path=_module_path(relative_path),
            )
        except (SyntaxError, UnicodeDecodeError):
            return False

        indexed_at = datetime.now(tz=UTC)
        scip_symbols = self._scip_symbols_for(relative_path, scip)
        result = extract(module, git_sha=sha, indexed_at=indexed_at, scip_symbols=scip_symbols)
        with self._store.transaction():
            self._store.delete_nodes_for_file(relative_path)
            nodes = self._embed_cues(result.nodes)
            for node in nodes:
                self._store.upsert_node(node)
            for edge in result.edges:
                self._store.upsert_edge(edge)
            self._store.record_indexed_file(relative_path, sha, indexed_at)

        stats.indexed_files += 1
        stats.nodes_written += len(result.nodes)
        stats.edges_written += len(result.edges)
        return True

    def _embed_cues(self, nodes: list[Node]) -> list[Node]:
        cues: list[tuple[int, CueNode]] = [
            (i, node) for i, node in enumerate(nodes) if isinstance(node, CueNode)
        ]
        if not cues:
            return list(nodes)
        vectors = self._embedder.embed([cue.text for _, cue in cues])
        if len(vectors) != len(cues):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(cues)} cue(s)")
        out: list[Node] = list(nodes)
        for (index, cue), vector in zip(cues, vectors, strict=True):
            out[index] = cue.model_copy(
                update={
                    "embedding": vector,
                    "embedding_model": self._embedder.model,
                    "embedding_model_version": self._embedder.model_version,
                }
            )
        return out

    @staticmethod
    def _scip_symbols_for(relative_path: str, scip: ScipIndex | None) -> Mapping[int, str] | None:
        """Map a file's 1-based definition line numbers to their SCIP symbols.

        The join is positional: tree-sitter records ``lineno = start_point[0] +
        1`` and SCIP places the definition occurrence at the same source line,
        so converting SCIP's 0-based ``start_line`` with ``+ 1`` lets the
        extractor look up symbols directly by ``definition.lineno``.
        """
        if scip is None:
            return None
        return {occ.start_line + 1: occ.symbol for occ in scip.definitions(relative_path)}

    @staticmethod
    def _discover(root: Path) -> list[Path]:
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".")
            ]
            for name in filenames:
                if name.endswith(".py"):
                    found.append(Path(dirpath) / name)
        return sorted(found)
