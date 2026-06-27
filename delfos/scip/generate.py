"""Generate a SCIP index for a repository by shelling out to ``scip-python``.

SCIP indexing is whole-repo: a single ``scip-python index`` invocation walks the
project and emits one ``index.scip`` protobuf. The indexer runs this as a
pre-pass before its per-file loop, then loads the result with
:class:`~delfos.scip.reader.ScipIndex`.

Generation is *best effort*: ``scip-python`` is an external Node.js tool that
may not be installed. Callers treat a :class:`ScipGenerationError` as "no SCIP
available" and degrade gracefully (the ``scip_symbol`` foreign key is left
empty) rather than failing the whole index run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Canonical, fixed filename for the generated index. Both the indexer pre-pass
# and the MCP server load from ``<root>/index.scip``.
SCIP_INDEX_FILENAME = "index.scip"

# scip-python's console-script name (npm: @sourcegraph/scip-python).
_SCIP_BINARY = "scip-python"

# scip-python can be slow on large repos; cap it so a hang never wedges indexing.
_DEFAULT_TIMEOUT_S = 600.0


class ScipGenerationError(RuntimeError):
    """SCIP index generation could not be completed (missing binary or failure)."""


def scip_index_path(root: Path) -> Path:
    """Canonical location of the SCIP index for the repo rooted at ``root``."""
    return root / SCIP_INDEX_FILENAME


def generate_scip_index(
    root: Path,
    *,
    project_name: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> Path:
    """Regenerate ``<root>/index.scip`` with ``scip-python`` and return its path.

    ``--output`` is resolved relative to ``--cwd``, so passing the bare filename
    lands the index at ``root / "index.scip"`` regardless of the caller's working
    directory.

    Raises
    ------
    ScipGenerationError
        If ``scip-python`` is not on ``PATH``, the command exits non-zero, times
        out, or produces no index file. Indexing should treat this as "no SCIP".
    """
    binary = shutil.which(_SCIP_BINARY)
    if binary is None:
        raise ScipGenerationError(
            f"{_SCIP_BINARY!r} not found on PATH; install it with "
            "`npm install -g @sourcegraph/scip-python` to enable SCIP cross-references"
        )

    name = project_name or root.name
    cmd = [
        binary,
        "index",
        "--project-name",
        name,
        "--output",
        SCIP_INDEX_FILENAME,
        "--cwd",
        str(root),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ScipGenerationError(f"{_SCIP_BINARY} timed out after {timeout:.0f}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise ScipGenerationError(
            f"{_SCIP_BINARY} failed (exit {exc.returncode}): {detail}"
        ) from exc

    out = scip_index_path(root)
    if not out.is_file():
        raise ScipGenerationError(f"{_SCIP_BINARY} did not produce {out}")
    return out
