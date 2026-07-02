"""Generate a SCIP index for a repository by shelling out to ``scip-python``.

SCIP indexing is whole-repo: a single ``scip-python index`` invocation walks the
project and emits one ``index.scip`` protobuf. The indexer runs this as a
pre-pass before its per-file loop, writing the index into the repo's
``.delfos/`` workspace, then loads the result with
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

# scip-python's console-script name (npm: @sourcegraph/scip-python).
_SCIP_BINARY = "scip-python"

# scip-python can be slow on large repos; cap it so a hang never wedges indexing.
_DEFAULT_TIMEOUT_S = 600.0


class ScipGenerationError(RuntimeError):
    """SCIP index generation could not be completed (missing binary or failure)."""


def scip_binary_available() -> bool:
    """Whether the ``scip-python`` binary is on ``PATH``."""
    return shutil.which(_SCIP_BINARY) is not None


def generate_scip_index(
    root: Path,
    output_path: Path,
    *,
    project_name: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> Path:
    """Regenerate the SCIP index for ``root`` at ``output_path`` and return it.

    ``scip-python`` resolves ``--output`` relative to ``--cwd``; we pass an
    absolute ``output_path`` so the index lands in the workspace regardless of
    the caller's working directory.

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

    out = output_path.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    name = project_name or root.name
    cmd = [
        binary,
        "index",
        "--project-name",
        name,
        "--output",
        str(out),
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

    if not out.is_file():
        raise ScipGenerationError(f"{_SCIP_BINARY} did not produce {out}")
    return out
