"""Entry point: `python -m delfos.mcp` (and the `delfos-mcp` console script).

Wires workspace config -> store + embedder -> startup model check -> a
planner-less ReconstructionService -> FastMCP, then serves over stdio. The repo
to serve defaults to ``DELFOS_REPO`` (or the current directory) when not passed
explicitly; its ``.delfos/`` workspace supplies the store, the SCIP index, and
the embed model. A ``.env`` file at the repo root is loaded explicitly here,
anchored to the resolved ``repo_root`` — never at import time and never by
walking up from the process's CWD.
"""

from __future__ import annotations

import logging
import os

from delfos._logging import configure_cli_logging
from delfos.config import (
    build_embedder,
    build_scip_service,
    build_store,
    check_model_match,
    load_dotenv_values,
    resolve_config,
)
from delfos.mcp.server import build_server
from delfos.reconstruct import ReconstructionService

logger = logging.getLogger(__name__)


_TRUTHY = {"1", "true", "yes", "on"}


def main(repo_root: str | None = None, *, verbose: bool | None = None) -> None:
    """Serve ``repo_root``'s workspace over stdio.

    Both parameters default to the ``DELFOS_REPO``/``DELFOS_VERBOSE``
    environment variables when omitted, so the zero-arg ``delfos-mcp`` console
    script and ``python -m delfos.mcp`` keep working unchanged. ``delfos
    serve`` (see ``delfos/cli/app.py``) passes both explicitly instead of
    mutating ``os.environ``.
    """
    if verbose is None:
        verbose = os.environ.get("DELFOS_VERBOSE", "").strip().lower() in _TRUTHY
    if repo_root is None:
        repo_root = os.environ.get("DELFOS_REPO", ".")
    configure_cli_logging(verbose=verbose)
    logger.info("[1/5] resolving workspace for %s", repo_root)
    merged_env = {**load_dotenv_values(repo_root), **os.environ}
    cfg = resolve_config(merged_env, repo_root=repo_root)
    logger.info("[2/5] opening graph store at %s", cfg.index_path)
    store = build_store(cfg)
    logger.info("[3/5] embedder: model=%s dim=%d", cfg.embed_model, cfg.embed_dim)
    embedder = build_embedder(cfg)
    check_model_match(store, embedder)
    service = ReconstructionService(store, embedder)  # planner=None: agent is the planner
    scip = build_scip_service(cfg, store)
    logger.info(
        "[4/5] SCIP service: %s", "ready" if scip is not None else "unavailable (no index.scip)"
    )
    logger.info("[5/5] serving MCP over stdio (Ctrl-C to stop)")
    build_server(service, scip).run()


if __name__ == "__main__":
    main()
