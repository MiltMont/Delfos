"""Entry point: `python -m delfos.mcp` (and the `delfos-mcp` console script).

Wires workspace config -> store + embedder -> startup model check -> a
planner-less ReconstructionService -> FastMCP, then serves over stdio. The repo
to serve is taken from ``DELFOS_REPO`` (default: the current directory); its
``.delfos/`` workspace supplies the store, the SCIP index, and the embed model.
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
    resolve_config,
)
from delfos.mcp.server import build_server
from delfos.reconstruct import ReconstructionService

logger = logging.getLogger(__name__)


def main() -> None:
    configure_cli_logging()
    repo_root = os.environ.get("DELFOS_REPO", ".")
    logger.info("[1/5] resolving workspace for %s", repo_root)
    cfg = resolve_config(os.environ, repo_root=repo_root)
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
