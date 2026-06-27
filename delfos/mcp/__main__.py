"""Entry point: `python -m delfos.mcp` (and the `delfos-mcp` console script).

Wires workspace config -> store + embedder -> startup model check -> a
planner-less ReconstructionService -> FastMCP, then serves over stdio. The repo
to serve is taken from ``DELFOS_REPO`` (default: the current directory); its
``.delfos/`` workspace supplies the store, the SCIP index, and the embed model.
"""

from __future__ import annotations

import os

from delfos.config import (
    build_embedder,
    build_scip_service,
    build_store,
    check_model_match,
    resolve_config,
)
from delfos.mcp.server import build_server
from delfos.reconstruct import ReconstructionService


def main() -> None:
    cfg = resolve_config(os.environ, repo_root=os.environ.get("DELFOS_REPO", "."))
    store = build_store(cfg)
    embedder = build_embedder(cfg)
    check_model_match(store, embedder)
    service = ReconstructionService(store, embedder)  # planner=None: agent is the planner
    scip = build_scip_service(cfg, store)
    build_server(service, scip).run()


if __name__ == "__main__":
    main()
