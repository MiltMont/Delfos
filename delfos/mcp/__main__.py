"""Entry point: `python -m delfos.mcp` (and the `delfos-mcp` console script).

Wires env config -> store + embedder -> startup model check -> a planner-less
ReconstructionService -> FastMCP, then serves over stdio.
"""

from __future__ import annotations

import os

from delfos.config import (
    build_embedder,
    build_scip_service,
    build_store,
    check_model_match,
    config_from_env,
)
from delfos.mcp.server import build_server
from delfos.reconstruct import ReconstructionService


def main() -> None:
    cfg = config_from_env(os.environ)
    store = build_store(cfg)
    embedder = build_embedder(cfg)
    check_model_match(store, embedder)
    service = ReconstructionService(store, embedder)  # planner=None: agent is the planner
    scip = build_scip_service(cfg, store)
    build_server(service, scip).run()


if __name__ == "__main__":
    main()
