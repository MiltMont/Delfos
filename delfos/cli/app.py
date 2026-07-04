"""Argument parsing, workspace-driven wiring, and command dispatch for the `delfos` CLI.

Every command is anchored on a repo's ``.delfos/`` workspace (``--repo``,
default: the current directory). The workspace supplies the graph store, the
SCIP index, and — via the manifest — the embedding model the index was built
with, so queries need only credentials in the environment.
"""

from __future__ import annotations

import argparse
import os

from delfos._logging import configure_cli_logging
from delfos.config import (
    build_embedder,
    build_planner,
    build_store,
    check_model_match,
    load_dotenv_values,
    planner_config_from_merged,
    resolve_config,
)
from delfos.indexer import Indexer, IndexStats
from delfos.indexer.embedder import Embedder
from delfos.reconstruct import ReconstructionService
from delfos.schema import CueNode
from delfos.scip.generate import scip_binary_available
from delfos.store import GraphStore
from delfos.workspace import Manifest, Workspace

from .render import (
    Check,
    render_doctor,
    render_index_stats,
    render_reconstruct,
    render_search,
    render_status,
)


def _add_repo_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo", default=".", help="repo whose .delfos/ workspace to use (default: .)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delfos", description="Delfos graph-memory CLI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log per-file detail (DEBUG) instead of just the high-level steps",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build/update a repo's .delfos/ workspace")
    p_index.add_argument("repo", help="path to the repository to index")

    p_status = sub.add_parser("status", help="inspect the workspace manifest and store")
    _add_repo_arg(p_status)

    p_doctor = sub.add_parser("doctor", help="check the workspace + toolchain setup")
    _add_repo_arg(p_doctor)

    p_search = sub.add_parser("search", help="semantic seed lookup over cue nodes")
    p_search.add_argument("query", help="natural-language query")
    p_search.add_argument("-k", type=int, default=5, help="number of cues to return")
    _add_repo_arg(p_search)

    p_reconstruct = sub.add_parser(
        "reconstruct", help="LLM-driven depth-first reconstruction of relevant content"
    )
    p_reconstruct.add_argument("query", help="natural-language query")
    p_reconstruct.add_argument("--budget", type=int, default=3, help="max planner calls")
    p_reconstruct.add_argument("-k", type=int, default=5, help="number of seed cues")
    _add_repo_arg(p_reconstruct)

    p_serve = sub.add_parser("serve", help="run the MCP read server over stdio")
    _add_repo_arg(p_serve)

    return parser


def run_index(repo_path: str, store: GraphStore, embedder: Embedder) -> IndexStats:
    return Indexer(store, embedder).index(repo_path)


def run_status(
    store: GraphStore, embed_model: str, embed_dim: int, manifest: Manifest | None
) -> str:
    return render_status(embed_model, embed_dim, store.list_indexed_files(), manifest)


def run_search(query: str, k: int, service: ReconstructionService) -> list[CueNode]:
    return service.search(query, k)


def run_doctor(workspace: Workspace, manifest: Manifest | None, embed_model: str) -> list[Check]:
    """Verify the toolchain + workspace so setup problems are obvious.

    Local-only checks (no network): the ``scip-python`` binary, the workspace
    directory + store, manifest presence, and the graph↔SCIP consistency verdict.
    """
    checks: list[Check] = []
    checks.append(
        Check(
            "scip-python on PATH",
            scip_binary_available(),
            "install with `npm install -g @sourcegraph/scip-python` for cross-references",
        )
    )
    checks.append(Check("workspace dir", workspace.dir.is_dir(), str(workspace.dir)))
    checks.append(
        Check("graph store", workspace.store_path.is_dir(), f"run `delfos index {workspace.root}`")
    )
    if manifest is None:
        checks.append(Check("manifest", False, "no manifest.json — run `delfos index`"))
        return checks
    checks.append(Check("manifest", True, f"embed model {embed_model!r}, dim {manifest.embed.dim}"))
    checks.append(
        Check(
            "scip present",
            workspace.scip_path.is_file(),
            f"status={manifest.scip.status.value}",
        )
    )
    checks.append(
        Check(
            "graph/scip consistent",
            manifest.is_consistent,
            "SCIP index is stale relative to the graph; re-run `delfos index`"
            if not manifest.is_consistent
            else "same index run",
        )
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(verbose=args.verbose)

    # `index` takes a positional `repo`; every other command an optional --repo.
    # Both land on `args.repo`.
    repo_root = args.repo

    # serve owns its own store/server lifecycle via the MCP entry point, which
    # re-configures logging for the standalone `delfos-mcp` case; pass the
    # verbose flag through explicitly so `delfos -v serve` keeps DEBUG.
    if args.command == "serve":
        from delfos.mcp.__main__ import main as serve_main

        serve_main(repo_root, verbose=args.verbose)
        return 0

    merged_env = {**load_dotenv_values(repo_root), **os.environ}
    cfg = resolve_config(merged_env, repo_root=repo_root)

    if args.command == "doctor":
        manifest = cfg.workspace.load_manifest()
        print(render_doctor(run_doctor(cfg.workspace, manifest, cfg.embed_model)))
        return 0

    store = build_store(cfg)
    try:
        if args.command == "index":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            print(render_index_stats(run_index(args.repo, store, embedder)))
        elif args.command == "status":
            print(run_status(store, cfg.embed_model, cfg.embed_dim, cfg.workspace.load_manifest()))
        elif args.command == "search":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            service = ReconstructionService(store, embedder)
            print(render_search(run_search(args.query, args.k, service)))
        elif args.command == "reconstruct":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            planner = build_planner(planner_config_from_merged(merged_env, repo_root=repo_root))
            service = ReconstructionService(store, embedder, planner, seed_k=args.k)
            print(render_reconstruct(service.reconstruct(args.query, args.budget)))
        else:  # pragma: no cover - argparse `required=True` guards this
            return 2
    finally:
        store.close()
    return 0
