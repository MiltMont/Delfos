"""Argument parsing, env-driven wiring, and command dispatch for the `delfos` CLI."""

from __future__ import annotations

import argparse
import os

from delfos.config import (
    build_embedder,
    build_planner,
    build_store,
    check_model_match,
    config_from_env,
    planner_config_from_env,
)
from delfos.indexer import Indexer, IndexStats
from delfos.indexer.embedder import Embedder
from delfos.reconstruct import ReconstructionService
from delfos.schema import CueNode
from delfos.store import GraphStore

from .render import render_index_stats, render_reconstruct, render_search, render_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delfos", description="Delfos graph-memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build/update a persistent store from a repo")
    p_index.add_argument("repo", help="path to the repository to index")
    p_index.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    p_status = sub.add_parser("status", help="inspect a store's manifest")
    p_status.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    p_search = sub.add_parser("search", help="semantic seed lookup over cue nodes")
    p_search.add_argument("query", help="natural-language query")
    p_search.add_argument("-k", type=int, default=5, help="number of cues to return")
    p_search.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    p_reconstruct = sub.add_parser(
        "reconstruct", help="LLM-driven depth-first reconstruction of relevant content"
    )
    p_reconstruct.add_argument("query", help="natural-language query")
    p_reconstruct.add_argument("--budget", type=int, default=3, help="max planner calls")
    p_reconstruct.add_argument("-k", type=int, default=5, help="number of seed cues")
    p_reconstruct.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    sub.add_parser("serve", help="run the MCP read server over stdio")

    return parser


def run_index(repo_path: str, store: GraphStore, embedder: Embedder) -> IndexStats:
    return Indexer(store, embedder).index(repo_path)


def run_status(store: GraphStore, embed_model: str, embed_dim: int) -> str:
    return render_status(embed_model, embed_dim, store.list_indexed_files())


def run_search(query: str, k: int, service: ReconstructionService) -> list[CueNode]:
    return service.search(query, k)


def _override_index_path(args: argparse.Namespace) -> None:
    index_path: str | None = getattr(args, "index_path", None)
    if index_path:
        os.environ["DELFOS_INDEX_PATH"] = index_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # serve owns its own store/server lifecycle via the MCP entry point.
    if args.command == "serve":
        from delfos.mcp.__main__ import main as serve_main

        serve_main()
        return 0

    _override_index_path(args)
    cfg = config_from_env(os.environ)
    store = build_store(cfg)
    try:
        if args.command == "index":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            print(render_index_stats(run_index(args.repo, store, embedder)))
        elif args.command == "status":
            print(run_status(store, cfg.embed_model, cfg.embed_dim))
        elif args.command == "search":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            service = ReconstructionService(store, embedder)
            print(render_search(run_search(args.query, args.k, service)))
        elif args.command == "reconstruct":
            embedder = build_embedder(cfg)
            check_model_match(store, embedder)
            planner = build_planner(planner_config_from_env(os.environ))
            service = ReconstructionService(store, embedder, planner)
            print(render_reconstruct(service.reconstruct(args.query, args.budget)))
        else:  # pragma: no cover - argparse `required=True` guards this
            return 2
    finally:
        store.close()
    return 0
