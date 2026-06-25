"""Argument parsing, env-driven wiring, and command dispatch for the `delfos` CLI."""

from __future__ import annotations

import argparse
import os

from delfos.config import build_embedder, build_store, check_model_match, config_from_env
from delfos.indexer import Indexer, IndexStats
from delfos.indexer.embedder import Embedder
from delfos.store import GraphStore

from .render import render_index_stats, render_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delfos", description="Delfos graph-memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build/update a persistent store from a repo")
    p_index.add_argument("repo", help="path to the repository to index")
    p_index.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    p_status = sub.add_parser("status", help="inspect a store's manifest")
    p_status.add_argument("--index-path", default=None, help="override DELFOS_INDEX_PATH")

    # search / reconstruct / serve are added in Task 3.
    return parser


def run_index(repo_path: str, store: GraphStore, embedder: Embedder) -> IndexStats:
    return Indexer(store, embedder).index(repo_path)


def run_status(store: GraphStore, embed_model: str, embed_dim: int) -> str:
    return render_status(embed_model, embed_dim, store.list_indexed_files())


def _override_index_path(args: argparse.Namespace) -> None:
    index_path: str | None = getattr(args, "index_path", None)
    if index_path:
        os.environ["DELFOS_INDEX_PATH"] = index_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        else:  # pragma: no cover - argparse `required=True` guards this
            return 2
    finally:
        store.close()
    return 0
