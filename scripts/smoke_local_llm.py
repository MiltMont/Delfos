"""Manual smoke test: drive `search` + `reconstruct` against a local LLM.

This is a by-hand harness, not part of the test suite. It indexes a slice of
this repo into a fresh NativeGraphStore using a local embedding model for
semantic seeds, then points OpenAIHopPlanner at a local OpenAI-compatible
endpoint and runs a real reconstruct walk. Both the embedder and the planner
talk to local endpoints — nothing leaves the machine.

Configuration is read from a `.env` file at the repo root (see `.env.example`);
real environment variables take precedence over `.env`. Recognized keys
(defaults target Ollama):

    DELFOS_LLM_BASE_URL     default http://localhost:11434/v1
    DELFOS_LLM_MODEL        default qwen2.5
    DELFOS_LLM_API_KEY      default "ollama"  (local servers ignore the value)
    DELFOS_EMBED_BASE_URL   default = DELFOS_LLM_BASE_URL
    DELFOS_EMBED_MODEL      default nomic-embed-text
    DELFOS_EMBED_DIM        default 768       (must match the embed model)
    DELFOS_INDEX_PATH       default delfos/store   (relative to repo root)
    DELFOS_QUERY            default "how does the graph store persist nodes?"
    DELFOS_DEBUG            set to 1 to trace per-hop decisions + HTTP calls

Run:

    cp .env.example .env   # then edit
    uv run python scripts/smoke_local_llm.py

    # see the planner's per-hop decisions and each HTTP call:
    DELFOS_DEBUG=1 uv run python scripts/smoke_local_llm.py

    # also dump full request/response bodies (schema sent, raw completion):
    DELFOS_DEBUG=1 OPENAI_LOG=debug uv run python scripts/smoke_local_llm.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from delfos.indexer import Indexer, OpenAIEmbedder
from delfos.reconstruct import ReconstructionService
from delfos.reconstruct.planners.openai import OpenAIHopPlanner
from delfos.store import NativeGraphStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def _configure_debug() -> None:
    """Enable verbose tracing when DELFOS_DEBUG is set.

    Shows the service's per-hop decisions (delfos logger at DEBUG) and one line
    per HTTP call (httpx at DEBUG). For full request/response *bodies* — the
    exact JSON schema sent and the raw completion — prefix the command with
    OPENAI_LOG=debug (the OpenAI SDK reads that at import time).
    """
    flag = os.environ.get("DELFOS_DEBUG", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("delfos").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    print("debug tracing on (DELFOS_DEBUG). For request/response bodies: OPENAI_LOG=debug")


def _build_planner() -> OpenAIHopPlanner:
    base_url = os.environ.get("DELFOS_LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("DELFOS_LLM_MODEL", "qwen2.5")
    api_key = os.environ.get("DELFOS_LLM_API_KEY", "ollama")
    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"planner  -> {base_url}  model={model}")
    return OpenAIHopPlanner(model=model, client=client)


def _build_embedder() -> tuple[OpenAIEmbedder, int, str]:
    llm_base = os.environ.get("DELFOS_LLM_BASE_URL", "http://localhost:11434/v1")
    base_url = os.environ.get("DELFOS_EMBED_BASE_URL", llm_base)
    model = os.environ.get("DELFOS_EMBED_MODEL", "nomic-embed-text")
    dim = int(os.environ.get("DELFOS_EMBED_DIM", "768"))
    api_key = os.environ.get("DELFOS_LLM_API_KEY", "ollama")
    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"embedder -> {base_url}  model={model}  dim={dim}")
    # send_dimensions=False: local endpoints reject the `dimensions` request arg.
    embedder = OpenAIEmbedder(model, dimensions=dim, send_dimensions=False, client=client)
    return embedder, dim, model


def main() -> int:
    # Load .env from the repo root; real env vars still win (override=False).
    load_dotenv(REPO_ROOT / ".env", override=False)
    _configure_debug()

    index_path = os.environ.get("DELFOS_INDEX_PATH", "delfos/store")
    query = os.environ.get("DELFOS_QUERY", "how does the graph store persist nodes?")

    planner = _build_planner()
    embedder, embed_dim, embed_model = _build_embedder()

    with tempfile.TemporaryDirectory(prefix="delfos_smoke_") as snap_dir:
        store = NativeGraphStore(
            Path(snap_dir), embedding_dim=embed_dim, embedding_model=embed_model
        )
        store.initialize()

        target = REPO_ROOT / index_path
        print(f"\nindexing {target} (embeds cues via the local model) ...")
        try:
            stats = Indexer(store, embedder).index(target)
        except Exception as exc:  # noqa: BLE001 - smoke harness, surface anything
            print(f"indexing failed talking to the embedding endpoint: {exc!r}")
            print("Check the embed endpoint is up and the model is pulled.")
            return 2
        print(
            f"indexed {stats.indexed_files} file(s), "
            f"{stats.nodes_written} node(s), {stats.edges_written} edge(s)"
        )
        if stats.nodes_written == 0:
            print("nothing indexed — pick a path with Python files via DELFOS_INDEX_PATH")
            return 1

        svc = ReconstructionService(store, embedder, planner)

        print(f"\nquery: {query!r}")
        cues = svc.search(query, k=5)
        print(f"\nsearch -> {len(cues)} cue(s):")
        for cue in cues:
            print(f"  - {cue.id}")

        print("\nreconstruct (this calls the local LLM per hop) ...")
        try:
            contents = svc.reconstruct(query, budget=4)
        except Exception as exc:  # noqa: BLE001 - smoke harness, surface anything
            print(f"reconstruct failed talking to the LLM: {exc!r}")
            print("Check the endpoint is up and the model supports structured outputs.")
            return 2

        print(f"\nreconstruct -> {len(contents)} content node(s):")
        for content in contents:
            print(f"  - [{content.memory_layer.value}] {content.id}")
        if not contents:
            print("  (empty — the model may not support JSON-schema structured outputs,")
            print("   in which case every hop returns a terminal stop)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
