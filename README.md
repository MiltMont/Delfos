# Delfos

A graph-memory **MCP server** for codebases. Delfos implements the active
*reconstruction* model from ["Memory is Reconstructed, Not Retrieved: Graph
Memory for LLM Agents"](https://arxiv.org/abs/2606.06036) (arXiv 2606.06036)
over a code repository, and exposes it as tools any MCP-compatible agent
(Claude Code, Cursor, etc.) can consume — no knowledge of the graph internals
required.

Memory access is an iterative, LLM-driven traversal of a persistent
**Cue → Tag → Content** graph rather than one-shot retrieval. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the architecture overview and
[`docs/decisions.md`](docs/decisions.md) for the design decisions and rationale.

## Architecture

Two layers meet at a single boundary — the `GraphStore` ABC. No component ever
touches the C++ engine directly. The summary below is the short version; the
full contributor-facing walkthrough is [`ARCHITECTURE.md`](ARCHITECTURE.md).

- **`libdelfos/`** — the C++ storage engine: an in-memory CSR directed property
  graph (`graph.hpp`), an HNSW vector index over USearch (`vector_index.hpp`),
  and crash-safe snapshot persistence via FlatBuffers + USearch native format
  (`snapshot.hpp`). Exposed to Python as the `delfos._delfos` extension through
  nanobind bindings.
- **`delfos.schema`** — the code-specific Cue-Tag-Content schema as Pydantic
  models (`CueNode`, `TagNode`, `ContentNode`), the `Edge` model, and the closed
  enum vocabularies. Cue and content nodes carry `source_file` + `git_sha`
  provenance (for delete-and-reindex; tags are shared across files and carry
  none) and optional embedding metadata (`embedding_model`) for embedding
  versioning.
- **`delfos.store`** — the `GraphStore` abstract base class (the single database
  boundary every other component goes through) and `NativeGraphStore`, the
  concrete backend over the C++ engine.
- **`delfos.indexer`** — the construction pipeline (`parser`, `extractor`,
  `embedder`, `pipeline`) that turns a repository into graph nodes and edges.

## Configuration

Delfos is configured via `DELFOS_*` environment variables, resolved with this
precedence (highest first):

1. Real environment variables
2. A `.env` file at the repo root passed to `--repo` / `DELFOS_REPO` — loaded
   explicitly at startup (copy `.env.example` to `.env` and edit)
3. `.delfos/config.toml` (non-secret settings only — keep API keys out of it)
4. `.delfos/manifest.json`, recorded at index time (`embed.model`/`embed.dim`
   only — these must match what the index was built with)
5. Built-in defaults (`nomic-embed-text`, dim 768)

| Variable | Purpose |
|---|---|
| `DELFOS_EMBED_MODEL` | Embedding model name |
| `DELFOS_EMBED_DIM` | Embedding output dimension (must match the model) |
| `DELFOS_EMBED_BASE_URL` | OpenAI-compatible embedding endpoint (unset = OpenAI) |
| `DELFOS_EMBED_API_KEY` | Embedding endpoint API key (required when `DELFOS_EMBED_BASE_URL` is unset) |
| `DELFOS_LLM_MODEL` | Chat model for the `reconstruct` hop planner |
| `DELFOS_LLM_BASE_URL` | OpenAI-compatible chat endpoint (unset = OpenAI) |
| `DELFOS_LLM_API_KEY` | Chat endpoint API key |
| `DELFOS_REPO` | Repo whose `.delfos/` workspace to serve (`delfos-mcp` only) |
| `DELFOS_VERBOSE` | `1` for per-file DEBUG logging |

A query against an already-indexed repo needs only credentials — the model and
dimension come from the manifest. `DELFOS_EMBED_BASE_URL` and
`DELFOS_LLM_BASE_URL` each point at their own endpoint independently; there is
no fallback between them.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install Python deps + dev tools
uv pip install -e .     # build + install the _delfos C++ extension
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check (strict)
uv run pytest           # run tests
```

### C++ engine

The standalone C++ build/test flow uses the Ninja generator (`brew install
ninja`):

```bash
cmake --preset debug                              # configure (ASan + UBSan)
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
```

> **Note:** CMake 4.x dropped compatibility with `cmake_minimum_required < 3.5`,
> which the vendored USearch dependency still declares. The build passes
> `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (wired into `pyproject.toml` and the CMake
> presets) to configure under modern CMake.
