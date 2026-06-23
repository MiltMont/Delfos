# Delfos

A graph-memory **MCP server** for codebases. Delfos implements the active
*reconstruction* model from ["Memory is Reconstructed, Not Retrieved: Graph
Memory for LLM Agents"](https://arxiv.org/abs/2606.06036) (arXiv 2606.06036)
over a code repository, and exposes it as tools any MCP-compatible agent
(Claude Code, Cursor, etc.) can consume — no knowledge of the graph internals
required.

Memory access is an iterative, LLM-driven traversal of a persistent
**Cue → Tag → Content** graph rather than one-shot retrieval. See
[`docs/libdelfos-plan.md`](docs/libdelfos-plan.md) and
[`CLAUDE.md`](CLAUDE.md) for the architecture and design decisions.

## Architecture

Two layers meet at a single boundary — the `GraphStore` ABC. No component ever
touches the C++ engine directly.

- **`libdelfos/`** — the C++ storage engine: an in-memory CSR directed property
  graph (`graph.hpp`), an HNSW vector index over USearch (`vector_index.hpp`),
  and crash-safe snapshot persistence via FlatBuffers + USearch native format
  (`snapshot.hpp`). Exposed to Python as the `delfos._delfos` extension through
  nanobind bindings.
- **`delfos.schema`** — the code-specific Cue-Tag-Content schema as Pydantic
  models (`CueNode`, `TagNode`, `ContentNode`), the `Edge` model, and the closed
  enum vocabularies. Every node carries `source_file` + `git_sha` provenance
  (for delete-and-reindex) and optional embedding metadata (`embedding_model`)
  for embedding versioning.
- **`delfos.store`** — the `GraphStore` abstract base class (the single database
  boundary every other component goes through) and `NativeGraphStore`, the
  concrete backend over the C++ engine.
- **`delfos.indexer`** — the construction pipeline (`parser`, `extractor`,
  `embedder`, `pipeline`) that turns a repository into graph nodes and edges.

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
