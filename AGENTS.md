# AGENTS.md

## Cursor Cloud specific instructions

Delfos is a graph-memory **MCP server** for codebases, with a full CLI. The
`delfos` package provides the schema (`delfos.schema`), the graph store
(`delfos.store` — `NativeGraphStore` over the C++ `delfos._delfos` nanobind
extension; the earlier DuckDB store has been removed), the indexer
(`delfos.indexer`), the read path (`delfos.reconstruct`), SCIP cross-references
(`delfos.scip`), the FastMCP server (`delfos.mcp`), and the `delfos` CLI
(`delfos.cli`). See `ARCHITECTURE.md` for the overview and `docs/decisions.md`
for the rationale.

### Tooling
- Project is managed by `uv` (see `README.md`/`CLAUDE.md` for the standard
  commands). The cloud VM installs `uv` to `~/.local/bin`; it is added to PATH via
  `~/.bashrc`, so interactive shells have it. The startup update script runs
  `uv sync`.
- `uv pip install -e .` builds and installs the `_delfos` C++ extension (build
  isolation pulls scikit-build-core + nanobind).
- Standard commands (already documented in `CLAUDE.md`): `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pyright` (strict mode), `uv run pytest`.

### C++ toolchain (for `libdelfos`, see `CMakePresets.json` and `CLAUDE.md`)
The cloud VM has the toolchain preinstalled: `clang++` 18
(LLVM; the build standardizes on clang, not GCC), `cmake` 3.28, `ninja`, and the
clang sanitizer runtimes (ASan/UBSan via `libclang-rt-18-dev`).
`CMakePresets.json` uses the Ninja generator with `clang++`. Build/test commands
are in `CLAUDE.md` (`cmake --preset debug`, `cmake --build build/debug`, etc.).

Non-obvious gotchas:
- `clang++` auto-selects the **GCC 14** libstdc++ install (highest version
  present), so `libstdc++-14-dev` must be installed — not just `libstdc++-13-dev`.
  Both are installed here; without the 14 dev package, links fail with
  `cannot find -lstdc++` and `<span>`/`<cstdint>` headers are not found.
- All external C++ deps (USearch, FlatBuffers, nanobind, Catch2, nanobench) are
  pulled via CMake `FetchContent` at **configure time** — the first configure
  needs network access to GitHub and is slower while it clones/builds them.

### Running it end-to-end
- Entry points: the `delfos` CLI (`index`, `status`, `doctor`, `search`,
  `reconstruct`, `serve`) and `delfos-mcp` (the MCP read server over stdio).
  Every command anchors on a repo's `.delfos/` workspace (`--repo`, default:
  the current directory), which holds the store snapshot, `index.scip`,
  `manifest.json`, and optional `config.toml`.
- Configuration is via `DELFOS_*` env variables (precedence and full table in
  `README.md`). Defaults: `nomic-embed-text`, dim 768. Point
  `DELFOS_EMBED_BASE_URL` at any OpenAI-compatible endpoint, or leave it unset
  for OpenAI-hosted (then `DELFOS_EMBED_API_KEY` is required). Queries against
  an already-indexed repo read the model/dim from the manifest and need only
  credentials.
- For offline/deterministic runs, drive the library directly: supply any object
  satisfying the `Embedder` protocol (`delfos.indexer.embedder.Embedder`) — a
  deterministic hash embedder is enough to index, search, and traverse without
  network access (this is what the tests do, along with a `FakeHopPlanner`).
  The store's `embedding_dim`/`embedding_model` must match the embedder's, or
  writes are rejected.
