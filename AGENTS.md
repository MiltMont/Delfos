# AGENTS.md

## Cursor Cloud specific instructions

Delfos is an early-stage **Python library** (graph-memory engine for codebases),
not a runnable service yet. There is no MCP server, CLI, or web UI — the `delfos`
package currently provides the schema (`delfos.schema`), the DuckDB-backed graph
store (`delfos.store`), and the indexer pipeline (`delfos.indexer`). Note the
`CLAUDE.md`/`README.md` claim the store/indexer are unimplemented stubs; that is
out of date — both are fully implemented and covered by tests under `tests/`.

### Tooling
- Project is managed by `uv` (see `README.md`/`CLAUDE.md` for the standard
  commands). The cloud VM installs `uv` to `~/.local/bin`; it is added to PATH via
  `~/.bashrc`, so interactive shells have it. The startup update script runs
  `uv sync`.
- Standard commands (already documented in `CLAUDE.md`): `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pyright` (strict mode), `uv run pytest`.

### C++ toolchain (for `libdelfos`, see `docs/libdelfos-plan.md`)
The cloud VM has the toolchain the plan requires preinstalled: `clang++` 18
(LLVM; the plan standardizes on clang, not GCC), `cmake` 3.28, `ninja`, and the
clang sanitizer runtimes (ASan/UBSan via `libclang-rt-18-dev`). The plan's
`CMakePresets.json` uses the Ninja generator with `clang++`. Build/test commands
are in `docs/libdelfos-plan.md` section 11 (`cmake --preset debug`, etc.).

Non-obvious gotchas:
- `clang++` auto-selects the **GCC 14** libstdc++ install (highest version
  present), so `libstdc++-14-dev` must be installed — not just `libstdc++-13-dev`.
  Both are installed here; without the 14 dev package, links fail with
  `cannot find -lstdc++` and `<span>`/`<cstdint>` headers are not found.
- All external C++ deps (USearch, FlatBuffers, nanobind, Catch2, nanobench) are
  pulled via CMake `FetchContent` at **configure time** — the first configure
  needs network access to GitHub and is slower while it clones/builds them.
- The C++ library source (`libdelfos/`, top-level `CMakeLists.txt`,
  `CMakePresets.json`) does not exist in the repo yet; the toolchain is ready but
  there is nothing to build until the plan's phases are implemented.

### Running it end-to-end
- There is no server entrypoint. To exercise the core flow, drive the library
  directly: build a `DuckDBGraphStore`, wrap it in `Indexer` with an `Embedder`,
  and call `indexer.index(<repo_path>)`, then query via `store.vector_search(...)`
  and `store.neighbors(...)`.
- The default `OpenAIEmbedder` needs `OPENAI_API_KEY`. For local/offline runs,
  supply any object satisfying the `Embedder` protocol
  (`delfos.indexer.embedder.Embedder`) — a deterministic hash embedder is enough
  to index, store, vector-search, and traverse without network/API access. The
  store's `embedding_dim`/`embedding_model` must match the embedder's, or writes
  are rejected.
