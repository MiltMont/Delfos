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
