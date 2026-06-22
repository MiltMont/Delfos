# Delfos

A graph-memory **MCP server** for codebases. Delfos implements the active
*reconstruction* model from ["Memory is Reconstructed, Not Retrieved: Graph
Memory for LLM Agents"](https://arxiv.org/abs/2606.06036) (arXiv 2606.06036)
over a code repository, and exposes it as tools any MCP-compatible agent
(Claude Code, Cursor, etc.) can consume — no knowledge of the graph internals
required.

Memory access is an iterative, LLM-driven traversal of a persistent
**Cue → Tag → Content** graph rather than one-shot retrieval. See `plan.md`,
`design.md`, `foundations.md`, and `decisions.md` for the full design.

## Status

Early foundations. This package currently provides:

- **`delfos.schema`** — the code-specific Cue-Tag-Content schema as Pydantic
  models (`CueNode`, `TagNode`, `ContentNode`), the `Edge` model, and the
  closed enum vocabularies. Every node carries `source_file` + `git_sha`
  provenance (for delete-and-reindex) and optional embedding metadata
  (`embedding_model`) for embedding versioning.
- **`delfos.store`** — the `GraphStore` abstract base class (the single
  database boundary every other component goes through) and a `DuckDBGraphStore`
  backend **stub** (interface only; methods raise `NotImplementedError`).

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install deps + dev tools
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check (strict)
```
