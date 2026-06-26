# TypeScript Indexing Support

**Date:** 2026-06-25
**Status:** Approved

## Summary

Extend the delfos indexer to parse `.ts` and `.tsx` files using tree-sitter,
producing the same `ParsedModule` / `ParsedDefinition` IR already consumed by
the extractor. Python indexing is unchanged.

## Scope

Extract all top-level TypeScript constructs:

- Functions and async functions (`function_declaration`)
- Arrow functions assigned to `const` (`lexical_declaration` → `arrow_function`)
- Classes with their methods (`class_declaration`, `method_definition`)
- Interfaces (`interface_declaration`)
- Type aliases (`type_alias_declaration`)
- Enums (`enum_declaration`)
- Error messages from `throw new Error("…")` calls (`throw_statement`)

## Dependencies

Add to `pyproject.toml` runtime dependencies:

```
tree-sitter>=0.23
tree-sitter-typescript>=0.23
```

Two language objects instantiated once at module level (not per-call):

```python
TS_LANGUAGE  = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())
```

## Files Changed

### `schema/enums.py`

Add three values to `ContentKind`:

```python
INTERFACE  = "interface"
TYPE_ALIAS = "type_alias"
ENUM       = "enum"
```

### `indexer/parser.py`

Add three values to `DefinitionKind` (the TypeScript parser imports this enum):

```python
INTERFACE  = "interface"
TYPE_ALIAS = "type_alias"
ENUM       = "enum"
```

Add `language: str` field to `ParsedModule`. The existing `parse_module`
function sets `language="python"`.

### `indexer/ts_parser.py` (new)

Public entry point:

```python
def parse_ts_module(
    source: str,
    *,
    source_file: str,
    module_path: str,
    tsx: bool = False,
) -> ParsedModule: ...
```

**Construct → `DefinitionKind` mapping:**

| TypeScript construct | tree-sitter node | `DefinitionKind` |
|---|---|---|
| `function foo() {}` | `function_declaration` | `FUNCTION` / `ASYNC_FUNCTION` |
| `const foo = () => {}` | `lexical_declaration` → `arrow_function` | `FUNCTION` / `ASYNC_FUNCTION` |
| `class Foo {}` | `class_declaration` | `CLASS` |
| methods inside class | `method_definition` | `METHOD` / `ASYNC_METHOD` |
| `interface Foo {}` | `interface_declaration` | `INTERFACE` |
| `type Foo = …` | `type_alias_declaration` | `TYPE_ALIAS` |
| `enum Foo {}` | `enum_declaration` | `ENUM` |

**Signatures** are reconstructed from node fields (name + parameters + return
type annotation), not by re-stringifying the full body.

**Error messages** are extracted by walking `throw_statement` nodes for
`new_expression` whose constructor name ends in `Error`, collecting the first
string argument literal.

**`is_test`** is `True` when the function name starts with `test`, `it`, or
`describe`, or when the source file path contains `.spec.` or `.test.`.

`ParsedModule.language` is set to `"typescript"` for `.ts` files and `"tsx"`
for `.tsx` files. These values become the `LANGUAGE` tag on every content node
produced from that file.

tree-sitter does not raise on syntax errors (it inserts `ERROR` nodes), so no
additional exception handling is required beyond the existing
`UnicodeDecodeError` catch in the pipeline.

### `indexer/pipeline.py`

**`_discover`** — extend file filter:

```python
if name.endswith((".py", ".ts", ".tsx")):
    found.append(...)
```

**`_module_path`** — handle TS extensions (`index.ts` / `index.tsx` are
equivalent to `__init__.py`):

```python
for ext, index_name in [(".tsx", "index"), (".ts", "index"), (".py", "__init__")]:
    if relative_path.endswith(ext):
        parts = relative_path[: -len(ext)].split("/")
        if parts and parts[-1] == index_name:
            parts = parts[:-1]
        return ".".join(parts)
return relative_path
```

**`_index_file`** — dispatch by extension:

```python
ext = Path(relative_path).suffix
if ext == ".py":
    module = parse_module(source, source_file=relative_path, module_path=mp)
else:
    module = parse_ts_module(source, source_file=relative_path, module_path=mp, tsx=(ext == ".tsx"))
```

### `indexer/extractor.py`

Remove `_LANGUAGE = "python"` constant; use `self._module.language` in
`_tag_content` instead.

Extend `_CONTENT_KINDS`:

```python
DefinitionKind.INTERFACE:  ContentKind.INTERFACE,
DefinitionKind.TYPE_ALIAS: ContentKind.TYPE_ALIAS,
DefinitionKind.ENUM:       ContentKind.ENUM,
```

## What Is Not Changed

- `GraphStore`, `NativeGraphStore`, the C++ engine
- MCP tools (`search`, `traverse_forward`, `traverse_reverse`, `reconstruct`)
- CLI commands
- The `_SKIP_DIRS` list (already includes `node_modules`)

## Testing

- Unit tests for `parse_ts_module` covering each construct type and the error-message extractor
- Pipeline integration test: index a small mixed `.py` + `.ts` directory and assert both file types appear in the store
- Regression: existing Python parser tests must continue to pass unchanged
