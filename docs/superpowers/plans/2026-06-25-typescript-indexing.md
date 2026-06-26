# TypeScript Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the delfos indexer to parse `.ts` and `.tsx` files using tree-sitter, producing the same graph nodes Python files produce today.

**Architecture:** A new `ts_parser.py` module uses tree-sitter to walk TypeScript/TSX ASTs and emit `ParsedModule`/`ParsedDefinition` IR — the same frozen dataclasses already consumed by the extractor. `pipeline.py`'s `_discover` and `_index_file` are extended to route `.ts`/`.tsx` files through the new parser; Python files are unchanged.

**Tech Stack:** `tree-sitter>=0.23`, `tree-sitter-typescript>=0.23` (both pure Python at install time, no C compilation required beyond what already exists).

## Global Constraints

- Python 3.12+; pyright strict mode — all new code must be fully typed with no `# type: ignore`
- `extra="forbid"` on all Pydantic models (already enforced; don't create new ones)
- Run `uv run ruff check . && uv run ruff format . && uv run pyright && uv run pytest` to verify after each task
- Branch: `feat/typescript-indexing`; commit after every task

## Spec Deviation Note

The approved spec called for new `ContentKind` values (`INTERFACE`, `TYPE_ALIAS`, `ENUM`). The C++ store (`_KIND_TO_NATIVE` in `native_store.py`) requires a corresponding C++ constant for every `ContentKind`. Adding C++ constants is out of scope. Instead:

- `DefinitionKind.INTERFACE`, `DefinitionKind.TYPE_ALIAS`, `DefinitionKind.ENUM` are added (Python-only IR layer)
- All three map to `ContentKind.CLASS` in `_CONTENT_KINDS`
- The `LANG_CONSTRUCT` tag records the precise kind ("interface", "type_alias", "enum") so agents can still filter by it

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `pyproject.toml` | Add tree-sitter runtime deps |
| Modify | `delfos/indexer/parser.py` | Add `language` field to `ParsedModule`; add 3 `DefinitionKind` values |
| Modify | `delfos/indexer/extractor.py` | Remove `_LANGUAGE` constant; extend `_CONTENT_KINDS` |
| Create | `delfos/indexer/ts_parser.py` | Full TypeScript/TSX parser |
| Modify | `delfos/indexer/__init__.py` | Export `parse_ts_module` |
| Modify | `delfos/indexer/pipeline.py` | Extend `_discover`, `_module_path`, `_index_file` |
| Create | `tests/indexer/__init__.py` | Empty; makes tests/indexer a package |
| Create | `tests/indexer/test_ir_changes.py` | Tests for Task 1 changes |
| Create | `tests/indexer/test_ts_parser.py` | Tests for Task 2 |
| Create | `tests/indexer/test_pipeline_ts.py` | Tests for Task 3 |

---

## Task 1: Dependencies and IR Groundwork

**Files:**
- Modify: `pyproject.toml`
- Modify: `delfos/indexer/parser.py`
- Modify: `delfos/indexer/extractor.py`
- Create: `tests/indexer/__init__.py`
- Create: `tests/indexer/test_ir_changes.py`

**Interfaces:**
- Produces: `ParsedModule.language: str` consumed by Task 2 and Task 3
- Produces: `DefinitionKind.INTERFACE`, `DefinitionKind.TYPE_ALIAS`, `DefinitionKind.ENUM` consumed by Task 2
- Produces: extractor handles new `DefinitionKind` values without `KeyError`

- [ ] **Step 1: Write failing tests**

Create `tests/indexer/__init__.py` (empty file), then create `tests/indexer/test_ir_changes.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from delfos.indexer.extractor import extract
from delfos.indexer.parser import DefinitionKind, ParsedModule, parse_module


def test_parse_module_sets_language_python() -> None:
    module = parse_module("x = 1", source_file="a.py", module_path="a")
    assert module.language == "python"


def test_parsed_module_language_field_exists() -> None:
    module = ParsedModule(
        module_path="a",
        source_file="a.py",
        language="typescript",
        docstring=None,
        source="",
        definitions=(),
    )
    assert module.language == "typescript"


def test_definition_kind_has_interface() -> None:
    assert DefinitionKind.INTERFACE == "interface"


def test_definition_kind_has_type_alias() -> None:
    assert DefinitionKind.TYPE_ALIAS == "type_alias"


def test_definition_kind_has_enum() -> None:
    assert DefinitionKind.ENUM == "enum"


def test_extractor_handles_interface_kind() -> None:
    from delfos.indexer.parser import ParsedDefinition

    module = ParsedModule(
        module_path="a",
        source_file="a.ts",
        language="typescript",
        docstring=None,
        source="interface Foo {}",
        definitions=(
            ParsedDefinition(
                name="Foo",
                qualified_name="Foo",
                kind=DefinitionKind.INTERFACE,
                signature="interface Foo",
                docstring=None,
                source="interface Foo {}",
                lineno=1,
                decorators=(),
                error_messages=(),
                is_test=False,
            ),
        ),
    )
    result = extract(module, git_sha="abc123", indexed_at=datetime.now(tz=UTC))
    content_ids = [n.id for n in result.nodes if n.id.startswith("content:")]
    assert any("Foo" in cid for cid in content_ids)


def test_extractor_uses_module_language_as_tag() -> None:
    from delfos.schema import TagNode

    module = ParsedModule(
        module_path="a",
        source_file="a.ts",
        language="typescript",
        docstring=None,
        source="",
        definitions=(),
    )
    result = extract(module, git_sha="abc", indexed_at=datetime.now(tz=UTC))
    tag_values = {n.value for n in result.nodes if isinstance(n, TagNode)}
    assert "typescript" in tag_values
    assert "python" not in tag_values
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/indexer/test_ir_changes.py -v
```

Expected: multiple failures — `ParsedModule` has no `language` field, `DefinitionKind` has no `INTERFACE`/`TYPE_ALIAS`/`ENUM`, extractor still uses hardcoded `"python"`.

- [ ] **Step 3: Add tree-sitter dependencies to `pyproject.toml`**

In `pyproject.toml`, change the `dependencies` list from:
```toml
dependencies = [
    "openai>=1.0.0",
    "pydantic>=2.9.0",
    "mcp>=1.28.0",
]
```
to:
```toml
dependencies = [
    "openai>=1.0.0",
    "pydantic>=2.9.0",
    "mcp>=1.28.0",
    "tree-sitter>=0.23",
    "tree-sitter-typescript>=0.23",
]
```

Then install:
```
uv sync
```

- [ ] **Step 4: Extend `DefinitionKind` and `ParsedModule` in `parser.py`**

In `delfos/indexer/parser.py`, change:

```python
class DefinitionKind(StrEnum):
    """Discriminator for the kind of definition extracted from source."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    CLASS = "class"
```

to:

```python
class DefinitionKind(StrEnum):
    """Discriminator for the kind of definition extracted from source."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    ENUM = "enum"
```

Change `ParsedModule` from:

```python
@dataclass(frozen=True)
class ParsedModule:
    """The full parse result for one Python module."""

    module_path: str
    source_file: str
    docstring: str | None
    source: str
    definitions: tuple[ParsedDefinition, ...]
```

to:

```python
@dataclass(frozen=True)
class ParsedModule:
    """The full parse result for one source module."""

    module_path: str
    source_file: str
    language: str
    docstring: str | None
    source: str
    definitions: tuple[ParsedDefinition, ...]
```

At the bottom of `parser.py`, in `parse_module`, change the return statement from:

```python
    return ParsedModule(
        module_path=module_path,
        source_file=source_file,
        docstring=ast.get_docstring(tree),
        source=source,
        definitions=tuple(definitions),
    )
```

to:

```python
    return ParsedModule(
        module_path=module_path,
        source_file=source_file,
        language="python",
        docstring=ast.get_docstring(tree),
        source=source,
        definitions=tuple(definitions),
    )
```

- [ ] **Step 5: Update `extractor.py` — remove `_LANGUAGE`, extend `_CONTENT_KINDS`, use `module.language`**

In `delfos/indexer/extractor.py`, change:

```python
_LANGUAGE = "python"
_MODULE_CONSTRUCT = "module"

_CONTENT_KINDS: dict[DefinitionKind, ContentKind] = {
    DefinitionKind.FUNCTION: ContentKind.FUNCTION,
    DefinitionKind.ASYNC_FUNCTION: ContentKind.FUNCTION,
    DefinitionKind.METHOD: ContentKind.FUNCTION,
    DefinitionKind.ASYNC_METHOD: ContentKind.FUNCTION,
    DefinitionKind.CLASS: ContentKind.CLASS,
}
```

to:

```python
_MODULE_CONSTRUCT = "module"

_CONTENT_KINDS: dict[DefinitionKind, ContentKind] = {
    DefinitionKind.FUNCTION: ContentKind.FUNCTION,
    DefinitionKind.ASYNC_FUNCTION: ContentKind.FUNCTION,
    DefinitionKind.METHOD: ContentKind.FUNCTION,
    DefinitionKind.ASYNC_METHOD: ContentKind.FUNCTION,
    DefinitionKind.CLASS: ContentKind.CLASS,
    DefinitionKind.INTERFACE: ContentKind.CLASS,
    DefinitionKind.TYPE_ALIAS: ContentKind.CLASS,
    DefinitionKind.ENUM: ContentKind.CLASS,
}
```

Then in `_Builder._tag_content` (line 181), change:

```python
    def _tag_content(self, content_id: str, construct: str) -> None:
        self._add_tag_edge(content_id, TagCategory.LANGUAGE, _LANGUAGE)
        self._add_tag_edge(content_id, TagCategory.MODULE_PATH, self._module.module_path)
        self._add_tag_edge(content_id, TagCategory.LANG_CONSTRUCT, construct)
```

to:

```python
    def _tag_content(self, content_id: str, construct: str) -> None:
        self._add_tag_edge(content_id, TagCategory.LANGUAGE, self._module.language)
        self._add_tag_edge(content_id, TagCategory.MODULE_PATH, self._module.module_path)
        self._add_tag_edge(content_id, TagCategory.LANG_CONSTRUCT, construct)
```

- [ ] **Step 6: Run tests and confirm they pass**

```
uv run pytest tests/indexer/test_ir_changes.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Run full suite to confirm no regressions**

```
uv run pytest --ignore=tests/test_e2e.py -v
```

Expected: all tests PASS (skip e2e — requires a live store).

- [ ] **Step 8: Lint, format, type-check**

```
uv run ruff check . && uv run ruff format . && uv run pyright
```

Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml delfos/indexer/parser.py delfos/indexer/extractor.py \
        tests/indexer/__init__.py tests/indexer/test_ir_changes.py
git commit -m "feat(indexer): add language field to ParsedModule and TS DefinitionKind values"
```

---

## Task 2: TypeScript Parser

**Files:**
- Create: `delfos/indexer/ts_parser.py`
- Modify: `delfos/indexer/__init__.py`
- Create: `tests/indexer/test_ts_parser.py`

**Interfaces:**
- Consumes: `ParsedModule`, `ParsedDefinition`, `DefinitionKind` from `delfos.indexer.parser` (Task 1)
- Produces: `parse_ts_module(source, *, source_file, module_path, tsx=False) -> ParsedModule`

- [ ] **Step 1: Write failing tests**

Create `tests/indexer/test_ts_parser.py`:

```python
from __future__ import annotations

import pytest

from delfos.indexer.parser import DefinitionKind
from delfos.indexer.ts_parser import parse_ts_module


# ── Functions ────────────────────────────────────────────────────────────────

def test_parse_sync_function() -> None:
    source = "function greet(name: string): string { return name; }"
    module = parse_ts_module(source, source_file="src/greet.ts", module_path="src.greet")
    assert len(module.definitions) == 1
    d = module.definitions[0]
    assert d.name == "greet"
    assert d.qualified_name == "greet"
    assert d.kind == DefinitionKind.FUNCTION
    assert "greet" in (d.signature or "")
    assert d.lineno == 1


def test_parse_async_function() -> None:
    source = "async function fetchData(url: string): Promise<void> {}"
    module = parse_ts_module(source, source_file="src/api.ts", module_path="src.api")
    assert len(module.definitions) == 1
    assert module.definitions[0].kind == DefinitionKind.ASYNC_FUNCTION


def test_parse_exported_function() -> None:
    source = "export function greet(): void {}"
    module = parse_ts_module(source, source_file="src/greet.ts", module_path="src.greet")
    assert len(module.definitions) == 1
    assert module.definitions[0].name == "greet"
    assert module.definitions[0].kind == DefinitionKind.FUNCTION


def test_parse_exported_async_function() -> None:
    source = "export async function load(): Promise<string> { return ''; }"
    module = parse_ts_module(source, source_file="src/api.ts", module_path="src.api")
    assert module.definitions[0].kind == DefinitionKind.ASYNC_FUNCTION


# ── Arrow functions ───────────────────────────────────────────────────────────

def test_parse_const_arrow_function() -> None:
    source = "const add = (a: number, b: number): number => a + b;"
    module = parse_ts_module(source, source_file="src/math.ts", module_path="src.math")
    assert len(module.definitions) == 1
    d = module.definitions[0]
    assert d.name == "add"
    assert d.kind == DefinitionKind.FUNCTION


def test_parse_const_async_arrow_function() -> None:
    source = "const fetch = async (url: string): Promise<string> => url;"
    module = parse_ts_module(source, source_file="src/api.ts", module_path="src.api")
    assert module.definitions[0].kind == DefinitionKind.ASYNC_FUNCTION


def test_parse_exported_const_arrow() -> None:
    source = "export const handler = (req: Request) => req;"
    module = parse_ts_module(source, source_file="src/handler.ts", module_path="src.handler")
    assert len(module.definitions) == 1
    assert module.definitions[0].name == "handler"


def test_skips_non_arrow_const() -> None:
    source = "const x = 42;"
    module = parse_ts_module(source, source_file="src/const.ts", module_path="src.const")
    assert len(module.definitions) == 0


# ── Classes ───────────────────────────────────────────────────────────────────

def test_parse_class() -> None:
    source = "class Foo { bar(): void {} }"
    module = parse_ts_module(source, source_file="src/foo.ts", module_path="src.foo")
    assert module.definitions[0].kind == DefinitionKind.CLASS
    assert module.definitions[0].name == "Foo"


def test_parse_class_with_sync_and_async_methods() -> None:
    source = "class Repo { find(): string { return ''; } async save(): Promise<void> {} }"
    module = parse_ts_module(source, source_file="src/repo.ts", module_path="src.repo")
    assert len(module.definitions) == 3
    cls, find, save = module.definitions
    assert cls.kind == DefinitionKind.CLASS
    assert find.kind == DefinitionKind.METHOD
    assert find.qualified_name == "Repo.find"
    assert save.kind == DefinitionKind.ASYNC_METHOD
    assert save.qualified_name == "Repo.save"


def test_parse_exported_class() -> None:
    source = "export class Service { run(): void {} }"
    module = parse_ts_module(source, source_file="src/svc.ts", module_path="src.svc")
    assert module.definitions[0].name == "Service"


# ── Interfaces ────────────────────────────────────────────────────────────────

def test_parse_interface() -> None:
    source = "interface Shape { area(): number; }"
    module = parse_ts_module(source, source_file="src/shapes.ts", module_path="src.shapes")
    assert len(module.definitions) == 1
    d = module.definitions[0]
    assert d.kind == DefinitionKind.INTERFACE
    assert d.name == "Shape"
    assert "Shape" in (d.signature or "")


def test_parse_exported_interface() -> None:
    source = "export interface Config { port: number; }"
    module = parse_ts_module(source, source_file="src/config.ts", module_path="src.config")
    assert module.definitions[0].name == "Config"
    assert module.definitions[0].kind == DefinitionKind.INTERFACE


# ── Type aliases ──────────────────────────────────────────────────────────────

def test_parse_type_alias() -> None:
    source = "type Point = { x: number; y: number };"
    module = parse_ts_module(source, source_file="src/types.ts", module_path="src.types")
    assert len(module.definitions) == 1
    d = module.definitions[0]
    assert d.kind == DefinitionKind.TYPE_ALIAS
    assert d.name == "Point"


def test_parse_exported_type_alias() -> None:
    source = "export type ID = string | number;"
    module = parse_ts_module(source, source_file="src/types.ts", module_path="src.types")
    assert module.definitions[0].kind == DefinitionKind.TYPE_ALIAS


# ── Enums ─────────────────────────────────────────────────────────────────────

def test_parse_enum() -> None:
    source = "enum Direction { Up, Down, Left, Right }"
    module = parse_ts_module(source, source_file="src/dir.ts", module_path="src.dir")
    assert len(module.definitions) == 1
    d = module.definitions[0]
    assert d.kind == DefinitionKind.ENUM
    assert d.name == "Direction"
    assert "Direction" in (d.signature or "")


def test_parse_exported_enum() -> None:
    source = "export enum Status { Active, Inactive }"
    module = parse_ts_module(source, source_file="src/status.ts", module_path="src.status")
    assert module.definitions[0].kind == DefinitionKind.ENUM


# ── Error messages ────────────────────────────────────────────────────────────

def test_extracts_throw_error_message() -> None:
    source = 'function fail(): never { throw new Error("something went wrong"); }'
    module = parse_ts_module(source, source_file="src/util.ts", module_path="src.util")
    assert "something went wrong" in module.definitions[0].error_messages


def test_extracts_custom_error_message() -> None:
    source = 'function fail(): never { throw new TypeError("bad type"); }'
    module = parse_ts_module(source, source_file="src/util.ts", module_path="src.util")
    assert "bad type" in module.definitions[0].error_messages


def test_no_error_message_without_throw() -> None:
    source = "function ok(): void {}"
    module = parse_ts_module(source, source_file="src/ok.ts", module_path="src.ok")
    assert module.definitions[0].error_messages == ()


# ── Language and is_test ──────────────────────────────────────────────────────

def test_language_is_typescript() -> None:
    module = parse_ts_module("", source_file="src/a.ts", module_path="src.a")
    assert module.language == "typescript"


def test_language_is_tsx() -> None:
    module = parse_ts_module("", source_file="src/a.tsx", module_path="src.a", tsx=True)
    assert module.language == "tsx"


def test_is_test_from_file_name() -> None:
    source = "function doSomething(): void {}"
    module = parse_ts_module(source, source_file="src/foo.spec.ts", module_path="src.foo_spec")
    assert module.definitions[0].is_test is True


def test_is_test_from_function_prefix() -> None:
    source = "function testAddition(): void {}"
    module = parse_ts_module(source, source_file="src/math.ts", module_path="src.math")
    assert module.definitions[0].is_test is True


def test_is_not_test_for_regular_function() -> None:
    source = "function add(a: number, b: number): number { return a + b; }"
    module = parse_ts_module(source, source_file="src/math.ts", module_path="src.math")
    assert module.definitions[0].is_test is False


# ── TSX ───────────────────────────────────────────────────────────────────────

def test_tsx_parses_arrow_returning_jsx() -> None:
    source = "const App = (): JSX.Element => <div>hello</div>;"
    module = parse_ts_module(source, source_file="src/App.tsx", module_path="src.App", tsx=True)
    assert len(module.definitions) == 1
    assert module.definitions[0].name == "App"
    assert module.definitions[0].kind == DefinitionKind.FUNCTION


def test_tsx_parses_component_class() -> None:
    source = "class Button extends React.Component { render() { return null; } }"
    module = parse_ts_module(source, source_file="src/Button.tsx", module_path="src.Button", tsx=True)
    assert module.definitions[0].kind == DefinitionKind.CLASS
    assert module.definitions[0].name == "Button"


# ── Module metadata ───────────────────────────────────────────────────────────

def test_module_path_preserved() -> None:
    module = parse_ts_module("", source_file="src/utils.ts", module_path="src.utils")
    assert module.module_path == "src.utils"


def test_source_file_preserved() -> None:
    module = parse_ts_module("", source_file="src/utils.ts", module_path="src.utils")
    assert module.source_file == "src/utils.ts"


def test_docstring_is_none() -> None:
    source = "/** a doc */ function foo(): void {}"
    module = parse_ts_module(source, source_file="src/foo.ts", module_path="src.foo")
    assert module.docstring is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/indexer/test_ts_parser.py -v
```

Expected: `ImportError` — `delfos.indexer.ts_parser` does not exist yet.

- [ ] **Step 3: Implement `delfos/indexer/ts_parser.py`**

Create `delfos/indexer/ts_parser.py`:

```python
"""Parse TypeScript and TSX source files into the shared ParsedModule IR.

Uses tree-sitter with the official TypeScript grammar. Produces the same
ParsedModule / ParsedDefinition frozen dataclasses consumed by the extractor,
so the rest of the pipeline is unaware of the source language.

Docstring extraction is not supported (JSDoc is not structurally part of the
AST in tree-sitter-typescript); all definitions get docstring=None.
"""

from __future__ import annotations

import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

from .parser import DefinitionKind, ParsedDefinition, ParsedModule

_TS_LANGUAGE: Language = Language(tsts.language_typescript())
_TSX_LANGUAGE: Language = Language(tsts.language_tsx())

_TEST_NAME_PREFIXES = ("test", "it", "describe")


def _is_test_file(source_file: str) -> bool:
    return ".spec." in source_file or ".test." in source_file


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _is_async(node: Node) -> bool:
    return any(child.type == "async" for child in node.children)


def _collect_throw_messages(node: Node, source_bytes: bytes) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    stack: list[Node] = list(node.children)
    while stack:
        n = stack.pop()
        if n.type == "throw_statement":
            for child in n.children:
                if child.type == "new_expression":
                    ctor = child.child_by_field_name("constructor")
                    if ctor is not None and _node_text(ctor, source_bytes).endswith("Error"):
                        args_node = child.child_by_field_name("arguments")
                        if args_node is not None:
                            for arg in args_node.children:
                                if arg.type in ("string", "template_string"):
                                    text = _node_text(arg, source_bytes).strip("'\"` ")
                                    if text not in seen:
                                        seen.add(text)
                                        result.append(text)
        stack.extend(n.children)
    return tuple(result)


def _build_function_sig(node: Node, name: str, source_bytes: bytes) -> str:
    prefix = "async function" if _is_async(node) else "function"
    params = node.child_by_field_name("parameters")
    ret = node.child_by_field_name("return_type")
    sig = f"{prefix} {name}"
    sig += _node_text(params, source_bytes) if params is not None else "()"
    if ret is not None:
        sig += _node_text(ret, source_bytes)
    return sig


def _build_method_sig(node: Node, name: str, class_name: str, source_bytes: bytes) -> str:
    prefix = "async " if _is_async(node) else ""
    params = node.child_by_field_name("parameters")
    ret = node.child_by_field_name("return_type")
    sig = f"{prefix}{class_name}.{name}"
    sig += _node_text(params, source_bytes) if params is not None else "()"
    if ret is not None:
        sig += _node_text(ret, source_bytes)
    return sig


def _build_arrow_sig(node: Node, name: str, source_bytes: bytes) -> str:
    prefix = "async " if _is_async(node) else ""
    params = node.child_by_field_name("parameters")
    ret = node.child_by_field_name("return_type")
    sig = f"const {name} = {prefix}"
    sig += _node_text(params, source_bytes) if params is not None else "()"
    if ret is not None:
        sig += _node_text(ret, source_bytes)
    sig += " =>"
    return sig


def _parse_function(
    node: Node,
    source_bytes: bytes,
    source_file: str,
    is_test_file: bool,
    class_name: str | None = None,
) -> ParsedDefinition | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    name = _node_text(name_node, source_bytes)
    is_async = _is_async(node)

    if class_name is not None:
        kind = DefinitionKind.ASYNC_METHOD if is_async else DefinitionKind.METHOD
        qualified_name = f"{class_name}.{name}"
        sig = _build_method_sig(node, name, class_name, source_bytes)
    else:
        kind = DefinitionKind.ASYNC_FUNCTION if is_async else DefinitionKind.FUNCTION
        qualified_name = name
        sig = _build_function_sig(node, name, source_bytes)

    is_test = is_test_file or name.startswith(_TEST_NAME_PREFIXES)
    return ParsedDefinition(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        signature=sig,
        docstring=None,
        source=_node_text(node, source_bytes),
        lineno=node.start_point[0] + 1,
        decorators=(),
        error_messages=_collect_throw_messages(node, source_bytes),
        is_test=is_test,
    )


def _parse_class(
    node: Node,
    source_bytes: bytes,
    source_file: str,
    is_test_file: bool,
) -> list[ParsedDefinition]:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return []
    name = _node_text(name_node, source_bytes)

    sig = f"class {name}"
    for child in node.children:
        if child.type == "class_heritage":
            sig += f" {_node_text(child, source_bytes)}"
            break

    is_test = is_test_file or name.startswith("Test")
    results: list[ParsedDefinition] = [
        ParsedDefinition(
            name=name,
            qualified_name=name,
            kind=DefinitionKind.CLASS,
            signature=sig,
            docstring=None,
            source=_node_text(node, source_bytes),
            lineno=node.start_point[0] + 1,
            decorators=(),
            error_messages=(),
            is_test=is_test,
        )
    ]

    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            if child.type == "method_definition":
                method = _parse_function(child, source_bytes, source_file, is_test_file, class_name=name)
                if method is not None:
                    results.append(method)

    return results


def _parse_interface(node: Node, source_bytes: bytes, is_test_file: bool) -> ParsedDefinition | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    name = _node_text(name_node, source_bytes)

    sig = f"interface {name}"
    for child in node.children:
        if child.type in ("extends_type_clause",):
            sig += f" {_node_text(child, source_bytes)}"
            break

    return ParsedDefinition(
        name=name,
        qualified_name=name,
        kind=DefinitionKind.INTERFACE,
        signature=sig,
        docstring=None,
        source=_node_text(node, source_bytes),
        lineno=node.start_point[0] + 1,
        decorators=(),
        error_messages=(),
        is_test=False,
    )


def _parse_type_alias(node: Node, source_bytes: bytes) -> ParsedDefinition | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    name = _node_text(name_node, source_bytes)

    type_params = node.child_by_field_name("type_parameters")
    sig = f"type {name}"
    if type_params is not None:
        sig += _node_text(type_params, source_bytes)
    value = node.child_by_field_name("value")
    if value is not None:
        sig += f" = {_node_text(value, source_bytes)}"

    return ParsedDefinition(
        name=name,
        qualified_name=name,
        kind=DefinitionKind.TYPE_ALIAS,
        signature=sig,
        docstring=None,
        source=_node_text(node, source_bytes),
        lineno=node.start_point[0] + 1,
        decorators=(),
        error_messages=(),
        is_test=False,
    )


def _parse_enum(node: Node, source_bytes: bytes) -> ParsedDefinition | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    name = _node_text(name_node, source_bytes)
    return ParsedDefinition(
        name=name,
        qualified_name=name,
        kind=DefinitionKind.ENUM,
        signature=f"enum {name}",
        docstring=None,
        source=_node_text(node, source_bytes),
        lineno=node.start_point[0] + 1,
        decorators=(),
        error_messages=(),
        is_test=False,
    )


def _parse_lexical_declaration(
    node: Node,
    source_bytes: bytes,
    source_file: str,
    is_test_file: bool,
) -> list[ParsedDefinition]:
    results: list[ParsedDefinition] = []
    for child in node.children:
        if child.type != "variable_declarator":
            continue
        name_node = child.child_by_field_name("name")
        value_node = child.child_by_field_name("value")
        if name_node is None or value_node is None:
            continue
        if value_node.type != "arrow_function":
            continue
        name = _node_text(name_node, source_bytes)
        is_async = _is_async(value_node)
        kind = DefinitionKind.ASYNC_FUNCTION if is_async else DefinitionKind.FUNCTION
        is_test = is_test_file or name.startswith(_TEST_NAME_PREFIXES)
        results.append(
            ParsedDefinition(
                name=name,
                qualified_name=name,
                kind=kind,
                signature=_build_arrow_sig(value_node, name, source_bytes),
                docstring=None,
                source=_node_text(node, source_bytes),
                lineno=node.start_point[0] + 1,
                decorators=(),
                error_messages=_collect_throw_messages(value_node, source_bytes),
                is_test=is_test,
            )
        )
    return results


def _unwrap_export(node: Node) -> Node:
    if node.type != "export_statement":
        return node
    for child in node.children:
        if child.is_named and child.type not in ("identifier", "string"):
            return child
    return node


def _process_node(
    node: Node,
    source_bytes: bytes,
    source_file: str,
    is_test_file: bool,
) -> list[ParsedDefinition]:
    inner = _unwrap_export(node)
    t = inner.type

    if t == "function_declaration":
        d = _parse_function(inner, source_bytes, source_file, is_test_file)
        return [d] if d is not None else []
    if t == "class_declaration":
        return _parse_class(inner, source_bytes, source_file, is_test_file)
    if t in ("lexical_declaration", "variable_declaration"):
        return _parse_lexical_declaration(inner, source_bytes, source_file, is_test_file)
    if t == "interface_declaration":
        d = _parse_interface(inner, source_bytes, is_test_file)
        return [d] if d is not None else []
    if t == "type_alias_declaration":
        d = _parse_type_alias(inner, source_bytes)
        return [d] if d is not None else []
    if t == "enum_declaration":
        d = _parse_enum(inner, source_bytes)
        return [d] if d is not None else []
    return []


def parse_ts_module(
    source: str,
    *,
    source_file: str,
    module_path: str,
    tsx: bool = False,
) -> ParsedModule:
    """Parse TypeScript or TSX *source* into a :class:`ParsedModule`.

    Parameters
    ----------
    source:
        Full source text.
    source_file:
        Filesystem path of the source file (stored verbatim on the result).
    module_path:
        Dotted module path (e.g. ``"src.components.Button"``).
    tsx:
        ``True`` to use the TSX grammar (for ``.tsx`` files).
    """
    lang = _TSX_LANGUAGE if tsx else _TS_LANGUAGE
    parser = Parser(lang)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    is_test_file = _is_test_file(source_file)
    definitions: list[ParsedDefinition] = []
    for node in tree.root_node.children:
        definitions.extend(_process_node(node, source_bytes, source_file, is_test_file))

    return ParsedModule(
        module_path=module_path,
        source_file=source_file,
        language="tsx" if tsx else "typescript",
        docstring=None,
        source=source,
        definitions=tuple(definitions),
    )
```

- [ ] **Step 4: Export `parse_ts_module` from `delfos/indexer/__init__.py`**

Change `delfos/indexer/__init__.py` from:

```python
from .parser import (
    DefinitionKind,
    ParsedDefinition,
    ParsedModule,
    parse_module,
)
from .pipeline import Indexer, IndexStats

__all__ = [
    "DefinitionKind",
    "Embedder",
    "ExtractionResult",
    "IndexStats",
    "Indexer",
    "OpenAIEmbedder",
    "ParsedDefinition",
    "ParsedModule",
    "extract",
    "parse_module",
]
```

to:

```python
from .parser import (
    DefinitionKind,
    ParsedDefinition,
    ParsedModule,
    parse_module,
)
from .pipeline import Indexer, IndexStats
from .ts_parser import parse_ts_module

__all__ = [
    "DefinitionKind",
    "Embedder",
    "ExtractionResult",
    "IndexStats",
    "Indexer",
    "OpenAIEmbedder",
    "ParsedDefinition",
    "ParsedModule",
    "extract",
    "parse_module",
    "parse_ts_module",
]
```

- [ ] **Step 5: Run tests and confirm they pass**

```
uv run pytest tests/indexer/test_ts_parser.py -v
```

Expected: all tests PASS. If a test fails due to a subtle tree-sitter field name difference (e.g., `"parameters"` vs `"formal_parameters"`), inspect with:

```python
# scratch script to print node fields
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser
lang = Language(tsts.language_typescript())
p = Parser(lang)
src = b"function foo(x: number): void {}"
tree = p.parse(src)
def dump(n, indent=0):
    print(" " * indent + f"{n.type!r} [{n.start_byte}:{n.end_byte}]")
    for c in n.children:
        dump(c, indent + 2)
dump(tree.root_node)
```

- [ ] **Step 6: Run full suite to confirm no regressions**

```
uv run pytest --ignore=tests/test_e2e.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Lint, format, type-check**

```
uv run ruff check . && uv run ruff format . && uv run pyright
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add delfos/indexer/ts_parser.py delfos/indexer/__init__.py \
        tests/indexer/test_ts_parser.py
git commit -m "feat(indexer): TypeScript/TSX parser via tree-sitter"
```

---

## Task 3: Pipeline Integration

**Files:**
- Modify: `delfos/indexer/pipeline.py`
- Create: `tests/indexer/test_pipeline_ts.py`

**Interfaces:**
- Consumes: `parse_ts_module` from Task 2; `_module_path`, `_discover`, `_index_file` internal to `pipeline.py`
- Produces: `Indexer.index()` picks up `.ts` and `.tsx` files alongside `.py`

- [ ] **Step 1: Write failing tests**

Create `tests/indexer/test_pipeline_ts.py`:

```python
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from delfos.indexer import Embedder, Indexer
from delfos.indexer.pipeline import _module_path
from delfos.store import NativeGraphStore

# ── Deterministic embedder (no network) ──────────────────────────────────────

HASH_DIM = 32
HASH_MODEL = "hash-sha256-d32"


class HashEmbedder:
    @property
    def model(self) -> str:
        return HASH_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return HASH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(b) - 128.0 for b in digest]
            length = math.sqrt(sum(x * x for x in raw)) or 1.0
            results.append([x / length for x in raw])
        return results


assert isinstance(HashEmbedder(), Embedder)


# ── _module_path unit tests ────────────────────────────────────────────────────

def test_module_path_ts_regular() -> None:
    assert _module_path("src/utils.ts") == "src.utils"


def test_module_path_ts_index() -> None:
    assert _module_path("src/index.ts") == "src"


def test_module_path_tsx_regular() -> None:
    assert _module_path("src/App.tsx") == "src.App"


def test_module_path_tsx_index() -> None:
    assert _module_path("src/index.tsx") == "src"


def test_module_path_py_regular() -> None:
    assert _module_path("delfos/store/base.py") == "delfos.store.base"


def test_module_path_py_init() -> None:
    assert _module_path("delfos/__init__.py") == "delfos"


# ── Pipeline integration ───────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path: Path) -> NativeGraphStore:
    s = NativeGraphStore(tmp_path / "store", embedding_dim=HASH_DIM, embedding_model=HASH_MODEL)
    s.initialize()
    return s


def test_discovers_ts_files(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def hello(): pass")
    (tmp_path / "app.ts").write_text("function greet(): void {}")
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.indexed_files == 2
    assert len(stats.failed_files) == 0


def test_discovers_tsx_files(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "App.tsx").write_text("const App = (): JSX.Element => <div/>;" )
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.indexed_files == 1


def test_ts_nodes_written_to_store(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "greeter.ts").write_text("export function greet(name: string): string { return name; }")
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.nodes_written > 0
    cue = store.get_node("cue:symbol:greeter.ts::greet")
    assert cue is not None


def test_python_files_still_indexed(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "util.py").write_text("def helper(): pass")
    indexer = Indexer(store, HashEmbedder())
    stats = indexer.index(tmp_path)
    assert stats.indexed_files == 1
    cue = store.get_node("cue:symbol:util.py::helper")
    assert cue is not None


def test_ts_incremental_skip(store: NativeGraphStore, tmp_path: Path) -> None:
    (tmp_path / "app.ts").write_text("function foo(): void {}")
    indexer = Indexer(store, HashEmbedder())
    indexer.index(tmp_path)
    stats2 = indexer.index(tmp_path)
    assert stats2.indexed_files == 0
    assert stats2.skipped_files == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run pytest tests/indexer/test_pipeline_ts.py -v
```

Expected: `_module_path` tests fail (no `.ts`/`.tsx` support yet), pipeline tests fail (only `.py` discovered).

- [ ] **Step 3: Update `pipeline.py` — `_module_path`, `_discover`, `_index_file`**

In `delfos/indexer/pipeline.py`, replace the existing `_module_path` function:

```python
def _module_path(relative_path: str) -> str:
    """Convert a posix relative path to a dotted module path.

    ``delfos/schema/nodes.py`` -> ``delfos.schema.nodes``;
    ``delfos/__init__.py`` -> ``delfos``.
    """
    parts = relative_path[: -len(".py")].split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)
```

with:

```python
def _module_path(relative_path: str) -> str:
    """Convert a posix relative path to a dotted module path.

    ``delfos/schema/nodes.py`` -> ``delfos.schema.nodes``;
    ``delfos/__init__.py`` -> ``delfos``;
    ``src/index.ts`` -> ``src``;
    ``src/App.tsx`` -> ``src.App``.
    """
    for ext, index_name in [(".tsx", "index"), (".ts", "index"), (".py", "__init__")]:
        if relative_path.endswith(ext):
            parts = relative_path[: -len(ext)].split("/")
            if parts and parts[-1] == index_name:
                parts = parts[:-1]
            return ".".join(parts)
    return relative_path
```

In `_discover`, replace:

```python
            for name in filenames:
                if name.endswith(".py"):
                    found.append(Path(dirpath) / name)
```

with:

```python
            for name in filenames:
                if name.endswith((".py", ".ts", ".tsx")):
                    found.append(Path(dirpath) / name)
```

Add the import for `parse_ts_module` at the top of the file, alongside the existing parser import. Change:

```python
from .parser import parse_module
```

to:

```python
from .parser import parse_module
from .ts_parser import parse_ts_module
```

In `_index_file`, replace:

```python
        try:
            source = data.decode("utf-8")
            module = parse_module(
                source,
                source_file=relative_path,
                module_path=_module_path(relative_path),
            )
        except (SyntaxError, UnicodeDecodeError):
            return False
```

with:

```python
        try:
            source = data.decode("utf-8")
            mp = _module_path(relative_path)
            if relative_path.endswith(".py"):
                module = parse_module(source, source_file=relative_path, module_path=mp)
            else:
                tsx = relative_path.endswith(".tsx")
                module = parse_ts_module(source, source_file=relative_path, module_path=mp, tsx=tsx)
        except (SyntaxError, UnicodeDecodeError):
            return False
```

- [ ] **Step 4: Run tests and confirm they pass**

```
uv run pytest tests/indexer/test_pipeline_ts.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

```
uv run pytest --ignore=tests/test_e2e.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Lint, format, type-check**

```
uv run ruff check . && uv run ruff format . && uv run pyright
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add delfos/indexer/pipeline.py tests/indexer/test_pipeline_ts.py
git commit -m "feat(indexer): extend pipeline to discover and index .ts and .tsx files"
```

---

## Final Verification

- [ ] Run the full test suite one last time (including e2e if a live store is available):

```
uv run pytest -v
```

- [ ] Verify `index.sh` now indexes TypeScript files in the target repo:

```
bash index.sh
```

Expected output should show `indexed: N` where N > 0 for any `.ts`/`.tsx` files in `~/Developer/delfos/`.
