from __future__ import annotations

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
    module = parse_ts_module(
        source, source_file="src/Button.tsx", module_path="src.Button", tsx=True
    )
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
