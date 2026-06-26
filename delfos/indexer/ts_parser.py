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
                method = _parse_function(
                    child, source_bytes, source_file, is_test_file, class_name=name
                )
                if method is not None:
                    results.append(method)

    return results


def _parse_interface(
    node: Node, source_bytes: bytes, is_test_file: bool
) -> ParsedDefinition | None:
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
