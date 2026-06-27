"""Parse Python source files into a ``ParsedModule`` intermediate representation.

Uses tree-sitter for error-tolerant parsing. Unlike the stdlib :mod:`ast` module,
files with syntax errors are parsed partially (using tree-sitter error recovery)
rather than raising :exc:`SyntaxError`; definitions that cannot be extracted are
silently skipped.

Extracts top-level functions, async functions, and classes (with one level of
method recursion) from Python source text.
"""

from __future__ import annotations

import ast
import inspect
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import tree_sitter_python as _tsp
from tree_sitter import Language, Node, Parser

_PY_LANGUAGE: Language = Language(_tsp.language())
_PARSER: Parser = Parser(_PY_LANGUAGE)

_FUNC_TYPES: frozenset[str] = frozenset({"function_definition"})
_TRIVIAL_NODE_TYPES: frozenset[str] = frozenset({"newline", "comment", "encoding_declaration"})


class DefinitionKind(StrEnum):
    """Discriminator for the kind of definition extracted from source."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    CLASS = "class"


@dataclass(frozen=True)
class ParsedDefinition:
    """A single definition (function, method, or class) extracted from source."""

    name: str
    qualified_name: str
    kind: DefinitionKind
    signature: str | None
    docstring: str | None
    source: str
    lineno: int
    decorators: tuple[str, ...]
    error_messages: tuple[str, ...]
    is_test: bool


@dataclass(frozen=True)
class ParsedModule:
    """The full parse result for one Python module."""

    module_path: str
    source_file: str
    docstring: str | None
    source: str
    definitions: tuple[ParsedDefinition, ...]


def _get_module_docstring(root: Node) -> str | None:
    """Return the module docstring if the first non-trivial statement is a string literal."""
    for child in root.children:
        if child.type in _TRIVIAL_NODE_TYPES:
            continue
        if child.type == "expression_statement":
            for expr in child.children:
                if expr.type == "string":
                    value = _parse_string_node(expr)
                    if value is not None:
                        return inspect.cleandoc(value)
        return None
    return None


def _node_text(node: Node) -> str:
    raw = node.text
    return raw.decode("utf-8") if raw is not None else ""


def _parse_string_node(node: Node) -> str | None:
    raw = node.text
    if raw is None:
        return None
    try:
        value = ast.literal_eval(raw.decode("utf-8"))
        return value if isinstance(value, str) else None
    except Exception:
        return None


def _get_docstring(body: Node) -> str | None:
    """Return the docstring from a function/class/module body node, or None."""
    if not body.children:
        return None
    first = body.children[0]
    if first.type != "expression_statement":
        return None
    for child in first.children:
        if child.type == "string":
            value = _parse_string_node(child)
            if value is not None:
                return inspect.cleandoc(value)
    return None


def _get_decorators(decorated_node: Node) -> tuple[str, ...]:
    """Extract decorator expression texts from a ``decorated_definition`` node."""
    result: list[str] = []
    for child in decorated_node.children:
        if child.type == "decorator":
            text = _node_text(child).strip()
            if text.startswith("@"):
                text = text[1:].strip()
            if text:
                result.append(text)
    return tuple(result)


def _build_function_signature(node: Node, *, is_async: bool) -> str:
    prefix = "async def" if is_async else "def"
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node is not None else "?"
    params_node = node.child_by_field_name("parameters")
    params = _node_text(params_node) if params_node is not None else "()"
    ret_node = node.child_by_field_name("return_type")
    sig = f"{prefix} {name}{params}"
    if ret_node is not None:
        sig += f" -> {_node_text(ret_node)}"
    return sig


def _build_class_signature(node: Node) -> str:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node is not None else "?"
    superclasses_node = node.child_by_field_name("superclasses")
    sig = f"class {name}"
    if superclasses_node is not None:
        sig += _node_text(superclasses_node)
    return sig


def _collect_error_messages(node: Node) -> tuple[str, ...]:
    """Extract string arguments from ``raise`` calls inside a function subtree."""
    seen: set[str] = set()
    result: list[str] = []
    queue: deque[Node] = deque(node.children)
    while queue:
        n = queue.popleft()
        if n.type == "raise_statement":
            for child in n.children:
                if child.type == "call":
                    args = child.child_by_field_name("arguments")
                    if args is not None:
                        for arg in args.children:
                            if arg.type == "string":
                                value = _parse_string_node(arg)
                                if value is not None and value not in seen:
                                    seen.add(value)
                                    result.append(value)
        queue.extend(n.children)
    return tuple(result)


def _parse_function_node(
    node: Node,
    *,
    class_name: str | None = None,
    decorators: tuple[str, ...] = (),
) -> ParsedDefinition:
    is_async = any(c.type == "async" for c in node.children)
    if class_name is not None:
        kind = DefinitionKind.ASYNC_METHOD if is_async else DefinitionKind.METHOD
    else:
        kind = DefinitionKind.ASYNC_FUNCTION if is_async else DefinitionKind.FUNCTION

    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node is not None else ""
    qualified_name = f"{class_name}.{name}" if class_name is not None else name

    body_node = node.child_by_field_name("body")
    docstring = _get_docstring(body_node) if body_node is not None else None

    return ParsedDefinition(
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        signature=_build_function_signature(node, is_async=is_async),
        docstring=docstring,
        source=_node_text(node),
        lineno=node.start_point[0] + 1,
        decorators=decorators,
        error_messages=_collect_error_messages(node),
        is_test=name.startswith("test"),
    )


def _parse_class_node(
    node: Node,
    *,
    decorators: tuple[str, ...] = (),
) -> list[ParsedDefinition]:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node is not None else ""

    body_node = node.child_by_field_name("body")
    docstring = _get_docstring(body_node) if body_node is not None else None

    class_def = ParsedDefinition(
        name=name,
        qualified_name=name,
        kind=DefinitionKind.CLASS,
        signature=_build_class_signature(node),
        docstring=docstring,
        source=_node_text(node),
        lineno=node.start_point[0] + 1,
        decorators=decorators,
        error_messages=(),
        is_test=name.startswith("Test"),
    )

    definitions: list[ParsedDefinition] = [class_def]

    if body_node is not None:
        for child in body_node.children:
            if child.type in _FUNC_TYPES:
                definitions.append(_parse_function_node(child, class_name=name))
            elif child.type == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is not None and inner.type in _FUNC_TYPES:
                    method_decorators = _get_decorators(child)
                    definitions.append(
                        _parse_function_node(inner, class_name=name, decorators=method_decorators)
                    )

    return definitions


def parse_module(source: str, *, source_file: str, module_path: str) -> ParsedModule:
    """Parse Python *source* into a :class:`ParsedModule`.

    Parameters
    ----------
    source:
        The full Python source text to parse.
    source_file:
        Filesystem path of the source file (stored verbatim on the result).
    module_path:
        Dotted module path (e.g. ``"delfos.indexer.parser"``).

    Returns
    -------
    ParsedModule
        The intermediate representation consumed by the downstream extractor.
        Syntax errors do not raise; tree-sitter recovers what it can.
    """
    tree = _PARSER.parse(source.encode("utf-8"))
    root = tree.root_node

    definitions: list[ParsedDefinition] = []

    for child in root.children:
        node_type = child.type

        if node_type in _FUNC_TYPES:
            definitions.append(_parse_function_node(child))
        elif node_type == "class_definition":
            definitions.extend(_parse_class_node(child))
        elif node_type == "decorated_definition":
            inner = child.child_by_field_name("definition")
            if inner is not None:
                deco_tuple = _get_decorators(child)
                if inner.type in _FUNC_TYPES:
                    definitions.append(_parse_function_node(inner, decorators=deco_tuple))
                elif inner.type == "class_definition":
                    definitions.extend(_parse_class_node(inner, decorators=deco_tuple))

    return ParsedModule(
        module_path=module_path,
        source_file=source_file,
        docstring=_get_module_docstring(root),
        source=source,
        definitions=tuple(definitions),
    )
