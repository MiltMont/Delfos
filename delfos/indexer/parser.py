"""Parse Python source files into a ``ParsedModule`` intermediate representation.

Uses the stdlib :mod:`ast` module to extract top-level functions, async functions,
and classes (with one level of method recursion) from Python source text.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum


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


def _build_signature_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """Reconstruct the signature line for a function or method definition."""
    is_async = isinstance(node, ast.AsyncFunctionDef)
    prefix = "async def" if is_async else "def"
    sig = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def _build_signature_class(node: ast.ClassDef) -> str:
    """Reconstruct the signature line for a class definition."""
    sig = f"class {node.name}"
    if node.bases:
        sig += f"({', '.join(ast.unparse(b) for b in node.bases)})"
    return sig


def _collect_error_messages(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Extract string arguments from ``raise`` calls inside a function body.

    Walks the entire subtree of *node*. For every ``ast.Raise`` whose ``exc``
    is an ``ast.Call``, each positional argument that is a string constant is
    collected. Results are deduped in first-seen order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise):
            continue
        exc = child.exc
        if not isinstance(exc, ast.Call):
            continue
        for arg in exc.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value not in seen:
                    seen.add(arg.value)
                    result.append(arg.value)
    return tuple(result)


def _get_source(source: str, node: ast.AST) -> str:
    """Return the source segment for *node*, falling back to ``""``."""
    segment = ast.get_source_segment(source, node)
    if segment is None:
        return ""
    return segment


def _parse_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    *,
    class_name: str | None = None,
) -> ParsedDefinition:
    """Build a ``ParsedDefinition`` for a function or method node."""
    if class_name is not None:
        if isinstance(node, ast.AsyncFunctionDef):
            kind = DefinitionKind.ASYNC_METHOD
        else:
            kind = DefinitionKind.METHOD
        qualified_name = f"{class_name}.{node.name}"
    else:
        if isinstance(node, ast.AsyncFunctionDef):
            kind = DefinitionKind.ASYNC_FUNCTION
        else:
            kind = DefinitionKind.FUNCTION
        qualified_name = node.name

    return ParsedDefinition(
        name=node.name,
        qualified_name=qualified_name,
        kind=kind,
        signature=_build_signature_function(node),
        docstring=ast.get_docstring(node),
        source=_get_source(source, node),
        lineno=node.lineno,
        decorators=tuple(ast.unparse(d) for d in node.decorator_list),
        error_messages=_collect_error_messages(node),
        is_test=node.name.startswith("test"),
    )


def _parse_class(node: ast.ClassDef, source: str) -> list[ParsedDefinition]:
    """Build ``ParsedDefinition`` entries for a class and its direct methods."""
    definitions: list[ParsedDefinition] = []

    class_def = ParsedDefinition(
        name=node.name,
        qualified_name=node.name,
        kind=DefinitionKind.CLASS,
        signature=_build_signature_class(node),
        docstring=ast.get_docstring(node),
        source=_get_source(source, node),
        lineno=node.lineno,
        decorators=tuple(ast.unparse(d) for d in node.decorator_list),
        error_messages=(),
        is_test=node.name.startswith("Test"),
    )
    definitions.append(class_def)

    for child in node.body:
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            definitions.append(
                _parse_function(child, source, class_name=node.name),
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

    Raises
    ------
    SyntaxError
        Propagated from :func:`ast.parse` when *source* is not valid Python.
    """
    tree = ast.parse(source)

    definitions: list[ParsedDefinition] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            definitions.append(_parse_function(node, source))
        elif isinstance(node, ast.ClassDef):
            definitions.extend(_parse_class(node, source))

    return ParsedModule(
        module_path=module_path,
        source_file=source_file,
        docstring=ast.get_docstring(tree),
        source=source,
        definitions=tuple(definitions),
    )
