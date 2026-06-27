"""Tests for delfos.indexer.parser (tree-sitter-based)."""

from __future__ import annotations

from delfos.indexer.parser import DefinitionKind, parse_module


def _parse(source: str) -> tuple[str | None, list[tuple[str, str, DefinitionKind]]]:
    """Helper: return (module_docstring, [(name, qualified_name, kind), ...])."""
    m = parse_module(source, source_file="test.py", module_path="test")
    return m.docstring, [(d.name, d.qualified_name, d.kind) for d in m.definitions]


# ---------------------------------------------------------------------------
# Module docstring
# ---------------------------------------------------------------------------


def test_module_docstring_extracted() -> None:
    m = parse_module('"""Module doc."""\n', source_file="f.py", module_path="f")
    assert m.docstring == "Module doc."


def test_module_docstring_triple_quotes_cleaned() -> None:
    src = '"""\n    Indented doc.\n    """\n'
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.docstring == "Indented doc."


def test_module_docstring_none_when_import_first() -> None:
    src = "import os\n\ndef foo(): pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.docstring is None


def test_module_docstring_none_for_empty_file() -> None:
    m = parse_module("", source_file="f.py", module_path="f")
    assert m.docstring is None


# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------


def test_plain_function() -> None:
    src = 'def foo(x: int, y: str = "hi") -> bool:\n    """Doc."""\n    return True\n'
    m = parse_module(src, source_file="f.py", module_path="f")
    assert len(m.definitions) == 1
    d = m.definitions[0]
    assert d.name == "foo"
    assert d.qualified_name == "foo"
    assert d.kind == DefinitionKind.FUNCTION
    assert d.docstring == "Doc."
    assert d.signature == 'def foo(x: int, y: str = "hi") -> bool'
    assert d.lineno == 1
    assert not d.is_test
    assert d.decorators == ()


def test_async_function() -> None:
    src = "async def fetch(url: str) -> None:\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert len(m.definitions) == 1
    d = m.definitions[0]
    assert d.kind == DefinitionKind.ASYNC_FUNCTION
    assert d.signature == "async def fetch(url: str) -> None"


def test_function_no_return_annotation() -> None:
    src = "def bare(x):\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    d = m.definitions[0]
    assert d.signature == "def bare(x)"


def test_function_no_docstring() -> None:
    src = "def nodoc():\n    return 1\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].docstring is None


def test_function_is_test_flag() -> None:
    src = "def test_something():\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].is_test is True


def test_decorated_function() -> None:
    src = "@staticmethod\n@cache\ndef decorated():\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert len(m.definitions) == 1
    d = m.definitions[0]
    assert d.name == "decorated"
    assert d.kind == DefinitionKind.FUNCTION
    assert d.decorators == ("staticmethod", "cache")


# ---------------------------------------------------------------------------
# Error message extraction
# ---------------------------------------------------------------------------


def test_error_messages_extracted() -> None:
    src = (
        "def validate(x: int) -> None:\n"
        "    if x < 0:\n"
        '        raise ValueError("negative value")\n'
        "    if x > 100:\n"
        '        raise RuntimeError("too large")\n'
    )
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].error_messages == ("negative value", "too large")


def test_error_messages_deduplicated() -> None:
    src = 'def f() -> None:\n    raise ValueError("dup")\n    raise ValueError("dup")\n'
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].error_messages == ("dup",)


def test_no_error_messages_when_none_raised() -> None:
    src = "def f():\n    return 42\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].error_messages == ()


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------


def test_class_with_methods() -> None:
    src = (
        "class MyClass(Base):\n"
        '    """Class doc."""\n'
        "\n"
        "    def method(self) -> str:\n"
        '        """Method doc."""\n'
        '        return ""\n'
        "\n"
        "    async def async_method(self) -> None:\n"
        "        pass\n"
    )
    m = parse_module(src, source_file="f.py", module_path="f")
    assert len(m.definitions) == 3

    cls = m.definitions[0]
    assert cls.name == "MyClass"
    assert cls.qualified_name == "MyClass"
    assert cls.kind == DefinitionKind.CLASS
    assert cls.docstring == "Class doc."
    assert cls.signature == "class MyClass(Base)"

    method = m.definitions[1]
    assert method.name == "method"
    assert method.qualified_name == "MyClass.method"
    assert method.kind == DefinitionKind.METHOD
    assert method.docstring == "Method doc."

    async_method = m.definitions[2]
    assert async_method.name == "async_method"
    assert async_method.qualified_name == "MyClass.async_method"
    assert async_method.kind == DefinitionKind.ASYNC_METHOD


def test_class_no_bases() -> None:
    src = "class Plain:\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].signature == "class Plain"


def test_class_multiple_bases() -> None:
    src = "class Multi(A, B, C):\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].signature == "class Multi(A, B, C)"


def test_class_is_test_flag() -> None:
    src = "class TestFoo:\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].is_test is True


def test_class_non_test_name() -> None:
    src = "class Foo:\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert m.definitions[0].is_test is False


def test_decorated_method_inside_class() -> None:
    src = "class Cls:\n    @staticmethod\n    def static_m():\n        pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert len(m.definitions) == 2
    method = m.definitions[1]
    assert method.name == "static_m"
    assert method.qualified_name == "Cls.static_m"
    assert method.decorators == ("staticmethod",)


# ---------------------------------------------------------------------------
# Mixed top-level content
# ---------------------------------------------------------------------------


def test_multiple_top_level_definitions() -> None:
    src = "def foo(): pass\nclass Bar: pass\nasync def baz(): pass\n"
    docstring, items = _parse(src)
    assert docstring is None
    names = [name for name, _, _ in items]
    assert names == ["foo", "Bar", "baz"]


def test_source_field_populated() -> None:
    src = "def f():\n    pass\n"
    m = parse_module(src, source_file="f.py", module_path="f")
    assert "def f():" in m.definitions[0].source


# ---------------------------------------------------------------------------
# Error recovery — syntax errors do not raise
# ---------------------------------------------------------------------------


def test_syntax_error_returns_partial_results() -> None:
    src = (
        "def good():\n"
        "    pass\n"
        "\n"
        "def bad(:\n"  # syntax error
        "    pass\n"
        "\n"
        "class Fine:\n"
        "    pass\n"
    )
    m = parse_module(src, source_file="broken.py", module_path="broken")
    # tree-sitter recovers; at minimum `good` and `Fine` should be extracted
    names = [d.name for d in m.definitions]
    assert "good" in names
    assert "Fine" in names
    # no exception raised
