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
