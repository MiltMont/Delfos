"""Tests for the SCIP index reader (delfos.scip.reader)."""

from __future__ import annotations

from pathlib import Path

from delfos.scip.reader import ScipIndex

from .builders import (
    document,
    legacy_occurrence,
    malformed_occurrence,
    occurrence,
    relationship,
    symbol_information,
    write_index,
)

SYM = "scip-python python . a/foo()."
OTHER = "scip-python python . a/bar()."


def _index(path: Path) -> ScipIndex:
    write_index(
        path,
        documents=[
            document(
                "a.py",
                occurrences=[
                    occurrence(SYM, 4, definition=True),
                    occurrence(OTHER, 10),
                ],
                symbols=[
                    symbol_information(
                        SYM,
                        [
                            relationship("iface#", is_implementation=True),
                            relationship("Type#", is_type_definition=True),
                            relationship("ref#", is_reference=True),
                        ],
                    )
                ],
            ),
            document(
                "b.py",
                occurrences=[
                    occurrence(SYM, 7),
                    occurrence(SYM, 12),
                ],
            ),
        ],
    )
    return ScipIndex(path)


def test_occurrences_decode_single_line_range(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    occs = idx.occurrences("a.py")
    first = occs[0]
    assert first.symbol == SYM
    assert (first.start_line, first.start_col, first.end_line, first.end_col) == (4, 0, 4, 10)
    assert first.is_definition is True


def test_occurrences_missing_file_is_empty(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    assert idx.occurrences("does/not/exist.py") == []


def test_definitions_only_returns_definition_occurrences(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    defs = idx.definitions("a.py")
    assert [d.symbol for d in defs] == [SYM]
    assert defs[0].start_line == 4


def test_references_excludes_definition_and_spans_files(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    refs = idx.references(SYM)
    # The definition in a.py is excluded; the two usages in b.py remain.
    assert [(path, occ.start_line) for path, occ in refs] == [("b.py", 7), ("b.py", 12)]
    assert all(not occ.is_definition for _, occ in refs)


def test_references_unknown_symbol_is_empty(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    assert idx.references("scip-python python . nope#") == []


def test_symbol_info_decodes_relationships(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    info = idx.symbol_info(SYM, relative_path="a.py")
    assert info is not None
    impls = [r.symbol for r in info.relationships if r.is_implementation]
    types = [r.symbol for r in info.relationships if r.is_type_definition]
    assert impls == ["iface#"]
    assert types == ["Type#"]


def test_symbol_info_falls_back_to_external_symbols(tmp_path: Path) -> None:
    path = tmp_path / "index.scip"
    write_index(
        path,
        documents=[document("a.py", occurrences=[occurrence(SYM, 1)])],
        external_symbols=[symbol_information(SYM, [relationship("ext#", is_implementation=True)])],
    )
    idx = ScipIndex(path)
    info = idx.symbol_info(SYM)
    assert info is not None
    assert [r.symbol for r in info.relationships] == ["ext#"]


def test_symbol_info_unknown_symbol_is_none(tmp_path: Path) -> None:
    idx = _index(tmp_path / "index.scip")
    assert idx.symbol_info("scip-python python . nope#") is None


def test_decode_range_legacy_three_and_four_int_forms(tmp_path: Path) -> None:
    path = tmp_path / "index.scip"
    write_index(
        path,
        documents=[
            document(
                "legacy.py",
                occurrences=[
                    legacy_occurrence("three#", [3, 1, 9]),
                    legacy_occurrence("four#", [3, 1, 5, 9]),
                ],
            )
        ],
    )
    idx = ScipIndex(path)
    three, four = idx.occurrences("legacy.py")
    assert (three.start_line, three.start_col, three.end_line, three.end_col) == (3, 1, 3, 9)
    assert (four.start_line, four.start_col, four.end_line, four.end_col) == (3, 1, 5, 9)


def test_decode_range_malformed_returns_sentinel(tmp_path: Path) -> None:
    path = tmp_path / "index.scip"
    write_index(
        path,
        documents=[document("bad.py", occurrences=[malformed_occurrence("bad#")])],
    )
    idx = ScipIndex(path)
    (occ,) = idx.occurrences("bad.py")
    assert (occ.start_line, occ.start_col, occ.end_line, occ.end_col) == (0, 0, 0, 0)
