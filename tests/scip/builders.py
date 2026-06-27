"""Helpers to synthesize minimal SCIP protobuf indexes for tests.

These build ``scip_pb2`` messages directly and serialize them to a file so we
can exercise :class:`delfos.scip.reader.ScipIndex` (and everything layered on
top of it) without invoking the external ``scip-python`` binary.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from delfos.scip import scip_pb2  # type: ignore[attr-defined]

DEFINITION_ROLE: int = int(scip_pb2.Definition)


def occurrence(
    symbol: str,
    line: int,
    *,
    start_col: int = 0,
    end_col: int = 10,
    definition: bool = False,
) -> scip_pb2.Occurrence:
    """A single-line occurrence; ``line`` is SCIP 0-based."""
    occ = scip_pb2.Occurrence(
        symbol=symbol,
        single_line_range=scip_pb2.SingleLineRange(
            line=line, start_character=start_col, end_character=end_col
        ),
    )
    if definition:
        occ.symbol_roles = DEFINITION_ROLE
    return occ


def legacy_occurrence(symbol: str, range_ints: list[int]) -> scip_pb2.Occurrence:
    """An occurrence using the deprecated compact repeated-int32 ``range`` field."""
    return scip_pb2.Occurrence(symbol=symbol, range=range_ints)


def malformed_occurrence(symbol: str) -> scip_pb2.Occurrence:
    """An occurrence carrying no range information at all."""
    return scip_pb2.Occurrence(symbol=symbol)


def relationship(
    symbol: str,
    *,
    is_reference: bool = False,
    is_implementation: bool = False,
    is_type_definition: bool = False,
    is_definition: bool = False,
) -> scip_pb2.Relationship:
    return scip_pb2.Relationship(
        symbol=symbol,
        is_reference=is_reference,
        is_implementation=is_implementation,
        is_type_definition=is_type_definition,
        is_definition=is_definition,
    )


def symbol_information(
    symbol: str, relationships: Iterable[scip_pb2.Relationship] = ()
) -> scip_pb2.SymbolInformation:
    return scip_pb2.SymbolInformation(symbol=symbol, relationships=list(relationships))


def document(
    relative_path: str,
    occurrences: Iterable[scip_pb2.Occurrence] = (),
    symbols: Iterable[scip_pb2.SymbolInformation] = (),
) -> scip_pb2.Document:
    return scip_pb2.Document(
        relative_path=relative_path,
        occurrences=list(occurrences),
        symbols=list(symbols),
    )


def write_index(
    path: Path,
    documents: Iterable[scip_pb2.Document] = (),
    external_symbols: Iterable[scip_pb2.SymbolInformation] = (),
) -> Path:
    """Serialize an :class:`Index` of ``documents`` to ``path`` and return it."""
    index = scip_pb2.Index(documents=list(documents), external_symbols=list(external_symbols))
    path.write_bytes(index.SerializeToString())
    return path
