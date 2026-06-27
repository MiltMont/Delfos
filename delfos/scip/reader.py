from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from delfos.scip import scip_pb2  # type: ignore[attr-defined]

_DEFINITION_ROLE: int = int(scip_pb2.Definition)


def _decode_range(occ: scip_pb2.Occurrence) -> tuple[int, int, int, int]:
    """Return (start_line, start_col, end_line, end_col) from any range encoding."""
    if occ.HasField("single_line_range"):
        r = occ.single_line_range
        return r.line, r.start_character, r.line, r.end_character
    if occ.HasField("multi_line_range"):
        r = occ.multi_line_range
        return r.start_line, r.start_character, r.end_line, r.end_character
    # Legacy compact encoding: [start_line, start_col, end_col] or [start_line, start_col, end_line, end_col]
    r = occ.range
    if len(r) == 3:
        return r[0], r[1], r[0], r[2]
    return r[0], r[1], r[2], r[3]


@dataclass(frozen=True)
class Occurrence:
    symbol: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    is_definition: bool


@dataclass(frozen=True)
class Relationship:
    symbol: str
    is_reference: bool
    is_implementation: bool
    is_type_definition: bool
    is_definition: bool


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    relationships: tuple[Relationship, ...]


class ScipIndex:
    def __init__(self, path: Path) -> None:
        index = scip_pb2.Index()
        index.ParseFromString(path.read_bytes())

        self._docs: dict[str, scip_pb2.Document] = {
            doc.relative_path: doc for doc in index.documents
        }
        self._external: dict[str, scip_pb2.SymbolInformation] = {
            sym.symbol: sym for sym in index.external_symbols
        }

    @property
    def files(self) -> list[str]:
        return list(self._docs)

    def occurrences(self, relative_path: str) -> list[Occurrence]:
        doc = self._docs.get(relative_path)
        if doc is None:
            return []
        result: list[Occurrence] = []
        for occ in doc.occurrences:
            start_line, start_col, end_line, end_col = _decode_range(occ)
            is_def = bool(occ.symbol_roles & _DEFINITION_ROLE)
            result.append(Occurrence(occ.symbol, start_line, start_col, end_line, end_col, is_def))
        return result

    def definitions(self, relative_path: str) -> list[Occurrence]:
        return [o for o in self.occurrences(relative_path) if o.is_definition]

    def references(self, symbol: str) -> list[tuple[str, Occurrence]]:
        """All occurrences of *symbol* across every file as (relative_path, occurrence) pairs."""
        out: list[tuple[str, Occurrence]] = []
        for path in self._docs:
            for occ in self.occurrences(path):
                if occ.symbol == symbol:
                    out.append((path, occ))
        return out

    def symbol_info(self, symbol: str, relative_path: str | None = None) -> SymbolInfo | None:
        """Look up SymbolInformation for *symbol*.

        Checks the document-local symbols first (if relative_path given), then
        falls back to external_symbols.
        """
        sym_pb: scip_pb2.SymbolInformation | None = None
        if relative_path is not None:
            doc = self._docs.get(relative_path)
            if doc is not None:
                for s in doc.symbols:
                    if s.symbol == symbol:
                        sym_pb = s
                        break
        if sym_pb is None:
            sym_pb = self._external.get(symbol)
        if sym_pb is None:
            return None
        rels = tuple(
            Relationship(
                symbol=r.symbol,
                is_reference=r.is_reference,
                is_implementation=r.is_implementation,
                is_type_definition=r.is_type_definition,
                is_definition=r.is_definition,
            )
            for r in sym_pb.relationships
        )
        return SymbolInfo(symbol=symbol, relationships=rels)
