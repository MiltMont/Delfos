"""The construction pipeline: source files -> Cue-Tag-Content graph.

Parses Python with the stdlib :mod:`ast` (:mod:`~delfos.indexer.parser`),
extracts nodes and edges (:mod:`~delfos.indexer.extractor`), embeds cue text
via an :class:`Embedder` (:mod:`~delfos.indexer.embedder`), and persists each
file atomically through a :class:`~delfos.store.base.GraphStore`
(:class:`~delfos.indexer.pipeline.Indexer`).
"""

from .embedder import Embedder, OpenAIEmbedder
from .extractor import ExtractionResult, extract
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
