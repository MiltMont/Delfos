"""Turn a :class:`~delfos.indexer.parser.ParsedModule` into graph nodes and edges.

This is the pure, side-effect-free heart of indexing: given the AST-derived
intermediate representation for one file (plus the file's ``git_sha`` and an
``indexed_at`` timestamp), it produces the Cue-Tag-Content nodes and the typed
edges that connect them. It performs **no** I/O, **no** embedding, and **no**
persistence — the :class:`~delfos.indexer.pipeline.Indexer` owns those, so the
embedding decision (``decisions.md`` section 5: vectors are attached per-file,
inside the transaction) stays out of here.

Graph shape produced per file:

- one ``MODULE`` :class:`ContentNode` (memory layer ``TOPIC``) for the file;
- one ``ContentNode`` per top-level function/class and per method (layer
  ``SEMANTIC``), each linked to the module node by a ``PART_OF_TOPIC`` edge;
- one ``SYMBOL`` :class:`CueNode` per definition, linked to its content by a
  ``CUE_OF`` edge;
- one ``ERROR_MESSAGE`` :class:`CueNode` per distinct literal raised inside a
  definition, linked to that definition's content by ``CUE_OF``;
- shared :class:`TagNode` s (``LANGUAGE``, ``MODULE_PATH``, ``LANG_CONSTRUCT``)
  reached from content via ``TAGGED_WITH`` edges. Tags carry no provenance and
  are shared across files; the file-scoped edges are what a re-index drops.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from delfos.schema import (
    ContentKind,
    ContentNode,
    CueNode,
    CueType,
    Edge,
    EdgeType,
    MemoryLayer,
    Node,
    TagCategory,
    TagNode,
)

from .parser import DefinitionKind, ParsedDefinition, ParsedModule

_LANGUAGE = "python"
_MODULE_CONSTRUCT = "module"

_CONTENT_KINDS: dict[DefinitionKind, ContentKind] = {
    DefinitionKind.FUNCTION: ContentKind.FUNCTION,
    DefinitionKind.ASYNC_FUNCTION: ContentKind.FUNCTION,
    DefinitionKind.METHOD: ContentKind.FUNCTION,
    DefinitionKind.ASYNC_METHOD: ContentKind.FUNCTION,
    DefinitionKind.CLASS: ContentKind.CLASS,
}


@dataclass
class ExtractionResult:
    """Nodes and edges extracted from a single file, ready to upsert.

    Nodes are de-duplicated by ``id`` and edges by ``(source_id, target_id,
    edge_type)`` so a caller can upsert them in any order without conflict.
    """

    nodes: list[Node]
    edges: list[Edge]


def _message_slug(message: str) -> str:
    return hashlib.sha1(message.encode("utf-8")).hexdigest()[:12]


def _module_content_id(source_file: str) -> str:
    return f"content:{source_file}::<module>"


def _definition_content_id(source_file: str, qualified_name: str) -> str:
    return f"content:{source_file}::{qualified_name}"


def _symbol_cue_id(source_file: str, qualified_name: str) -> str:
    return f"cue:symbol:{source_file}::{qualified_name}"


def _error_cue_id(source_file: str, message: str) -> str:
    return f"cue:error:{source_file}::{_message_slug(message)}"


def _tag_id(category: TagCategory, value: str) -> str:
    return f"tag:{category.value}:{value}"


class _Builder:
    """Accumulates de-duplicated nodes and edges for one file."""

    def __init__(self, module: ParsedModule, *, git_sha: str, indexed_at: datetime) -> None:
        self._module = module
        self._git_sha = git_sha
        self._indexed_at = indexed_at
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, EdgeType], Edge] = {}

    def build(self) -> ExtractionResult:
        module_id = _module_content_id(self._module.source_file)
        self._add_content(
            node_id=module_id,
            kind=ContentKind.MODULE,
            memory_layer=MemoryLayer.TOPIC,
            symbol_name=self._module.module_path,
            signature=None,
            docstring=self._module.docstring,
            body=self._module.source,
        )
        self._tag_content(module_id, _MODULE_CONSTRUCT)

        for definition in self._module.definitions:
            self._add_definition(definition, module_id)

        return ExtractionResult(nodes=list(self._nodes.values()), edges=list(self._edges.values()))

    def _add_definition(self, definition: ParsedDefinition, module_id: str) -> None:
        source_file = self._module.source_file
        content_id = _definition_content_id(source_file, definition.qualified_name)
        self._add_content(
            node_id=content_id,
            kind=ContentKind.TEST if definition.is_test else _CONTENT_KINDS[definition.kind],
            memory_layer=MemoryLayer.SEMANTIC,
            symbol_name=definition.qualified_name,
            signature=definition.signature,
            docstring=definition.docstring,
            body=definition.source,
        )
        self._tag_content(content_id, definition.kind.value)
        self._add_edge(content_id, module_id, EdgeType.PART_OF_TOPIC)

        symbol_cue_id = _symbol_cue_id(source_file, definition.qualified_name)
        self._add_cue(symbol_cue_id, CueType.SYMBOL, definition.name)
        self._add_edge(symbol_cue_id, content_id, EdgeType.CUE_OF)

        for message in definition.error_messages:
            error_cue_id = _error_cue_id(source_file, message)
            self._add_cue(error_cue_id, CueType.ERROR_MESSAGE, message)
            self._add_edge(error_cue_id, content_id, EdgeType.CUE_OF)

    def _add_content(
        self,
        *,
        node_id: str,
        kind: ContentKind,
        memory_layer: MemoryLayer,
        symbol_name: str | None,
        signature: str | None,
        docstring: str | None,
        body: str,
    ) -> None:
        self._nodes[node_id] = ContentNode(
            id=node_id,
            source_file=self._module.source_file,
            git_sha=self._git_sha,
            indexed_at=self._indexed_at,
            kind=kind,
            memory_layer=memory_layer,
            symbol_name=symbol_name,
            signature=signature,
            docstring=docstring,
            body=body,
        )

    def _add_cue(self, node_id: str, cue_type: CueType, text: str) -> None:
        self._nodes[node_id] = CueNode(
            id=node_id,
            source_file=self._module.source_file,
            git_sha=self._git_sha,
            indexed_at=self._indexed_at,
            cue_type=cue_type,
            text=text,
        )

    def _tag_content(self, content_id: str, construct: str) -> None:
        self._add_tag_edge(content_id, TagCategory.LANGUAGE, _LANGUAGE)
        self._add_tag_edge(content_id, TagCategory.MODULE_PATH, self._module.module_path)
        self._add_tag_edge(content_id, TagCategory.LANG_CONSTRUCT, construct)

    def _add_tag_edge(self, content_id: str, category: TagCategory, value: str) -> None:
        tag_id = _tag_id(category, value)
        if tag_id not in self._nodes:
            self._nodes[tag_id] = TagNode(
                id=tag_id,
                indexed_at=self._indexed_at,
                category=category,
                value=value,
            )
        self._add_edge(content_id, tag_id, EdgeType.TAGGED_WITH)

    def _add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        self._edges[(source_id, target_id, edge_type)] = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            source_file=self._module.source_file,
            git_sha=self._git_sha,
            indexed_at=self._indexed_at,
        )


def extract(module: ParsedModule, *, git_sha: str, indexed_at: datetime) -> ExtractionResult:
    """Extract all nodes and edges for ``module``.

    ``git_sha`` is the per-file content SHA stamped on every sourced node and
    edge so the delete-and-reindex strategy can find them; ``indexed_at`` is the
    shared timestamp for this indexing pass. Cue nodes are returned without
    embeddings — the pipeline attaches those inside the file's transaction.
    """
    return _Builder(module, git_sha=git_sha, indexed_at=indexed_at).build()
