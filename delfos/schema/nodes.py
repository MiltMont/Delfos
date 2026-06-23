"""Pydantic node models for the Cue-Tag-Content graph.

Every node carries provenance (``source_file`` + ``git_sha``) so the
delete-and-reindex stale-handling strategy can find and drop all nodes sourced
from a changed file (see ``decisions.md`` section 4). Nodes that carry a vector
also record which embedding model produced it (section 5), so a future
migration can identify which vectors need re-embedding.

These models define types only; persistence is the responsibility of a
``GraphStore`` backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    ContentKind,
    CueType,
    MemoryLayer,
    NodeStatus,
    NodeType,
    TagCategory,
)


class BaseNode(BaseModel):
    """Identity, lifecycle, and tombstone fields common to every node type.

    ``id`` is the stable identity used for upserts (the indexer assigns it, e.g.
    derived from ``(source_file, symbol_name)``). Tombstone metadata
    (``deleted_at`` / ``deleted_by_commit``) is populated when ``status`` is
    flipped to :attr:`NodeStatus.DELETED`.

    Provenance lives on the subclasses, not here: :class:`SourcedNode` (cues and
    content) requires it, while :class:`TagNode` keeps it optional because a tag
    is shared across files.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    indexed_at: datetime
    status: NodeStatus = NodeStatus.ACTIVE
    deleted_at: datetime | None = None
    deleted_by_commit: str | None = None


class SourcedNode(BaseNode):
    """A node attributable to a single source file.

    Cues and content are extracted from one file and carry mandatory
    provenance so the delete-and-reindex strategy can drop every node sourced
    from a changed file (``decisions.md`` section 4). Tags, by contrast, are
    cross-file bridges and keep provenance optional (see :class:`TagNode`).
    """

    source_file: str
    git_sha: str


class EmbeddedMixin(BaseModel):
    """Mixin for nodes that may carry a vector embedding.

    The embedding is optional, but if present ``embedding_model`` is mandatory:
    every stored vector must be attributable to a model so a partial or future
    re-embedding migration is possible (``decisions.md`` section 5).
    """

    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_model_version: str | None = None

    @model_validator(mode="after")
    def _require_model_for_embedding(self) -> EmbeddedMixin:
        if self.embedding is not None and self.embedding_model is None:
            raise ValueError("embedding_model is required when embedding is set")
        return self


class CueNode(SourcedNode, EmbeddedMixin):
    """An entry point a developer might query by.

    Cues are the only nodes searched by vector similarity in the read path
    (``search(query) -> list[CueNode]``).
    """

    node_type: Literal[NodeType.CUE] = NodeType.CUE
    cue_type: CueType
    text: str


class TagNode(BaseNode):
    """A semantic bridge connecting cues to content.

    Tags are categorical, not embedded; traversal filters on them rather than
    ranking them by similarity. A single tag (e.g. ``LANGUAGE=python`` or
    ``MODULE_PATH=delfos.schema``) is shared by content from many files, so its
    provenance fields stay unset — it outlives the re-index of any one file.
    The file-scoped ``TAGGED_WITH`` edges that point at it are what get cleaned
    up by :meth:`~delfos.store.base.GraphStore.delete_nodes_for_file`.
    """

    node_type: Literal[NodeType.TAG] = NodeType.TAG
    category: TagCategory
    value: str
    source_file: str | None = None
    git_sha: str | None = None


class ContentNode(SourcedNode, EmbeddedMixin):
    """The actual implementation artifact returned to agents.

    ``reconstruct`` returns a flat, score-ordered list of these.
    """

    node_type: Literal[NodeType.CONTENT] = NodeType.CONTENT
    kind: ContentKind
    memory_layer: MemoryLayer
    symbol_name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    body: str


Node = Annotated[
    CueNode | TagNode | ContentNode,
    Field(discriminator="node_type"),
]
"""Discriminated union over all node types, keyed on ``node_type``."""
