"""Edge model for the Cue-Tag-Content graph."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from .enums import EdgeType


class Edge(BaseModel):
    """A directed, typed relationship from ``source_id`` to ``target_id``.

    Provenance (``source_file`` / ``git_sha`` / ``indexed_at``) is optional,
    unlike on :class:`~delfos.schema.nodes.BaseNode` where it is required. Most
    edges are file-scoped (``CUE_OF``, ``TAGGED_WITH``, ``PART_OF_TOPIC``) and
    are removed implicitly when their incident nodes are dropped during a
    re-index, so they do not strictly need their own provenance. Edges that span
    files or commits — chiefly ``REDIRECTS_TO``, which encodes a rename and must
    be followed transparently during traversal — cannot be attributed to a
    single source file, which is why these fields are nullable. When an edge is
    file-scoped, the indexer should still stamp its provenance so
    ``delete_nodes_for_file`` can clean it up directly.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: EdgeType
    source_file: str | None = None
    git_sha: str | None = None
    indexed_at: datetime | None = None
