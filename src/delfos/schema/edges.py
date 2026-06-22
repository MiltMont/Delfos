"""Edge model for the Cue-Tag-Content graph."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from .enums import EdgeType


class Edge(BaseModel):
    """A directed, typed relationship from ``source_id`` to ``target_id``.

    Edges carry the same provenance as nodes so they can be dropped alongside
    the nodes of a re-indexed file. ``REDIRECTS_TO`` edges encode renames and
    must be followed transparently during traversal.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: EdgeType
    source_file: str | None = None
    git_sha: str | None = None
    indexed_at: datetime | None = None
