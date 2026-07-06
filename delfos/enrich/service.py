"""The write path: agent-driven enrichment of content nodes.

The calling agent is the extractor (the write-path extension of the read
path's "the calling agent is the planner"): it supplies concept phrases and
semantic tag values for content it has actually read, via the MCP ``annotate``
tool. This module never calls a chat LLM. Sits entirely on top of
:class:`~delfos.store.base.GraphStore`.
"""

from __future__ import annotations

import hashlib

MAX_CONCEPTS_PER_CALL = 10
MAX_PHRASE_LENGTH = 100


class EnrichmentError(ValueError):
    """Raised when an annotate call is invalid; the message is agent-facing."""


def _normalize_phrase(phrase: str) -> str:  # pyright: ignore[reportUnusedFunction]
    """Lowercase and collapse all whitespace runs to single spaces."""
    return " ".join(phrase.split()).lower()


def _normalize_tag_value(value: str) -> str:  # pyright: ignore[reportUnusedFunction]
    """Lowercase and replace whitespace runs with hyphens.

    ``Storage Engine`` -> ``storage-engine``.
    """
    return "-".join(value.split()).lower()


def _concept_cue_id(source_file: str, phrase: str) -> str:  # pyright: ignore[reportUnusedFunction]
    """Deterministic concept-cue id; mirrors the error-cue scheme in the extractor."""
    slug = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:12]
    return f"cue:concept:{source_file}::{slug}"
