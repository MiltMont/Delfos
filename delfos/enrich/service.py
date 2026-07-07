"""The write path: agent-driven enrichment of content nodes.

The calling agent is the extractor (the write-path extension of the read
path's "the calling agent is the planner"): it supplies concept phrases and
semantic tag values for content it has actually read, via the MCP ``annotate``
tool. This module never calls a chat LLM. Sits entirely on top of
:class:`~delfos.store.base.GraphStore`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from delfos.indexer.embedder import Embedder
from delfos.schema import ContentNode, CueNode, CueType, Edge, EdgeType, TagCategory, TagNode
from delfos.store import GraphStore

MAX_CONCEPTS_PER_CALL = 10
MAX_PHRASE_LENGTH = 100


class EnrichmentError(ValueError):
    """Raised when an annotate call is invalid; the message is agent-facing."""


def _normalize_phrase(phrase: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return " ".join(phrase.split()).lower()


def _normalize_tag_value(value: str) -> str:
    """Lowercase and replace whitespace runs with hyphens.

    ``Storage Engine`` -> ``storage-engine``.
    """
    return "-".join(value.split()).lower()


def _concept_cue_id(source_file: str, phrase: str) -> str:
    """Deterministic concept-cue id; mirrors the error-cue scheme in the extractor."""
    slug = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:12]
    return f"cue:concept:{source_file}::{slug}"


@dataclass(frozen=True)
class AnnotationOutcome:
    """What one ``annotate`` call wrote, dropped, and found already in the graph."""

    content_id: str
    written_cue_ids: list[str]
    written_tag_ids: list[str]
    dropped_phrases: list[str]
    existing_values: dict[str, list[str]]


class EnrichmentService:
    """Write-path operations: agent-supplied concept cues and semantic tags."""

    def __init__(self, store: GraphStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def annotate(
        self,
        content_id: str,
        concepts: Sequence[str] = (),
        *,
        arch_layer: str | None = None,
        pattern_type: str | None = None,
    ) -> AnnotationOutcome:
        """Attach concept cues and semantic tags to one content node.

        Idempotent: ids are deterministic, so retries upsert the same nodes.
        Embedding happens before the transaction opens — an embedder failure
        writes nothing. Everything written carries the target's provenance,
        so a re-index of the file wipes it (delete-and-reindex).

        A transaction is opened only when there is something to write: a
        vocab-only call (no concepts, no tags) never commits, since the C++
        store rewrites the full snapshot on every commit.
        """
        if len(concepts) > MAX_CONCEPTS_PER_CALL:
            raise EnrichmentError(
                f"got {len(concepts)} concepts; pass at most {MAX_CONCEPTS_PER_CALL} per call"
            )
        target = self._store.get_node(content_id)
        if target is None:
            raise EnrichmentError(f"unknown node id {content_id!r}")
        if not isinstance(target, ContentNode):
            raise EnrichmentError(
                f"{content_id!r} is a {target.node_type.value} node, not a content node"
            )

        accepted, dropped = self._screen_phrases(concepts, target.symbol_name)
        tags = self._screen_tags(arch_layer=arch_layer, pattern_type=pattern_type)

        vectors = self._embedder.embed(accepted) if accepted else []

        indexed_at = datetime.now(tz=UTC)
        cue_ids: list[str] = []
        tag_ids: list[str] = []
        if accepted or tags:
            with self._store.transaction():
                for phrase, vector in zip(accepted, vectors, strict=True):
                    cue_id = _concept_cue_id(target.source_file, phrase)
                    self._store.upsert_node(
                        CueNode(
                            id=cue_id,
                            source_file=target.source_file,
                            git_sha=target.git_sha,
                            indexed_at=indexed_at,
                            cue_type=CueType.CONCEPT,
                            text=phrase,
                            embedding=vector,
                            embedding_model=self._embedder.model,
                            embedding_model_version=self._embedder.model_version,
                        )
                    )
                    self._store.upsert_edge(
                        Edge(
                            source_id=cue_id,
                            target_id=content_id,
                            edge_type=EdgeType.CUE_OF,
                            source_file=target.source_file,
                            git_sha=target.git_sha,
                            indexed_at=indexed_at,
                        )
                    )
                    cue_ids.append(cue_id)
                for category, value in tags:
                    tag_id = f"tag:{category.value}:{value}"
                    self._store.upsert_node(
                        TagNode(id=tag_id, indexed_at=indexed_at, category=category, value=value)
                    )
                    self._store.upsert_edge(
                        Edge(
                            source_id=content_id,
                            target_id=tag_id,
                            edge_type=EdgeType.TAGGED_WITH,
                            source_file=target.source_file,
                            git_sha=target.git_sha,
                            indexed_at=indexed_at,
                        )
                    )
                    tag_ids.append(tag_id)

        return AnnotationOutcome(
            content_id=content_id,
            written_cue_ids=cue_ids,
            written_tag_ids=tag_ids,
            dropped_phrases=dropped,
            existing_values={
                TagCategory.ARCH_LAYER.value: self._store.list_tag_values(TagCategory.ARCH_LAYER),
                TagCategory.PATTERN_TYPE.value: self._store.list_tag_values(
                    TagCategory.PATTERN_TYPE
                ),
            },
        )

    @staticmethod
    def _screen_phrases(
        concepts: Sequence[str], symbol_name: str | None
    ) -> tuple[list[str], list[str]]:
        """Normalize phrases; drop empties, overlong ones, duplicates, and the symbol name."""
        symbol = (symbol_name or "").lower()
        accepted: list[str] = []
        dropped: list[str] = []
        seen: set[str] = set()
        for raw in concepts:
            phrase = _normalize_phrase(raw)
            if not phrase or len(phrase) > MAX_PHRASE_LENGTH or phrase == symbol or phrase in seen:
                dropped.append(raw)
                continue
            seen.add(phrase)
            accepted.append(phrase)
        return accepted, dropped

    @staticmethod
    def _screen_tags(
        *, arch_layer: str | None, pattern_type: str | None
    ) -> list[tuple[TagCategory, str]]:
        """Normalize the two agent-writable tag values; reject ones that normalize to empty."""
        out: list[tuple[TagCategory, str]] = []
        pairs = ((TagCategory.ARCH_LAYER, arch_layer), (TagCategory.PATTERN_TYPE, pattern_type))
        for category, raw in pairs:
            if raw is None:
                continue
            value = _normalize_tag_value(raw)
            if not value:
                raise EnrichmentError(f"empty tag value for {category.value}")
            if len(value) > MAX_PHRASE_LENGTH:
                raise EnrichmentError(
                    f"tag value for {category.value} exceeds {MAX_PHRASE_LENGTH} characters"
                )
            out.append((category, value))
        return out
