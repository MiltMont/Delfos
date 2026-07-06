"""Tests for the agent-driven enrichment write path."""

from __future__ import annotations

import pytest

from delfos.enrich import AnnotationOutcome, EnrichmentError, EnrichmentService
from delfos.enrich.service import (
    _concept_cue_id,  # pyright: ignore[reportPrivateUsage]
    _normalize_phrase,  # pyright: ignore[reportPrivateUsage]
    _normalize_tag_value,  # pyright: ignore[reportPrivateUsage]
)
from delfos.schema import CueNode, CueType, Direction, EdgeType
from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import FakeEmbedder, load, make_content, make_cue, vec


def test_normalize_phrase_lowercases_and_collapses_whitespace() -> None:
    assert _normalize_phrase("  Crash \t Recovery ") == "crash recovery"
    assert _normalize_phrase("") == ""
    assert _normalize_phrase("   ") == ""


def test_normalize_tag_value_lowercases_and_hyphenates() -> None:
    assert _normalize_tag_value("Storage Engine") == "storage-engine"
    assert _normalize_tag_value("  CLI  ") == "cli"
    assert _normalize_tag_value("   ") == ""


def test_concept_cue_id_mirrors_error_cue_scheme_and_is_stable() -> None:
    a = _concept_cue_id("a.py", "crash recovery")
    b = _concept_cue_id("a.py", "crash recovery")
    assert a == b
    assert a.startswith("cue:concept:a.py::")
    assert len(a.split("::")[-1]) == 12
    assert _concept_cue_id("b.py", "crash recovery") != a
    assert _concept_cue_id("a.py", "rate limiting") != a


def _service(store: NativeGraphStore, mapping: dict[str, list[float]]) -> EnrichmentService:
    return EnrichmentService(store, FakeEmbedder(mapping))


def test_annotate_writes_concept_cues_edges_and_tags(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    outcome: AnnotationOutcome = svc.annotate(
        "content:1", ["Crash  Recovery"], arch_layer="Storage Engine", pattern_type=None
    )

    assert len(outcome.written_cue_ids) == 1
    cue = store.get_node(outcome.written_cue_ids[0])
    assert isinstance(cue, CueNode)
    assert cue.cue_type == CueType.CONCEPT
    assert cue.text == "crash recovery"
    assert cue.embedding is not None

    # CUE_OF edge: cue -> content
    neighbors = store.neighbors(cue.id, edge_type=EdgeType.CUE_OF, direction=Direction.OUTGOING)
    assert [n.id for n in neighbors] == ["content:1"]

    # TAGGED_WITH edge: content -> normalized tag
    assert outcome.written_tag_ids == ["tag:arch_layer:storage-engine"]
    tags = store.neighbors(
        "content:1", edge_type=EdgeType.TAGGED_WITH, direction=Direction.OUTGOING
    )
    assert "tag:arch_layer:storage-engine" in [t.id for t in tags]


def test_annotate_stamps_target_provenance_on_cues(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    outcome = svc.annotate("content:1", ["crash recovery"])

    cue = store.get_node(outcome.written_cue_ids[0])
    assert isinstance(cue, CueNode)
    assert cue.source_file == "a.py"  # make_content uses source_file="a.py"
    assert cue.git_sha == "s"


def test_annotate_is_idempotent(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    first = svc.annotate("content:1", ["crash recovery"], arch_layer="storage")
    second = svc.annotate("content:1", ["crash recovery"], arch_layer="storage")

    assert first.written_cue_ids == second.written_cue_ids
    assert first.written_tag_ids == second.written_tag_ids


def test_annotate_drops_bad_phrases_but_keeps_good_ones(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})

    outcome = svc.annotate(
        "content:1",
        ["crash recovery", "Save_Snapshot", "   ", "x" * 101, "crash  recovery"],
    )

    assert len(outcome.written_cue_ids) == 1  # only "crash recovery"
    assert outcome.dropped_phrases == ["Save_Snapshot", "   ", "x" * 101, "crash  recovery"]


def test_annotate_rejects_unknown_and_non_content_ids(store: NativeGraphStore) -> None:
    load(store, [make_cue("cue:symbol:a.py::f", "f", embedding=vec(2.0))], [])
    svc = _service(store, {})

    with pytest.raises(EnrichmentError, match="unknown node id"):
        svc.annotate("content:nope", ["a concept"])
    with pytest.raises(EnrichmentError, match="not a content node"):
        svc.annotate("cue:symbol:a.py::f", ["a concept"])


def test_annotate_rejects_too_many_concepts(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])
    svc = _service(store, {})

    with pytest.raises(EnrichmentError, match="at most 10"):
        svc.annotate("content:1", [f"concept {i}" for i in range(11)])


def test_annotate_rejects_empty_tag_value(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])
    svc = _service(store, {})

    with pytest.raises(EnrichmentError, match="empty tag value"):
        svc.annotate("content:1", arch_layer="   ")


def test_annotate_with_only_content_id_is_a_vocab_query(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])
    svc = _service(store, {"crash recovery": vec(1.0)})
    svc.annotate("content:1", ["crash recovery"], arch_layer="storage", pattern_type="validation")

    outcome = svc.annotate("content:1")

    assert outcome.written_cue_ids == []
    assert outcome.written_tag_ids == []
    assert outcome.existing_values == {
        "arch_layer": ["storage"],
        "pattern_type": ["validation"],
    }


def test_annotate_writes_nothing_when_embedder_fails(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "f")], [])

    class Boom:
        @property
        def model(self) -> str:
            return "fake-v1"

        @property
        def model_version(self) -> str | None:
            return None

        @property
        def dimensions(self) -> int:
            return 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("dead endpoint")

    svc = EnrichmentService(store, Boom())
    with pytest.raises(RuntimeError, match="dead endpoint"):
        svc.annotate("content:1", ["crash recovery"], arch_layer="storage")

    assert store.neighbors("content:1", edge_type=EdgeType.TAGGED_WITH) == []
