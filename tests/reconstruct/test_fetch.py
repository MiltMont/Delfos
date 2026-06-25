from __future__ import annotations

import pytest

from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType, NodeStatus, TagCategory
from delfos.store.native_store import NativeGraphStore

from .conftest import (
    FakeEmbedder,
    edge,
    load,
    make_content,
    make_cue,
    make_tag,
)


def _service(store: NativeGraphStore) -> ReconstructionService:
    # No planner: exercises the planner-optional constructor.
    return ReconstructionService(store, FakeEmbedder({}))


def test_fetch_returns_active_content_and_skips_unknown(store: NativeGraphStore) -> None:
    content = make_content("c1", "login")
    load(store, [content], [])
    svc = _service(store)

    got = svc.fetch(["c1", "does-not-exist"])

    assert [c.id for c in got] == ["c1"]
    assert got[0].body == "def login(): ..."


def test_fetch_skips_deleted_content(store: NativeGraphStore) -> None:
    content = make_content("c1", "login")
    content.status = NodeStatus.DELETED
    load(store, [content], [])
    svc = _service(store)

    assert svc.fetch(["c1"]) == []


def test_content_tags_renders_sorted_category_value(store: NativeGraphStore) -> None:
    content = make_content("c1", "login")
    tag = make_tag("t1", TagCategory.LANGUAGE, "python")
    load(store, [content, tag], [edge("c1", "t1", EdgeType.TAGGED_WITH)])
    svc = _service(store)

    assert svc.content_tags("c1") == ["language=python"]


def test_fetch_skips_non_content_node(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    load(store, [cue], [])
    svc = _service(store)
    assert svc.fetch(["cue-1"]) == []


def test_reconstruct_without_planner_raises(store: NativeGraphStore) -> None:
    svc = _service(store)
    with pytest.raises(RuntimeError, match="planner"):
        svc.reconstruct("anything")
