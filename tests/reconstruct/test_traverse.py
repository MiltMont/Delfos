from __future__ import annotations

from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType, TagCategory
from delfos.store.native_store import NativeGraphStore

from .conftest import FakeEmbedder, edge, load, make_content, make_cue, make_tag


def _service(store: NativeGraphStore) -> ReconstructionService:
    return ReconstructionService(store, FakeEmbedder({}), FakeHopPlanner([]))


def test_traverse_forward_follows_cue_of(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    content = make_content("content-1", "login")
    load(store, [cue, content], [edge("cue-1", "content-1", EdgeType.CUE_OF)])

    result = _service(store).traverse_forward(["cue-1"])

    assert [c.id for c in result] == ["content-1"]


def test_traverse_forward_filters_by_tag(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    py = make_content("content-py", "login")
    js = make_content("content-js", "logon")
    tag_py = make_tag("tag-py", TagCategory.LANGUAGE, "python")
    edges = [
        edge("cue-1", "content-py", EdgeType.CUE_OF),
        edge("cue-1", "content-js", EdgeType.CUE_OF),
        edge("content-py", "tag-py", EdgeType.TAGGED_WITH),
    ]
    load(store, [cue, py, js, tag_py], edges)

    result = _service(store).traverse_forward(
        ["cue-1"], tag_filters=[(TagCategory.LANGUAGE, "python")]
    )

    assert [c.id for c in result] == ["content-py"]


def test_traverse_forward_dedups_across_cues(store: NativeGraphStore) -> None:
    cues = [make_cue("cue-1", "a"), make_cue("cue-2", "b")]
    content = make_content("content-1", "login")
    edges = [
        edge("cue-1", "content-1", EdgeType.CUE_OF),
        edge("cue-2", "content-1", EdgeType.CUE_OF),
    ]
    load(store, [*cues, content], edges)

    result = _service(store).traverse_forward(["cue-1", "cue-2"])

    assert [c.id for c in result] == ["content-1"]


def test_traverse_forward_follows_redirect(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    old = make_content("content-old", "login")
    new = make_content("content-new", "login")
    edges = [
        edge("cue-1", "content-old", EdgeType.CUE_OF),
        edge("content-old", "content-new", EdgeType.REDIRECTS_TO),
    ]
    load(store, [cue, old, new], edges)

    result = _service(store).traverse_forward(["cue-1"])

    assert [c.id for c in result] == ["content-new"]


def test_traverse_reverse_finds_sibling_cues(store: NativeGraphStore) -> None:
    content = make_content("content-1", "login")
    cue_a = make_cue("cue-a", "auth")
    cue_b = make_cue("cue-b", "signin")
    edges = [
        edge("cue-a", "content-1", EdgeType.CUE_OF),
        edge("cue-b", "content-1", EdgeType.CUE_OF),
    ]
    load(store, [content, cue_a, cue_b], edges)

    result = _service(store).traverse_reverse(["content-1"])

    assert {c.id for c in result} == {"cue-a", "cue-b"}


def test_traverse_reverse_dedups(store: NativeGraphStore) -> None:
    c1 = make_content("content-1", "login")
    c2 = make_content("content-2", "logout")
    cue = make_cue("cue-a", "auth")
    edges = [
        edge("cue-a", "content-1", EdgeType.CUE_OF),
        edge("cue-a", "content-2", EdgeType.CUE_OF),
    ]
    load(store, [c1, c2, cue], edges)

    result = _service(store).traverse_reverse(["content-1", "content-2"])

    assert [c.id for c in result] == ["cue-a"]
