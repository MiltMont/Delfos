from __future__ import annotations

from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType, TagCategory
from delfos.store.duckdb_store import DuckDBGraphStore

from .conftest import FakeEmbedder, edge, load, make_content, make_cue, make_tag


def _service(store: DuckDBGraphStore) -> ReconstructionService:
    return ReconstructionService(store, FakeEmbedder({}), FakeHopPlanner([]))


def test_traverse_forward_follows_cue_of(store: DuckDBGraphStore) -> None:
    cue = make_cue("cue-1", "auth")
    content = make_content("content-1", "login")
    load(store, [cue, content], [edge("cue-1", "content-1", EdgeType.CUE_OF)])

    result = _service(store).traverse_forward(["cue-1"])

    assert [c.id for c in result] == ["content-1"]


def test_traverse_forward_filters_by_tag(store: DuckDBGraphStore) -> None:
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


def test_traverse_forward_dedups_across_cues(store: DuckDBGraphStore) -> None:
    cues = [make_cue("cue-1", "a"), make_cue("cue-2", "b")]
    content = make_content("content-1", "login")
    edges = [
        edge("cue-1", "content-1", EdgeType.CUE_OF),
        edge("cue-2", "content-1", EdgeType.CUE_OF),
    ]
    load(store, [*cues, content], edges)

    result = _service(store).traverse_forward(["cue-1", "cue-2"])

    assert [c.id for c in result] == ["content-1"]
