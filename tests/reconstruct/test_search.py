from __future__ import annotations

from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.store.native_store import NativeGraphStore

from .conftest import FakeEmbedder, load, make_cue, vec


def test_search_returns_nearest_cues(store: NativeGraphStore) -> None:
    near = make_cue("cue-near", "auth", embedding=vec(0.10))
    far = make_cue("cue-far", "billing", embedding=vec(9.0))
    load(store, [near, far], [])

    embedder = FakeEmbedder({"how does auth work": vec(0.10)})
    service = ReconstructionService(store, embedder, FakeHopPlanner([]))

    hits = service.search("how does auth work", k=2)

    assert [c.id for c in hits] == ["cue-near", "cue-far"]


def test_search_only_returns_cue_nodes(store: NativeGraphStore) -> None:
    near = make_cue("cue-near", "auth", embedding=vec(0.10))
    load(store, [near], [])
    embedder = FakeEmbedder({"q": vec(0.10)})
    service = ReconstructionService(store, embedder, FakeHopPlanner([]))

    hits = service.search("q", k=5)

    assert all(isinstance(c, type(near)) for c in hits)
    assert hits[0].id == "cue-near"
