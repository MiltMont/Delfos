from __future__ import annotations

from delfos.reconstruct.planner import Collected, HopDecision
from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType
from delfos.store.native_store import NativeGraphStore

from .conftest import FakeEmbedder, edge, load, make_content, make_cue, vec


def _build_two_hop_graph(store: NativeGraphStore) -> None:
    # cue-auth -> content-login -> (sibling) cue-session -> content-token
    seed = make_cue("cue-auth", "auth", embedding=vec(0.1))
    login = make_content("content-login", "login")
    session = make_cue("cue-session", "session")
    token = make_content("content-token", "make_token")
    edges = [
        edge("cue-auth", "content-login", EdgeType.CUE_OF),
        edge("cue-session", "content-login", EdgeType.CUE_OF),
        edge("cue-session", "content-token", EdgeType.CUE_OF),
    ]
    load(store, [seed, login, session, token], edges)


def _service(store: NativeGraphStore, planner: FakeHopPlanner) -> ReconstructionService:
    embedder = FakeEmbedder({"q": vec(0.1)})
    return ReconstructionService(store, embedder, planner, seed_k=5)


def test_reconstruct_collects_from_first_hop(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [HopDecision(collect=[Collected(id="content-login", relevance=0.9)], stop=True)]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 1


def test_reconstruct_descends_and_orders_by_relevance(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [
            # Hop 1 at cue-auth: collect login (0.5), descend into content-login
            HopDecision(
                collect=[Collected(id="content-login", relevance=0.5)],
                descend_into="content-login",
            ),
            # Hop 2 at content-login: sibling cue-session is a candidate; collect it
            # (resolves to its first content) with higher relevance, then stop.
            HopDecision(collect=[Collected(id="cue-session", relevance=0.95)], stop=True),
        ]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    # cue-session resolves to content-login (its first CUE_OF target); already
    # collected, so relevance is upgraded to 0.95. Single deduped result.
    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 2


def test_reconstruct_stops_at_budget(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    # Always descend, never stop: only budget halts the walk.
    planner = FakeHopPlanner(
        [
            HopDecision(collect=[], descend_into="content-login"),
            HopDecision(collect=[], descend_into="cue-session"),
            HopDecision(collect=[], descend_into="content-token"),
            HopDecision(collect=[], descend_into="content-login"),
        ]
    )
    _service(store, planner).reconstruct("q", budget=2)

    assert planner.call_count == 2


def test_reconstruct_empty_when_no_seed_cues(store: NativeGraphStore) -> None:
    # No cue carries an embedding, so vector_search returns nothing.
    load(store, [make_content("content-1", "login")], [])
    planner = FakeHopPlanner([])
    result = _service(store, planner).reconstruct("q", budget=3)

    assert result == []
    assert planner.call_count == 0


def test_reconstruct_ignores_hallucinated_ids(store: NativeGraphStore) -> None:
    # Two embedded seeds: when hop 1's descend_into is invalid and the stack is
    # empty, the walk falls back to the second seed and keeps going.
    seed1 = make_cue("cue-auth", "auth", embedding=vec(0.1))
    seed2 = make_cue("cue-extra", "extra", embedding=vec(0.11))
    login = make_content("content-login", "login")
    edges = [
        edge("cue-auth", "content-login", EdgeType.CUE_OF),
        edge("cue-extra", "content-login", EdgeType.CUE_OF),
    ]
    load(store, [seed1, seed2, login], edges)

    planner = FakeHopPlanner(
        [
            HopDecision(
                collect=[Collected(id="does-not-exist", relevance=0.9)],
                descend_into="also-fake",
                stop=False,
            ),
            HopDecision(collect=[Collected(id="content-login", relevance=0.4)], stop=True),
        ]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    # Hallucinated collect dropped; invalid descend_into forced a fallback to the
    # second seed, where a real collect succeeded.
    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 2


def test_reconstruct_returns_partial_on_planner_error(store: NativeGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [
            HopDecision(
                collect=[Collected(id="content-login", relevance=0.7)],
                descend_into="content-login",
            )
        ],
        error_after=1,  # 2nd call raises
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    # First hop collected before the second call blew up.
    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 2


def test_reconstruct_backtracks_to_parent_via_stack(store: NativeGraphStore) -> None:
    # Single seed: descend into content, dead-end there with a non-empty stack,
    # and the walk must pop back to the parent cue (the stack branch, not the
    # empty-stack seed-queue fallback) and resume.
    seed = make_cue("cue-auth", "auth", embedding=vec(0.1))
    login = make_content("content-login", "login")
    edges = [edge("cue-auth", "content-login", EdgeType.CUE_OF)]
    load(store, [seed, login], edges)

    planner = FakeHopPlanner(
        [
            # Hop 1 at cue-auth: descend into content-login (pushes cue-auth on stack).
            HopDecision(collect=[], descend_into="content-login"),
            # Hop 2 at content-login: invalid descend, stack non-empty -> pop to cue-auth.
            HopDecision(collect=[], descend_into="nope", stop=False),
            # Hop 3 back at the popped parent cue-auth: collect and stop.
            HopDecision(collect=[Collected(id="content-login", relevance=0.8)], stop=True),
        ]
    )
    result = _service(store, planner).reconstruct("q", budget=5)

    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 3
    # Hop 3's current node is the popped parent cue, proving the stack backtrack
    # path (seed_queue was empty, so this could only come from stack.pop()).
    assert planner.requests[2].current.label == "auth"
    assert planner.requests[2].current.node_kind == "cue"
