from __future__ import annotations

from delfos.reconstruct.planner import Collected, HopDecision
from delfos.reconstruct.planners.fake import FakeHopPlanner
from delfos.reconstruct.service import ReconstructionService
from delfos.schema import EdgeType
from delfos.store.duckdb_store import DuckDBGraphStore

from .conftest import FakeEmbedder, edge, load, make_content, make_cue, vec


def _build_two_hop_graph(store: DuckDBGraphStore) -> None:
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


def _service(store: DuckDBGraphStore, planner: FakeHopPlanner) -> ReconstructionService:
    embedder = FakeEmbedder({"q": vec(0.1)})
    return ReconstructionService(store, embedder, planner, seed_k=5)


def test_reconstruct_collects_from_first_hop(store: DuckDBGraphStore) -> None:
    _build_two_hop_graph(store)
    planner = FakeHopPlanner(
        [HopDecision(collect=[Collected(id="content-login", relevance=0.9)], stop=True)]
    )
    result = _service(store, planner).reconstruct("q", budget=3)

    assert [c.id for c in result] == ["content-login"]
    assert planner.call_count == 1


def test_reconstruct_descends_and_orders_by_relevance(store: DuckDBGraphStore) -> None:
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


def test_reconstruct_stops_at_budget(store: DuckDBGraphStore) -> None:
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
