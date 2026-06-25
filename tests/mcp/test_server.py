from __future__ import annotations

import asyncio

import pytest
from mcp.server.fastmcp import FastMCP

from delfos.mcp.server import (
    _fetch,  # pyright: ignore[reportPrivateUsage]
    _search,  # pyright: ignore[reportPrivateUsage]
    _traverse_forward,  # pyright: ignore[reportPrivateUsage]
    _traverse_reverse,  # pyright: ignore[reportPrivateUsage]
    build_server,
    reconstruct_prompt,
)
from delfos.reconstruct import ReconstructionService
from delfos.schema import EdgeType, TagCategory
from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import (
    EMB_DIM,
    EMB_MODEL,
    edge,
    load,
    make_content,
    make_cue,
    make_tag,
)

from .conftest import make_service, vec


class BoomEmbedder:
    """Embedder that always raises on embed — simulates a dead endpoint."""

    @property
    def model(self) -> str:
        return EMB_MODEL

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return EMB_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ConnectionError("endpoint unreachable")


def _seed(store: NativeGraphStore) -> None:
    cue = make_cue("cue-1", "auth", embedding=vec(0.10))
    content = make_content("c1", "login")
    tag = make_tag("t1", TagCategory.LANGUAGE, "python")
    load(
        store,
        [cue, content, tag],
        [
            edge("cue-1", "c1", EdgeType.CUE_OF),
            edge("c1", "t1", EdgeType.TAGGED_WITH),
        ],
    )


def test_search_returns_cue_summaries(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _search(svc, "q", k=5)

    assert [s.id for s in out] == ["cue-1"]
    assert out[0].kind == "cue"
    assert out[0].label == "auth"


def test_search_translates_embedder_failure(store: NativeGraphStore) -> None:
    _seed(store)
    svc = ReconstructionService(store, BoomEmbedder())

    with pytest.raises(RuntimeError, match="embedding endpoint"):
        _search(svc, "q", k=5)


def test_traverse_forward_returns_content_summaries_with_tags(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _traverse_forward(svc, ["cue-1"])

    assert [s.id for s in out] == ["c1"]
    assert out[0].kind == "content"
    assert out[0].tags == ["language=python"]


def test_traverse_forward_unknown_tag_category_errors(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    try:
        _traverse_forward(svc, ["cue-1"], [("not_a_category", "x")])
    except ValueError as exc:
        assert "not_a_category" in str(exc)
        assert "language" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError for unknown tag category")


def test_traverse_reverse_returns_sibling_cues(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _traverse_reverse(svc, ["c1"])

    assert [s.id for s in out] == ["cue-1"]
    assert out[0].kind == "cue"


def test_fetch_returns_full_bodies(store: NativeGraphStore) -> None:
    _seed(store)
    svc = make_service(store, vec(0.10))

    out = _fetch(svc, ["c1", "missing"])

    assert [d.id for d in out] == ["c1"]
    assert out[0].body == "def login(): ..."


def test_reconstruct_prompt_contains_protocol_and_args() -> None:
    text = reconstruct_prompt("how does auth work", budget=4)
    lowered = text.lower()
    assert "how does auth work" in text
    assert "4" in text
    assert "search" in lowered
    assert "fetch" in lowered
    assert "budget" in lowered


def test_build_server_registers_tools_and_prompt(store: NativeGraphStore) -> None:
    svc = make_service(store, vec(0.10))
    server = build_server(svc)
    assert isinstance(server, FastMCP)
    tool_names = {t.name for t in asyncio.run(server.list_tools())}
    assert tool_names == {"search", "traverse_forward", "traverse_reverse", "fetch"}
    prompt_names = {p.name for p in asyncio.run(server.list_prompts())}
    assert "reconstruct" in prompt_names
