from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from delfos.enrich import EnrichmentService
from delfos.mcp.server import (
    _annotate,  # pyright: ignore[reportPrivateUsage]
    _fetch,  # pyright: ignore[reportPrivateUsage]
    _implementations,  # pyright: ignore[reportPrivateUsage]
    _references,  # pyright: ignore[reportPrivateUsage]
    _search,  # pyright: ignore[reportPrivateUsage]
    _traverse_forward,  # pyright: ignore[reportPrivateUsage]
    _traverse_reverse,  # pyright: ignore[reportPrivateUsage]
    _type_definition,  # pyright: ignore[reportPrivateUsage]
    build_server,
    enrich_prompt,
    reconstruct_prompt,
)
from delfos.reconstruct import ReconstructionService
from delfos.schema import EdgeType, TagCategory
from delfos.scip.reader import ScipIndex
from delfos.scip.service import ScipService
from delfos.store.native_store import NativeGraphStore
from tests.reconstruct.conftest import (
    EMB_DIM,
    EMB_MODEL,
    FakeEmbedder,
    edge,
    load,
    make_content,
    make_cue,
    make_tag,
)
from tests.scip.builders import (
    document,
    occurrence,
    relationship,
    symbol_information,
    write_index,
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
    assert tool_names == {
        "search",
        "traverse_forward",
        "traverse_reverse",
        "fetch",
        "references",
        "implementations",
        "type_definition",
        "annotate",
    }
    prompt_names = {p.name for p in asyncio.run(server.list_prompts())}
    assert "reconstruct" in prompt_names


SCIP_SYM = "scip-python python . a.py/login()."


def _scip_service(store: NativeGraphStore, tmp_path: Path) -> ScipService:
    path = write_index(
        tmp_path / "index.scip",
        documents=[
            document(
                "a.py",
                occurrences=[
                    occurrence(SCIP_SYM, 0, definition=True),
                    occurrence(SCIP_SYM, 9),
                ],
                symbols=[
                    symbol_information(
                        SCIP_SYM,
                        [
                            relationship("iface#", is_implementation=True),
                            relationship("Type#", is_type_definition=True),
                        ],
                    )
                ],
            )
        ],
    )
    return ScipService(store, ScipIndex(path))


def test_scip_tools_resolve_references_and_relationships(
    store: NativeGraphStore, tmp_path: Path
) -> None:
    # Node id IS the SCIP symbol — no separate scip_symbol field.
    content = make_content(SCIP_SYM, "login")
    load(store, [content], [])
    scip = _scip_service(store, tmp_path)

    refs = _references(scip, SCIP_SYM)
    assert [(r.relative_path, r.start_line) for r in refs] == [("a.py", 9)]
    assert [r.symbol for r in _implementations(scip, SCIP_SYM)] == ["iface#"]
    assert [r.symbol for r in _type_definition(scip, SCIP_SYM)] == ["Type#"]


def test_scip_tools_unavailable_when_no_index() -> None:
    with pytest.raises(RuntimeError, match="SCIP index not available"):
        _references(None, "c1")
    with pytest.raises(RuntimeError, match="SCIP index not available"):
        _implementations(None, "c1")
    with pytest.raises(RuntimeError, match="SCIP index not available"):
        _type_definition(None, "c1")


def test_annotate_tool_writes_and_echoes_vocab(store: NativeGraphStore) -> None:
    load(store, [make_content("content:1", "save_snapshot")], [])
    enrich = EnrichmentService(store, FakeEmbedder({"crash recovery": vec(3.0)}))

    result = _annotate(
        enrich, "content:1", ["crash recovery"], arch_layer="storage", pattern_type=None
    )

    assert result.content_id == "content:1"
    assert len(result.written_cue_ids) == 1
    assert result.written_tag_ids == ["tag:arch_layer:storage"]
    assert result.existing_values["arch_layer"] == ["storage"]


def test_annotate_tool_errors_without_service() -> None:
    with pytest.raises(RuntimeError, match="enrichment unavailable"):
        _annotate(None, "content:1", ["x"], arch_layer=None, pattern_type=None)


def test_enrich_prompt_teaches_the_annotate_discipline() -> None:
    text = enrich_prompt()
    assert "annotate" in text
    assert "1-5 concept phrases" in text
    assert "reuse" in text.lower()

    focused = enrich_prompt(focus="the storage layer")
    assert "the storage layer" in focused


def test_build_server_registers_annotate_and_enrich(store: NativeGraphStore) -> None:
    svc = make_service(store, vec(0.0))
    enrich = EnrichmentService(store, FakeEmbedder({}))

    server = build_server(svc, None, enrich)

    tools = asyncio.run(server.list_tools())
    assert "annotate" in [t.name for t in tools]
    prompts = asyncio.run(server.list_prompts())
    assert "enrich" in [p.name for p in prompts]
