from __future__ import annotations

from datetime import datetime

from delfos.mcp.views import (
    SNIPPET_LIMIT,
    ContentDetail,
    NodeSummary,
    content_to_detail,
    content_to_summary,
    cue_to_summary,
)
from delfos.schema import ContentKind, ContentNode, CueNode, CueType, MemoryLayer

NOW = datetime(2026, 6, 24, 12, 0, 0)


def _content(**over: object) -> ContentNode:
    base: dict[str, object] = dict(
        id="c1",
        source_file="a.py",
        git_sha="sha",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="login",
        signature="def login()",
        docstring=None,
        body="def login(): ...",
    )
    base.update(over)
    return ContentNode(**base)  # type: ignore[arg-type]


def test_cue_to_summary_has_no_snippet_or_tags() -> None:
    cue = CueNode(
        id="q1",
        source_file="a.py",
        git_sha="sha",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="auth",
    )
    summary = cue_to_summary(cue)
    assert summary == NodeSummary(id="q1", kind="cue", label="auth", snippet=None, tags=[])


def test_content_summary_prefers_signature_and_uses_docstring() -> None:
    summary = content_to_summary(_content(docstring="Logs a user in."), ["language=python"])
    assert summary.kind == "content"
    assert summary.label == "def login()"
    assert summary.snippet == "Logs a user in."
    assert summary.tags == ["language=python"]


def test_content_summary_truncates_body_when_no_docstring() -> None:
    summary = content_to_summary(_content(body="x" * 600, docstring=None), [])
    assert summary.snippet is not None
    assert len(summary.snippet) == SNIPPET_LIMIT + 1  # 500 chars + ellipsis
    assert summary.snippet.endswith("…")


def test_content_detail_omits_embedding_and_carries_provenance() -> None:
    detail = content_to_detail(
        _content(embedding=[0.1] * 4, embedding_model="fake-v1", body="def login(): ...")
    )
    assert isinstance(detail, ContentDetail)
    assert detail.body == "def login(): ..."
    assert detail.source_file == "a.py"
    assert detail.memory_layer == "semantic"
    assert "embedding" not in detail.model_dump()
