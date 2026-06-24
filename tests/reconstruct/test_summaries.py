from __future__ import annotations

from datetime import datetime

from delfos.reconstruct.summaries import summarize
from delfos.schema import ContentKind, ContentNode, CueNode, CueType, MemoryLayer

NOW = datetime(2026, 6, 23, 12, 0, 0)


def _cue() -> CueNode:
    return CueNode(
        id="cue-1",
        source_file="a.py",
        git_sha="s",
        indexed_at=NOW,
        cue_type=CueType.SYMBOL,
        text="load_config",
    )


def _content(body: str) -> ContentNode:
    return ContentNode(
        id="content-1",
        source_file="a.py",
        git_sha="s",
        indexed_at=NOW,
        kind=ContentKind.FUNCTION,
        memory_layer=MemoryLayer.SEMANTIC,
        symbol_name="load_config",
        signature="def load_config() -> Config",
        docstring="Load it.",
        body=body,
    )


def test_summarize_cue_uses_text_as_label() -> None:
    s = summarize(_cue())
    assert s.node_kind == "cue"
    assert s.label == "load_config"
    assert s.snippet is None


def test_summarize_content_prefers_signature_and_docstring() -> None:
    s = summarize(_content(body="def load_config(): ..."))
    assert s.node_kind == "content"
    assert s.label == "def load_config() -> Config"
    assert s.snippet == "Load it."


def test_summarize_truncates_long_body_when_no_docstring() -> None:
    long_body = "x" * 1000
    content = _content(body=long_body)
    content.docstring = None
    s = summarize(content)
    assert s.snippet is not None
    assert len(s.snippet) <= 501  # 500 chars + ellipsis
    assert s.snippet.endswith("…")


def test_summarize_passes_through_tags() -> None:
    s = summarize(_cue(), tags=["language=python"])
    assert s.tags == ["language=python"]
