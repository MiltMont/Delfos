"""Tests for the agent-driven enrichment write path."""

from __future__ import annotations

from delfos.enrich.service import (
    _concept_cue_id,  # pyright: ignore[reportPrivateUsage]
    _normalize_phrase,  # pyright: ignore[reportPrivateUsage]
    _normalize_tag_value,  # pyright: ignore[reportPrivateUsage]
)


def test_normalize_phrase_lowercases_and_collapses_whitespace() -> None:
    assert _normalize_phrase("  Crash \t Recovery ") == "crash recovery"
    assert _normalize_phrase("") == ""
    assert _normalize_phrase("   ") == ""


def test_normalize_tag_value_lowercases_and_hyphenates() -> None:
    assert _normalize_tag_value("Storage Engine") == "storage-engine"
    assert _normalize_tag_value("  CLI  ") == "cli"
    assert _normalize_tag_value("   ") == ""


def test_concept_cue_id_mirrors_error_cue_scheme_and_is_stable() -> None:
    a = _concept_cue_id("a.py", "crash recovery")
    b = _concept_cue_id("a.py", "crash recovery")
    assert a == b
    assert a.startswith("cue:concept:a.py::")
    assert len(a.split("::")[-1]) == 12
    assert _concept_cue_id("b.py", "crash recovery") != a
    assert _concept_cue_id("a.py", "rate limiting") != a
