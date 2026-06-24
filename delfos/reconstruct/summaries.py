"""Map a graph node to the compact `CandidateSummary` the planner sees.

This is the single place that decides how much of each node is exposed to the
LLM, which keeps token cost controlled and prompt-shaping out of the traversal
loop.
"""

from __future__ import annotations

from collections.abc import Sequence

from delfos.schema import ContentNode, CueNode

from .planner import CandidateSummary

_SNIPPET_LIMIT = 500


def _truncate(text: str, limit: int = _SNIPPET_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def summarize(node: CueNode | ContentNode, tags: Sequence[str] = ()) -> CandidateSummary:
    """Build the planner-facing summary of ``node``."""
    if isinstance(node, CueNode):
        return CandidateSummary(
            id=node.id, node_kind="cue", label=node.text, snippet=None, tags=list(tags)
        )

    label = node.signature or node.symbol_name or node.kind.value
    snippet = node.docstring if node.docstring is not None else _truncate(node.body)
    return CandidateSummary(
        id=node.id, node_kind="content", label=label, snippet=snippet, tags=list(tags)
    )
