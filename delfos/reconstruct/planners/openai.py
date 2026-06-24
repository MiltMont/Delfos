"""OpenAI-backed :class:`HopPlanner` — the LLM in the reconstruct loop.

Mirrors :class:`delfos.indexer.embedder.OpenAIEmbedder`: the concrete
:class:`~openai.OpenAI` client is injectable so tests run without a network. Each
hop is one structured-output call whose response is parsed straight into a
:class:`HopDecision`. If the model declines to produce a decision the planner
returns a terminal ``stop`` so the walk always halts; id hallucinations are left
for the service to filter against the real candidate set.
"""

from __future__ import annotations

from openai import OpenAI

from delfos.reconstruct.planner import HopDecision, HopRequest

_SYSTEM_PROMPT = (
    "You traverse a code-memory graph to answer a developer's query. At each "
    "hop you see the current node and its candidate neighbors. Choose which "
    "candidates to collect (with a relevance in [0, 1]), optionally one "
    "candidate id to descend into next, and whether to stop. Only ever "
    "reference ids that appear in the candidate list."
)


class OpenAIHopPlanner:
    """Per-hop planner backed by the OpenAI structured-output API."""

    _model: str
    _client: OpenAI

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def decide(self, request: HopRequest) -> HopDecision:
        """Ask the model for a single hop decision."""
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": request.model_dump_json()},
            ],
            response_format=HopDecision,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            return HopDecision(stop=True)
        return parsed
