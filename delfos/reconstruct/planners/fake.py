"""A scripted `HopPlanner` for tests and demos — no network, fully deterministic."""

from __future__ import annotations

from collections.abc import Sequence

from delfos.reconstruct.planner import HopDecision, HopRequest


class FakeHopPlanner:
    """Returns pre-scripted decisions in order.

    After the script is exhausted it returns a terminal ``stop`` decision, so a
    walk always halts. With ``error_after=n`` the ``n``-th call (0-indexed) and
    every call after it raise ``RuntimeError``, to exercise partial-result
    handling.
    """

    def __init__(self, decisions: Sequence[HopDecision], *, error_after: int | None = None) -> None:
        self._decisions = list(decisions)
        self._error_after = error_after
        self._calls = 0
        self.requests: list[HopRequest] = []

    def decide(self, request: HopRequest) -> HopDecision:
        index = self._calls
        self._calls += 1
        self.requests.append(request)
        if self._error_after is not None and index >= self._error_after:
            raise RuntimeError("FakeHopPlanner scripted failure")
        if index < len(self._decisions):
            return self._decisions[index]
        return HopDecision(collect=[], descend_into=None, stop=True)

    @property
    def call_count(self) -> int:
        return self._calls
