"""Tests for OpenAIEmbedder, driven against a fake OpenAI client (no network).

Focus: the `dimensions` API argument. OpenAI's text-embedding-3 models accept a
`dimensions` request param, but most OpenAI-compatible local endpoints (Ollama,
LM Studio) reject unknown params. The embedder must let callers record a known
dimension for the store without forcing that param onto the request.
"""

from __future__ import annotations

from typing import Any, cast

from openai import OpenAI

from delfos.indexer.embedder import OpenAIEmbedder


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: list[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddings:
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeEmbeddingResponse:
        self.calls.append(kwargs)
        texts: list[str] = list(kwargs["input"])
        data = [_FakeEmbeddingItem(i, [0.1] * self._dim) for i, _ in enumerate(texts)]
        return _FakeEmbeddingResponse(data)


class _FakeClient:
    def __init__(self, dim: int) -> None:
        self.embeddings = _FakeEmbeddings(dim)


def test_omits_dimensions_arg_when_send_dimensions_false() -> None:
    fake = _FakeClient(dim=768)
    embedder = OpenAIEmbedder(
        "nomic-embed-text",
        dimensions=768,
        send_dimensions=False,
        client=cast(OpenAI, fake),
    )

    vectors = embedder.embed(["hello"])

    assert embedder.dimensions == 768
    assert len(vectors[0]) == 768
    assert "dimensions" not in fake.embeddings.calls[0]


def test_sends_dimensions_arg_by_default_when_explicit() -> None:
    fake = _FakeClient(dim=512)
    embedder = OpenAIEmbedder(
        "text-embedding-3-small",
        dimensions=512,
        client=cast(OpenAI, fake),
    )

    embedder.embed(["hello"])

    assert fake.embeddings.calls[0]["dimensions"] == 512
