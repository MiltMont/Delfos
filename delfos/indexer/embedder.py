"""Embedding layer for Delfos cue-node vector generation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openai import OpenAI
from openai.types import CreateEmbeddingResponse

_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding backends."""

    @property
    def model(self) -> str: ...

    @property
    def model_version(self) -> str | None: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Embedder implementation backed by the OpenAI embeddings API."""

    _model: str
    _model_version: str | None
    _dimensions: int
    _dimensions_explicit: bool
    _client: OpenAI

    def __init__(
        self,
        model: str,
        *,
        dimensions: int | None = None,
        model_version: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._model_version = model_version

        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(api_key=api_key)

        if dimensions is not None:
            self._dimensions = dimensions
            self._dimensions_explicit = True
        elif model in _KNOWN_DIMENSIONS:
            self._dimensions = _KNOWN_DIMENSIONS[model]
            self._dimensions_explicit = False
        else:
            msg = (
                f"Unknown model {model!r}: pass `dimensions` explicitly "
                f"when using a model not in the known-dimensions dict."
            )
            raise ValueError(msg)

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_version(self) -> str | None:
        return self._model_version

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        if self._dimensions_explicit:
            resp: CreateEmbeddingResponse = self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
        else:
            resp = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )

        sorted_data = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in sorted_data]
