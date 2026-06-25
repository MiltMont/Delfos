from __future__ import annotations

from tests.reconstruct.conftest import EMB_DIM, EMB_MODEL


class FixedEmbedder:
    """Embedder protocol double: one fixed vector for any text (model matches the store)."""

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
        return [[0.0] * EMB_DIM for _ in texts]
