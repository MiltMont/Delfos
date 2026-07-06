"""End-to-end proof of the two enrichment guarantees.

1. Staleness: annotations die with their file on re-index (delete-and-reindex).
2. Retrieval win: an annotated concept phrase is findable via `search` and
   leads to the content node via `traverse_forward`.

SCIP generation is forced to fail so content ids use the deterministic
fallback scheme ``content:{source_file}::{qualified_name}`` regardless of
whether scip-python is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from delfos.enrich import EnrichmentService
from delfos.indexer import Indexer
from delfos.indexer import pipeline as pipeline_mod
from delfos.reconstruct import ReconstructionService
from delfos.scip.generate import ScipGenerationError
from delfos.store.native_store import NativeGraphStore
from delfos.workspace import Workspace

from .conftest import HASH_DIM, HASH_MODEL, HashEmbedder

CONTENT_ID = "content:mod.py::save_snapshot"


@pytest.fixture(autouse=True)
def _no_scip(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    def _raise(*_a: object, **_k: object) -> object:
        raise ScipGenerationError("scip disabled for deterministic ids")

    monkeypatch.setattr(pipeline_mod, "generate_scip_index", _raise)


def _index(repo: Path, store: NativeGraphStore) -> None:
    Indexer(store, HashEmbedder()).index(repo, workspace=Workspace(repo))


def _make_repo(tmp_path: Path, body: str) -> tuple[Path, NativeGraphStore]:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "mod.py").write_text(body)
    store = NativeGraphStore(tmp_path / "graph", embedding_dim=HASH_DIM, embedding_model=HASH_MODEL)
    store.initialize()
    return repo, store


def test_annotations_die_when_their_file_is_reindexed(tmp_path: Path) -> None:
    repo, store = _make_repo(tmp_path, "def save_snapshot():\n    return 1\n")
    _index(repo, store)
    enrich = EnrichmentService(store, HashEmbedder())

    outcome = enrich.annotate(CONTENT_ID, ["crash recovery"], arch_layer="storage")
    cue_id = outcome.written_cue_ids[0]
    assert store.get_node(cue_id) is not None

    (repo / "mod.py").write_text("def save_snapshot():\n    return 2\n")
    _index(repo, store)

    assert store.get_node(cue_id) is None  # annotation died with the file
    assert store.get_node(CONTENT_ID) is not None  # content was re-indexed
    store.close()


def test_search_finds_content_via_concept_cue(tmp_path: Path) -> None:
    repo, store = _make_repo(tmp_path, "def save_snapshot():\n    return 1\n")
    _index(repo, store)
    EnrichmentService(store, HashEmbedder()).annotate(CONTENT_ID, ["crash recovery"])

    service = ReconstructionService(store, HashEmbedder())
    cues = service.search("crash recovery", k=1)

    assert len(cues) == 1
    assert cues[0].text == "crash recovery"
    contents = service.traverse_forward([cues[0].id])
    assert [c.id for c in contents] == [CONTENT_ID]
    store.close()
