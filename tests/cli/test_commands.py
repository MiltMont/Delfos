from __future__ import annotations

from pathlib import Path

from delfos.cli.app import run_index, run_search, run_status
from delfos.cli.render import render_search
from delfos.reconstruct import ReconstructionService
from delfos.store.native_store import NativeGraphStore
from tests.cli.conftest import FixedEmbedder
from tests.reconstruct.conftest import EMB_DIM, EMB_MODEL, FakeEmbedder, load, make_cue, vec


def _store(tmp_path: Path) -> NativeGraphStore:
    s = NativeGraphStore(tmp_path / "graph", embedding_dim=EMB_DIM, embedding_model=EMB_MODEL)
    s.initialize()
    return s


def test_index_then_status_reflects_written_files(tmp_path: Path) -> None:
    # A tiny repo to index.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def hello():\n    return 1\n")

    store = _store(tmp_path)
    # FixedEmbedder returns a constant vector for any cue text it is asked to embed.
    embedder = FixedEmbedder()

    stats = run_index(str(repo), store, embedder)
    assert stats.indexed_files == 1
    assert stats.failed_files == []

    out = run_status(store, EMB_MODEL, EMB_DIM)
    assert "mod.py" in out
    assert "1 file" in out
    store.close()


def test_search_renders_seeded_cue(tmp_path: Path) -> None:
    store = _store(tmp_path)
    load(store, [make_cue("cue-auth", "authentication", embedding=vec(1.0))], [])
    # The query text is known here, so a keyed FakeEmbedder is fine.
    embedder = FakeEmbedder({"how does auth work": vec(1.0)})
    service = ReconstructionService(store, embedder)

    cues = run_search("how does auth work", 5, service)
    out = render_search(cues)
    assert "cue-auth" in out
    assert "authentication" in out
    store.close()
