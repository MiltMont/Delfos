"""The ``.delfos/`` workspace: one self-describing directory per indexed repo.

Delfos used to scatter its artifacts across three independently-configured
locations — the graph store (``DELFOS_INDEX_PATH``), the SCIP index written at
index time (``<repo>/index.scip``), and the SCIP index read at serve time
(``DELFOS_SCIP_PATH``) — with nothing tying them together. A
:class:`Workspace` consolidates them under ``<repo>/.delfos/``::

    <repo>/.delfos/
    ├── store/          # NativeGraphStore snapshot (graph + vectors)
    ├── index.scip      # SCIP cross-reference index
    ├── manifest.json   # provenance + consistency metadata
    └── config.toml     # optional non-secret config (embed/llm settings)

The :class:`Manifest` records *which run* produced the graph and the SCIP index
so staleness is detectable: every ``delfos index`` mints one ``run_id`` stamped
on the graph (always) and on the SCIP index (only when it regenerates), and
:attr:`Manifest.is_consistent` is simply ``scip.run_id == graph.last_run_id``.
"""

from __future__ import annotations

import tomllib
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict

WORKSPACE_DIRNAME = ".delfos"
STORE_DIRNAME = "store"
SCIP_INDEX_FILENAME = "index.scip"
MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.toml"

MANIFEST_FORMAT_VERSION = 1


class ScipStatus(StrEnum):
    """Whether a SCIP index was produced by the most recent index run."""

    PRESENT = "present"
    ABSENT = "absent"
    FAILED = "failed"


class EmbedInfo(BaseModel):
    """Embedding model the graph was indexed with (authoritative for queries)."""

    model_config = ConfigDict(extra="forbid")

    model: str
    dim: int
    model_version: str | None = None
    base_url: str | None = None


class GraphInfo(BaseModel):
    """Provenance of the persisted graph store."""

    model_config = ConfigDict(extra="forbid")

    last_run_id: str
    updated_at: datetime
    files: int


class ScipInfo(BaseModel):
    """Provenance of the persisted SCIP index."""

    model_config = ConfigDict(extra="forbid")

    status: ScipStatus
    run_id: str | None = None
    generated_at: datetime | None = None
    generator: str = "scip-python"


class Manifest(BaseModel):
    """Top-level provenance record for a ``.delfos/`` workspace."""

    model_config = ConfigDict(extra="forbid")

    format_version: int = MANIFEST_FORMAT_VERSION
    repo_root: str
    embed: EmbedInfo
    graph: GraphInfo
    scip: ScipInfo

    @property
    def is_consistent(self) -> bool:
        """True when the SCIP index and the graph come from the same index run."""
        return self.scip.status is ScipStatus.PRESENT and self.scip.run_id == self.graph.last_run_id


class Workspace:
    """Resolves the ``.delfos/`` paths for the repo rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @property
    def dir(self) -> Path:
        return self.root / WORKSPACE_DIRNAME

    @property
    def store_path(self) -> Path:
        return self.dir / STORE_DIRNAME

    @property
    def scip_path(self) -> Path:
        return self.dir / SCIP_INDEX_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_FILENAME

    @property
    def config_path(self) -> Path:
        return self.dir / CONFIG_FILENAME

    def ensure_dirs(self) -> None:
        """Create ``.delfos/`` and the store directory if absent."""
        self.store_path.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> Manifest | None:
        """Read and validate ``manifest.json``; ``None`` if absent or unreadable."""
        path = self.manifest_path
        if not path.is_file():
            return None
        try:
            return Manifest.model_validate_json(path.read_text())
        except Exception:
            return None

    def write_manifest(self, manifest: Manifest) -> None:
        """Persist ``manifest.json`` (creating ``.delfos/`` if needed)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    def load_config(self) -> dict[str, str]:
        """Read ``config.toml`` into ``DELFOS_*`` env-style keys (all optional).

        Lets a repo persist non-secret settings (embed/llm model, dim, base
        URLs) without exporting environment variables. Secrets (API keys) are
        deliberately *not* read from here — keep them in the environment.
        """
        path = self.config_path
        if not path.is_file():
            return {}
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        return _config_to_env(data)


_TRUTHY = {"1", "true", "yes", "on"}

# (toml table, key) -> DELFOS_* env name. Secrets are intentionally omitted.
_CONFIG_KEY_MAP: dict[tuple[str, str], str] = {
    ("embed", "model"): "DELFOS_EMBED_MODEL",
    ("embed", "dim"): "DELFOS_EMBED_DIM",
    ("embed", "base_url"): "DELFOS_EMBED_BASE_URL",
    ("embed", "send_dimensions"): "DELFOS_EMBED_SEND_DIM",
    ("llm", "model"): "DELFOS_LLM_MODEL",
    ("llm", "base_url"): "DELFOS_LLM_BASE_URL",
}


def _config_to_env(data: dict[str, object]) -> dict[str, str]:
    out: dict[str, str] = {}
    for (table, key), env_name in _CONFIG_KEY_MAP.items():
        section = data.get(table)
        if not isinstance(section, dict):
            continue
        typed_section = cast("dict[str, object]", section)
        if key not in typed_section:
            continue
        out[env_name] = _stringify(typed_section[key])
    return out


def _stringify(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)
