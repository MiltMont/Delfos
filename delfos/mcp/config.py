"""Backwards-compatible re-export of the shared config (now in :mod:`delfos.config`)."""

from __future__ import annotations

from delfos.config import (
    ServerConfig,
    build_embedder,
    build_store,
    check_model_match,
    config_from_env,
)

__all__ = [
    "ServerConfig",
    "build_embedder",
    "build_store",
    "check_model_match",
    "config_from_env",
]
