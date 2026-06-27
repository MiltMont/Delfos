"""CLI logging setup: human-readable step logs on stderr.

The library itself never configures logging (a library shouldn't); only the
``delfos`` / ``delfos-mcp`` entry points call :func:`configure_cli_logging`, so
importing and using Delfos as a library stays silent. Logs go to **stderr** on
purpose: the MCP server speaks JSON-RPC over stdout, and step logs must not
corrupt that stream.
"""

from __future__ import annotations

import logging
import sys

_HANDLER_NAME = "delfos-cli-stderr"


def configure_cli_logging(*, verbose: bool = False) -> None:
    """Attach a stderr handler to the ``delfos`` logger (idempotent).

    Emits one line per pipeline step at INFO; ``verbose`` drops to DEBUG for
    per-file detail. Safe to call more than once — the handler is installed only
    once, but the level is updated on every call.
    """
    root = logging.getLogger("delfos")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.propagate = False
    if any(h.name == _HANDLER_NAME for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.name = _HANDLER_NAME
    handler.setFormatter(logging.Formatter("delfos: %(message)s"))
    root.addHandler(handler)
