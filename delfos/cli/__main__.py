"""Entry point: `python -m delfos.cli` and the `delfos` console script."""

from __future__ import annotations

import sys

from delfos.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
