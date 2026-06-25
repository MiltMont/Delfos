from __future__ import annotations

import pytest

from delfos.cli.app import build_parser


def test_index_requires_repo() -> None:
    parser = build_parser()
    ns = parser.parse_args(["index", "some/repo"])
    assert ns.command == "index"
    assert ns.repo == "some/repo"


def test_index_path_flag_overrides() -> None:
    parser = build_parser()
    ns = parser.parse_args(["status", "--index-path", "/data/g"])
    assert ns.index_path == "/data/g"


def test_no_command_errors() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
