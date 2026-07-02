from __future__ import annotations

import pytest

from delfos.cli.app import build_parser


def test_index_requires_repo() -> None:
    parser = build_parser()
    ns = parser.parse_args(["index", "some/repo"])
    assert ns.command == "index"
    assert ns.repo == "some/repo"


def test_repo_flag_defaults_to_cwd() -> None:
    parser = build_parser()
    ns = parser.parse_args(["status"])
    assert ns.repo == "."


def test_repo_flag_overrides() -> None:
    parser = build_parser()
    ns = parser.parse_args(["status", "--repo", "/data/proj"])
    assert ns.repo == "/data/proj"


def test_doctor_takes_repo() -> None:
    parser = build_parser()
    ns = parser.parse_args(["doctor", "--repo", "/x"])
    assert ns.command == "doctor"
    assert ns.repo == "/x"


def test_no_command_errors() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_search_parses_k() -> None:
    parser = build_parser()
    ns = parser.parse_args(["search", "how does auth work", "-k", "7"])
    assert ns.command == "search"
    assert ns.query == "how does auth work"
    assert ns.k == 7  # parsed as int, not "7"


def test_reconstruct_parses_budget() -> None:
    parser = build_parser()
    ns = parser.parse_args(["reconstruct", "x", "--budget", "2"])
    assert ns.command == "reconstruct"
    assert ns.budget == 2  # parsed as int


def test_serve_takes_no_args() -> None:
    parser = build_parser()
    ns = parser.parse_args(["serve"])
    assert ns.command == "serve"
