"""Tests for the CLI (T-1.14) — typer-based ``initialize`` subcommand."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from on_prem_llm_lab.cli.main import app
from on_prem_llm_lab.services.hardware_scanner import (
    DEFAULT_PLACEHOLDER_END,
    DEFAULT_PLACEHOLDER_START,
)


def _make_valid_setup(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    setup = tmp_path / "config" / "setup.json"
    setup.write_text(
        json.dumps({
            "version": "1.00",
            "init": {
                "doc_targets": [],  # no doc patching for the happy-path CLI test
                "placeholder_start": DEFAULT_PLACEHOLDER_START,
                "placeholder_end": DEFAULT_PLACEHOLDER_END,
                "keep_bak": False,
            },
            "hardware_constraints": None,
            "airllm": {"layer_shards_saving_path": str(tmp_path / "shards")},
        }, indent=2),
        encoding="utf-8",
    )
    return setup


def test_cli_help_lists_initialize_subcommand() -> None:
    """``--help`` exits 0 and mentions the ``initialize`` subcommand."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "initialize" in result.stdout.lower()


def test_cli_initialize_exits_zero_on_success(tmp_path: Path) -> None:
    setup = _make_valid_setup(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["initialize", "--config", str(setup), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "Hardware scan captured at" in result.stdout
    assert "config_setup_json" in result.stdout


def test_cli_initialize_exits_nonzero_when_config_missing(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "initialize",
            "--config", str(tmp_path / "missing.json"),
            "--repo-root", str(tmp_path),
        ],
    )
    assert result.exit_code == 1, result.stdout
    assert "FAILURES" in result.stdout
