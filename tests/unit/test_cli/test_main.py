"""Tests for the CLI (T-1.14 ``initialize`` + T-2a.2 ``run-plumbing-test``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from on_prem_llm_lab import (
    EnvironmentNotInitializedError,
    OnPremLlmSDK,
    PlumbingStageError,
)
from on_prem_llm_lab.cli.main import app
from on_prem_llm_lab.services.hardware_scanner import (
    DEFAULT_PLACEHOLDER_END,
    DEFAULT_PLACEHOLDER_START,
)
from on_prem_llm_lab.services.plumbing_test_runner import PlumbingResult, StageOutcome


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


def _fake_result(tmp_path: Path, overall: str = "ok") -> PlumbingResult:
    """Cheap PlumbingResult builder for monkeypatch-based CLI tests."""
    return PlumbingResult(
        captured_at="2026-06-27T12:00:00Z",
        plumbing_test_model={"id": "x", "label": "y"},
        stages={
            "download": StageOutcome("ok", 1.0),
            "mmap_allocation": StageOutcome("ok", 2.0),
            "metric_collection": StageOutcome("ok", 3.0),
            "manifest_write": StageOutcome("ok", 0.0, extras={"path": str(tmp_path / "m.json")}),
        },
        overall=overall,
        remediation_hint=None if overall == "ok" else "try harder",
        manifest_path=tmp_path / "m.json",
    )


def test_cli_help_lists_run_plumbing_test(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-plumbing-test" in result.stdout


def test_cli_run_plumbing_test_exits_zero_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _make_valid_setup(tmp_path)
    monkeypatch.setattr(
        OnPremLlmSDK, "run_plumbing_test", lambda self, **kw: _fake_result(tmp_path),
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run-plumbing-test", "--config", str(setup), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "OK" in result.stdout
    assert "metric_collection" in result.stdout


def test_cli_run_plumbing_test_exits_nonzero_on_stage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _make_valid_setup(tmp_path)
    fake = _fake_result(tmp_path, overall="fail")

    def raises(self: OnPremLlmSDK, **_kw: object) -> PlumbingResult:
        raise PlumbingStageError("download", "network down", fake)

    monkeypatch.setattr(OnPremLlmSDK, "run_plumbing_test", raises)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run-plumbing-test", "--config", str(setup), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 1, result.stdout
    assert "download" in result.stdout
    assert "network down" in result.stdout
    assert "try harder" in result.stdout


def test_cli_run_plumbing_test_exits_nonzero_when_env_not_initialised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = _make_valid_setup(tmp_path)

    def raises(self: OnPremLlmSDK, **_kw: object) -> PlumbingResult:
        raise EnvironmentNotInitializedError("constraints null. Run `uv run init_env.py`.")

    monkeypatch.setattr(OnPremLlmSDK, "run_plumbing_test", raises)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run-plumbing-test", "--config", str(setup), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 1, result.stdout
    assert "env not initialised" in result.stdout
