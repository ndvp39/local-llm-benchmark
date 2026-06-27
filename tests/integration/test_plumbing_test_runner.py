"""Integration test for PlumbingTestRunner end-to-end wiring (T-2a.3 · ADR-010).

Drives the production path (``SDK.run_plumbing_test()`` →
``build_default_stages()`` → ``PlumbingTestRunner.run()``) with the three heavy
external dependencies (``huggingface_hub.snapshot_download``, ``airllm.AutoModel``,
``psutil.Process``) stubbed at their import boundary so the test doesn't pull a
real model off HF Hub or instantiate a real AirLLM model. T-2a.5 exercises the
same wiring against the real ``plumbing_test_model`` on the student's machine.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from on_prem_llm_lab.cli.main import app
from on_prem_llm_lab.sdk import OnPremLlmSDK

_PLUMBING_MODEL = {
    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "quantization": "q2",
    "label": "tiny-q2",
}


def _make_setup(tmp_path: Path) -> Path:
    """Materialise an env-initialised setup.json under tmp_path/config/."""
    setup = tmp_path / "config" / "setup.json"
    setup.parent.mkdir(parents=True)
    setup.write_text(
        json.dumps({
            "version": "1.00",
            "init": {"max_age_hours": 99999},
            "hardware_constraints": {
                "captured_at": "2026-06-27T10:00:00Z",
                "cpu": {"cores_logical": 8},
            },
            "plumbing_test_model": _PLUMBING_MODEL,
            "airllm": {"layer_shards_saving_path": str(tmp_path / "shards")},
            "generation": {"plumbing_max_new_tokens": 4},
        }),
        encoding="utf-8",
    )
    return setup


class _FakeTokenized:
    input_ids = [[1, 2, 3]]


class _FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt") -> _FakeTokenized:
        return _FakeTokenized()


class _FakeAirLLMModel:
    layers = [None] * 22
    tokenizer = _FakeTokenizer()

    def generate(self, input_ids: Any, max_new_tokens: int = 1) -> None:
        return None


class _FakeAirLLMAutoModel:
    @staticmethod
    def from_pretrained(model_path: str, **kwargs: Any) -> _FakeAirLLMModel:
        return _FakeAirLLMModel()


class _FakeMemInfo:
    rss = 100 * 1024 * 1024


class _FakeProcess:
    def memory_info(self) -> _FakeMemInfo:
        return _FakeMemInfo()


def _stub_externals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Patch the three heavy deps at their import boundary inside the closures."""
    fake_airllm = types.ModuleType("airllm")
    fake_airllm.AutoModel = _FakeAirLLMAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "airllm", fake_airllm)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, token=None: str(tmp_path / "fake_local_model"),
    )
    monkeypatch.setattr("psutil.Process", _FakeProcess)


@pytest.mark.integration
def test_full_plumbing_pipeline_succeeds_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All four stages MUST reach ``ok`` and the manifest MUST land on disk (DoD)."""
    setup = _make_setup(tmp_path)
    _stub_externals(monkeypatch, tmp_path)

    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    result = sdk.run_plumbing_test()

    assert result.overall == "ok"
    assert result.remediation_hint is None
    for name in ("download", "mmap_allocation", "metric_collection", "manifest_write"):
        assert result.stages[name].status == "ok", f"{name} did not reach ok"

    assert result.manifest_path is not None and result.manifest_path.exists()
    assert str(result.manifest_path).startswith(str(tmp_path / "results"))
    on_disk = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert on_disk["overall"] == "ok"
    assert on_disk["plumbing_test_model"]["id"] == _PLUMBING_MODEL["id"]
    assert on_disk["stages"]["mmap_allocation"]["layers"] == 22
    assert "ttft_ms" in on_disk["stages"]["metric_collection"]
    assert "peak_ram_mb" in on_disk["stages"]["metric_collection"]


@pytest.mark.integration
def test_cli_run_plumbing_test_succeeds_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI subcommand MUST exit 0 and print per-stage statuses on success."""
    setup = _make_setup(tmp_path)
    _stub_externals(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run-plumbing-test", "--config", str(setup), "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "OK: plumbing test passed" in result.stdout
    for name in ("download", "mmap_allocation", "metric_collection", "manifest_write"):
        assert name in result.stdout
