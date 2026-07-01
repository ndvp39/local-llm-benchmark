"""Integration test for the ``run_sweep`` plumbing precondition (T-2a.4 · ADR-010).

Confirms end-to-end that ``SDK.run_sweep`` refuses to start without a successful
plumbing manifest, accepts a real manifest produced by the real
``PlumbingTestRunner``, and supports ``skip_plumbing=True`` as an escape hatch.
External deps are stubbed at the import boundary as in
``test_plumbing_test_runner.py`` so we don't pull a real model off HF Hub.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.sdk import (
    OnPremLlmSDK,
    PlumbingNotRunError,
)

_PLUMBING_MODEL = {
    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "quantization": "q4",
    "label": "tinyllama-1b-q4-plumbing",
    "loader": "airllm",
}


def _make_setup(tmp_path: Path) -> Path:
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
    fake_airllm = types.ModuleType("airllm")
    fake_airllm.AutoModel = _FakeAirLLMAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "airllm", fake_airllm)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo_id, token=None: str(tmp_path / "fake_local_model"),
    )
    monkeypatch.setattr("psutil.Process", _FakeProcess)


_SENTINEL_CSV = Path("sentinel_sweep.csv")


def _stub_sweep_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit the sweep service so the guard-precondition test stays fast."""
    monkeypatch.setattr(
        "on_prem_llm_lab.sdk.sdk._run_sweep",
        lambda **kwargs: _SENTINEL_CSV,
    )


@pytest.mark.integration
def test_run_sweep_without_plumbing_manifest_raises_end_to_end(tmp_path: Path) -> None:
    """DoD: SDK.run_sweep MUST raise PlumbingNotRunError when no manifest exists."""
    setup = _make_setup(tmp_path)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    with pytest.raises(PlumbingNotRunError, match="No plumbing manifest"):
        sdk.run_sweep()


@pytest.mark.integration
def test_run_sweep_passes_guard_after_real_plumbing_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A real PlumbingTestRunner manifest MUST unlock run_sweep end-to-end."""
    setup = _make_setup(tmp_path)
    _stub_externals(monkeypatch, tmp_path)

    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    plumbing = sdk.run_plumbing_test()
    assert plumbing.overall == "ok"
    assert plumbing.manifest_path is not None and plumbing.manifest_path.exists()

    _stub_sweep_service(monkeypatch)
    assert sdk.run_sweep() == _SENTINEL_CSV


@pytest.mark.integration
def test_run_sweep_skip_plumbing_bypass_without_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``skip_plumbing=True`` MUST reach the sweep service even with no manifest."""
    setup = _make_setup(tmp_path)
    _stub_sweep_service(monkeypatch)
    sdk = OnPremLlmSDK(config_path=setup, repo_root=tmp_path)
    assert sdk.run_sweep(skip_plumbing=True) == _SENTINEL_CSV
