"""Unit tests for backends/airllm_backend.py (T-3.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psutil
import pytest
import torch

from on_prem_llm_lab.backends.airllm_backend import (
    AirLLMBackend,
    AirLLMConfigError,
    AirLLMDiskError,
)
from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant
from on_prem_llm_lab.shared.automodel_factory import UnsupportedQuantizationError

from .conftest import StepClock


class TestConstruction:
    def test_all_required_kwargs(self, tmp_path: Path, make_airllm: Any) -> None:
        assert make_airllm(tmp_path)._quantization is Quant.FP16

    @pytest.mark.parametrize("q", ["nf4", "q2", "fp32"])
    def test_unsupported_quantization_raises(
        self, tmp_path: Path, make_airllm: Any, q: str,
    ) -> None:
        with pytest.raises(UnsupportedQuantizationError):
            make_airllm(tmp_path, quantization=q)

    def test_empty_shard_path_raises_config_error(self) -> None:
        with pytest.raises(AirLLMConfigError):
            AirLLMBackend(
                target_label="t", model_id="m", quantization=Quant.FP16,
                layer_shards_saving_path="", factory=lambda *a, **k: None,
            )

    def test_positional_construction_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            AirLLMBackend("t", "m", Quant.FP16, str(tmp_path))  # type: ignore[misc]


class TestPreflightDisk:
    def test_shortfall_raises_disk_error(
        self, tmp_path: Path, make_airllm: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Usage:
            free = 1 * 1024 ** 3
        monkeypatch.setattr(psutil, "disk_usage", lambda p: _Usage())
        with pytest.raises(AirLLMDiskError, match="1.00 GiB"):
            make_airllm(tmp_path, min_free_disk_gb=25.0).load()

    def test_creates_shard_dir_if_missing(
        self, tmp_path: Path, make_airllm: Any,
    ) -> None:
        shard_dir = tmp_path / "new_shards"
        make_airllm(shard_dir).load()
        assert shard_dir.exists()


class TestLoadKwargs:
    @pytest.mark.parametrize(
        ("cuda", "expected"), [(False, "cpu"), (True, "cuda:0")],
    )
    def test_device_pin(
        self, tmp_path: Path, make_airllm: Any, monkeypatch: pytest.MonkeyPatch,
        cuda: bool, expected: str,
    ) -> None:
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        make_airllm(tmp_path, calls=calls).load()
        assert calls[0]["device"] == expected

    @pytest.mark.parametrize(
        ("quant", "expected"), [(Quant.Q4, "4bit"), (Quant.Q8, "8bit")],
    )
    def test_compression_mapping(
        self, tmp_path: Path, make_airllm: Any, quant: Quant, expected: str,
    ) -> None:
        calls: list[dict[str, Any]] = []
        make_airllm(tmp_path, calls=calls, quantization=quant).load()
        assert calls[0]["compression"] == expected

    def test_fp16_omits_compression_kwarg(
        self, tmp_path: Path, make_airllm: Any,
    ) -> None:
        calls: list[dict[str, Any]] = []
        make_airllm(tmp_path, calls=calls).load()
        assert "compression" not in calls[0]

    def test_load_idempotent(self, tmp_path: Path, make_airllm: Any) -> None:
        calls: list[dict[str, Any]] = []
        be = make_airllm(tmp_path, calls=calls)
        be.load()
        be.load()
        assert len(calls) == 1

    def test_layer_shards_path_stringified(
        self, tmp_path: Path, make_airllm: Any,
    ) -> None:
        calls: list[dict[str, Any]] = []
        make_airllm(tmp_path, calls=calls).load()
        assert calls[0]["layer_shards_saving_path"] == str(tmp_path)


class TestHfTokenResolution:
    def test_explicit_kwarg_wins_over_env(
        self, tmp_path: Path, make_airllm: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "env_token")
        calls: list[dict[str, Any]] = []
        make_airllm(tmp_path, calls=calls, hf_token="explicit").load()
        assert calls[0]["hf_token"] == "explicit"

    def test_env_fallback_when_no_kwarg(
        self, tmp_path: Path, make_airllm: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "env_token")
        calls: list[dict[str, Any]] = []
        make_airllm(tmp_path, calls=calls).load()
        assert calls[0]["hf_token"] == "env_token"

    def test_no_token_omits_kwarg(
        self, tmp_path: Path, make_airllm: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        calls: list[dict[str, Any]] = []
        make_airllm(tmp_path, calls=calls).load()
        assert "hf_token" not in calls[0]


class TestGenerate:
    def test_before_load_raises(self, tmp_path: Path, make_airllm: Any) -> None:
        with pytest.raises(RuntimeError, match="before load"):
            make_airllm(tmp_path).generate("hi", max_new_tokens=4, params={})

    def test_returns_well_shaped_result(
        self, tmp_path: Path, make_airllm: Any,
    ) -> None:
        clock = StepClock([0.0, 0.5, 5.0])
        be = make_airllm(tmp_path, clock=clock)
        be.load()
        r = be.generate("Hello", max_new_tokens=4, params={})
        assert isinstance(r, BackendRunResult)
        assert r.backend is BackendId.AIRLLM
        assert r.target_label == "test-target"
        assert r.model_id == "test/model"
        assert r.prompt_tokens == 8
        assert r.completion_tokens == 4
        assert r.ttft_ms == pytest.approx(500.0)
        assert r.wall_s == pytest.approx(5.0)
        assert r.completion_text == "airllm stub completion"


class TestUnload:
    def test_releases_and_idempotent(
        self, tmp_path: Path, make_airllm: Any,
    ) -> None:
        be = make_airllm(tmp_path)
        be.load()
        assert be._loaded is not None
        be.unload()
        be.unload()
        assert be._loaded is None
