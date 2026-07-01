"""Integration test — AirLLMBackend end-to-end via `_default_factory` (T-3.1).

Proves the full wiring: backend construction → pre-flight → the
real ``_default_factory`` path → ``airllm.AutoModel.from_pretrained``
(faked via ``sys.modules["airllm"]`` injection so no download fires)
→ ``generate()`` → ``BackendRunResult``. Complements the unit tests
which inject ``factory=`` directly and bypass the default path.

Corresponds to PRD_airllm_integration.md §9.2 I-AL-1.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import psutil
import pytest

from on_prem_llm_lab.backends.airllm_backend import AirLLMBackend
from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant


class _Tensor:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def __getitem__(self, idx: int) -> _Tensor:
        return self


class _Tokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt") -> Any:
        return types.SimpleNamespace(input_ids=_Tensor(6))

    def decode(self, output: Any, **kwargs: Any) -> str:
        return "airllm integration ok"


class _StubAirLLMModel:
    layers = list(range(35))

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.init_kwargs = kwargs
        self.tokenizer = _Tokenizer()

    def generate(self, input_ids: _Tensor, max_new_tokens: int) -> _Tensor:
        return _Tensor(input_ids.shape[-1] + max_new_tokens)


class _StubAirLLMModule:
    """Stands in for the real ``airllm`` package."""

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _StubAirLLMModel:
            return _StubAirLLMModel(model_id, **kwargs)


@pytest.mark.integration
class TestAirLLMBackendEndToEnd:
    def test_full_wiring_via_default_factory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Inject the fake airllm module BEFORE ``_default_factory`` imports it.
        monkeypatch.setitem(sys.modules, "airllm", _StubAirLLMModule)

        # Pin available disk high enough for the 25 GiB pre-flight to pass.
        class _Usage:
            free = 100 * 1024 ** 3
        monkeypatch.setattr(psutil, "disk_usage", lambda p: _Usage())

        backend = AirLLMBackend(
            target_label="llama3-8b-fp16",
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            quantization=Quant.FP16,
            layer_shards_saving_path=tmp_path / "shards",
            hf_token="ignored-in-test",
            min_free_disk_gb=25.0,
            # NOTE: no ``factory=`` override — exercises _default_factory.
        )
        backend.load()
        result = backend.generate(
            "Hello, world.", max_new_tokens=4, params={},
        )
        assert isinstance(result, BackendRunResult)
        assert result.backend is BackendId.AIRLLM
        assert result.target_label == "llama3-8b-fp16"
        assert result.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert result.quantization is Quant.FP16
        assert result.completion_text == "airllm integration ok"
        assert result.completion_tokens == 4
        assert result.prompt_tokens == 6
        # peak_ram_mb + energy_wh stay 0 — the runner fills those in.
        assert result.peak_ram_mb == 0.0
        assert result.energy_wh == 0.0

        # Verify _default_factory reached the stub AirLLM.AutoModel with the
        # right kwargs (device pinned, no compression for fp16, shard path str).
        assert backend._loaded is not None
        assert isinstance(backend._loaded, _StubAirLLMModel)
        assert backend._loaded.init_kwargs["device"] in ("cpu", "cuda:0")
        assert "compression" not in backend._loaded.init_kwargs  # fp16 default
        assert backend._loaded.init_kwargs["hf_token"] == "ignored-in-test"
        assert backend._loaded.init_kwargs["layer_shards_saving_path"] == \
            str(tmp_path / "shards")
        assert backend._n_layers == 35

        backend.unload()
        assert backend._loaded is None
