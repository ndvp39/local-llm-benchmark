"""Tests for ``services/plumbing_default_stages.py`` (T-2a.2).

Behavior of each closure (real HF / AirLLM / psutil work) is exercised by the
integration tests in T-2a.3 and the real-machine run in T-2a.5. Here we only
verify the shape contract: the factory returns the three keys
:class:`PlumbingTestRunner` expects, each value is callable, and missing
``plumbing_test_model`` config keys are surfaced cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from on_prem_llm_lab.services.plumbing_default_stages import build_default_stages

_MIN_CFG = {
    "plumbing_test_model": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "quantization": "fp16",
        "label": "tinyllama-1b-fp16-plumbing",
        "loader": "transformers",
    },
    "airllm": {"layer_shards_saving_path": "D:/airllm_shards"},
    "generation": {"plumbing_max_new_tokens": 8},
}


def test_returns_three_callables_with_expected_keys(tmp_path: Path) -> None:
    """Factory MUST return exactly the three stage keys PlumbingTestRunner consumes."""
    stages = build_default_stages(_MIN_CFG, tmp_path)
    assert set(stages.keys()) == {"download", "mmap_allocation", "metric_collection"}
    for fn in stages.values():
        assert callable(fn)


def test_missing_plumbing_test_model_raises_at_build_time(tmp_path: Path) -> None:
    """If the caller hands us a config without plumbing_test_model, surface it now."""
    with pytest.raises(KeyError, match="plumbing_test_model"):
        build_default_stages({}, tmp_path)


def test_optional_sections_default_to_empty(tmp_path: Path) -> None:
    """Missing ``airllm`` / ``generation`` blocks MUST not crash factory construction."""
    minimal = {"plumbing_test_model": _MIN_CFG["plumbing_test_model"]}
    stages = build_default_stages(minimal, tmp_path)
    assert callable(stages["download"])
    assert callable(stages["mmap_allocation"])
    assert callable(stages["metric_collection"])
