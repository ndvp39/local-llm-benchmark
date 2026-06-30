"""Unit tests for mixins/manifest_logging_mixin.py (T-2.4)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from typing import Any

import pytest

from on_prem_llm_lab.mixins.manifest_logging_mixin import (
    ManifestLoggingMixin,
    ManifestRecord,
    resolve_git_hash,
)


class _Mixed(ManifestLoggingMixin):
    """Minimal concrete subclass."""


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "20260630T200000Z",
        "target_label": "llama3-8b-fp16",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "backend": "airllm",
        "quantization": "fp16",
        "seed": 42,
        "prompt": "Hello, world.",
        "max_new_tokens": 128,
        "config_snapshot": {"version": "1.00", "sampling": {"memory_hz": 5}},
        "git_hash": "deadbeef" * 5,  # 40-char synthetic hash
        "started_at": "2026-06-30T20:00:00Z",
    }
    base.update(overrides)
    return base


class TestBuildManifest:
    def test_happy_path_populates_every_required_field(self) -> None:
        m = _Mixed().build_manifest(**_kwargs())
        assert m.run_id == "20260630T200000Z"
        assert m.target_label == "llama3-8b-fp16"
        assert m.model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
        assert m.backend == "airllm"
        assert m.quantization == "fp16"
        assert m.seed == 42
        assert m.prompt == "Hello, world."
        assert m.max_new_tokens == 128
        assert m.config_snapshot == {"version": "1.00", "sampling": {"memory_hz": 5}}
        assert m.git_hash == "deadbeef" * 5
        assert m.started_at == "2026-06-30T20:00:00Z"
        assert m.run_result is None
        assert m.python  # populated from sys.version
        assert m.package_version  # populated from shared.version.__version__

    def test_no_git_hash_records_none(self) -> None:
        m = _Mixed().build_manifest(**_kwargs(git_hash=None))
        assert m.git_hash is None

    def test_started_at_default_uses_iso_utc_now(self) -> None:
        """No ``started_at`` -> auto-fills with ISO UTC `YYYY-MM-DDTHH:MM:SSZ`."""
        kw = _kwargs()
        del kw["started_at"]
        m = _Mixed().build_manifest(**kw)
        assert m.started_at.endswith("Z")
        assert len(m.started_at) == 20  # YYYY-MM-DDTHH:MM:SSZ

    def test_run_result_attached_when_provided(self) -> None:
        result = {"ttft_ms": 367_282.43, "tpot_ms": 368_350.11, "n_tokens": 2}
        m = _Mixed().build_manifest(**_kwargs(run_result=result))
        assert m.run_result == result

    def test_config_snapshot_is_defensively_copied(self) -> None:
        """Mutating the caller's dict after build MUST NOT affect the record."""
        cfg: dict[str, Any] = {"a": 1}
        m = _Mixed().build_manifest(**_kwargs(config_snapshot=cfg))
        cfg["a"] = 999
        assert m.config_snapshot == {"a": 1}


class TestWriteManifest:
    def test_writes_json_at_expected_path(self, tmp_path: Path) -> None:
        m = _Mixed().build_manifest(**_kwargs())
        results = tmp_path / "results"
        path = _Mixed().write_manifest(m, results)
        assert path == results / "run_20260630T200000Z.json"
        assert path.exists()

    def test_creates_results_dir_if_missing(self, tmp_path: Path) -> None:
        m = _Mixed().build_manifest(**_kwargs())
        results = tmp_path / "fresh" / "results"
        assert not results.exists()
        path = _Mixed().write_manifest(m, results)
        assert path.exists()
        assert results.is_dir()

    def test_written_json_round_trips(self, tmp_path: Path) -> None:
        m = _Mixed().build_manifest(**_kwargs())
        path = _Mixed().write_manifest(m, tmp_path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == asdict(m)


class TestResolveGitHash:
    def test_returns_string_in_real_repo(self) -> None:
        """Running inside the project repo, this MUST resolve a real hash."""
        repo_root = Path(__file__).resolve().parents[3]
        sha = resolve_git_hash(repo_root)
        assert sha is not None
        assert len(sha) == 40  # full SHA-1
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_none_outside_repo(self, tmp_path: Path) -> None:
        """An empty tmp_path is not a git repo -> None."""
        assert resolve_git_hash(tmp_path) is None


class TestManifestRecordFrozen:
    def test_is_frozen_and_correct_type(self) -> None:
        m = _Mixed().build_manifest(**_kwargs())
        assert isinstance(m, ManifestRecord)
        with pytest.raises(FrozenInstanceError):
            m.run_id = "tampered"  # type: ignore[misc]
