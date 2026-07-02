"""Tests for `services/quality_matrix.py` (T-3.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from on_prem_llm_lab.services.quality_matrix import (
    QualityCell,
    collect_cells,
    format_markdown,
    write_quality_matrix,
)


def _manifest(tmp: Path, *, run_id: str, target: str, quant: str,
              text: str = "hi world", ttft: float = 100.0,
              tpot: float = 200.0, ram: float = 500.0) -> Path:
    p = tmp / f"run_{run_id}.json"
    p.write_text(json.dumps({
        "run_id": run_id, "target_label": target, "quantization": quant,
        "backend": "airllm", "prompt": "Hello, world.", "seed": 42,
        "max_new_tokens": 8,
        "run_result": {
            "completion_text": text, "ttft_ms": ttft, "tpot_ms": tpot,
            "peak_ram_mb": ram,
        },
    }), encoding="utf-8")
    return p


class TestCollectCells:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert collect_cells(tmp_path) == []

    def test_dedup_by_target_quant_backend(self, tmp_path: Path) -> None:
        _manifest(tmp_path, run_id="a", target="llama", quant="fp16", text="first")
        _manifest(tmp_path, run_id="b", target="llama", quant="fp16", text="second")
        cells = collect_cells(tmp_path)
        assert len(cells) == 1
        assert cells[0].completion_text == "first"

    def test_groups_across_cells(self, tmp_path: Path) -> None:
        _manifest(tmp_path, run_id="a", target="llama", quant="fp16")
        _manifest(tmp_path, run_id="b", target="llama", quant="q4")
        _manifest(tmp_path, run_id="c", target="qwen", quant="fp16")
        assert len({(c.target_label, c.quantization) for c in collect_cells(tmp_path)}) == 3

    def test_skips_empty_completion(self, tmp_path: Path) -> None:
        _manifest(tmp_path, run_id="empty", target="llama", quant="fp16", text="")
        assert collect_cells(tmp_path) == []


class TestFormatMarkdown:
    def test_header_and_prompt(self) -> None:
        md = format_markdown([], "Hi.")
        assert "Quality Matrix" in md
        assert "Hi." in md
        assert "TTFT (ms)" in md

    def test_row_per_cell(self) -> None:
        cells = [
            QualityCell(target_label="llama", quantization="fp16", backend="airllm",
                         prompt="p", completion_text="alpha", ttft_ms=100, tpot_ms=200,
                         peak_ram_mb=500, seed=42),
            QualityCell(target_label="llama", quantization="q4", backend="airllm",
                         prompt="p", completion_text="beta", ttft_ms=80, tpot_ms=150,
                         peak_ram_mb=300, seed=42),
        ]
        md = format_markdown(cells, "p")
        assert "alpha" in md
        assert "beta" in md
        assert md.count("| `llama`") == 2

    def test_completion_truncated_at_300(self) -> None:
        long = "x" * 500
        cells = [QualityCell(
            target_label="llama", quantization="fp16", backend="airllm",
            prompt="p", completion_text=long, ttft_ms=0, tpot_ms=0,
            peak_ram_mb=0, seed=0,
        )]
        md = format_markdown(cells, "p")
        assert "..." in md
        assert md.count("x") < 500

    def test_pipe_escaping(self) -> None:
        cells = [QualityCell(
            target_label="t", quantization="fp16", backend="airllm",
            prompt="p", completion_text="a|b", ttft_ms=0, tpot_ms=0,
            peak_ram_mb=0, seed=0,
        )]
        md = format_markdown(cells, "p")
        assert "a\\|b" in md


class TestWriteQualityMatrix:
    def test_default_output_path(self, tmp_path: Path) -> None:
        _manifest(tmp_path, run_id="a", target="llama", quant="fp16")
        out = write_quality_matrix(tmp_path)
        assert out == tmp_path / "quality_matrix.md"
        assert out.exists()
        assert "llama" in out.read_text(encoding="utf-8")

    def test_explicit_output_path(self, tmp_path: Path) -> None:
        _manifest(tmp_path, run_id="a", target="llama", quant="fp16")
        out = write_quality_matrix(tmp_path, tmp_path / "custom.md")
        assert out.name == "custom.md"
        assert out.exists()


@pytest.fixture
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("random.seed", lambda x: None)
