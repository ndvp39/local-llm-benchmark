"""Unit tests for services/sweep_runner.py (T-3.5 / DP-4 §9.2 U-BM-7..15)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from on_prem_llm_lab.backends.base import BackendId, BackendRunResult, Quant
from on_prem_llm_lab.services.sweep_runner import SweepRunner


def _make_result(**overrides: Any) -> BackendRunResult:
    base: dict[str, Any] = {
        "run_id": "r", "started_at": "2026-07-01T00:00:00Z",
        "backend": BackendId.AIRLLM, "target_label": "t",
        "model_id": "m", "quantization": Quant.FP16,
        "prompt_tokens": 8, "completion_tokens": 128,
        "ttft_ms": 500.0, "tpot_ms": 200.0, "throughput_tps": 1.0,
        "peak_ram_mb": 100.0, "wall_s": 1.0, "energy_wh": 0.05,
        "completion_text": "ok",
    }
    base.update(overrides)
    return BackendRunResult(**base)


def _make_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "target_models": [
            {"id": "meta-llama/M", "label": "target-a", "quantization": "fp16"},
        ],
        "generation": {
            "max_new_tokens": 128, "seed": 42, "baseline_prompt": "Hello.",
        },
        "sampling": {"repeat": 3, "warmup_repeats": 1},
    }
    base.update(overrides)
    return base


class _Rec:
    """Callable cell_runner: records calls, replays results, raises on demand."""

    def __init__(
        self, results: list[BackendRunResult] | None = None,
        raises_on: list[int] | None = None,
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._results = results or []
        self._raises = set(raises_on or [])
        self._n = 0

    def __call__(self, *args: Any) -> BackendRunResult:
        idx = self._n
        self._n += 1
        self.calls.append(args)
        if idx in self._raises:
            raise RuntimeError(f"boom-{idx}")
        return self._results[idx] if idx < len(self._results) else _make_result()


def _first_supported_row(csv_text: str) -> tuple[list[str], list[str]]:
    """Return (columns, first row whose skip_reason is empty)."""
    header, *rows = csv_text.splitlines()
    cols = header.split(",")
    ir = cols.index("skip_reason")
    row = next(r.split(",") for r in rows if r.split(",")[ir] == "")
    return cols, row


def _first_with_skip(csv_text: str, needle: str) -> tuple[list[str], list[str]]:
    header, *rows = csv_text.splitlines()
    cols = header.split(",")
    ir = cols.index("skip_reason")
    row = next(r.split(",") for r in rows if needle in r.split(",")[ir])
    return cols, row


def _run(tmp_path: Path, cfg: dict[str, Any] | None = None,
         rec: _Rec | None = None) -> Path:
    rec = rec or _Rec()
    return SweepRunner(
        cfg or _make_config(), tmp_path, cell_runner=rec, clock=lambda: "T",
    ).run(["airllm"])


class TestSweepRunner:
    def test_iterates_full_matrix(self, tmp_path: Path) -> None:
        # U-BM-7: 1 target × 4 quantizations × 1 backend = 4 rows.
        csv_path = _run(tmp_path)
        assert csv_path == tmp_path / "sweep_T.csv"
        assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 5

    def test_skips_unsupported_cells(self, tmp_path: Path) -> None:
        # U-BM-8: only fp16/q4/q8 supported for airllm -> 3 × (1+3) = 12 calls.
        rec = _Rec()
        _run(tmp_path, rec=rec)
        assert len(rec.calls) == 12

    def test_warmup_excluded_from_stats(self, tmp_path: Path) -> None:
        # U-BM-9: warmup=1 at TTFT=1000; N=3 measurements at 500 -> mean=500.
        results = (
            [_make_result(ttft_ms=1000.0)]
            + [_make_result(ttft_ms=500.0)] * 3
        ) * 3
        rec = _Rec(results=results)
        csv_path = _run(tmp_path, rec=rec)
        cols, row = _first_supported_row(csv_path.read_text(encoding="utf-8"))
        assert float(row[cols.index("ttft_ms_mean")]) == 500.0

    def test_failed_measurement_bumps_n_failed(self, tmp_path: Path) -> None:
        # U-BM-10: raise on measurement idx 1 -> n_failed=1.
        rec = _Rec(raises_on=[1])
        csv_path = _run(tmp_path, rec=rec)
        cols, row = _first_supported_row(csv_path.read_text(encoding="utf-8"))
        assert row[cols.index("n_failed")] == "1"

    def test_zero_success_produces_nan_row(self, tmp_path: Path) -> None:
        # U-BM-11: warmup ok (idx 0), all 3 measurements raise -> NaN row.
        rec = _Rec(raises_on=[1, 2, 3])
        csv_path = _run(tmp_path, rec=rec)
        cols, row = _first_supported_row(csv_path.read_text(encoding="utf-8"))
        assert row[cols.index("n_success")] == "0"
        assert row[cols.index("ttft_ms_mean")] == "nan"

    def test_method_note_on_fast_decoder(self, tmp_path: Path) -> None:
        # U-BM-12: TTFT=100, TPOT=50, N=8 -> visible_second_prefill=True.
        cfg = _make_config(
            generation={"max_new_tokens": 8, "seed": 42, "baseline_prompt": "H"},
            sampling={"repeat": 3, "warmup_repeats": 1},
        )
        rec = _Rec(results=[_make_result(ttft_ms=100.0, tpot_ms=50.0)] * 12)
        csv_path = _run(tmp_path, cfg=cfg, rec=rec)
        cols, row = _first_supported_row(csv_path.read_text(encoding="utf-8"))
        assert "second-prefill" in row[cols.index("method_note")]

    def test_manifest_written_alongside_csv(self, tmp_path: Path) -> None:
        # U-BM-14
        _run(tmp_path)
        assert (tmp_path / "sweep_T.json").exists()

    def test_warmup_failure_aborts_cell_with_skip_reason(
        self, tmp_path: Path,
    ) -> None:
        # C-BM-5 — warmup raises, cell aborts, skip_reason set.
        rec = _Rec(raises_on=[0])
        csv_path = _run(tmp_path, rec=rec)
        cols, row = _first_with_skip(
            csv_path.read_text(encoding="utf-8"), "warmup_failed",
        )
        assert row[cols.index("n_success")] == "0"

    def test_falls_back_to_baseline_prompt(self, tmp_path: Path) -> None:
        # Q3 answer: sweep_prompt absent -> use baseline_prompt.
        cfg = _make_config(
            generation={"max_new_tokens": 4, "seed": 42,
                        "baseline_prompt": "fallback"},
        )
        rec = _Rec()
        _run(tmp_path, cfg=cfg, rec=rec)
        # every cell call's prompt (index 5) should be "fallback"
        assert all(call[5] == "fallback" for call in rec.calls)
