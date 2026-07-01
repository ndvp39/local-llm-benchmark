"""Unit tests for services/sweep_stats.py (T-3.5 / DP-4 §9.1 U-BM-1..6)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from on_prem_llm_lab.services.sweep_stats import (
    METRICS,
    NAN_STATS,
    MetricStats,
    SweepRow,
    aggregate,
    csv_columns,
    has_visible_second_prefill,
    row_to_csv_dict,
    write_csv,
    write_manifest,
)


class TestAggregate:
    def test_empty_returns_nan_stats(self) -> None:
        # U-BM-1 boundary
        result = aggregate([])
        assert result == NAN_STATS
        assert math.isnan(result.mean)

    def test_single_value(self) -> None:
        # U-BM-1
        s = aggregate([1.0])
        assert s.mean == 1.0
        assert s.median == 1.0
        assert s.std == 0.0
        assert s.min == 1.0
        assert s.max == 1.0
        assert s.p95 == 1.0

    def test_five_values(self) -> None:
        # U-BM-2
        s = aggregate([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s.mean == 3.0
        assert s.median == 3.0
        assert s.std == pytest.approx(1.5811, rel=1e-3)
        assert s.min == 1.0
        assert s.max == 5.0
        # < 20 values -> p95 collapses to max
        assert s.p95 == 5.0

    def test_twenty_values_uses_real_p95(self) -> None:
        # U-BM-3
        s = aggregate(list(range(1, 21)))  # 1..20
        assert s.min == 1.0
        assert s.max == 20.0
        # statistics.quantiles(n=20)[18] is the 19th quantile boundary
        assert s.p95 < s.max


class TestHasVisibleSecondPrefill:
    def test_fast_decoder_case_true(self) -> None:
        # U-BM-4: TTFT=100, TPOT=50, N=8 -> 100/7 ≈ 14.3 > 0.1*50=5 -> True
        assert has_visible_second_prefill(100.0, 50.0, 8) is True

    def test_airllm_128_token_case_false(self) -> None:
        # U-BM-5: real AirLLM run — TTFT=367 282, TPOT=368 350 ms, N=128.
        # ttft/(N-1) = 367 282/127 ≈ 2 892; 0.1*tpot = 36 835. 2 892 < 36 835 -> False.
        assert has_visible_second_prefill(367_282.0, 368_350.0, 128) is False

    def test_max_new_tokens_less_than_two_short_circuits_false(self) -> None:
        # U-BM-6a: N<2 short-circuit
        assert has_visible_second_prefill(1000.0, 100.0, 1) is False

    def test_zero_tpot_edge_returns_false(self) -> None:
        # tpot_ms == 0 short-circuits to False
        assert has_visible_second_prefill(100.0, 0.0, 8) is False


class TestSweepRowFrozen:
    def test_row_is_frozen(self) -> None:
        row = _sample_row()
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            row.target_label = "other"  # type: ignore[misc]


class TestCsvSerialization:
    def test_columns_include_all_metrics_and_stats(self) -> None:
        cols = csv_columns()
        assert "target_label" in cols
        # 6 metrics × 6 stat fields = 36 metric columns
        for m in METRICS:
            for f in ("mean", "median", "std", "min", "max", "p95"):
                assert f"{m}_{f}" in cols

    def test_row_to_csv_dict_flattens_nested_stats(self) -> None:
        row = _sample_row(stats_ttft_mean=100.0)
        d = row_to_csv_dict(row)
        assert d["ttft_ms_mean"] == 100.0
        assert "stats" not in d

    def test_write_csv_roundtrip(self, tmp_path: Path) -> None:
        # U-BM-13 / SC-BM-7
        rows = [_sample_row(target_label="t1"), _sample_row(target_label="t2")]
        path = tmp_path / "sweep.csv"
        write_csv(path, rows)
        with path.open(encoding="utf-8") as f:
            read_rows = list(csv.DictReader(f))
        assert len(read_rows) == 2
        assert read_rows[0]["target_label"] == "t1"
        assert read_rows[1]["target_label"] == "t2"


class TestManifest:
    def test_write_manifest_counts_supported_and_skipped(
        self, tmp_path: Path,
    ) -> None:
        # SC-BM-8
        rows = [
            _sample_row(skip_reason=None),
            _sample_row(skip_reason="unsupported_quantization"),
        ]
        path = tmp_path / "sweep.json"
        write_manifest(path, {"version": "1.00"}, rows, "20260101T000000Z")
        m = json.loads(path.read_text(encoding="utf-8"))
        assert m["captured_at"] == "20260101T000000Z"
        assert m["n_cells"] == 2
        assert m["n_supported"] == 1
        assert m["n_skipped"] == 1
        assert m["config_snapshot"] == {"version": "1.00"}


def _sample_row(**overrides: object) -> SweepRow:
    stats_ttft_mean = overrides.pop("stats_ttft_mean", None)
    ttft = (
        MetricStats(mean=float(stats_ttft_mean), median=0.0, std=0.0,
                    min=0.0, max=0.0, p95=0.0)
        if stats_ttft_mean is not None else NAN_STATS
    )
    kwargs = {
        "target_label": "t", "backend": "direct", "quantization": "fp16",
        "seed": 42, "prompt_tokens": 8, "max_new_tokens": 128,
        "completion_tokens": 128, "repeat": 5, "warmup_repeats": 1,
        "n_success": 5, "n_failed": 0, "skip_reason": None,
        "method_note": "", "stats": dict.fromkeys(METRICS, NAN_STATS) | {"ttft_ms": ttft},
    }
    kwargs.update(overrides)
    return SweepRow(**kwargs)  # type: ignore[arg-type]
