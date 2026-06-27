"""Tests for ``shared/plumbing_guard.py`` (T-2a.4 · ADR-010)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from on_prem_llm_lab.shared.plumbing_guard import (
    PlumbingNotRunError,
    require_current_plumbing,
)


def _write_manifest(results_dir: Path, captured_at_stamp: str, overall: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"plumbing_{captured_at_stamp}.json"
    path.write_text(
        json.dumps({
            "captured_at": captured_at_stamp,
            "overall": overall,
            "stages": {},
        }),
        encoding="utf-8",
    )
    return path


def test_raises_when_no_manifest_exists(tmp_path: Path) -> None:
    """Empty results/ MUST raise PlumbingNotRunError with the remediation hint."""
    with pytest.raises(PlumbingNotRunError, match="No plumbing manifest"):
        require_current_plumbing(tmp_path)


def test_returns_latest_ok_manifest_path(tmp_path: Path) -> None:
    """Lex-sortable timestamps → the latest entry MUST win."""
    _write_manifest(tmp_path, "20260101T000000Z", "ok")
    latest = _write_manifest(tmp_path, "20260627T120000Z", "ok")
    _write_manifest(tmp_path, "20260301T120000Z", "ok")
    assert require_current_plumbing(tmp_path) == latest


def test_raises_when_latest_manifest_is_fail(tmp_path: Path) -> None:
    """An older successful manifest doesn't excuse a failed latest one."""
    _write_manifest(tmp_path, "20260101T000000Z", "ok")
    _write_manifest(tmp_path, "20260627T120000Z", "fail")
    with pytest.raises(PlumbingNotRunError, match="overall='fail'"):
        require_current_plumbing(tmp_path)


def test_raises_when_latest_manifest_has_no_overall_field(tmp_path: Path) -> None:
    """``overall`` missing is treated the same as a failure (defensive)."""
    path = tmp_path / "plumbing_20260627T120000Z.json"
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PlumbingNotRunError, match="overall=None"):
        require_current_plumbing(tmp_path)


def test_raises_when_latest_manifest_is_corrupt_json(tmp_path: Path) -> None:
    """Unreadable manifest → PlumbingNotRunError chained from JSONDecodeError."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "plumbing_20260627T120000Z.json").write_text(
        "not json", encoding="utf-8",
    )
    with pytest.raises(PlumbingNotRunError, match="Failed to read"):
        require_current_plumbing(tmp_path)
