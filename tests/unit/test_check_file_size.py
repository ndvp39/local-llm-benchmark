"""Tests for ``tools/check_file_size.py`` (T-1.6 DoD).

Verifies (a) the LOC counter ignores blank + pure-comment lines per
constitution §2.2; (b) the script returns non-zero on a fixture file
that exceeds the limit; (c) the script returns zero on a clean tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_check_script() -> ModuleType:
    """Load ``tools/check_file_size.py`` without making ``tools/`` a package."""
    repo = Path(__file__).resolve().parents[2]
    path = repo / "tools" / "check_file_size.py"
    spec = importlib.util.spec_from_file_location("check_file_size", path)
    assert spec is not None and spec.loader is not None, f"no loader for {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_file_size"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_count_loc_skips_blank_and_comment_only_lines() -> None:
    mod = _load_check_script()
    src = '"""docstring"""\n\n# a comment\n   \nx = 1\n   # indented comment\ny = 2\n'
    assert mod.count_loc(src) == 3  # docstring + x=1 + y=2


def test_scan_flags_oversized_fixture(tmp_path: Path) -> None:
    """DoD: script returns non-zero on a fixture file > 150 LOC."""
    mod = _load_check_script()
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "big.py").write_text("x = 1\n" * 200)  # 200 LOC, all counted
    rc = mod.main(["--root", str(tmp_path), "--quiet"])
    assert rc == 1


def test_scan_passes_on_clean_tree(tmp_path: Path) -> None:
    mod = _load_check_script()
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "small.py").write_text("x = 1\n" * 50)  # 50 LOC
    rc = mod.main(["--root", str(tmp_path), "--quiet"])
    assert rc == 0
