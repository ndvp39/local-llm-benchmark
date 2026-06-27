"""Smoke test for T-1.5 — verifies the package skeleton is importable end-to-end.

This is the minimum signal we need to know that pytest + pytest-cov are wired
correctly against the freshly synced venv. It also walk-imports every submodule
under :mod:`on_prem_llm_lab`, which keeps coverage near 100 % during M1 (when
every module is still a docstring-only stub). It will be progressively replaced
by per-Building-Block unit tests as M2+ tasks land.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

import on_prem_llm_lab


def _walk_submodules(package: ModuleType) -> list[str]:
    """Recursively list every submodule fully-qualified name under ``package``."""
    names: list[str] = []
    for info in pkgutil.iter_modules(package.__path__, prefix=package.__name__ + "."):
        names.append(info.name)
        if info.ispkg:
            names.extend(_walk_submodules(importlib.import_module(info.name)))
    return names


def test_top_level_package_imports() -> None:
    """The root package MUST be importable under ``uv run``."""
    assert on_prem_llm_lab.__name__ == "on_prem_llm_lab"


def test_every_submodule_imports_cleanly() -> None:
    """Walk-import every module to validate the skeleton wires up cleanly."""
    modules = _walk_submodules(on_prem_llm_lab)
    # PLAN §8 enumerates 6 subpackages + ~26 leaf modules; expect at least 25.
    assert len(modules) >= 25, f"unexpected module count: {len(modules)}"
    for name in modules:
        importlib.import_module(name)  # raises if any stub is broken
