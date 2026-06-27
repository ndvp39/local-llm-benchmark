"""Tests for ``shared/version.py`` (T-1.11)."""

from __future__ import annotations

import re

import on_prem_llm_lab
from on_prem_llm_lab.shared import version as version_mod


def test_version_value_is_initial_per_constitution() -> None:
    """Constitution §7.1 — initial version MUST be ``1.00``."""
    assert version_mod.__version__ == "1.00"


def test_version_shape_is_two_decimals() -> None:
    """Convention: ``MAJOR.MINOR`` with a 2-digit minor (e.g. ``1.00``, ``1.10``)."""
    assert re.fullmatch(r"\d+\.\d{2}", version_mod.__version__)


def test_root_package_reexports_version() -> None:
    """``on_prem_llm_lab.__version__`` MUST mirror ``shared.version.__version__``."""
    assert on_prem_llm_lab.__version__ == version_mod.__version__
