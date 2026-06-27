"""Project code version (constitution §7.1 — initial value 1.00).

This is the single source of truth for the runtime version. Config files
(``config/*.json``) each carry their own ``"version"`` field which the
config loader (T-3.8) compares against this value to detect schema drift.
"""

from __future__ import annotations

__version__ = "1.00"

__all__ = ["__version__"]
