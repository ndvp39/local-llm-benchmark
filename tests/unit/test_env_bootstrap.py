"""Regression guard for T-3.5c — HF cache MUST land on the D: path from ``.env``.

Two-part bug prevented here:

1. ``load_dotenv()`` must fire BEFORE ``transformers`` is transitively imported
   (via ``shared.automodel_factory``), since ``transformers`` reads
   ``HF_HOME`` / ``TRANSFORMERS_CACHE`` at import time and caches the resolved
   path. Deferring to ``cli/main.py`` was too late — the SDK import in
   ``on_prem_llm_lab/__init__.py`` already pulled ``transformers`` in.

2. ``load_dotenv()`` defaults to ``override=False`` and silently skips when
   the parent shell passes ``HF_HOME=""`` (empty). We MUST pass
   ``override=True`` so ``.env`` wins over an empty inherited value.

Combined breakage: HF downloads land in ``C:\\Users\\<u>\\.cache\\huggingface``
even though ``.env`` says D:, triggering the ENOSPC disk-full incident
(prompts_book §11.9). This test asserts both invariants at the source-text
level so a future refactor can't silently reintroduce either.
"""

from __future__ import annotations

import ast
from pathlib import Path

_INIT = Path("src/on_prem_llm_lab/__init__.py")


def _load_dotenv_call() -> ast.Call:
    """Return the ``load_dotenv(...)`` Call node in the package ``__init__``."""
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in {"load_dotenv", "_load_dotenv"}:
                return node
    raise AssertionError("on_prem_llm_lab/__init__.py must call load_dotenv()")


def test_package_init_calls_load_dotenv_with_override_true() -> None:
    """``on_prem_llm_lab/__init__.py`` MUST call ``load_dotenv(override=True)``."""
    call = _load_dotenv_call()
    override = next(
        (kw for kw in call.keywords if kw.arg == "override"), None,
    )
    assert override is not None, (
        "load_dotenv() must pass override=True — otherwise an empty "
        "HF_HOME='' inherited from the parent shell silently keeps HF "
        "downloads on the default C: cache (see T-3.5c)."
    )
    assert isinstance(override.value, ast.Constant) and override.value.value is True


def test_load_dotenv_fires_before_child_imports() -> None:
    """``load_dotenv`` MUST appear BEFORE any ``from on_prem_llm_lab...`` import.

    Rationale: ``on_prem_llm_lab.sdk`` transitively imports
    ``shared.automodel_factory``, which does ``from transformers import ...``
    at module top. ``transformers`` reads HF cache env at import time; if
    ``.env`` hasn't been applied yet, downloads silently fall back to C:.
    """
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    load_dotenv_line: int | None = None
    first_child_import_line: int | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) in {"load_dotenv", "_load_dotenv"}
            and load_dotenv_line is None
        ):
            load_dotenv_line = node.lineno
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("on_prem_llm_lab")
            and first_child_import_line is None
        ):
            first_child_import_line = node.lineno
    assert load_dotenv_line is not None, "load_dotenv() call not found"
    assert first_child_import_line is not None, "no child imports found"
    assert load_dotenv_line < first_child_import_line, (
        f"load_dotenv() at line {load_dotenv_line} must fire BEFORE the "
        f"first `from on_prem_llm_lab...` import at line "
        f"{first_child_import_line} — otherwise transformers reads HF_HOME "
        f"stale (see T-3.5c)."
    )


def test_hf_cache_resolves_to_env_value() -> None:
    """After importing the package, ``HF_HUB_CACHE`` MUST resolve to ``.env``'s HF_HOME."""
    from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415

    import on_prem_llm_lab  # noqa: F401 — side effect: load_dotenv fires

    env_path = Path(".env")
    if not env_path.exists():
        return  # no .env in this checkout — skip silently
    hf_home = None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("HF_HOME="):
            hf_home = line.split("=", 1)[1].strip()
            break
    if not hf_home:
        return  # .env doesn't set HF_HOME — skip
    assert HF_HUB_CACHE.replace("\\", "/").startswith(hf_home.rstrip("/")), (
        f"HF_HUB_CACHE={HF_HUB_CACHE!r} does not start with .env HF_HOME={hf_home!r} "
        f"— downloads will silently fall back to C: (see T-3.5c)."
    )
