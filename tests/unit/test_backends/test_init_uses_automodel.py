"""ADR-009 enforcement: backends MUST use ``AutoModel*`` factories only.

Walks every ``.py`` file under ``src/on_prem_llm_lab/backends/`` with the
``ast`` module and flags any direct import of a concrete causal-LM class
from ``transformers`` (e.g. ``LlamaForCausalLM``, ``Qwen2ForCausalLM``).
The rule exists because hard-coding concrete classes defeats the
``AutoModel*`` indirection: a future quantization swap would silently
fall back to the wrong implementation. The single sanctioned path is
``shared.automodel_factory.load_causal_lm()`` (T-1.15).

DoD (T-2.0): passes on real backend code, fails on a fixture that
imports ``LlamaForCausalLM``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKENDS_DIR = _REPO_ROOT / "src" / "on_prem_llm_lab" / "backends"
_FIXTURE_DIR = Path(__file__).parent / "fixtures"

_FORBIDDEN_SUFFIX = "ForCausalLM"
_TRANSFORMERS_HEADS = ("transformers",)


def _is_transformers_module(module: str | None) -> bool:
    if not module:
        return False
    return module.split(".", 1)[0] in _TRANSFORMERS_HEADS


def _find_concrete_causal_lm_imports(file_path: Path) -> list[str]:
    """Return concrete ``*ForCausalLM`` names imported from ``transformers``.

    ``AutoModelForCausalLM`` (and any other ``Auto*ForCausalLM``) is
    explicitly allowed — that's the sanctioned ADR-009 entry point.
    """
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not _is_transformers_module(node.module):
            continue
        for alias in node.names:
            name = alias.name
            if name.endswith(_FORBIDDEN_SUFFIX) and not name.startswith("Auto"):
                hits.append(name)
    return hits


def test_no_concrete_causal_lm_in_backends() -> None:
    """Real backend code under ``src/`` satisfies ADR-009."""
    violations: dict[str, list[str]] = {}
    for py in sorted(_BACKENDS_DIR.rglob("*.py")):
        hits = _find_concrete_causal_lm_imports(py)
        if hits:
            violations[str(py.relative_to(_REPO_ROOT))] = hits
    assert not violations, (
        "ADR-009 violation: concrete *ForCausalLM imports found in backends/.\n"
        f"{violations}\nUse shared/automodel_factory.load_causal_lm() instead."
    )


def test_fixture_with_concrete_import_is_flagged() -> None:
    """Synthetic fixture importing ``LlamaForCausalLM`` trips the guard."""
    fixture = _FIXTURE_DIR / "_bad_concrete_import.py"
    hits = _find_concrete_causal_lm_imports(fixture)
    assert "LlamaForCausalLM" in hits, (
        f"AST guard failed to detect the deliberate violation in {fixture}; "
        f"hits={hits!r}"
    )


def test_automodel_for_causal_lm_is_allowed() -> None:
    """``AutoModelForCausalLM`` is the sanctioned import — never flagged."""
    sample = "from transformers import AutoModelForCausalLM\n"
    tmp = _FIXTURE_DIR / "_auto_ok_inline.py"
    tmp.write_text(sample, encoding="utf-8")
    try:
        hits = _find_concrete_causal_lm_imports(tmp)
        assert hits == []
    finally:
        tmp.unlink()


def test_submodule_import_path_is_flagged() -> None:
    """``from transformers.models.llama... import LlamaForCausalLM`` is also caught."""
    sample = (
        "from transformers.models.llama.modeling_llama import LlamaForCausalLM"
        "  # noqa: F401\n"
    )
    tmp = _FIXTURE_DIR / "_submodule_bad_inline.py"
    tmp.write_text(sample, encoding="utf-8")
    try:
        hits = _find_concrete_causal_lm_imports(tmp)
        assert hits == ["LlamaForCausalLM"]
    finally:
        tmp.unlink()
