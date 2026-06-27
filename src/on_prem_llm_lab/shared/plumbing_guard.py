"""Plumbing-first precondition for ``OnPremLlmSDK.run_sweep`` (T-2a.4 · ADR-010).

The constitution rule (PRD FR-PT-2): an oversized sweep MUST NOT start until a
successful plumbing manifest exists under ``results/``. Same pattern as
:mod:`shared.env_guard` — the exception class + the boolean check live here, and
the SDK calls :func:`require_current_plumbing` as a one-liner.

``skip_plumbing=True`` at the SDK is for tests / dry-runs only; production
callers MUST run the plumbing test first.
"""

from __future__ import annotations

import json
from pathlib import Path


class PlumbingNotRunError(RuntimeError):
    """Raised by ``run_sweep`` when no successful plumbing manifest is present."""


REMEDIATION = (
    "Run `uv run on-prem-llm run-plumbing-test` first, or pass skip_plumbing=True."
)


def require_current_plumbing(results_dir: Path) -> Path:
    """Return the path of the latest successful plumbing manifest, or raise.

    Manifest filenames embed the captured-at timestamp with all separators
    stripped (see ``PlumbingTestRunner.run``), so lexicographic sort of the
    glob = chronological order. We take the last entry as "latest".
    """
    manifests = sorted(results_dir.glob("plumbing_*.json"))
    if not manifests:
        raise PlumbingNotRunError(
            f"No plumbing manifest under {results_dir}. {REMEDIATION}"
        )
    latest = manifests[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlumbingNotRunError(
            f"Failed to read plumbing manifest {latest.name}: {exc}. {REMEDIATION}"
        ) from exc
    overall = payload.get("overall")
    if overall != "ok":
        raise PlumbingNotRunError(
            f"Latest plumbing manifest {latest.name} has overall={overall!r}. "
            f"{REMEDIATION}"
        )
    return latest


__all__ = ["REMEDIATION", "PlumbingNotRunError", "require_current_plumbing"]
