"""Reproducibility-manifest mixin (T-2.4).

Builds and writes a JSON manifest capturing everything a reviewer needs
to reproduce a run: the seed, the prompt + max_new_tokens, the target
label + model id, the backend identifier, the quantization label, the
git hash of the source tree, the on_prem_llm_lab version, and a frozen
snapshot of the runtime config. Optional ``run_result`` carries the
post-run :class:`BackendRunResult` (as a plain dict to avoid coupling
this mixin to ``backends/base.py``).

Manifest path convention (PRD FR-10): ``results/run_<run_id>.json``.
The caller controls ``run_id`` — pass a timestamp ID like
``"20260630T200000Z"`` for the canonical filename shape.

Production-default helpers (``_utc_now_iso``, ``resolve_git_hash``) are
module-level so tests can inject deterministic values via the mixin's
public API rather than monkeypatching.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from on_prem_llm_lab.shared.version import __version__


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_git_hash(repo_root: Path) -> str | None:
    """Return the full 40-char HEAD commit hash, or ``None`` if unavailable.

    Failure modes that all collapse to ``None``: ``git`` not on PATH,
    ``repo_root`` not a git repo, ``git`` returns non-zero. The mixin
    treats missing git as a soft warning recorded as ``None`` — not a
    hard failure — because the project may be run from a tarball clone
    by a reviewer.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


@dataclass(frozen=True, kw_only=True)
class ManifestRecord:
    """Frozen reproducibility manifest (PRD FR-10).

    ``run_result`` is left as an opaque dict to avoid an import-cycle
    risk with ``backends/base.py``; callers pass
    ``dataclasses.asdict(backend_run_result)`` if they want to attach.
    """

    run_id: str
    started_at: str
    target_label: str
    model_id: str
    backend: str
    quantization: str
    seed: int
    prompt: str
    max_new_tokens: int
    config_snapshot: dict[str, Any]
    git_hash: str | None
    python: str
    package_version: str
    run_result: dict[str, Any] | None = field(default=None)


class ManifestLoggingMixin:
    """Mixin: build + write a :class:`ManifestRecord` JSON file."""

    def build_manifest(
        self,
        *,
        run_id: str,
        target_label: str,
        model_id: str,
        backend: str,
        quantization: str,
        seed: int,
        prompt: str,
        max_new_tokens: int,
        config_snapshot: dict[str, Any],
        git_hash: str | None,
        run_result: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> ManifestRecord:
        """Assemble a :class:`ManifestRecord`. Pure — no I/O."""
        return ManifestRecord(
            run_id=run_id,
            started_at=started_at or _utc_now_iso(),
            target_label=target_label,
            model_id=model_id,
            backend=backend,
            quantization=quantization,
            seed=seed,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            config_snapshot=dict(config_snapshot),
            git_hash=git_hash,
            python=sys.version.split()[0],
            package_version=__version__,
            run_result=dict(run_result) if run_result is not None else None,
        )

    def write_manifest(
        self,
        manifest: ManifestRecord,
        results_dir: Path,
    ) -> Path:
        """Serialise ``manifest`` to ``results_dir/run_<run_id>.json``."""
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"run_{manifest.run_id}.json"
        path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
        return path


__all__ = [
    "ManifestLoggingMixin",
    "ManifestRecord",
    "resolve_git_hash",
]
