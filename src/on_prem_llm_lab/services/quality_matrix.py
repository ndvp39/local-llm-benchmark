"""Quality-matrix harness (T-3.6) — side-by-side `completion_text` per cell.

Reads per-run manifests under ``results/run_<uuid>.json``, groups them by
``(target_label, quantization, backend)``, picks a representative completion
text per cell (first successful measurement), and writes a Markdown table to
``results/quality_matrix.md``. The report embeds this file verbatim in the
§Quantization Quality section.

Contract: FR-11 (`docs/PRD.md`) — "same prompt, all quantization levels, per
target model, side-by-side outputs captured in ``results/quality_matrix.md``".
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class QualityCell:
    """One (target, quantization, backend) cell + its representative text."""

    target_label: str
    quantization: str
    backend: str
    prompt: str
    completion_text: str
    ttft_ms: float
    tpot_ms: float
    peak_ram_mb: float
    seed: int


def _load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_cells(results_dir: Path) -> list[QualityCell]:
    """Walk ``results_dir/run_*.json`` + group into one QualityCell per group."""
    cells: dict[tuple[str, str, str], QualityCell] = {}
    for p in sorted(results_dir.glob("run_*.json")):
        m = _load_manifest(p)
        if m is None:
            continue
        rr = m.get("run_result") or {}
        completion = rr.get("completion_text") or ""
        if not completion:
            continue
        key = (
            m.get("target_label", "?"),
            m.get("quantization", "?"),
            m.get("backend", "?"),
        )
        if key in cells:
            continue
        cells[key] = QualityCell(
            target_label=key[0],
            quantization=key[1],
            backend=key[2],
            prompt=m.get("prompt", ""),
            completion_text=completion,
            ttft_ms=float(rr.get("ttft_ms", 0.0)),
            tpot_ms=float(rr.get("tpot_ms", 0.0)),
            peak_ram_mb=float(rr.get("peak_ram_mb", 0.0)),
            seed=int(m.get("seed", 0)),
        )
    return list(cells.values())


def format_markdown(cells: Iterable[QualityCell], prompt: str) -> str:
    """Render `results/quality_matrix.md` — FR-11 side-by-side table."""
    lines = [
        "# Quality Matrix — per-cell completion side-by-side",
        "",
        f'**Prompt (identical for every cell):** `{prompt}`',
        "",
        "| target | quantization | backend | TTFT (ms) | TPOT (ms) | peak RAM (MB) | completion |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda x: (x.target_label, x.quantization)):
        completion_cell = c.completion_text.replace("|", "\\|").replace("\n", " ")
        if len(completion_cell) > 300:
            completion_cell = completion_cell[:297] + "..."
        lines.append(
            f"| `{c.target_label}` | `{c.quantization}` | `{c.backend}` | "
            f"{c.ttft_ms:.0f} | {c.tpot_ms:.0f} | {c.peak_ram_mb:.0f} | "
            f"{completion_cell} |",
        )
    lines.append("")
    return "\n".join(lines)


def write_quality_matrix(
    results_dir: Path, out_path: Path | None = None, prompt: str = "Hello, world.",
) -> Path:
    """Read per-run manifests, format Markdown, write to disk. Returns path."""
    cells = collect_cells(results_dir)
    target = out_path if out_path is not None else results_dir / "quality_matrix.md"
    target.write_text(format_markdown(cells, prompt), encoding="utf-8")
    return target


__all__ = [
    "QualityCell",
    "collect_cells",
    "format_markdown",
    "write_quality_matrix",
]
