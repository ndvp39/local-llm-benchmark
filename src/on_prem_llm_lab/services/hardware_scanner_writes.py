"""Atomic config injection + doc placeholder patching for HardwareScanner.

Implements ADR-015 steps 2 and 3 with strict atomicity invariants:

- :func:`inject_into_setup_json` sets ``hardware_constraints`` in ``setup.json``.
- :func:`patch_doc_placeholders` replaces the content between the placeholder
  pair (``init.placeholder_start`` / ``init.placeholder_end``) in a doc file.

Atomic recipe: write ``<file>.tmp`` then :func:`os.replace`. When ``keep_bak``
is true, the prior content is first copied to ``<file>.bak``. Idempotent: if
the new content equals the existing content byte-for-byte, no write occurs
and the receipt status is ``ok-noop`` (with no ``.bak`` created).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from on_prem_llm_lab.services.hardware_scanner_types import (
    HardwareScanResult,
    WriteReceipt,
)

log = logging.getLogger(__name__)


def _atomic_write(path: Path, new_text: str, keep_bak: bool) -> Path | None:
    """Replace ``path`` content with ``new_text`` atomically; return .bak path if made."""
    bak: Path | None = None
    if path.exists() and keep_bak:
        bak = path.with_name(path.name + ".bak")
        shutil.copy2(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, path)
    return bak


def _result_payload_for_config(r: HardwareScanResult) -> dict:
    """Strip ``write_receipts`` from a result before injecting into ``setup.json``."""
    d = asdict(r)
    d.pop("write_receipts", None)
    return d


def inject_into_setup_json(
    path: Path, result: HardwareScanResult, *, keep_bak: bool
) -> WriteReceipt:
    """Set ``hardware_constraints`` in ``setup.json`` to the scan payload."""
    if not path.exists():
        return WriteReceipt(
            status="fail", path=str(path), reason="config file not found"
        )
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return WriteReceipt(
            status="fail", path=str(path), reason=f"invalid JSON: {exc}"
        )
    payload = _result_payload_for_config(result)
    if cfg.get("hardware_constraints") == payload:
        return WriteReceipt(status="ok-noop", path=str(path))
    cfg["hardware_constraints"] = payload
    new_text = json.dumps(cfg, indent=4) + "\n"
    try:
        bak = _atomic_write(path, new_text, keep_bak)
    except OSError as exc:
        return WriteReceipt(
            status="fail", path=str(path), reason=f"write error: {exc}"
        )
    return WriteReceipt(
        status="ok", path=str(path), bak=str(bak) if bak else None
    )


def patch_doc_placeholders(
    path: Path,
    table_md: str,
    start_marker: str,
    end_marker: str,
    *,
    keep_bak: bool,
) -> WriteReceipt:
    """Replace content between the marker pair with ``table_md``."""
    if not path.exists():
        return WriteReceipt(status="skipped-file-missing", path=str(path))
    text = path.read_text(encoding="utf-8")
    si = text.find(start_marker)
    if si < 0:
        return WriteReceipt(
            status="skipped-no-markers", path=str(path),
            reason="start marker not found",
        )
    ei = text.find(end_marker, si + len(start_marker))
    if ei < 0:
        return WriteReceipt(
            status="skipped-no-markers", path=str(path),
            reason="end marker not found after start",
        )
    head = text[: si + len(start_marker)]
    tail = text[ei:]
    new_text = head + "\n\n" + table_md + "\n\n" + tail
    if new_text == text:
        return WriteReceipt(status="ok-noop", path=str(path))
    try:
        bak = _atomic_write(path, new_text, keep_bak)
    except OSError as exc:
        return WriteReceipt(
            status="fail", path=str(path), reason=f"write error: {exc}"
        )
    return WriteReceipt(
        status="ok", path=str(path), bak=str(bak) if bak else None
    )


__all__ = ["inject_into_setup_json", "patch_doc_placeholders"]
