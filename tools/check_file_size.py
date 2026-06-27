"""Enforce constitution §2.2 — every source file MUST stay within 150 LOC.

LOC = lines that are *neither* blank *nor* pure-comment. A "pure-comment" line
is one whose stripped content starts with ``#``. Docstrings count as LOC
(they are statements, not comments). Multi-line string literals likewise count.

Default scan covers ``src/**/*.py``, ``tests/**/*.py``, ``tools/**/*.py``, and
the top-level ``init_env.py`` (when it lands in T-1.16).

Exit code 0 on clean, 1 on any violation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

MAX_LOC_DEFAULT = 150

DEFAULT_INCLUDE: tuple[str, ...] = (
    "src/**/*.py",
    "tests/**/*.py",
    "tools/**/*.py",
    "init_env.py",
)


def count_loc(text: str) -> int:
    """Return the number of non-blank, non-pure-comment lines in ``text``."""
    n = 0
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
    return n


def collect_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    """Resolve glob patterns relative to ``root``; return a sorted file list."""
    seen: set[Path] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def scan(
    files: Iterable[Path], root: Path, max_loc: int
) -> tuple[list[str], list[str]]:
    """Return (per-file LOC rows, violation messages)."""
    rows: list[str] = []
    violations: list[str] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            violations.append(f"{rel}: cannot read ({exc})")
            continue
        loc = count_loc(text)
        rows.append(f"  {loc:>4}  {rel}")
        if loc > max_loc:
            violations.append(f"{rel}: {loc} LOC > {max_loc} limit")
    return rows, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=MAX_LOC_DEFAULT,
        help=f"LOC limit per file (default: {MAX_LOC_DEFAULT}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print violations + summary (skip per-file LOC table).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    files = collect_files(root, DEFAULT_INCLUDE)
    rows, violations = scan(files, root, args.max_lines)
    if not args.quiet:
        print(" LOC  file")
        for r in rows:
            print(r)
    for v in violations:
        print(f"VIOLATION {v}")
    print(
        f"---\nScanned {len(files)} file(s); "
        f"{len(violations)} violation(s); limit={args.max_lines}."
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
