"""Scan the repo for invocations of forbidden tools (constitution §7.4 / ADR-001).

`uv` is the sole package manager and task runner. Direct use of `pip install`,
`python -m`, `virtualenv`, or the bare `venv` command is forbidden in execution-
scope files. Docs are excluded by default (they *discuss* the rule); pass
``--include-docs`` to audit them too.

Per-line opt-out marker (rare): ``# ALLOW-FORBIDDEN: <reason>``. Each opt-out
is reported as a WARNING so reviewers see it without the build failing.

Exit code 0 on clean, 1 if any non-allowlisted invocation is found.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

# (regex, human label) — patterns are deliberately precise to avoid false hits
# on tokens like ".venv/" (uv's virtual-env dir) or "venv" inside an English
# sentence about what's forbidden.
FORBIDDEN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpip\s+install\b"), "pip install"),
    (re.compile(r"\bpython\s+-m\b"), "python -m"),
    (re.compile(r"\bvirtualenv\b"), "virtualenv"),
    # `venv` not preceded by `.` or word char, not followed by word char.
    # Skips `.venv/`, `noenv`, `myvenv2`, the substring inside `eventfilter`, etc.
    (re.compile(r"(?<![.\w])venv\b"), "venv"),
]

ALLOW_MARKER = "ALLOW-FORBIDDEN:"

DEFAULT_INCLUDE: tuple[str, ...] = (
    "src/**/*.py",
    "tests/**/*.py",
    "tools/**/*.py",
    "init_env.py",
    "pyproject.toml",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "*.sh",
    "scripts/**/*.py",
    "scripts/**/*.sh",
)

DOC_INCLUDE: tuple[str, ...] = ("docs/**/*.md", "README.md")

# Files that are themselves part of the enforcement machinery and must not
# trigger on their own definitions.
SELF_EXCLUDE: tuple[str, ...] = ("tools/check_forbidden_tools.py",)


def collect_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    """Resolve glob patterns to a sorted, de-duplicated, existing-file list."""
    seen: set[Path] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def scan_file(path: Path, root: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one file."""
    rel = path.relative_to(root).as_posix()
    if rel in SELF_EXCLUDE:
        return [], []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"{rel}: cannot read ({exc})"], []
    errors: list[str] = []
    warnings: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for regex, label in FORBIDDEN:
            if regex.search(line):
                msg = f"{rel}:{lineno}: forbidden token `{label}` -> {line.strip()}"
                if ALLOW_MARKER in line:
                    warnings.append("ALLOWED  " + msg)
                else:
                    errors.append("FORBIDDEN " + msg)
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Also scan docs/ and README.md (advisory sweep).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    patterns: list[str] = list(DEFAULT_INCLUDE)
    if args.include_docs:
        patterns.extend(DOC_INCLUDE)
    files = collect_files(root, patterns)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for f in files:
        errs, warns = scan_file(f, root)
        all_errors.extend(errs)
        all_warnings.extend(warns)
    for w in all_warnings:
        print(w)
    for e in all_errors:
        print(e)
    print(
        f"---\nScanned {len(files)} file(s); "
        f"{len(all_errors)} error(s), {len(all_warnings)} allowed warning(s)."
    )
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
