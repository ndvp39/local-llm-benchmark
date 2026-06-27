"""Env-init bootstrap (ADR-016) — ``uv run init_env.py`` before any pipeline action.

Delegate-only per constitution §3.1: parse args → SDK → ``initialize_environment()``.
Prints the per-file write receipts; exits non-zero on any ``fail`` receipt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from on_prem_llm_lab import OnPremLlmSDK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bootstrap on_prem_llm_lab env (ADR-016).")
    p.add_argument("--config", type=Path, default=Path("config/setup.json"))
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    a = p.parse_args(argv)
    sdk = OnPremLlmSDK(config_path=a.config, env=dict(os.environ), repo_root=a.repo_root)
    result = sdk.initialize_environment()
    print(f"Hardware scan captured at {result.scan.captured_at}")
    for key, r in result.scan.write_receipts.items():
        suffix = f" ({r.reason})" if r.reason else ""
        print(f"  [{key}] {r.status} -> {r.path}{suffix}")
    for f in result.failures:
        print(f"FAIL: {f}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
