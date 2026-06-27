"""CLI entry point — delegate-only (constitution §3.1 / ADR-002).

Every subcommand here MUST do the absolute minimum: argument parsing, a single
call into :class:`~on_prem_llm_lab.sdk.OnPremLlmSDK`, formatting the result for
the terminal, and choosing an exit code. ANY branch that computes a metric,
decides a strategy, or formats a number is a bug — push it into the SDK.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from on_prem_llm_lab.sdk import OnPremLlmSDK

app = typer.Typer(
    help="On-Premises LLM benchmark lab CLI.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    """No-op root callback that keeps Typer in multi-command mode.

    Without this, a Typer app with a single ``@app.command()`` collapses to
    single-command mode and refuses to recognise the subcommand name on the
    command line. The callback is invoked once on every CLI run; it deliberately
    does nothing so the per-command handlers stay the sole entry points.
    """


_DEFAULT_CONFIG = Path("config/setup.json")

_ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to setup.json (default: ./config/setup.json)."),
]
_RepoRootOpt = Annotated[
    Path | None,
    typer.Option("--repo-root", help="Repository root for resolving doc placeholders (default: cwd)."),
]


@app.command()
def initialize(
    config: _ConfigOpt = _DEFAULT_CONFIG,
    repo_root: _RepoRootOpt = None,
) -> None:
    """Run the env-init bootstrap (ADR-016).

    Scans hardware, atomically injects ``hardware_constraints`` into
    ``setup.json``, and patches the ``<!-- HARDWARE_SPECS_PLACEHOLDER:* -->``
    block in each path listed in ``init.doc_targets``. Prints a per-file
    receipt summary. Exits 0 on success, non-zero on any ``fail`` receipt.
    """
    sdk = OnPremLlmSDK(
        config_path=config,
        env=dict(os.environ),
        repo_root=repo_root or Path.cwd(),
    )
    result = sdk.initialize_environment()
    typer.echo(f"Hardware scan captured at {result.scan.captured_at}")
    typer.echo("")
    typer.echo("Per-file write receipts:")
    for key, receipt in result.scan.write_receipts.items():
        marker = "x" if receipt.status == "fail" else "."
        typer.echo(f"  [{marker}] {key:24s} {receipt.status:25s} -> {receipt.path}")
        if receipt.reason:
            typer.echo(f"       reason: {receipt.reason}")
    if result.failures:
        typer.echo("")
        typer.echo("FAILURES:")
        for f in result.failures:
            typer.echo(f"  - {f}")
    raise typer.Exit(0 if result.ok else 1)


if __name__ == "__main__":
    app()
