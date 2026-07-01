"""CLI entry point — delegate-only (constitution §3.1 / ADR-002).

Every subcommand here MUST do the absolute minimum: argument parsing, a single
call into :class:`~on_prem_llm_lab.sdk.OnPremLlmSDK`, formatting the result for
the terminal, and choosing an exit code. ANY branch that computes a metric,
decides a strategy, or formats a number is a bug — push it into the SDK.
"""

from __future__ import annotations

# CRITICAL LOAD ORDER: .env must load BEFORE `on_prem_llm_lab` is imported,
# because that package transitively imports `transformers` / `huggingface_hub`
# which read HF_HOME at *import time* and cache it — a late load_dotenv()
# leaves HF downloads going to the default C: cache even though the .env
# says D:. See prompts_book §11.9 for the disk-full incident.
# override=True: parent shell may pass HF_HOME="" (empty) which python-dotenv
# treats as "already set" and skips; without override, downloads silently
# fall back to C:\Users\<user>\.cache\huggingface\ even with .env present.
from dotenv import load_dotenv

load_dotenv(override=True)

import os  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Annotated  # noqa: E402

import typer  # noqa: E402

from on_prem_llm_lab import (  # noqa: E402
    EnvironmentNotInitializedError,
    OnPremLlmSDK,
    PlumbingStageError,
)

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


@app.command("run-plumbing-test")
def run_plumbing_test(
    config: _ConfigOpt = _DEFAULT_CONFIG,
    repo_root: _RepoRootOpt = None,
) -> None:
    """ADR-010 pre-flight on the small/Q2 plumbing model (T-2a.2)."""
    sdk = OnPremLlmSDK(
        config_path=config,
        env=dict(os.environ),
        repo_root=repo_root or Path.cwd(),
    )
    try:
        result = sdk.run_plumbing_test()
    except EnvironmentNotInitializedError as exc:
        typer.echo(f"FAIL: env not initialised -- {exc}")
        raise typer.Exit(1) from exc
    except PlumbingStageError as exc:
        typer.echo(f"FAIL: plumbing stage '{exc.stage}' failed")
        typer.echo(f"  message: {exc}")
        if exc.result.remediation_hint:
            typer.echo(f"  remediation: {exc.result.remediation_hint}")
        if exc.result.manifest_path:
            typer.echo(f"  partial manifest: {exc.result.manifest_path}")
        raise typer.Exit(1) from exc
    typer.echo("OK: plumbing test passed")
    typer.echo(f"  manifest: {result.manifest_path}")
    for name, outcome in result.stages.items():
        typer.echo(f"  {name}: {outcome.status} ({outcome.duration_s:.3f}s)")
    raise typer.Exit(0)


def _resolve_baseline_targets(sdk: OnPremLlmSDK, target_label: str | None) -> list[str]:
    import json as _json  # noqa: PLC0415
    cfg = _json.loads(sdk.config_path.read_text(encoding="utf-8"))
    labels = [t["label"] for t in cfg.get("target_models", [])]
    if target_label is None:
        return labels
    if target_label not in labels:
        raise typer.BadParameter(f"Unknown target_label {target_label!r}; known: {labels}")
    return [target_label]


@app.command("run-baseline")
def run_baseline(
    target_label: Annotated[
        str | None,
        typer.Argument(
            help="Target label from config.target_models; omit to iterate all.",
        ),
    ] = None,
    config: _ConfigOpt = _DEFAULT_CONFIG,
    repo_root: _RepoRootOpt = None,
    skip_preflight: Annotated[
        bool, typer.Option("--skip-preflight",
            help="Bypass the RAM pre-flight guard — force torch to attempt the naive load.",
        ),
    ] = False,
) -> None:
    """Direct back-end baseline run (T-2.10 / SC-1) — per target."""
    sdk = OnPremLlmSDK(
        config_path=config,
        env=dict(os.environ),
        repo_root=repo_root or Path.cwd(),
    )
    targets = _resolve_baseline_targets(sdk, target_label)
    failures = 0
    for label in targets:
        try:
            result = sdk.run_baseline(label, skip_preflight=skip_preflight)
        except EnvironmentNotInitializedError as exc:
            typer.echo(f"FAIL: env not initialised -- {exc}")
            raise typer.Exit(1) from exc
        except Exception as exc:  # noqa: BLE001 — SC-1 surfacing
            failures += 1
            typer.echo(f"FAIL: {label} -- {type(exc).__name__}: {exc}")
            continue
        typer.echo(f"OK: {label} -> {result.raw_log_path}")
    raise typer.Exit(1 if failures else 0)


from on_prem_llm_lab.cli import backend_cmds as _backend_cmds  # noqa: E402, F401

if __name__ == "__main__":
    app()
