"""Extra backend CLI subcommands split out of ``main.py`` (size-cap exception).

Registers ``run-airllm`` on the shared ``app`` from ``cli.main``. This
module is imported at the tail of ``main.py`` so ``uv run on-prem-llm``
sees the command without any explicit ``include_typer`` boilerplate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from on_prem_llm_lab import EnvironmentNotInitializedError, OnPremLlmSDK
from on_prem_llm_lab.cli.main import _DEFAULT_CONFIG, _ConfigOpt, _RepoRootOpt, app


@app.command("run-airllm")
def run_airllm(
    target_label: Annotated[
        str, typer.Argument(help="Target label from config.target_models."),
    ],
    quantization: Annotated[
        str | None,
        typer.Option(
            "--quantization", "-q",
            help="Override the target's default quantization (fp16|q4|q8).",
        ),
    ] = None,
    max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-new-tokens", "-n",
            help="Override generation.max_new_tokens for this cell (hero run).",
        ),
    ] = None,
    config: _ConfigOpt = _DEFAULT_CONFIG,
    repo_root: _RepoRootOpt = None,
) -> None:
    """Single-cell AirLLM run (T-3.5a) — writes results/airllm_<label>_<q>_<ts>.json."""
    sdk = OnPremLlmSDK(
        config_path=config, env=dict(os.environ),
        repo_root=repo_root or Path.cwd(),
    )
    try:
        result = sdk.run_airllm(
            target_label, quantization=quantization,
            max_new_tokens=max_new_tokens,
        )
    except EnvironmentNotInitializedError as exc:
        typer.echo(f"FAIL: env not initialised -- {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001 — SC-1 surfacing
        typer.echo(f"FAIL: {target_label} -- {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"OK: {target_label} -> {result.raw_log_path}")
    raise typer.Exit(0)


@app.command("run-sweep")
def run_sweep(
    skip_plumbing: Annotated[
        bool, typer.Option(
            "--skip-plumbing", help="Bypass the ADR-010 plumbing precondition.",
        ),
    ] = False,
    config: _ConfigOpt = _DEFAULT_CONFIG,
    repo_root: _RepoRootOpt = None,
) -> None:
    """Full target × quantization × backend sweep (T-3.5) — writes sweep CSV + manifest."""
    sdk = OnPremLlmSDK(
        config_path=config, env=dict(os.environ),
        repo_root=repo_root or Path.cwd(),
    )
    try:
        csv_path = sdk.run_sweep(skip_plumbing=skip_plumbing)
    except EnvironmentNotInitializedError as exc:
        typer.echo(f"FAIL: env not initialised -- {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"FAIL: sweep failed -- {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"OK: sweep complete -> {csv_path}")
    raise typer.Exit(0)
