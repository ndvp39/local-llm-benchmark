"""Witness the naive baseline load — sample RAM every second around the run.

Spawns ``uv run on-prem-llm run-baseline <target> --skip-preflight`` as a
subprocess and independently samples ``psutil.virtual_memory()`` in the
parent every ``--interval`` seconds. When the child dies (whether via
Python-level Exception, Windows ``OSError``, or OS-level ``SIGSEGV`` /
exit 139), records the exit code + wall time + peak process RSS + the
full RAM timeline into a JSON manifest at
``results/witness_baseline_<label>_<ts>.json``.

Purpose: prove that the naive load actually **attempted to run** and
crashed **because of RAM**, with a timestamped memory curve that
independent readers can audit. The pre-flight guard's ``MemoryError``
is a preemptive safety net; this script produces the reactive
"tried and died" evidence the SC-1 baseline story needs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


def _utc_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mem_gib(x: int) -> float:
    return x / (1024 ** 3)


def witness(target_label: str, interval: float = 1.0) -> Path:
    ts_start = _utc_ts()
    started_iso = _now_iso()
    t0 = time.perf_counter()

    env = dict(os.environ)
    env.setdefault("HF_HOME", "D:/AI_agents_course/hf_cache")
    cmd = [
        "uv", "run", "on-prem-llm", "run-baseline",
        target_label, "--skip-preflight",
    ]
    print(f"[witness] {started_iso}  spawn: {' '.join(cmd)}")
    child = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child_proc = psutil.Process(child.pid)

    samples: list[dict[str, float | str]] = []
    peak_rss = 0
    trough_avail = psutil.virtual_memory().available
    while True:
        try:
            vm = psutil.virtual_memory()
            rss = 0
            with contextlib.suppress(psutil.Error):
                rss += child_proc.memory_info().rss
                for gc in child_proc.children(recursive=True):
                    with contextlib.suppress(psutil.Error):
                        rss += gc.memory_info().rss
            peak_rss = max(peak_rss, rss)
            trough_avail = min(trough_avail, vm.available)
            samples.append({
                "ts": _now_iso(),
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "sys_available_gib": round(_mem_gib(vm.available), 4),
                "sys_used_gib": round(_mem_gib(vm.used), 4),
                "sys_percent": vm.percent,
                "child_rss_gib": round(_mem_gib(rss), 4),
            })
            print(
                f"[witness] +{samples[-1]['elapsed_s']:>7.1f}s  "
                f"avail={samples[-1]['sys_available_gib']} GiB  "
                f"child_rss={samples[-1]['child_rss_gib']} GiB",
                flush=True,
            )
        except psutil.Error:
            pass
        if child.poll() is not None:
            break
        time.sleep(interval)

    wall_s = round(time.perf_counter() - t0, 3)
    exit_code = child.returncode
    stderr = ""
    with contextlib.suppress(OSError):
        stderr = (child.stderr.read() or b"").decode("utf-8", errors="replace")
    ended_iso = _now_iso()

    interpretation = _interpret(exit_code, stderr, samples)
    manifest = {
        "target_label": target_label,
        "started_at": started_iso,
        "ended_at": ended_iso,
        "wall_s": wall_s,
        "exit_code": exit_code,
        "interpretation": interpretation,
        "peak_child_rss_gib": round(_mem_gib(peak_rss), 4),
        "trough_sys_available_gib": round(_mem_gib(trough_avail), 4),
        "sample_count": len(samples),
        "sample_interval_s": interval,
        "stderr_tail": stderr[-2000:] if stderr else "",
        "samples": samples,
    }
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"witness_baseline_{target_label}_{ts_start}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[witness] {ended_iso}  exit={exit_code}  wall={wall_s}s  peak_rss={_mem_gib(peak_rss):.2f} GiB")
    print(f"[witness] manifest -> {path}")
    return path


def _interpret(exit_code: int, stderr: str, samples: list[dict[str, float | str]]) -> str:
    if exit_code == 0:
        return "load_succeeded"
    if exit_code == 139 or exit_code == -11:
        return "os_killed_signal_11_segfault"
    if "paging file" in stderr.lower() or "os error 1455" in stderr.lower():
        return "windows_paging_file_exhausted_os_error_1455"
    if "MemoryError" in stderr:
        return "python_memoryerror"
    if samples and float(samples[-1]["child_rss_gib"]) > 1.0:
        return f"child_died_after_ram_growth_to_{samples[-1]['child_rss_gib']}_gib"
    return f"unknown_exit_{exit_code}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target_label", help="Target label from config.target_models")
    ap.add_argument("--interval", type=float, default=1.0, help="RAM sample interval in seconds")
    args = ap.parse_args()
    witness(args.target_label, args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
