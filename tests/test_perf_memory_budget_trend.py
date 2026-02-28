"""System/performance test for streaming LPA under cgroup memory limits.

This test is intentionally gated (`LPKIT_RUN_PERF=1`) and should not run in the
default unit-test suite. It launches the CLI under `systemd-run --user --scope`
with different `MemoryMax` values, then checks:

1) Completed runs reach the same endpoint metadata (same graph/seed/params).
2) Lower memory budgets do not become materially faster (with slack).
3) OOM/kill under lower memory is treated as a worse outcome, not a false failure.

The test compares medians across repeated runs and randomizes run order to reduce
cache and scheduler bias.
"""

import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
import pytest
from lpkit.generate_graph import generate_large_graph

# run in console cli, doesnt work inside the intellij probably
#LPKIT_RUN_PERF=1 pytest -q -s tests/test_perf_memory_budget_trend.py
def test_memory_budget_trend(tmp_path: Path) -> None:
    if os.getenv("LPKIT_RUN_PERF") != "1":
        pytest.skip("Set LPKIT_RUN_PERF=1 to run perf test")

    if shutil.which("systemd-run") is None:
        pytest.skip("systemd-run not found")

    try:
        p = subprocess.run(
            ["systemd-run", "--user", "--scope", "true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if p.returncode != 0:
            pytest.skip("systemd --user scope unavailable")
    except Exception:
        pytest.skip("systemd --user scope unavailable")

    repo_root = Path(__file__).resolve().parents[1]
    summary_re = re.compile(
        r"\[STREAM\(bs=(?P<bs>\d+)\)]\s+"
        r"n=(?P<n>\d+)\s+"
        r"communities=(?P<comms>\d+)\s+"
        r"sweeps=(?P<sweeps>\d+)\s+"
        r"converged=(?P<conv>\d+)\s+"
        r"time=(?P<t>[0-9.]+)s"
    )

    n = int(os.getenv("LPKIT_PERF_N", "100000"))
    m = int(os.getenv("LPKIT_PERF_M", "300000"))
    block_size = int(os.getenv("LPKIT_PERF_BLOCK_SIZE", "5000"))
    max_sweeps = int(os.getenv("LPKIT_PERF_MAX_SWEEPS", "50"))
    repeats = int(os.getenv("LPKIT_PERF_REPEATS", "3"))
    seed = int(os.getenv("LPKIT_PERF_SEED", "1337"))
    mem_caps = [int(x) for x in os.getenv("LPKIT_PERF_MEM_CAPS_MB", "256,192,128").split(",") if x.strip()]
    slack = float(os.getenv("LPKIT_PERF_MONOTONE_SLACK", "0.90"))

    assert len(mem_caps) >= 2
    assert mem_caps == sorted(mem_caps, reverse=True)

    raw = tmp_path / "large_random.edgelist"
    generate_large_graph(str(raw), n=n, m=m, topology="random", seed=987654)

    venv = os.environ.get("VIRTUAL_ENV")
    venv_bin = Path(venv) / "bin" if venv else Path(sys.executable).resolve().parent
    python = (venv_bin / "python").resolve()
    env_path = f"{venv_bin}:{os.environ.get('PATH','')}"
    env_pp = [str(repo_root)]
    if (repo_root / "src").is_dir():
        env_pp.append(str(repo_root / "src"))
    if os.environ.get("PYTHONPATH"):
        env_pp.append(os.environ["PYTHONPATH"])
    env_pythonpath = ":".join(env_pp)

    runs = {cap: [] for cap in mem_caps}
    #shuffle order to reduce warm-cache or ordering bias across memory caps
    schedule = [(cap, rep) for rep in range(repeats) for cap in mem_caps]
    random.Random(20260226).shuffle(schedule)

    for cap, rep in schedule:
        out_labels = tmp_path / f"labels_mem{cap}_rep{rep}.npy"
        out_labels.parent.mkdir(parents=True, exist_ok=True)
        out_labels.touch(exist_ok=True)

        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            "--working-directory",
            str(repo_root),
            "-p",
            f"MemoryMax={cap}M",
            "env",
            f"VIRTUAL_ENV={venv or ''}",
            f"PATH={env_path}",
            f"PYTHONPATH={env_pythonpath}",
            str(python),
            "-m",
            "lpkit.cli",
            "stream",
            "--in",
            str(raw),
            "--out",
            str(out_labels),
            "--max-sweeps",
            str(max_sweeps),
            "--block-size",
            str(block_size),
            "--seed",
            str(seed),
        ]

        t0 = time.perf_counter()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wall = time.perf_counter() - t0

        if p.returncode != 0:
            # killed/oom => treat as "worse under low mem"
            if p.returncode in (-9, -15, 137, 143):
                runs[cap].append({"ok": False, "rc": p.returncode, "wall": wall, "out": p.stdout})
                continue
            raise AssertionError(
                f"Run failed for MemoryMax={cap}M\n"
                f"returncode={p.returncode}\n"
                f"cmd={' '.join(map(str, cmd))}\n"
                f"Output:\n{p.stdout}"
            )

        msum = summary_re.search(p.stdout)
        if not msum:
            raise AssertionError(
                "Could not parse lpkit summary line.\n"
                f"cmd={' '.join(map(str, cmd))}\n"
                f"Output:\n{p.stdout}"
            )

        runs[cap].append(
            {
                "ok": True,
                "rc": 0,
                "wall": wall,
                "t": float(msum.group("t")),
                "n": int(msum.group("n")),
                "comms": int(msum.group("comms")),
                "sweeps": int(msum.group("sweeps")),
                "conv": int(msum.group("conv")),
                "bs": int(msum.group("bs")),
            }
        )

    ok_caps = [cap for cap in mem_caps if any(r["ok"] for r in runs[cap])]
    if len(ok_caps) < 2:
        pytest.skip(f"Too many caps killed/OOM to compare runtimes: { {cap:[r['rc'] for r in runs[cap]] for cap in mem_caps} }")

    #successful runs should end at the same endpoint metadata for same graph/seed/params
    base = None
    for cap in ok_caps:
        for r in runs[cap]:
            if not r["ok"]:
                continue
            sig = (r["n"], r["comms"], r["sweeps"], r["conv"], r["bs"])
            base = sig if base is None else base
            assert sig == base, f"Endpoint changed under {cap}M: {sig} != {base}"

    #median to reduce chache noise or scheduling
    med = {cap: statistics.median([r["t"] for r in runs[cap] if r["ok"]]) for cap in ok_caps}

    for hi, lo in zip(mem_caps, mem_caps[1:]):
        if hi not in ok_caps:
            raise AssertionError(f"Higher cap {hi}M did not complete; returncodes={[r['rc'] for r in runs[hi]]}")
        if lo not in ok_caps:
            continue
        assert med[lo] >= slack * med[hi], (
            f"inversion: {lo}M {med[lo]:.3f}s < {slack:.2f}*{hi}M {med[hi]:.3f}s"
        )
