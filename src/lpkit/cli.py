"""Command-line interface for LPKit.

Subcommands
-----------
ram
    In-memory baseline LPA (useful for small/medium graphs and correctness checks).
stream
    Disk-backed streaming LPA pipeline for larger graphs.

The CLI intentionally mirrors the lower-level API functions, so README examples and
programmatic usage stay aligned.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple
from .label_propagation import label_propagation
from .stream import (
    symmetrize_and_sort,
    build_block_index,
    init_labels_memmap,
    split_sorted_sym_to_blocks,
    stream_multi_sweep_blocks)


def _load_adj_from_edgelist(path: str) -> List[List[int]]:
    """Load an undirected adjacency list from a 2-column edge list.

    Comment lines starting with `#` or `%` are ignored. Edges are inserted in both
    directions because RAM mode expects an undir adj list
    """
    edges: List[Tuple[int, int]] = []
    n = 0
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("%"):
                continue
            a, b = ln.split()
            u = int(a); v = int(b)
            edges.append((u, v))
            n = max(n, u + 1, v + 1)

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _write_labels(labels, out_path: str):
    """Write labels either as `.npy` or as one label per text line."""
    p = Path(out_path)
    if p.suffix == ".npy":
        np.save(out_path, np.asarray(labels, dtype=np.int64))
    else:
        with open(out_path, "w") as f:
            for x in labels:
                f.write(f"{int(x)}\n")


def _summarize(labels, info: dict, took: float, mode: str):
    """Print a compact run summary similar to benchmark output."""
    k = len(set(int(x) for x in labels))
    print(f"[{mode}] n={len(labels)}  communities={k}  sweeps={info.get('sweeps','?')}  "
        f"converged={info.get('converged','?')}  time={took:.3f}s")


def run_ram(args) -> int:
    """Execute in-memory LPA from CLI arguments."""
    adj = _load_adj_from_edgelist(args.in_path)
    t0 = time.time()
    labels, info = label_propagation(
        adj,
        seed=args.seed,
        max_sweeps=args.max_sweeps,
        min_sweeps=max(1, min(args.min_sweeps, args.max_sweeps)),
        verify_each_sweep=True,
        shuffle_each_sweep=not args.no_shuffle)

    took = time.time() - t0
    _write_labels(labels, args.out_path)
    _summarize(labels, info, took, "RAM")
    return 0


def run_stream(args) -> int:
    """Execute the full streaming pipeline from CLI arguments.

    Artifacts produced (unless custom paths are supplied):
    - `<base>.sorted.sym`
    - `<base>.blockidx.npy` (legacy helper index; not required by the new block split path)
    - `<base>.blocks/`
    - labels `.npy`
    """
    outp = Path(args.in_path)
    if not outp.exists():
        raise FileNotFoundError(
            f"Input edgelist not found: {outp}\n"
            f"Working directory: {Path.cwd()}\n"
            f"Tip: replace the placeholder path with a real file path."
        )
    base = outp.with_suffix("")

    sorted_sym = str(args.sorted if args.sorted else base.with_suffix(".sorted.sym"))
    block_index = str(args.index if args.index else base.with_suffix(".blockidx.npy"))
    labels_path = str(outp if outp.suffix == ".npy" else base.with_suffix(".labels.npy"))

    #1) externalsort + symm
    meta = symmetrize_and_sort(args.in_path, sorted_sym)  # returns {"n","m"}
    n = int(meta["n"])

    block_size = args.block_size
    if block_size is None or block_size <= 0:
        block_size = max(1000, n // 20)

    #2) build (byte-offset) block index + init labels
    build_block_index(sorted_sym, n=n, block_size=block_size, index_path=block_index)
    init_labels_memmap(labels_path, n=n)

    #3) split into binary block files and run streaming sweaping
    t0 = time.time()
    #split sorted_sym into per block files once
    blocks_dir = str(base.with_suffix(".blocks"))
    block_paths = split_sorted_sym_to_blocks(
        sorted_sym,
        n=n,
        block_size=block_size,
        out_dir=blocks_dir)

    info = stream_multi_sweep_blocks(
        block_paths,
        labels_path,
        n=meta["n"],
        block_size=args.block_size,
        seed=args.seed,
        max_sweeps=args.max_sweeps,
        min_sweeps=max(1, min(args.min_sweeps, args.max_sweeps)),
        tie_break=args.tie_break)
    took = time.time() - t0

    if args.block_size is None:
        args.block_size = 5000

    #4) summarize
    mm = np.lib.format.open_memmap(labels_path, mode="r+")
    _summarize(mm, info, took, f"STREAM(bs={block_size})")

    if Path(args.out_path).suffix != ".npy":
        _write_labels(mm, args.out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and dispatch to a subcommand."""
    p = argparse.ArgumentParser(prog="lpkit", description="Label Propagation (RAM or streaming).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        """Attach flags shared by RAM and streaming modes."""
        sp.add_argument("--in", dest="in_path", required=True, help="Input edgelist file (u v per line).")
        sp.add_argument("--out", dest="out_path", required=True, help="Output labels file (.npy recommended).")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max-sweeps", type=int, default=100)
        sp.add_argument("--min-sweeps", type=int, default=1)

    #RAM (essential only for some tests and correctness determination)
    pr = sub.add_parser("ram", help="Run in-RAM LPA (baseline).")
    add_common(pr)
    pr.add_argument("--no-shuffle", action="store_true", help="Disable per-sweep vertex shuffling.")
    pr.set_defaults(func=run_ram)

    #HDD
    ps = sub.add_parser("stream", help="Run streaming (HDD) LPA.")
    add_common(ps)
    ps.add_argument("--block-size", type=int, default=None, help="Vertices per block (default n//20, min 1000).")
    ps.add_argument("--tie-break", choices=["random", "min", "max"], default="min")
    ps.add_argument("--sorted", default=None, help="(optional) path for .sorted.sym")
    ps.add_argument("--index", default=None, help="(optional) path for .blockidx.npy")
    ps.set_defaults(func=run_stream)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130 #ctrl + c == sigint


if __name__ == "__main__":
    sys.exit(main())
