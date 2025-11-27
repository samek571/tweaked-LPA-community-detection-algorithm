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
    stream_multi_sweep_parallel_blocks)


def _load_adj_from_edgelist(path: str) -> List[List[int]]:
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
    p = Path(out_path)
    if p.suffix == ".npy":
        np.save(out_path, np.asarray(labels, dtype=np.int64))
    else:
        with open(out_path, "w") as f:
            for x in labels:
                f.write(f"{int(x)}\n")


def _summarize(labels, info: dict, took: float, mode: str):
    k = len(set(int(x) for x in labels))
    print(f"[{mode}] n={len(labels)}  communities={k}  sweeps={info.get('sweeps','?')}  "
        f"converged={info.get('converged','?')}  time={took:.3f}s")


def run_ram(args) -> int:
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
    outp = Path(args.out_path)
    base = outp.with_suffix("")

    sorted_sym = str(args.sorted if args.sorted else base.with_suffix(".sorted.sym"))
    block_index = str(args.index if args.index else base.with_suffix(".blockidx.npy"))
    labels_path = str(outp if outp.suffix == ".npy" else base.with_suffix(".labels.npy"))

    #1) sort + symm
    meta = symmetrize_and_sort(args.in_path, sorted_sym)  # returns {"n","m"}
    n = int(meta["n"])

    block_size = args.block_size
    if block_size is None or block_size <= 0:
        block_size = max(1000, n // 20)

    #2) build block index + init labels
    build_block_index(sorted_sym, n=n, block_size=block_size, index_path=block_index)
    init_labels_memmap(labels_path, n=n)

    #3) multi-sweep
    t0 = time.time()
    #split sorted_sym into per block files once
    blocks_dir = str(base.with_suffix(".blocks"))
    block_paths = split_sorted_sym_to_blocks(
        sorted_sym,
        n=n,
        block_size=block_size,
        out_dir=blocks_dir)

    workers = args.workers if getattr(args, "parallel", False) else 1
    #parallel sweeps over block files, only per-block adjacency stored inside RAM
    info = stream_multi_sweep_parallel_blocks(
        block_paths,
        labels_path,
        n=meta["n"],
        block_size=args.block_size,
        seed=args.seed,
        max_sweeps=args.max_sweeps,
        min_sweeps=max(1, min(args.min_sweeps, args.max_sweeps)),
        tie_break=args.tie_break,
        workers=workers)
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
    p = argparse.ArgumentParser(prog="lpkit", description="Label Propagation (RAM or streaming).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--in", dest="in_path", required=True, help="Input edgelist file (u v per line).")
        sp.add_argument("--out", dest="out_path", required=True, help="Output labels file (.npy recommended).")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max-sweeps", type=int, default=100)
        sp.add_argument("--min-sweeps", type=int, default=1)

    #RAM
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
    ps.add_argument("--parallel", action="store_true", help="Enable parallel block sweeps.")
    ps.add_argument("--workers", type=int, default=None, help="Number of worker threads.")
    ps.set_defaults(func=run_stream)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130 #ctrl + c == sigint


if __name__ == "__main__":
    sys.exit(main())
