import time
import numpy as np
from pathlib import Path
from util_topologies import write_grid_graph, write_clusters_graph
from lpkit import (label_propagation, symmetrize_and_sort, build_block_index, init_labels_memmap, stream_multi_sweep)


def _load_adj_from_edgelist(path: Path):
    with open(path) as f:
        edges = [tuple(map(int, ln.split())) for ln in f]
    n = (max(max(u, v) for u, v in edges) + 1) if edges else 0
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


def _run_ram(adj, seed=1337, max_sweeps=200):
    t0 = time.time()
    labels, info = label_propagation(adj, seed=seed, max_sweeps=max_sweeps)
    dt = time.time() - t0
    num = len(set(labels))
    return {"num": num, "time": dt, "sweeps": info.get("sweeps", None), "converged": info.get("converged", None)}


def _run_hdd(raw_path: Path, n: int, block_size: int, seed=1337, max_sweeps=200):
    sorted_sym = raw_path.with_suffix(".sorted.sym")
    idx = raw_path.with_suffix(".blockidx.npy")
    labels = raw_path.with_suffix(".labels.npy")

    meta = symmetrize_and_sort(str(raw_path), str(sorted_sym))
    bs = max(1, min(block_size, n))
    build_block_index(str(sorted_sym), n=meta["n"], block_size=bs, index_path=str(idx))
    init_labels_memmap(str(labels), n=meta["n"])
    t0 = time.time()
    info = stream_multi_sweep(
        str(sorted_sym),
        str(idx),
        str(labels),
        n=meta["n"],
        block_size=bs,
        seed=seed,
        max_sweeps=max_sweeps,
        tie_break="min",
    )
    dt = time.time() - t0
    mm = np.lib.format.open_memmap(str(labels), mode="r+")
    num = len(set(int(x) for x in mm))
    return {"num": num, "time": dt, "sweeps": info.get("sweeps", None)}


def test_grid_graph(tmp_path):
    raw = tmp_path / "grid.edgelist"
    side = 20
    n, _m = write_grid_graph(raw, side_len=side)

    #RAM
    adj = _load_adj_from_edgelist(raw)
    ram = _run_ram(adj, seed=1337, max_sweeps=200)

    #HDD
    hdd_n = _run_hdd(raw, n=n, block_size=n, seed=1337, max_sweeps=200)
    hdd_n4 = _run_hdd(raw, n=n, block_size=n // 4, seed=1337, max_sweeps=200)
    hdd_1k = _run_hdd(raw, n=n, block_size=1000, seed=1337, max_sweeps=200)

    print(f"GRID(side={side}): "
        f"\nRAM={ram['num']} "
        f"\nHDD_n={hdd_n['num']} "
        f"\nHDD_n4={hdd_n4['num']} "
        f"\nHDD_1k={hdd_1k['num']}\n")


def test_clusters_graph(tmp_path):
    raw = tmp_path / "clusters.edgelist"
    k, size = 5, 40
    n = k * size
    _n, _m = write_clusters_graph(raw, k_clusters=k, cluster_size=size, p_in=0.8, p_out=0.01, seed=123)

    #RAM
    adj = _load_adj_from_edgelist(raw)
    ram = _run_ram(adj, seed=1337, max_sweeps=200)

    #HDD
    hdd_n = _run_hdd(raw, n=n, block_size=n, seed=1337, max_sweeps=200)
    hdd_n4 = _run_hdd(raw, n=n, block_size=n //4, seed=1337, max_sweeps=200)
    hdd_1k = _run_hdd(raw, n=n, block_size=1000, seed=1337, max_sweeps=200)

    print(f"CLUSTERS(k={k},size={size}): "
        f"\nRAM={ram['num']} "
        f"\nHDD_n={hdd_n['num']} "
        f"\nHDD_n4={hdd_n4['num']} "
        f"\nHDD_1k={hdd_1k['num']}"
    )
