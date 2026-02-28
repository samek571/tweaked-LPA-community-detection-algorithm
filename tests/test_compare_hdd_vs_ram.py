"""Behavioral comparison of RAM LPA vs streaming (disk-backed) LPA.

This is an acceptance-style test, not an exact equivalence proof:
LPA is update-order sensitive, so community counts may differ slightly even
with the same seed. Assertions are intentionally loose on small graphs and
coarser on larger graphs.
"""

import time
import numpy as np
import pytest
from collections import Counter
from lpkit import generate_large_graph, label_propagation
from lpkit.stream import symmetrize_and_sort, split_sorted_sym_to_blocks, init_labels_memmap, stream_multi_sweep_parallel_blocks

@pytest.mark.parametrize("scale", [1, 10, 100])
def test_hdd_matches_ram(tmp_path, scale):
    """Streaming output should be in the same ballpark as RAM baseline across scales."""
    n, m = 1000 * scale, 3000 * scale

    raw         = tmp_path / f"g_{scale}.edgelist"
    labels      = tmp_path / f"g_{scale}.labels.npy"
    sorted_sym  = tmp_path / f"g_{scale}.sorted.sym"

    print(f"\n####### running tests at scale {scale} (n={n}, m={m}) ---")

    #gen Graph on disk
    generate_large_graph(str(raw), n=n, m=m, topology="random", seed=987654 + scale)

    #ram baseline - full adj list in memory build
    adj = [[] for _ in range(n)]
    with open(raw) as f:
        for tmp in f:
            u, v = map(int, tmp.split())
            adj[u].append(v)
            adj[v].append(u)

    t0 = time.time()
    labels_ram, _ = label_propagation(adj, seed=1337, max_sweeps=600)
    t_ram = time.time() - t0

    #hdd pipeline
    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    #i googled and this should estimate realistic HDD chunking, i am not expert in this tho...
    block_size = max(500, n // 20)

    block_paths = split_sorted_sym_to_blocks(
        str(sorted_sym),
        n=meta["n"],
        block_size=block_size,
        out_dir=str(tmp_path / f"blocks_{scale}"),
    )
    init_labels_memmap(str(labels), n=meta["n"])

    t0 = time.time()
    info_stream = stream_multi_sweep_parallel_blocks(
        block_paths,
        str(labels),
        n=meta["n"],
        block_size=block_size,
        seed=1337,
        max_sweeps=500,
        min_sweeps=1,
        tie_break="min",
        workers=1,
    )
    t_stream = time.time() -t0
    mm = np.lib.format.open_memmap(str(labels), mode="r+")

    def community_stats(labels_arr):
        c = Counter(int(l) for l in labels_arr)
        return len(c), sorted(c.values())

    n_ram, sizes_ram = community_stats(labels_ram)
    n_stream, sizes_stream = community_stats(mm)

    assert sum(sizes_ram) == sum(sizes_stream) == n #all vertices
    assert n_stream >= 2, "streaming converged onto a single label, which is fucking weird"

    if n <= 2_000:
        #smaller graphs are noisy- 3 vs 4 already gives 0.25
        diff = abs(n_ram - n_stream)
        ratio = diff / max(n_ram, n_stream)
        print(f"scale={scale}: RAM={n_ram}, HDD={n_stream}, diff={diff}, ratio={ratio:.3f}")
        assert ratio <= 0.35, f"Community ratio difference too large at n={n}: {ratio:.3f}"

    elif n <= 20_000:
        #small/medium graphs have similar results
        diff = abs(n_ram - n_stream)
        ratio = diff / max(n_ram, n_stream)
        print(f"scale={scale}: RAM={n_ram}, HDD={n_stream}, diff={diff}, ratio={ratio:.3f}")
        assert ratio <= 0.2, f"Community ratio difference too large at n={n}: {ratio:.3f}"
    else:
        #large graphs have larger deviation, depends on the max_sweeps param...
        print(f"scale={scale}: RAM={n_ram}, HDD={n_stream} "
              f"(RAM == micro-clusters, HDD == merged aggressively)")
        assert n_stream <= n_ram, (
            f"at large scale HDD should not fragment more than RAM "
            f"(RAM={n_ram}, HDD={n_stream})"
        )

    print(
        f"scale={scale}: RAM={t_ram:.2f}s, HDD={t_stream:.2f}s, "
        f"RAM_comms={n_ram}, HDD_comms={n_stream}, sweeps={info_stream.get('sweeps','?')}"
    )
