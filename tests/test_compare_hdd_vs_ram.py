import time
import numpy as np
import pytest
from collections import Counter
from lpkit import (
    generate_large_graph,
    label_propagation,
    symmetrize_and_sort,
    build_block_index,
    init_labels_memmap,
    stream_multi_sweep,
)

@pytest.mark.parametrize("scale", [1, 10, 100])
def test_hdd_matches_ram(tmp_path, scale):
    n, m = 1000 * scale, 3000 * scale

    raw         = tmp_path / f"g_{scale}.edgelist"
    idx         = tmp_path / f"g_{scale}.blockidx.npy"
    labels      = tmp_path / f"g_{scale}.labels.npy"
    sorted_sym  = tmp_path / f"g_{scale}.sorted.sym"

    print(f"\n####### running tests at scale {scale} (n={n}, m={m}) ---")

    #gen g on disk
    generate_large_graph(str(raw), n=n, m=m, topology="random", seed=987654 + scale)

    #ram baseline - full adj list in memory
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
    #i googled and this should estimate realistic HDD chunking, i am not expert in that expertise so dont take my word for it
    block_size = max(500, n // 20)

    build_block_index(str(sorted_sym), n=meta["n"], block_size=block_size, index_path=str(idx))
    init_labels_memmap(str(labels), n=meta["n"])

    t0 = time.time()
    info_stream = stream_multi_sweep(
        str(sorted_sym),
        str(idx),
        str(labels),
        n=meta["n"],
        block_size=block_size,
        seed=1337,
        max_sweeps=500,
        tie_break="min"
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

    if n <= 20_000:
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
